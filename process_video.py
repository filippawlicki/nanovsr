"""Video super-resolution with a pretrained NanoVSR checkpoint.

Upscales a video 4x with NanoVSR and writes it back, preserving audio
and metadata. The model is a *video* (recurrent, bidirectional) network,
so frames are processed in temporal windows.

Codec handling:
    * A quick ``ffprobe`` check decides the decoder.
    * AV1 videos are decoded/encoded with PyAV (OpenCV mis-handles AV1).
      PyAV is imported lazily, only on the AV1 path.
    * Everything else is decoded/encoded with OpenCV; audio + metadata are
      muxed back in with ``ffmpeg`` (OpenCV cannot write audio).

Quality choices:
    * Overlapping windows with triangular weight blending (no seams).
    * FP16 inference on CUDA.
    * LANCZOS downscaling if a target shorter-side resolution is set.

Safety:
    * All output is first written to a temp file, then atomically renamed
      to ``--output`` at the very end, so a crash never corrupts the
      destination.

Example:
    python process_video.py --input in.mp4 --output out.mp4 --target 1080
"""

import os
import shutil
import argparse
import subprocess
import tempfile
import fractions
import queue
import threading

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.nanovsr import NanoVSR
from utils import detect_model_config, load_checkpoint_state_dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def weight(length: int, idx: int) -> float:
    """Triangular window weight peaking at the centre frame.

    Used to blend overlapping temporal windows: centre frames (with full
    forward+backward context) are trusted more than edge frames.

    Args:
        length: Number of frames in the current window.
        idx: Position of this frame inside the window.

    Returns:
        A weight in (0, 1], 1.0 at the centre.
    """
    if length <= 1:
        return 1.0
    centre = (length - 1) / 2.0
    return 1.0 - abs(idx - centre) / (centre + 1e-9)


def compute_output_size(h: int, w: int, target: int | None) -> tuple[int, int]:
    """4x upscale, then optionally downscale to cap the shorter side.

    Args:
        h: Input frame height.
        w: Input frame width.
        target: If set, the shorter side of the output is capped to this
            value (aspect preserved). None keeps the full 4x result.

    Returns:
        (out_h, out_w) of the final frame.
    """
    oh, ow = h * 4, w * 4
    if target and min(oh, ow) > target:
        scale = target / min(oh, ow)
        oh = max(1, round(oh * scale))
        ow = max(1, round(ow * scale))
    return oh, ow


def downscale(rgb: np.ndarray, target: int | None) -> np.ndarray:
    """LANCZOS-downscale an RGB frame to cap its shorter side.

    Args:
        rgb: uint8 RGB frame [H, W, 3].
        target: Shorter-side cap, or None to skip.

    Returns:
        The (possibly resized) uint8 RGB frame.
    """
    if target is None:
        return rgb
    h, w = rgb.shape[:2]
    if min(h, w) <= target:
        return rgb
    scale = target / min(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def fourcc_for(ext: str) -> str:
    """Pick an OpenCV fourcc code from the output file extension."""
    ext = ext.lower()
    if ext == ".avi":
        return "XVID"
    if ext == ".mkv":
        print("[warn] .mkv via OpenCV 'MP4V' may be unreliable; prefer .mp4")
    return "MP4V"


def _clean_env() -> dict:
    """Env without LD_LIBRARY_PATH.

    ffmpeg/ffprobe can crash with a glib ``symbol lookup error`` when
    ``LD_LIBRARY_PATH`` is polluted by other apps (e.g. a Flatpak
    injecting conflicting libglib). Dropping the var lets the system
    libraries load correctly.
    """
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    return env


def detect_av1(path: str) -> bool:
    """Return True if the first video stream is AV1 (via ffprobe)."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True, env=_clean_env(),
        )
        return res.stdout.strip().lower() == "av1"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffprobe unavailable or errored: assume not AV1 (OpenCV path).
        return False


def mux_audio(in_path: str, video_tmp: str, out_path: str) -> None:
    """Copy original audio + metadata onto a silent SR video via ffmpeg.

    Args:
        in_path: Original (source) video, used for audio + metadata.
        video_tmp: Silent super-resolved video produced locally.
        out_path: Final destination (atomic replace at the end).
    """
    tmp_mux = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        "ffmpeg", "-y",
        "-i", video_tmp,
        "-i", in_path,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-map_metadata", "1",
        "-shortest", tmp_mux,
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True, env=_clean_env())
    if ret.returncode == 0:
        shutil.move(tmp_mux, out_path)
    else:
        print("[warn] ffmpeg audio mux failed; writing video without audio.")
        print(ret.stderr.strip().splitlines()[-15:])
        shutil.move(video_tmp, out_path)
    for f in (video_tmp, tmp_mux):
        if os.path.exists(f) and not os.path.samefile(f, out_path):
            os.remove(f)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: str, device: torch.device) -> NanoVSR:
    """Load a NanoVSR checkpoint, auto-detect arch, switch to deploy mode.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint.
        device: Target torch device.

    Returns:
        The eval-mode NanoVSR model on ``device``.
    """
    print(f"Loading model from {checkpoint_path}...")
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    num_feat, num_blocks = detect_model_config(state_dict)
    print(f"Detected architecture: num_feat={num_feat}, num_blocks={num_blocks}")
    model = NanoVSR(num_feat=num_feat or 32, num_blocks=num_blocks or 8)
    model.load_state_dict(state_dict, strict=False)
    model.switch_to_deploy()
    model.eval()
    return model.to(device)


@torch.inference_mode()
def process_window(
    model: NanoVSR,
    rgb_frames: list[np.ndarray],
    device: torch.device,
    use_fp16: bool,
) -> list[np.ndarray]:
    """Upscale a list of RGB frames 4x with the model.

    Args:
        model: Deploy-mode NanoVSR.
        rgb_frames: List of uint8 RGB frames [H, W, 3].
        device: Torch device.
        use_fp16: Enable FP16 autocast (CUDA only).

    Returns:
        List of uint8 RGB frames [4H, 4W, 3].
    """
    tensors = [torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0 for f in rgb_frames]
    batch = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    if use_fp16 and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
        out = out.float()
    else:
        out = model(batch)

    # Do clamp/scale/round/uint8 on the GPU, then a SINGLE host transfer for
    # the whole window (was one .cpu() sync per frame -> T syncs, which stalled
    # the GPU between windows and caused the utilisation sawtooth).
    out = out.squeeze(0).clamp(0, 1).mul_(255.0).round_().to(torch.uint8)  # [T,3,4H,4W]
    arr = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [T,4H,4W,3]
    return [arr[i] for i in range(arr.shape[0])]


# --------------------------------------------------------------------------- #
# Frame accumulator (overlap + blend)
# --------------------------------------------------------------------------- #
class FrameAccumulator:
    """Accumulates weighted SR frames across overlapping windows.

    Each output frame may be produced by two neighbouring windows; the
    weighted sum is finalised (and handed to the writer) as soon as the
    last contributing window has been processed.
    """

    def __init__(self, writer: "VideoWriterBase", stride: int) -> None:
        self.writer = writer
        self.stride = stride
        self.acc: dict[int, np.ndarray] = {}
        self.wsum: dict[int, float] = {}

    def add(self, srs: list[np.ndarray], start: int) -> None:
        length = len(srs)
        cur_k = start // self.stride
        for j, sr in enumerate(srs):
            g = start + j
            w = weight(length, j)
            if g not in self.acc:
                self.acc[g] = sr.astype(np.float32) * w
                self.wsum[g] = w
            else:
                self.acc[g] += sr.astype(np.float32) * w
                self.wsum[g] += w
            # Finalised once the current window is the last one covering g.
            if (g // self.stride) == cur_k:
                self._finalize(g)

    def flush(self) -> None:
        for g in sorted(self.acc):
            self._finalize(g)

    def _finalize(self, g: int) -> None:
        if g not in self.acc:
            return
        frame = (self.acc[g] / self.wsum[g]).clip(0, 255).round().astype(np.uint8)
        self.writer.write(frame)
        del self.acc[g], self.wsum[g]


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
class VideoWriterBase:
    def write(self, rgb: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVWriter(VideoWriterBase):
    """Writes silent video with OpenCV (RGB frames converted to BGR)."""

    def __init__(self, path: str, fps: float, size: tuple[int, int], ext: str) -> None:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_for(ext))
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open {path}")

    def write(self, rgb: np.ndarray) -> None:
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


class PyAVWriter(VideoWriterBase):
    """Writes video-only with PyAV at a fixed size.

    Audio + metadata are muxed back in afterwards via ``mux_audio``
    (PyAV's container metadata is read-only in this version, so ``ffmpeg``
    does the muxing — same as the OpenCV path).
    """

    def __init__(self, path: str, fps: float, size: tuple[int, int],
                 preset: str = "medium", crf: int = 18, nvenc: bool = False) -> None:
        import av  # lazy: only imported on the AV1 path

        self.av = av
        self.container = av.open(path, "w")
        rate = fractions.Fraction(fps).limit_denominator(1001)
        if nvenc:
            # Hardware encode: offloads the pipeline bottleneck to NVENC.
            vs = self.container.add_stream("h264_nvenc", rate=rate)
            vs.width, vs.height = size
            vs.pix_fmt = "yuv420p"
            # p1..p7 (fast..slow); rc=vbr with cq acting like crf.
            vs.options = {"preset": "p5", "rc": "vbr", "cq": str(crf), "b": "0"}
        else:
            vs = self.container.add_stream("libx264", rate=rate)
            vs.width, vs.height = size
            vs.pix_fmt = "yuv420p"
            vs.options = {"preset": preset, "crf": str(crf)}
        self.vs = vs
        self._idx = 0

    def write(self, rgb: np.ndarray) -> None:
        # from_ndarray already sizes the frame to rgb's HxW (= stream size),
        # and VideoFrame.width/height are read-only, so don't reassign them.
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self._idx
        self._idx += 1
        for packet in self.vs.encode(frame):
            self.container.mux(packet)

    def flush_video(self) -> None:
        for packet in self.vs.encode(None):
            self.container.mux(packet)

    def close(self) -> None:
        self.container.close()


class Consumer:
    """Background consumer: downscale + overlap-blend + encode off the GPU thread.

    The GPU thread only runs inference and hands each raw 4x window (already
    on the host as uint8) to this consumer. All remaining CPU work — LANCZOS
    downscaling, weighted blending and libx264 encoding — happens here, so the
    GPU is never blocked waiting on it (kills the utilisation sawtooth).
    """

    def __init__(self, inner: VideoWriterBase, stride: int, target: int | None,
                 maxsize: int = 8) -> None:
        # Encode on its own thread so this consumer only does downscale+blend;
        # two-stage pipeline keeps the GPU fed even when the encoder is slow.
        self.inner = ThreadedWriter(inner)
        self.acc = FrameAccumulator(self.inner, stride)
        self.target = target
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._err: BaseException | None = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            srs, pos = item
            try:
                srs = [downscale(sr, self.target) for sr in srs]
                self.acc.add(srs, pos)
            except BaseException as e:
                self._err = e
                break

    def add(self, srs: list[np.ndarray], pos: int) -> None:
        if self._err:
            raise self._err
        self.q.put((srs, pos))

    def finish(self) -> None:
        self.q.put(None)
        self.t.join()
        if self._err:
            raise self._err
        self.acc.flush()
        self.inner.flush_video()
        self.inner.close()


class ThreadedWriter(VideoWriterBase):
    """Wraps a writer so encoding runs on a background thread.

    Frame encoding (libx264, preset slow) is CPU-heavy and otherwise
    stalls the GPU between windows. Offloading it to a worker thread keeps
    the GPU continuously fed (removes the utilisation sawtooth).
    """

    def __init__(self, inner: VideoWriterBase, maxsize: int = 16) -> None:
        self.inner = inner
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._err: BaseException | None = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self.inner.write(item)
            except BaseException as e:  # surface to main thread
                self._err = e
                break

    def write(self, rgb: np.ndarray) -> None:
        if self._err:
            raise self._err
        self.q.put(rgb)

    def flush_video(self) -> None:
        self.q.put(None)
        self.t.join()
        if self._err:
            raise self._err
        self.inner.flush_video()

    def close(self) -> None:
        self.inner.close()


# --------------------------------------------------------------------------- #
# Windowed processing driver (shared by both backends)
# --------------------------------------------------------------------------- #
def iter_windows(
    reader,
    consumer: "Consumer",
    model: NanoVSR,
    device: torch.device,
    use_fp16: bool,
    window: int,
    stride: int,
    pbar: tqdm,
) -> None:
    """Pull frames from ``reader``, upscale in windows, feed the consumer.

    Downscaling and blending happen in the consumer thread; this loop only
    decodes (via reader), runs the GPU forward pass and enqueues raw output.

    Args:
        reader: Callable returning the next RGB frame or None at EOF.
        consumer: Background downscale/blend/encode consumer.
        model: NanoVSR model.
        device: Torch device.
        use_fp16: FP16 autocast enabled.
        window: Temporal window size T.
        stride: Step between windows (T for no overlap, T//2 for overlap).
        pbar: Progress bar.
    """
    buf: list[np.ndarray] = []
    pos = 0
    while True:
        while len(buf) < window:
            fr = reader()
            if fr is None:
                break
            buf.append(fr)
        if not buf:
            break

        length = min(window, len(buf))
        srs = process_window(model, buf[:length], device, use_fp16)
        consumer.add(srs, pos)

        step = stride if (stride != window and length == window) else length
        del buf[:step]
        pos += step
        pbar.update(step)

        if length < window:
            break


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def process_opencv(
    model: NanoVSR,
    device: torch.device,
    use_fp16: bool,
    target: int | None,
    window: int,
    overlap: bool,
    in_path: str,
    out_path: str,
    preset: str = "medium",
    crf: int = 18,
    nvenc: bool = False,
) -> None:
    """Decode with OpenCV, super-resolve, mux audio/metadata via ffmpeg."""
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, probe = cap.read()
    if not ok:
        raise RuntimeError(f"No frames in {in_path}")
    h, w = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sr_h, sr_w = compute_output_size(h, w, target)
    stride = window // 2 if overlap else window

    tmp_silent = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # PyAV/libx264 (crf18) writer; downscale+blend+encode run in the consumer
    # thread so the GPU stays fed.
    consumer = Consumer(
        PyAVWriter(tmp_silent, fps, (sr_w, sr_h), preset, crf, nvenc),
        stride, target)

    # Prefetch decode on a background thread so the GPU never waits on cv2.
    prefetch_q: queue.Queue = queue.Queue(maxsize=window * 2)

    def _decode_worker() -> None:
        while True:
            ok, fr = cap.read()
            if not ok:
                prefetch_q.put(None)
                return
            prefetch_q.put(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

    dec_t = threading.Thread(target=_decode_worker, daemon=True)
    dec_t.start()

    def reader() -> np.ndarray | None:
        return prefetch_q.get()

    try:
        with tqdm(unit="fr", desc="Upscaling (OpenCV)") as pbar:
            iter_windows(reader, consumer, model, device, use_fp16,
                         window, stride, pbar)
        consumer.finish()
        dec_t.join()
        cap.release()

        # Copy original audio + metadata back in (OpenCV can't write audio).
        mux_audio(in_path, tmp_silent, out_path)
    finally:
        if cap.isOpened():
            cap.release()
        if os.path.exists(tmp_silent) and not os.path.samefile(tmp_silent, out_path):
            os.remove(tmp_silent)


def process_av1(
    model: NanoVSR,
    device: torch.device,
    use_fp16: bool,
    target: int | None,
    window: int,
    overlap: bool,
    in_path: str,
    out_path: str,
    preset: str = "medium",
    crf: int = 18,
    nvenc: bool = False,
) -> None:
    """Decode AV1 with PyAV, super-resolve, mux audio/metadata via ffmpeg."""
    import av  # lazy: only imported on the AV1 path

    in_c = av.open(in_path)
    if not in_c.streams.video:
        in_c.close()
        raise RuntimeError(f"No video stream in {in_path}")
    vs_in = in_c.streams.video[0]
    fps = float(vs_in.average_rate)
    h, w = vs_in.height, vs_in.width
    sr_h, sr_w = compute_output_size(h, w, target)
    stride = window // 2 if overlap else window

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # downscale+blend+encode run in the consumer thread so the GPU stays fed.
    consumer = Consumer(
        PyAVWriter(tmp, fps, (sr_w, sr_h), preset, crf, nvenc),
        stride, target)

    pbar = tqdm(unit="fr", desc="Upscaling (PyAV/AV1)")
    pending: list[np.ndarray] = []
    pos = 0
    try:
        for packet in in_c.demux(vs_in):
            for frame in packet.decode():
                pending.append(frame.to_ndarray(format="rgb24"))
            while len(pending) >= window:
                chunk = pending[:window]
                consumer.add(process_window(model, chunk, device, use_fp16), pos)
                step = stride if overlap else window
                del pending[:step]
                pos += step
                pbar.update(step)
        if pending:
            consumer.add(process_window(model, pending, device, use_fp16), pos)
            pbar.update(len(pending))
        pbar.close()
    except av.error.EOFError:
        # demux may raise at the very end; flush whatever we have.
        if pending:
            consumer.add(process_window(model, pending, device, use_fp16), pos)
            pbar.update(len(pending))
        pbar.close()
    except Exception:
        pbar.close()
        try:
            in_c.close()
        except Exception:
            pass
        if os.path.exists(tmp) and not os.path.samefile(tmp, out_path):
            os.remove(tmp)
        raise

    consumer.finish()
    in_c.close()

    # Copy original audio + metadata back in (PyAV container metadata
    # is read-only in this version, so ffmpeg does the muxing).
    mux_audio(in_path, tmp, out_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NanoVSR 4x video super-resolution.")
    p.add_argument("--input", required=True, help="Input video file.")
    p.add_argument("--output", required=True, help="Output video file.")
    p.add_argument("--model", default=os.path.join(SCRIPT_DIR, "checkpoints", "nanovsr_644k.pth"),
                   help="Path to a NanoVSR .pth checkpoint.")
    p.add_argument("--batch", type=int, default=15, help="Temporal window size T.")
    p.add_argument("--target", type=int, default=None,
                   help="Cap the SHORTER side of the output (e.g. 1080). Omit for full 4x.")
    p.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=False, help="Enable FP16 on CUDA (default: FP32).")
    p.add_argument("--overlap", dest="overlap", action="store_true", default=False, help="Overlap windows + blend (default: off, faster).")
    p.add_argument("--lanczos", action="store_true", default=False,
                   help="Bypass the network: upscale with ffmpeg LANCZOS (for comparison).")
    p.add_argument("--preset", default="medium",
                   help="libx264 preset (ultrafast..veryslow). Faster = less encoder "
                        "bottleneck. Default: medium.")
    p.add_argument("--crf", type=int, default=18, help="Encoder quality (lower=better). Default: 18.")
    p.add_argument("--nvenc", action="store_true", default=False,
                   help="Hardware H.264 encode via NVENC (offloads the encoder bottleneck).")
    return p.parse_args()


def probe_size(path: str) -> tuple[int, int]:
    """Return (height, width) of the first video stream via ffprobe."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, env=_clean_env(),
    )
    w, h = (int(x) for x in res.stdout.strip().split(","))
    return h, w


def process_lanczos(in_path: str, out_path: str, target: int | None) -> None:
    """Bypass the network: 4x (then target cap) upscale via ffmpeg LANCZOS.

    Used purely for comparison against the NanoVSR output.

    Args:
        in_path: Input video.
        out_path: Output video.
        target: Shorter-side cap (same semantics as the network path).
    """
    h, w = probe_size(in_path)
    oh, ow = compute_output_size(h, w, target)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf", f"scale={ow}:{oh}:flags=lanczos",
        "-c:v", "libx264", "-crf", "18",
        # Map only video + audio (drop timecode/data streams mp4 can't mux).
        "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0", "-c:a", "copy",
        tmp,
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True, env=_clean_env())
    if ret.returncode == 0:
        shutil.move(tmp, out_path)
    else:
        print("[warn] ffmpeg LANCZOS upscale failed:")
        print(ret.stderr.strip().splitlines()[-15:])
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError("ffmpeg LANCZOS upscale failed")


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"[error] input not found: {args.input}")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("[error] --output must differ from --input")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    if args.batch < 1:
        raise SystemExit("[error] --batch must be >= 1")
    if args.target is not None and args.target < 1:
        raise SystemExit("[error] --target must be >= 1")

    if args.lanczos:
        # Pure ffmpeg upscale, no model — for A/B comparison.
        process_lanczos(args.input, args.output, args.target)
        print(f"Done (LANCZOS) -> {args.output}")
        return

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, device)
    use_fp16 = args.fp16 and device.type == "cuda"

    is_av1 = detect_av1(args.input)
    print(f"AV1 detected: {is_av1}")

    if is_av1:
        process_av1(model, device, use_fp16, args.target,
                     args.batch, args.overlap, args.input, args.output,
                     args.preset, args.crf, args.nvenc)
    else:
        process_opencv(model, device, use_fp16, args.target,
                       args.batch, args.overlap, args.input, args.output,
                       args.preset, args.crf, args.nvenc)

    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
