"""NanoVSR 4x video super-resolution.

Upscales a video with a recurrent NanoVSR checkpoint, preserving audio and
metadata. AV1 is decoded/encoded via PyAV (lazy import); everything else via
OpenCV decode + PyAV encode, with audio/metadata muxed back by ffmpeg.
Output is written to a temp file in ``--output``'s dir, then atomically
renamed. Optional LANCZOS downscale caps the shorter side (``--target``).
"""

import os
import argparse
import subprocess
import tempfile
import logging
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

log = logging.getLogger("nanovsr")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def compute_output_size(h: int, w: int, target: int | None) -> tuple[int, int]:
    """4x upscale, then optionally cap the shorter side to ``target``."""
    oh, ow = h * 4, w * 4
    if target and min(oh, ow) > target:
        scale = target / min(oh, ow)
        oh = round(oh * scale)
        ow = round(ow * scale)
    # libx264 / h264_nvenc force yuv420p, which needs even dimensions.
    oh = max(2, (oh // 2) * 2)
    ow = max(2, (ow // 2) * 2)
    return oh, ow


def downscale(rgb: np.ndarray, target: int | None) -> np.ndarray:
    """LANCZOS-downscale an RGB frame if its shorter side exceeds ``target``."""
    if target is None:
        return rgb
    h, w = rgb.shape[:2]
    if min(h, w) <= target:
        return rgb
    scale = target / min(h, w)
    nh, nw = round(h * scale), round(w * scale)
    nh, nw = max(2, (nh // 2) * 2), max(2, (nw // 2) * 2)
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def _clean_env() -> dict:
    """Copy env without LD_LIBRARY_PATH (avoids glib symbol clashes)."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    return env


def detect_av1(path: str) -> bool:
    """True if the first video stream is AV1 (via ffprobe)."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True, timeout=300,
            env=_clean_env(),
        )
        return res.stdout.strip().lower() == "av1"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # ffprobe unavailable, errored, or hung: assume not AV1 (OpenCV path).
        return False


def mux_audio(in_path: str, video_tmp: str, out_path: str) -> None:
    """Mux original audio + metadata onto the silent SR video via ffmpeg."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    tmp_mux = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4",
                                          dir=out_dir).name
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
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                             env=_clean_env())
    except subprocess.TimeoutExpired:
        ret = None
    if ret is not None and ret.returncode == 0:
        os.replace(tmp_mux, out_path)
    else:
        log.warning("ffmpeg audio mux failed; writing video without audio.")
        if ret is not None:
            log.warning("\n".join(ret.stderr.strip().splitlines()[-15:]))
        if os.path.exists(video_tmp):
            os.replace(video_tmp, out_path)
    for f in (video_tmp, tmp_mux):
        if os.path.exists(f) and not os.path.samefile(f, out_path):
            os.remove(f)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: str, device: torch.device) -> NanoVSR:
    """Load a NanoVSR checkpoint, auto-detect arch, switch to deploy mode."""
    print(f"Loading model from {checkpoint_path}...")
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    num_feat, num_blocks = detect_model_config(state_dict)
    if num_feat is None or num_blocks is None:
        raise RuntimeError(
            f"Could not auto-detect model architecture from {checkpoint_path} "
            f"(num_feat={num_feat}, num_blocks={num_blocks}). The checkpoint "
            f"uses unexpected key names; a wrong arch would load silently and "
            f"produce garbage output."
        )
    print(f"Detected architecture: num_feat={num_feat}, num_blocks={num_blocks}")
    model = NanoVSR(num_feat=num_feat, num_blocks=num_blocks)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint weights do not match the detected architecture "
            f"(num_feat={num_feat}, num_blocks={num_blocks}): {e}"
        ) from e
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
    """Upscale a list of RGB frames 4x and return uint8 RGB frames."""
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
# Writers
# --------------------------------------------------------------------------- #
class VideoWriterBase:
    def write(self, rgb: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class PyAVWriter(VideoWriterBase):
    """Writes video-only with PyAV at a fixed size (yuv420p)."""

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
            vs.options = {"preset": "p5", "rc": "vbr", "cq": str(crf)}
        else:
            vs = self.container.add_stream("libx264", rate=rate)
            vs.width, vs.height = size
            vs.pix_fmt = "yuv420p"
            vs.options = {"preset": preset, "crf": str(crf)}
        self.vs = vs
        self._idx = 0

    def write(self, rgb: np.ndarray) -> None:
        # The stream size is fixed at construction; the consumer downscales each
        # frame to match, but if a 1px rounding drift occurs this resize makes
        # the writer authoritative instead of crashing the encoder.
        h, w = self.vs.height, self.vs.width
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h))
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
    """Background consumer: downscale + encode off the GPU thread.

    The GPU thread only runs inference and hands each raw 4x window to this
    consumer; LANCZOS downscale + libx264 encode happen here so the GPU
    is never blocked on CPU work.
    """

    def __init__(self, inner: VideoWriterBase, target: int | None,
                 maxsize: int = 8) -> None:
        self.inner = ThreadedWriter(inner)
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
                for sr in srs:
                    self.inner.write(downscale(sr, self.target))
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
        self.inner.flush_video()
        self.inner.close()


class ThreadedWriter(VideoWriterBase):
    """Runs frame encoding on a background thread (offloads the GPU)."""

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
                # Discard anything still queued so a producer calling put()
                # can never block forever on a worker that has already died.
                while True:
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break
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
    pbar: tqdm,
) -> None:
    """Decode frames via ``reader``, upscale in windows of ``window``, enqueue."""
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

        del buf[:length]
        pos += length
        pbar.update(length)

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

    out_dir = os.path.dirname(os.path.abspath(out_path))
    tmp_silent = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4",
                                             dir=out_dir).name
    consumer = Consumer(
        PyAVWriter(tmp_silent, fps, (sr_w, sr_h), preset, crf, nvenc),
        target)

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
                         window, pbar)
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
    fps = float(vs_in.average_rate) or 30.0
    h, w = vs_in.height, vs_in.width
    sr_h, sr_w = compute_output_size(h, w, target)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=out_dir).name
    consumer = Consumer(
        PyAVWriter(tmp, fps, (sr_w, sr_h), preset, crf, nvenc),
        target)

    # Prefetch decode on a background thread so the GPU never waits on PyAV.
    frame_q: queue.Queue = queue.Queue(maxsize=window * 2)
    decode_err: list[BaseException] = []

    def _decode_worker() -> None:
        try:
            for packet in in_c.demux(vs_in):
                for frame in packet.decode():
                    frame_q.put(frame.to_ndarray(format="rgb24"))
        except av.error.EOFError:
            pass
        except BaseException as e:  # surface unexpected decode errors
            decode_err.append(e)
        finally:
            frame_q.put(None)

    dec_t = threading.Thread(target=_decode_worker, daemon=True)
    dec_t.start()

    def reader() -> np.ndarray | None:
        return frame_q.get()

    try:
        with tqdm(unit="fr", desc="Upscaling (PyAV/AV1)") as pbar:
            iter_windows(reader, consumer, model, device, use_fp16, window, pbar)
        consumer.finish()
        dec_t.join()
        if decode_err:
            raise decode_err[0]
        in_c.close()
        mux_audio(in_path, tmp, out_path)
    finally:
        if os.path.exists(tmp) and not os.path.samefile(tmp, out_path):
            try:
                os.remove(tmp)
            except OSError:
                pass


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
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True, timeout=300,
            env=_clean_env(),
        )
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as e:
        raise RuntimeError(
            f"ffprobe failed to read {path}; is ffmpeg/ffprobe installed?"
        ) from e
    w, h = (int(x) for x in res.stdout.strip().split(","))
    return h, w


def process_lanczos(in_path: str, out_path: str, target: int | None,
                    preset: str = "medium", crf: int = 18) -> None:
    """4x (then target cap) upscale via ffmpeg LANCZOS — for A/B comparison."""
    h, w = probe_size(in_path)
    oh, ow = compute_output_size(h, w, target)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=out_dir).name
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf", f"scale={ow}:{oh}:flags=lanczos",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        # Map only video + audio (drop timecode/data streams mp4 can't mux).
        "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0", "-c:a", "copy",
        tmp,
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                             env=_clean_env())
    except subprocess.TimeoutExpired:
        ret = None
    if ret is not None and ret.returncode == 0:
        os.replace(tmp, out_path)
    else:
        log.warning("ffmpeg LANCZOS upscale failed:")
        if ret is not None:
            log.warning("\n".join(ret.stderr.strip().splitlines()[-15:]))
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError("ffmpeg LANCZOS upscale failed")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
        process_lanczos(args.input, args.output, args.target, args.preset, args.crf)
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
                     args.batch, args.input, args.output,
                     args.preset, args.crf, args.nvenc)
    else:
        process_opencv(model, device, use_fp16, args.target,
                       args.batch, args.input, args.output,
                       args.preset, args.crf, args.nvenc)

    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
