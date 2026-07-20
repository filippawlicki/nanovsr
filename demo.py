"""Upscale a video (or a directory of frames) 4x with NanoVSR.

Frames are processed in chunks (default T=15, the paper's edge setting) to bound
memory on long videos. --compare writes a side-by-side LR vs. NanoVSR video.

If ffmpeg is on PATH, output is encoded with libx264 (CRF 18) and the original
audio/metadata are muxed back in; otherwise it falls back to OpenCV's mp4v
writer (video only). No extra Python dependencies are required either way.

Examples:
    python demo.py --checkpoint checkpoints/nanovsr_644k.pth --input clip.mp4
    python demo.py --checkpoint checkpoints/nanovsr_644k.pth --input clip.mp4 --compare
    python demo.py --checkpoint checkpoints/nanovsr_644k.pth --input frames_dir/ --fps 30
"""

import os
import glob
import time
import shutil
import argparse
import subprocess
import contextlib
from itertools import islice

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import load_model, img2tensor_batch

SCALE = 4


def open_input(input_path, fps_override):
    """Return (frame_iterator, total_frames_or_None, fps) for a video file or a frame directory."""
    if os.path.isdir(input_path):
        frame_paths = sorted(glob.glob(os.path.join(input_path, '*.png')) +
                             glob.glob(os.path.join(input_path, '*.jpg')))
        if not frame_paths:
            raise FileNotFoundError(f"No .png/.jpg frames found in {input_path}")

        def frames():
            for p in frame_paths:
                yield cv2.imread(p, cv2.IMREAD_COLOR)

        return frames(), len(frame_paths), fps_override or 25.0

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    return frames(), total, fps


def chunked(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def tensor2img_batch_bgr(chunk):
    """[T, 3, H, W] float in [0, 1] -> contiguous [T, H, W, 3] uint8 BGR.

    Clamp/scale/round/uint8 run on-device, then a single host transfer moves the
    whole window at once (vs. one .cpu() sync per frame, which stalls the GPU
    between windows).
    """
    arr = (chunk.clamp(0, 1).mul_(255.0).round_().to(torch.uint8)
           .permute(0, 2, 3, 1)
           .contiguous().cpu().numpy())
    return np.ascontiguousarray(arr[..., ::-1])


def draw_label(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, img.shape[0] / 720.0)
    thickness = max(1, int(round(2 * scale)))
    origin = (int(20 * scale), int(45 * scale))
    cv2.putText(img, text, origin, font, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def make_comparison(lq_bgr, sr_bgr):
    h, w = sr_bgr.shape[:2]
    base = cv2.resize(lq_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    draw_label(base, f'Input (LR x{SCALE})')
    draw_label(sr_bgr, 'NanoVSR')
    divider = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.hstack([base, divider, sr_bgr])


def default_output_path(input_path, compare):
    base = input_path.rstrip('/\\')
    stem = os.path.splitext(base)[0] if os.path.isfile(base) else base
    suffix = f'_x{SCALE}_compare.mp4' if compare else f'_x{SCALE}.mp4'
    return stem + suffix


class FrameSink:
    """Writes BGR frames to an mp4.

    With ffmpeg on PATH: pipes raw frames to a single libx264 encode and, when
    ``audio_from`` is given, muxes that source's audio/metadata in the same pass.
    Without ffmpeg: falls back to OpenCV's mp4v writer (video only).
    """

    def __init__(self, path, fps, size, crf, audio_from=None):
        self.path = path
        self.mode = None
        w, h = size

        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            cmd = [ffmpeg, '-y', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
            if audio_from:
                cmd += ['-i', audio_from, '-map', '0:v:0', '-map', '1:a:0?',
                        '-map_metadata', '1', '-c:a', 'copy', '-shortest']
            cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
                    '-pix_fmt', 'yuv420p', path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.mode = 'ffmpeg+audio' if audio_from else 'ffmpeg'
        else:
            self.proc = None
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            if not self.cv.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")
            self.mode = 'mp4v'

    def write(self, bgr):
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
        else:
            self.cv.write(bgr)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg encode failed (exit {self.proc.returncode}) for {self.path}")
        else:
            self.cv.release()


def main():
    parser = argparse.ArgumentParser(description=f'NanoVSR {SCALE}x video upscaling demo')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a NanoVSR checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Low-resolution video file, or a directory of PNG/JPG frames')
    parser.add_argument('--output', type=str, default=None,
                        help=f"Output video path (default: '<input>_x{SCALE}.mp4' next to the input)")
    parser.add_argument('--chunk_size', type=int, default=15,
                        help='Temporal window T per forward pass (paper edge setting: 15); '
                             'memory scales with chunk size and resolution')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output frame rate (default: input FPS, or 25 for frame directories)')
    parser.add_argument('--max_frames', type=int, default=None, help='Process only the first N frames')
    parser.add_argument('--compare', action='store_true',
                        help='Write a side-by-side LR vs. NanoVSR comparison video')
    parser.add_argument('--crf', type=int, default=18,
                        help='libx264 quality when ffmpeg is used (lower=better, ignored for mp4v fallback)')
    parser.add_argument('--no_audio', action='store_true',
                        help="Don't mux the source video's audio/metadata into the output")
    parser.add_argument('--fp16', action='store_true', help='Half-precision inference (CUDA only)')
    parser.add_argument('--no_deploy', action='store_true',
                        help='Skip structural reparameterization (keep the multi-branch training topology)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available - running on CPU, this will be slow.")
    use_fp16 = args.fp16 and device == 'cuda'

    model = load_model(args.checkpoint, device=device, use_switch_to_deploy=not args.no_deploy)

    frames_iter, total_frames, fps = open_input(args.input, args.fps)
    if args.max_frames is not None:
        frames_iter = islice(frames_iter, args.max_frames)
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    output_path = args.output or default_output_path(args.input, args.compare)
    audio_from = args.input if (os.path.isfile(args.input) and not args.no_audio) else None

    sink = None
    processed = 0
    start = time.time()
    autocast_ctx = torch.autocast('cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()
    pbar = tqdm(total=total_frames, unit='frame', desc='Upscaling',
                ncols=shutil.get_terminal_size().columns - 1)

    with torch.inference_mode():
        for chunk in chunked(frames_iter, args.chunk_size):
            if processed == 0:
                h, w = chunk[0].shape[:2]
                tqdm.write(f"Input: {w}x{h} @ {fps:.2f} FPS -> output: {w * SCALE}x{h * SCALE}")
                if h * w > 540 * 960:
                    tqdm.write("Note: NanoVSR is designed for low-resolution input (e.g. 180x320 - 270x480); "
                               "large inputs are slow and memory-hungry. Consider --chunk_size 5 or a smaller clip.")

            lq_tensor = img2tensor_batch(chunk).unsqueeze(0).to(device)

            with autocast_ctx:
                output = model(lq_tensor)
            sr_batch = tensor2img_batch_bgr(output.float().squeeze(0))

            for idx in range(sr_batch.shape[0]):
                sr_img = sr_batch[idx]
                out_frame = make_comparison(chunk[idx], sr_img) if args.compare else sr_img

                if sink is None:
                    out_h, out_w = out_frame.shape[:2]
                    sink = FrameSink(output_path, fps, (out_w, out_h), args.crf, audio_from=audio_from)

                sink.write(out_frame)
                processed += 1
                pbar.update(1)

    pbar.close()
    if processed == 0:
        raise RuntimeError(f"No frames were read from {args.input}")
    sink.close()

    elapsed = time.time() - start
    print(f"\nDone: {processed} frames -> {output_path}  [{sink.mode}]")
    print(f"Elapsed: {elapsed:.1f} s ({processed / elapsed:.2f} FPS end-to-end, including video I/O)")


if __name__ == '__main__':
    main()
