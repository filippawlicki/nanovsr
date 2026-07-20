<div align="center">

# NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices

### Accepted to ECCV 2026

[Filip Pawlicki](https://orcid.org/0009-0001-3375-8091) · [Marcel Kańduła](https://orcid.org/0009-0001-1314-5511) · [Marcin Pucek](https://orcid.org/0009-0003-9879-8195) · [Kamil Dobies](https://orcid.org/0009-0007-0441-2140)

Gdańsk University of Technology

[![arXiv](https://img.shields.io/badge/arXiv-2607.10495-b31b1b)](https://arxiv.org/abs/2607.10495)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/teaser.gif" width="720" alt="Low-resolution input vs. NanoVSR-644k output (4x upscaling)"/>

*4× video super-resolution with NanoVSR-644k — 27.2 FPS on a Jetson Orin NX (25 W).*

</div>

## Overview

**NanoVSR** is a scalable, fully convolutional video super-resolution (VSR) architecture designed for resource-constrained edge devices. Instead of relying on transformers or explicit optical flow, NanoVSR:

- uses a **bidirectional recurrent** design with a **direct additive propagation** scheme that avoids channel concatenation, reducing memory bandwidth;
- is built from **reparameterizable multi-branch blocks** (3×3 + 1×1 + identity) that collapse into a single stream of plain 3×3 convolutions at inference — no custom CUDA ops, natively ONNX/TensorRT compatible;
- learns spatio-temporal alignment **implicitly** through a two-stage progressive training curriculum (short Vimeo-90K sequences → long REDS sequences).

The baseline NanoVSR-644k reaches **28.64 dB PSNR on REDS4 at 27.20 FPS** on an NVIDIA Jetson Orin NX 16GB (25 W), and the scaled NanoVSR-1.7M reaches **29.15 dB at 19.58 FPS**.

## News

- **2026-07**: Code and pretrained models are released.
- **2026-06**: NanoVSR is accepted to **ECCV 2026**! 🎉

## Model Zoo

All models perform 4× upscaling and were trained with the two-stage curriculum (50k iterations on Vimeo-90K, then 100k on REDS). PSNR/SSIM: RGB for REDS4, Y-channel for Vid4 and Vimeo-90K-T. Runtime is ms/frame for a 180×320 input on an H100 (FP32); FPS is measured on Jetson Orin NX 16GB (25 W) with TensorRT FP16, 180×320 input, T=15.

| Model | Params | REDS4 | Vid4 | Vimeo-90K-T | H100 (ms) | Orin NX (FPS) |                                                Download                                                 |
| :--- | ---: | :---: | :---: | :---: | ---: | ---: |:-------------------------------------------------------------------------------------------------------:|
| NanoVSR-226k | 226k | 28.23 / 0.8057 | 25.26 / 0.7252 | 34.31 / 0.9130 | 1.910 | 43.86 |       [weights](https://github.com/filippawlicki/nanovsr/releases/download/v1.0/nanovsr_226k.pth)       |
| **NanoVSR-644k** (baseline) | 644k | 28.64 / 0.8215 | 26.05 / 0.7761 | 35.00 / 0.9226 | 2.982 | 27.20 |       [weights](https://github.com/filippawlicki/nanovsr/releases/download/v1.0/nanovsr_644k.pth)       |
| NanoVSR-1.7M | 1.7M | 29.15 / 0.8364 | 26.44 / 0.7964 | 35.49 / 0.9294 | 4.268 | 19.58 |       [weights](https://github.com/filippawlicki/nanovsr/releases/download/v1.0/nanovsr_1.7m.pth)       |
| NanoVSR-5.4M | 5.4M | 29.73 / 0.8526 | 26.76 / 0.8089 | 35.85 / 0.9335 | 8.547 | 8.66 |             [weights](https://github.com/filippawlicki/nanovsr/releases/download/v1.0/nanovsr_5.4m.pth)              |

<details>
<summary>All architectural configurations (scaling study)</summary>

| Model | Parameters | Blocks N | Channels F | REDS4 PSNR |
| :--- | ---: | ---: | ---: | ---: |
| NanoVSR-22k | 22,107 | 3 | 10 | 27.62 |
| NanoVSR-31k | 31,153 | 4 | 12 | 27.79 |
| NanoVSR-48k | 48,245 | 4 | 16 | 27.96 |
| NanoVSR-226k | 225,797 | 8 | 32 | 28.23 |
| NanoVSR-644k | 644,245 | 12 | 48 | 28.64 |
| NanoVSR-1.7M | 1,709,605 | 20 | 64 | 29.15 |
| NanoVSR-5.4M | 5,447,365 | 30 | 96 | 29.73 |
| NanoVSR-9.6M | 9,630,309 | 30 | 128 | 29.90 |

</details>

## Installation

```bash
git clone https://github.com/filippawlicki/nanovsr.git
cd nanovsr

conda create -n nanovsr python=3.11 -y
conda activate nanovsr

# Install PyTorch first (pick the right CUDA build for your system):
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The code was tested with Python 3.13, PyTorch 2.10.0, CUDA 12.8 (training/inference on H100) and TensorRT 10.3 (deployment on Jetson Orin NX).

## Quick Demo

Upscale any low-resolution video (or a directory of frames) 4× with a pretrained checkpoint — no datasets required:

```bash
python demo.py --checkpoint checkpoints/nanovsr_644k.pth --input my_clip.mp4
```

The result is written next to the input as `my_clip_x4.mp4`. Useful flags:

- `--compare` — write a labeled side-by-side *LR (nearest-neighbor) vs. NanoVSR* video (like the teaser above);
- `--input frames_dir/ --fps 30` — read a directory of PNG/JPG frames instead of a video file;
- `--chunk_size 15` — temporal window per forward pass (the paper's edge setting); lower it if you run out of memory;
- `--fp16` — half-precision inference on CUDA;
- `--max_frames 100` — quick test on the first N frames.

NanoVSR expects genuinely low-resolution input (the paper operates on 180×320 and 270×480 frames); feeding an HD video will be slow and memory-hungry.

## Data Preparation

We train on **Vimeo-90K** (phase 1) and **REDS** (phase 2), and evaluate on **REDS4**, **Vid4** and **Vimeo-90K-T**.

1. **REDS** — download `train_sharp` (GT) and `train_sharp_bicubic` (LR, X4) from the [official REDS page](https://seungjunnah.github.io/Datasets/reds.html). The four REDS4 clips (`000`, `011`, `015`, `020`) are excluded from training automatically and used for testing.
2. **Vimeo-90K** — download the septuplet dataset from the [OpenDataLab page](https://opendatalab.com/OpenDataLab/Vimeo90K) (includes `sep_trainlist.txt` / `sep_testlist.txt`). The 4× LR frames (`vimeo_septuplet_matlabLRx4`) are generated with MATLAB bicubic downsampling, following standard practice ([BasicSR guide](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)). If the LR folder is missing, the training dataloader falls back to on-the-fly PIL bicubic downsampling — convenient for a quick start, but use the MATLAB LR data to reproduce paper numbers.
3. **Vid4** — download GT and BIx4 using [MMagic dataset guide](https://github.com/open-mmlab/mmagic/blob/main/docs/en/user_guides/dataset_prepare.md).

Organize everything under `data/`:

```
data/
├── REDS/
│   ├── GT/
│   │   └── train/
│   │       └── train_sharp/                # 000, 001, ..., 269
│   └── LR/
│       └── train/
│           └── train_sharp_bicubic/
│               └── X4/                     # 000, 001, ..., 269
├── vimeo_septuplet/
│   ├── sequences/                          # 00001/0001/im1.png ... im7.png
│   ├── sep_trainlist.txt
│   └── sep_testlist.txt
├── vimeo_septuplet_matlabLRx4/
│   └── sequences/
└── Vid4/
    ├── GT/                                 # calendar, city, foliage, walk
    └── BIx4/
```

## Training

The two-stage curriculum is handled automatically: 7-frame Vimeo-90K sequences for the first 50k iterations, then 30-frame REDS sequences until 150k. 256×256 GT patches, Charbonnier loss, cosine annealing from 3e-4 to 1e-7, BF16 AMP and gradient clipping are the script defaults.

**Multi-GPU (paper setup)** — launch with `torchrun`; the paper uses 4 GPUs with a per-GPU batch size of 3 (**global batch size 12**):

```bash
# NanoVSR-644k (baseline)
torchrun --nproc_per_node=4 train.py \
    --vimeo_root data/vimeo_septuplet \
    --reds_root data/REDS \
    --output_dir experiments/nanovsr_644k \
    --num_blocks 12 \
    --num_feat 48 \
    --batch_size 3
```

**Single GPU** — run the same script with plain `python` (DDP is bypassed automatically):

```bash
python train.py \
    --vimeo_root data/vimeo_septuplet \
    --reds_root data/REDS \
    --output_dir experiments/nanovsr_644k \
    --num_blocks 12 \
    --num_feat 48 \
    --batch_size 12
```

The global batch size is `#GPUs × --batch_size`; to reproduce paper results keep it at 12 (reduce `--batch_size` if you run out of memory, at the cost of a slightly different training trajectory).

To train other variants, change the architecture flags according to the Model Zoo table, e.g. `--num_blocks 8 --num_feat 32` for NanoVSR-226k or `--num_blocks 20 --num_feat 64` for NanoVSR-1.7M.

<details>
<summary>All training options</summary>

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--vimeo_root` | — | Path to `vimeo_septuplet/` |
| `--reds_root` | — | Path to `REDS/` |
| `--output_dir` | `output_auto_curriculum` | Checkpoint directory |
| `--num_feat` | 32 | Feature channels F |
| `--num_blocks` | 8 | Propagation blocks N per direction |
| `--batch_size` | 3 | Per-GPU batch size |
| `--lr` | 3e-4 | Initial learning rate |
| `--patch_size` | 256 | GT patch size |
| `--switch_iter` | 50000 | Iteration to switch Vimeo-90K → REDS |
| `--total_iterations` | 150000 | Total iterations |
| `--long_num_frames` | 30 | Sequence length in the REDS phase |
| `--num_workers` | 10 | Dataloader workers per GPU |

</details>

## Evaluation

A single command runs the model on the benchmarks and reports PSNR/SSIM with the paper's protocol (REDS4 in RGB; Vid4 and Vimeo-90K-T on the Y channel; Vimeo-90K-T on the center frame of each septuplet). The architecture is detected from the checkpoint and the model is automatically reparameterized into its fused deploy form:

```bash
python evaluate.py --checkpoint checkpoints/nanovsr_644k.pth --data_root data
```

- `--datasets REDS4 Vid4` evaluates a subset; benchmarks missing under `--data_root` are skipped.
- `--save_images` additionally writes the SR frames to `results/NanoVSR/{REDS4,Vid4,Vimeo90K}/<clip>/` (off by default).

Pre-computed SR frames — e.g. produced by a TensorRT engine — can be scored directly, without running the model:

```bash
python evaluate.py --datasets REDS4 \
    --sr_root results/trt/REDS4 \
    --gt_root data/REDS/GT/train/train_sharp
```

## Pretrained Models

Download the checkpoints from the [Releases page](https://github.com/filippawlicki/nanovsr/releases) (direct links in the [Model Zoo](#model-zoo)) and place them in `checkpoints/`:

```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/filippawlicki/nanovsr/releases/download/v1.0/nanovsr_644k.pth
```

Checkpoints store the multi-branch (training) topology; `demo.py`, `evaluate.py` and `export_onnx.py` fuse them into the single-branch deploy form on load.

## ONNX Export & TensorRT Deployment

Export a reparameterized model to ONNX (the paper uses a fixed temporal window of T=15 for chunk-based execution on Jetson):

```bash
python export_onnx.py \
    --checkpoint checkpoints/nanovsr_644k.pth \
    --num_frames 15 --height 180 --width 320
```

Then build a TensorRT engine on the target device (paper setup: TensorRT 10.3, FP16):

```bash
trtexec --onnx=checkpoints/nanovsr_644k.onnx \
        --saveEngine=checkpoints/nanovsr_644k.engine \
        --fp16
```

Measured edge throughput (TensorRT FP16, T=15):

| Model | Resolution | Orin NX 8GB / 15 W | Orin NX 16GB / 25 W |
| :--- | :--- | ---: | ---: |
| NanoVSR-226k | 180×320 | 23.55 FPS | 43.86 FPS |
| NanoVSR-644k | 180×320 | 16.12 FPS | 27.20 FPS |
| NanoVSR-1.7M | 180×320 | 11.57 FPS | 19.58 FPS |
| NanoVSR-226k | 270×480 | 10.51 FPS | 19.55 FPS |
| NanoVSR-644k | 270×480 | 7.19 FPS | 12.82 FPS |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{pawlicki2026nanovsrrealtimevideosuperresolution,
      title={NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices}, 
      author={Filip Pawlicki and Marcel Kańduła and Marcin Pucek and Kamil Dobies},
      year={2026},
      eprint={2607.10495},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.10495}, 
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

For questions, please open an [issue](https://github.com/filippawlicki/nanovsr/issues) or contact the authors:
- Filip Pawlicki (s198371@student.pg.edu.pl)
- Marcel Kańduła (s197677@student.pg.edu.pl)
- Marcin Pucek (s197893@student.pg.edu.pl)
- Kamil Dobies (s197875@student.pg.edu.pl)
