import os
import glob
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import DataLoader
    from utils import load_model, img2tensor_batch, tensor2img_fast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

script_dir = Path(__file__).resolve().parent

DATASETS = {
    'Vid4': {
        'gt_path': 'Vid4/GT',
        'lq_path': 'Vid4/BIx4',
        'test_y_channel': True,
        'center_frame_only': False,
        'flip_seq': False
    },
    'REDS4': {
        'gt_path': 'REDS/GT/train/train_sharp',
        'lq_path': 'REDS/LR/train/train_sharp_bicubic/X4',
        'clips': ['000', '011', '015', '020'],
        'test_y_channel': False,
        'center_frame_only': False,
        'flip_seq': False
    },
    'Vimeo90K': {
        'gt_path': 'vimeo_septuplet/sequences',
        'lq_path': 'vimeo_septuplet_matlabLRx4/sequences',
        'file_list': 'vimeo_septuplet/sep_testlist.txt',
        'test_y_channel': True,
        'center_frame_only': True,
        'flip_seq': True
    }
}


def resolve_path(path_value):
    if path_value is None:
        return None
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((script_dir / path_obj).resolve())


def bgr2y(img):
    img = img.astype(np.float64) / 255.
    return 65.481 * img[..., 2] + 128.553 * img[..., 1] + 24.966 * img[..., 0] + 16.0


def calculate_psnr(img1, img2, crop_border=0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


def _ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if img1.ndim == 2:
        return _ssim(img1, img2)
    return np.mean([_ssim(img1[..., c], img2[..., c]) for c in range(img1.shape[2])])


def frame_metrics(sr_bgr, gt_bgr, test_y_channel, crop_border):
    if sr_bgr.shape != gt_bgr.shape:
        raise ValueError(f"Shape mismatch between SR {sr_bgr.shape} and GT {gt_bgr.shape}")
    if test_y_channel:
        sr_bgr = bgr2y(sr_bgr)
        gt_bgr = bgr2y(gt_bgr)
    return calculate_psnr(sr_bgr, gt_bgr, crop_border), calculate_ssim(sr_bgr, gt_bgr, crop_border)


def list_frames(directory):
    return sorted(glob.glob(os.path.join(directory, '*.png')) + glob.glob(os.path.join(directory, '*.jpg')))


class BenchmarkDataset:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.clips = []
        self._scan_dataset()

    def _scan_dataset(self):
        gt_root = self.config['gt_path']
        lq_root = self.config['lq_path']

        if self.name == 'Vid4':
            clip_names = sorted([d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))])
            for clip in clip_names:
                self.clips.append((os.path.join(gt_root, clip), os.path.join(lq_root, clip), clip))

        elif self.name == 'REDS4':
            target_clips = self.config.get('clips', [])
            for clip in target_clips:
                self.clips.append((os.path.join(gt_root, clip), os.path.join(lq_root, clip), clip))

        elif self.name == 'Vimeo90K':
            file_list_path = self.config.get('file_list')
            if file_list_path and os.path.exists(file_list_path):
                with open(file_list_path, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        clip_name = line.replace('/', '_')
                        self.clips.append((os.path.join(gt_root, line), os.path.join(lq_root, line), clip_name))
            else:
                print(f"Warning: Vimeo90K file list not found at {file_list_path}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        gt_dir, lq_dir, clip_name = self.clips[idx]

        gt_paths = list_frames(gt_dir)
        lq_paths = list_frames(lq_dir)
        min_len = min(len(gt_paths), len(lq_paths))
        lq_paths = lq_paths[:min_len]

        lq_imgs = [cv2.imread(p) for p in lq_paths]
        lq_tensor = img2tensor_batch(lq_imgs)

        return lq_tensor, clip_name, gt_dir


def _to_sequence_tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        output = torch.stack(output, dim=1)
    if output.dim() == 4:
        output = output.unsqueeze(1)
    return output


def infer_sequence(model, lq_tensor, flip_seq=False):
    output = _to_sequence_tensor(model(lq_tensor))
    if flip_seq:
        flip_input = torch.flip(lq_tensor, [1])
        output_flip = _to_sequence_tensor(model(flip_input))
        output_flip = torch.flip(output_flip, [1])
        output = (output + output_flip) / 2.0

    frame_mapping = list(range(lq_tensor.shape[1]))
    if output.shape[1] != len(frame_mapping):
        if output.shape[1] == 1:
            frame_mapping = [frame_mapping[len(frame_mapping) // 2]]
        else:
            frame_mapping = frame_mapping[:output.shape[1]]
            output = output[:, :len(frame_mapping)]

    return output, frame_mapping


def report_dataset(ds_name, per_clip, test_y_channel, center_frame_only):
    channel = 'Y' if test_y_channel else 'RGB'
    suffix = ', center frame only' if center_frame_only else ''

    if len(per_clip) <= 8:
        for clip_name, psnr, ssim in per_clip:
            print(f"  {clip_name:<12s} PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}")

    avg_psnr = float(np.mean([p for _, p, _ in per_clip]))
    avg_ssim = float(np.mean([s for _, _, s in per_clip]))
    print(f"{ds_name} average over {len(per_clip)} clips ({channel}{suffix}): "
          f"PSNR: {avg_psnr:.2f} dB  SSIM: {avg_ssim:.4f}\n")
    return avg_psnr, avg_ssim


def evaluate_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device=device,
                       use_switch_to_deploy=not args.no_deploy, model_name=args.model_name)

    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    dataset_names = args.datasets if args.datasets else list(DATASETS.keys())

    summary = {}
    for ds_name in dataset_names:
        ds_conf = dict(DATASETS[ds_name])
        ds_conf['gt_path'] = os.path.join(data_root, ds_conf['gt_path'])
        ds_conf['lq_path'] = os.path.join(data_root, ds_conf['lq_path'])
        if 'file_list' in ds_conf:
            ds_conf['file_list'] = os.path.join(data_root, ds_conf['file_list'])

        if not os.path.exists(ds_conf['gt_path']):
            print(f"Skipping dataset {ds_name} - not found at {ds_conf['gt_path']}")
            continue

        dataset = BenchmarkDataset(ds_name, ds_conf)
        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        test_y_channel = ds_conf['test_y_channel']
        center_frame_only = ds_conf['center_frame_only']
        flip_seq = ds_conf['flip_seq']

        per_clip = []
        pbar = tqdm(loader, total=len(loader), desc=f"Evaluating {args.model_name} on {ds_name}",
                    ncols=shutil.get_terminal_size().columns - 1)

        with torch.inference_mode():
            for lq_tensor, clip_name_tuple, gt_dir_tuple in pbar:
                clip_name = clip_name_tuple[0]
                gt_dir = gt_dir_tuple[0]
                lq_tensor = lq_tensor.to(device)

                output, frame_mapping = infer_sequence(model, lq_tensor, flip_seq=flip_seq)
                output = output.squeeze(0)

                gt_paths = list_frames(gt_dir)
                num_frames = output.shape[0]
                eval_indices = [num_frames // 2] if center_frame_only else list(range(num_frames))

                if args.save_images:
                    clip_save_dir = os.path.join(output_dir, args.model_name, ds_name, clip_name)
                    os.makedirs(clip_save_dir, exist_ok=True)

                psnr_values, ssim_values = [], []
                for idx in range(num_frames):
                    if not args.save_images and idx not in eval_indices:
                        continue

                    sr_img = tensor2img_fast(output[idx])
                    mapped_idx = frame_mapping[idx]

                    if args.save_images:
                        cv2.imwrite(os.path.join(clip_save_dir, f"{mapped_idx:04d}.png"), sr_img)

                    if idx in eval_indices:
                        gt_img = cv2.imread(gt_paths[mapped_idx], cv2.IMREAD_COLOR)
                        psnr, ssim = frame_metrics(sr_img, gt_img, test_y_channel, args.crop_border)
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                per_clip.append((clip_name, float(np.mean(psnr_values)), float(np.mean(ssim_values))))
                pbar.set_postfix(psnr=f"{np.mean([p for _, p, _ in per_clip]):.2f}")

        summary[ds_name] = report_dataset(ds_name, per_clip, test_y_channel, center_frame_only)

    if not summary:
        raise RuntimeError("No datasets were evaluated - check --data_root (see README for the expected layout).")

    if len(summary) > 1:
        print("Summary:")
        for ds_name, (psnr, ssim) in summary.items():
            print(f"  {ds_name:<10s} PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}")


def evaluate_folders(args):
    ds_name = args.datasets[0]
    ds_conf = DATASETS[ds_name]
    test_y_channel = ds_conf['test_y_channel']
    center_frame_only = ds_conf['center_frame_only']

    sr_root = resolve_path(args.sr_root)
    gt_root = resolve_path(args.gt_root)

    clip_names = sorted([d for d in os.listdir(sr_root) if os.path.isdir(os.path.join(sr_root, d))])
    if not clip_names:
        raise FileNotFoundError(f"No clip directories found in {sr_root}")

    per_clip = []
    for clip in tqdm(clip_names, desc=f"Scoring {ds_name}", disable=len(clip_names) <= 8,
                     ncols=shutil.get_terminal_size().columns - 1):
        sr_dir = os.path.join(sr_root, clip)
        gt_rel = clip.replace('_', os.sep) if ds_name == 'Vimeo90K' else clip
        gt_dir = os.path.join(gt_root, gt_rel)

        if not os.path.isdir(gt_dir):
            print(f"Warning: GT directory not found for clip '{clip}' ({gt_dir}); skipping.")
            continue

        sr_paths = list_frames(sr_dir)
        gt_paths = list_frames(gt_dir)
        if len(sr_paths) != len(gt_paths):
            print(f"Warning: '{clip}' has {len(sr_paths)} SR frames but {len(gt_paths)} GT frames; "
                  f"evaluating the first {min(len(sr_paths), len(gt_paths))}.")
        num_frames = min(len(sr_paths), len(gt_paths))
        if num_frames == 0:
            print(f"Warning: no frames found for clip '{clip}'; skipping.")
            continue

        indices = [num_frames // 2] if center_frame_only else range(num_frames)

        psnr_values, ssim_values = [], []
        for i in indices:
            sr = cv2.imread(sr_paths[i], cv2.IMREAD_COLOR)
            gt = cv2.imread(gt_paths[i], cv2.IMREAD_COLOR)
            psnr, ssim = frame_metrics(sr, gt, test_y_channel, args.crop_border)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

        per_clip.append((clip, float(np.mean(psnr_values)), float(np.mean(ssim_values))))

    if not per_clip:
        raise RuntimeError("No clips were evaluated - check --sr_root and --gt_root.")

    report_dataset(ds_name, per_clip, test_y_channel, center_frame_only)


def main():
    parser = argparse.ArgumentParser(description='NanoVSR evaluation (PSNR / SSIM, optional frame saving)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a NanoVSR checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory containing REDS/, Vid4/ and vimeo_septuplet/ (see README)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        choices=list(DATASETS.keys()),
                        help='Subset of benchmarks to run (default: all found under --data_root)')
    parser.add_argument('--save_images', action='store_true',
                        help='Additionally write SR frames as PNG (default: metrics only)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for SR frames when --save_images is set')
    parser.add_argument('--model_name', type=str, default='NanoVSR',
                        help='Name used in logs and for the output subdirectory')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--crop_border', type=int, default=0)
    parser.add_argument('--no_deploy', action='store_true',
                        help='Skip structural reparameterization (keep the multi-branch training topology)')
    parser.add_argument('--sr_root', type=str, default=None,
                        help='Score existing SR frames (one subdirectory per clip) instead of running a checkpoint')
    parser.add_argument('--gt_root', type=str, default=None,
                        help='Ground-truth root, required together with --sr_root')
    args = parser.parse_args()

    if args.sr_root:
        if not args.gt_root:
            parser.error('--gt_root is required together with --sr_root')
        if not args.datasets or len(args.datasets) != 1:
            parser.error('--sr_root requires exactly one --datasets value (it selects the metric protocol)')
        evaluate_folders(args)
    else:
        if not args.checkpoint:
            parser.error('--checkpoint is required (or use --sr_root to score existing results)')
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required to evaluate a checkpoint; '
                              'install requirements.txt or score existing frames with --sr_root.')
        evaluate_model(args)


if __name__ == '__main__':
    main()
