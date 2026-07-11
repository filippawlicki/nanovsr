import os
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

REDS4_CLIPS = ['000', '011', '015', '020']

class GeometryAug:
    def __call__(self, lr_list, gt_list):
        if random.random() < 0.5:
            lr_list = [TF.hflip(img) for img in lr_list]
            gt_list = [TF.hflip(img) for img in gt_list]

        if random.random() < 0.5:
            lr_list = [TF.vflip(img) for img in lr_list]
            gt_list = [TF.vflip(img) for img in gt_list]

        if random.random() < 0.5:
            lr_list = lr_list[::-1]
            gt_list = gt_list[::-1]

        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            lr_list = [TF.rotate(img, angle) for img in lr_list]
            gt_list = [TF.rotate(img, angle) for img in gt_list]

        return lr_list, gt_list


class CutBlur:
    def __init__(self, prob=0.5, alpha=0.7, scale=4):
        self.prob = prob
        self.alpha = alpha
        self.scale = scale

    def __call__(self, lr, gt):
        if self.prob <= 0 or random.random() >= self.prob:
            return lr, gt

        T, C, h_lr, w_lr = lr.shape
        lam = np.random.beta(self.alpha, self.alpha)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w_lr * cut_rat)
        cut_h = int(h_lr * cut_rat)

        if cut_w < 2 or cut_h < 2:
            return lr, gt

        cx = np.random.randint(w_lr)
        cy = np.random.randint(h_lr)

        x1 = np.clip(cx - cut_w // 2, 0, w_lr)
        y1 = np.clip(cy - cut_h // 2, 0, h_lr)
        x2 = np.clip(cx + cut_w // 2, 0, w_lr)
        y2 = np.clip(cy + cut_h // 2, 0, h_lr)

        if (x2 - x1) < 1 or (y2 - y1) < 1:
            return lr, gt

        gt_x1, gt_y1 = x1 * self.scale, y1 * self.scale
        gt_x2, gt_y2 = x2 * self.scale, y2 * self.scale

        gt_patch = gt[:, :, gt_y1:gt_y2, gt_x1:gt_x2]
        gt_patch_down = torch.nn.functional.interpolate(
            gt_patch, size=(y2 - y1, x2 - x1), mode='bicubic', align_corners=False
        )

        lr_aug = lr.clone()
        lr_aug[:, :, y1:y2, x1:x2] = gt_patch_down

        return lr_aug, gt


class REDSDataset(Dataset):
    def __init__(self, data_root, num_frames=7, patch_size=256, scale=4, split='train'):
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.scale = scale
        self.split = split

        self.geo_aug = GeometryAug()
        self.cutblur = CutBlur(prob=0.5 if split == 'train' else 0.0, scale=scale)

        self.lr_root = self.data_root / 'train' / 'train_sharp_bicubic' / 'X4'
        self.gt_root = self.data_root / 'train' / 'train_sharp'

        if not self.lr_root.exists():
            self.lr_root = self.data_root / 'LR' / 'train' / 'train_sharp_bicubic' / 'X4'
            self.gt_root = self.data_root / 'GT' / 'train' / 'train_sharp'

        if not self.lr_root.exists():
            self.lr_root = self.data_root / 'train_sharp_bicubic' / 'X4'
            self.gt_root = self.data_root / 'train_sharp'

        if not self.lr_root.exists():
            raise FileNotFoundError(f"Could not find REDS folders in {data_root}. Check structure.")

        all_sequences = sorted([d.name for d in self.lr_root.iterdir() if d.is_dir()])
        self.sequences = []

        if split == 'train':
            for seq in all_sequences:
                if seq not in REDS4_CLIPS:
                    self.sequences.append(seq)
            print(f"[Dataset] REDS Train: Loaded {len(self.sequences)} clips (REDS4 excluded).")

        elif split == 'test':
            for seq in all_sequences:
                if seq in REDS4_CLIPS:
                    self.sequences.append(seq)
            print(f"[Dataset] REDS test (REDS4): Loaded {len(self.sequences)} clips {self.sequences}")

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for seq in self.sequences:
            seq_path = self.lr_root / seq
            total_frames = len(list(seq_path.glob('*.png')))

            if total_frames < self.num_frames:
                continue

            num_starts = total_frames - self.num_frames + 1

            if self.split == 'train':
                for i in range(num_starts):
                    samples.append((seq, i))
            else:
                samples.append((seq, 0))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, start = self.samples[idx]
        lr_frames, gt_frames = [], []

        for i in range(self.num_frames):
            frame_idx = start + i
            frame_name = f'{frame_idx:08d}.png'

            lr_path = self.lr_root / seq / frame_name
            gt_path = self.gt_root / seq / frame_name

            lr = Image.open(lr_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')
            lr_frames.append(lr)
            gt_frames.append(gt)

        if self.split == 'train' and self.patch_size is not None:
            gt_w, gt_h = gt_frames[0].size

            top = random.randint(0, gt_h - self.patch_size)
            left = random.randint(0, gt_w - self.patch_size)

            top = (top // self.scale) * self.scale
            left = (left // self.scale) * self.scale

            lr_top, lr_left = top // self.scale, left // self.scale
            lr_patch = self.patch_size // self.scale

            lr_frames = [f.crop((lr_left, lr_top, lr_left + lr_patch, lr_top + lr_patch)) for f in lr_frames]
            gt_frames = [f.crop((left, top, left + self.patch_size, top + self.patch_size)) for f in gt_frames]

            lr_frames, gt_frames = self.geo_aug(lr_frames, gt_frames)

        lr_tensors = torch.stack([TF.to_tensor(f) for f in lr_frames])
        gt_tensors = torch.stack([TF.to_tensor(f) for f in gt_frames])

        if self.split == 'train':
            lr_tensors, gt_tensors = self.cutblur(lr_tensors, gt_tensors)

        return {'lr': lr_tensors, 'gt': gt_tensors}


class Vimeo90KDataset(Dataset):
    def __init__(self, data_root, num_frames=7, patch_size=256, scale=4, split='train'):
        self.data_root = Path(data_root)
        self.num_frames = 7
        self.patch_size = patch_size
        self.scale = scale
        self.split = split

        self.geo_aug = GeometryAug()
        self.cutblur = CutBlur(prob=0.5 if split == 'train' else 0.0, scale=scale)

        if (self.data_root / 'sequences').exists():
            self.gt_root = self.data_root / 'sequences'
        else:
            self.gt_root = self.data_root

        self.lr_root = self.data_root.parent / 'vimeo_septuplet_matlabLRx4' / 'sequences'
        if not self.lr_root.exists():
            self.lr_root = self.data_root / 'sequences_LR'
            if not self.lr_root.exists():
                self.lr_root = None

        list_filename = 'sep_trainlist.txt' if split == 'train' else 'sep_testlist.txt'
        list_file = self.data_root / list_filename

        if list_file.exists():
            with open(list_file, 'r') as f:
                self.samples = [x.strip() for x in f.readlines()]
            print(f"[Dataset] Vimeo90K ({split}) loaded via list: {len(self.samples)} sequences")
        else:
            self.samples = []
            print(f"[Dataset] Warning: {list_filename} not found in {self.data_root}. Scanning...")
            for folder in self.gt_root.iterdir():
                if folder.is_dir():
                    for subfolder in folder.iterdir():
                        if subfolder.is_dir():
                            self.samples.append(f"{folder.name}/{subfolder.name}")
            print(f"[Dataset] Vimeo90K scanned: {len(self.samples)} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path = self.samples[idx]
        lr_frames, gt_frames = [], []

        for i in range(1, 8):
            fname = f'im{i}.png'
            gt_path = self.gt_root / seq_path / fname

            gt = Image.open(gt_path).convert('RGB')
            gt_frames.append(gt)

            if self.lr_root and (self.lr_root / seq_path / fname).exists():
                lr = Image.open(self.lr_root / seq_path / fname).convert('RGB')
            else:
                w, h = gt.size
                lr = gt.resize((w // self.scale, h // self.scale), Image.BICUBIC)

            lr_frames.append(lr)

        if self.split == 'train' and self.patch_size is not None:
            gt_w, gt_h = gt_frames[0].size

            pad_h = max(0, self.patch_size - gt_h)
            pad_w = max(0, self.patch_size - gt_w)
            if pad_h > 0 or pad_w > 0:
                gt_frames = [TF.pad(img, (0, 0, pad_w, pad_h), padding_mode='reflect') for img in gt_frames]
                lr_pad_h = pad_h // self.scale
                lr_pad_w = pad_w // self.scale
                lr_frames = [TF.pad(img, (0, 0, lr_pad_w, lr_pad_h), padding_mode='reflect') for img in lr_frames]
                gt_w, gt_h = gt_frames[0].size

            top = random.randint(0, gt_h - self.patch_size)
            left = random.randint(0, gt_w - self.patch_size)

            top = (top // self.scale) * self.scale
            left = (left // self.scale) * self.scale

            lr_top, lr_left = top // self.scale, left // self.scale
            lr_patch = self.patch_size // self.scale

            lr_frames = [f.crop((lr_left, lr_top, lr_left + lr_patch, lr_top + lr_patch)) for f in lr_frames]
            gt_frames = [f.crop((left, top, left + self.patch_size, top + self.patch_size)) for f in gt_frames]

            lr_frames, gt_frames = self.geo_aug(lr_frames, gt_frames)

        lr_tensors = torch.stack([TF.to_tensor(f) for f in lr_frames])
        gt_tensors = torch.stack([TF.to_tensor(f) for f in gt_frames])

        if self.split == 'train':
            lr_tensors, gt_tensors = self.cutblur(lr_tensors, gt_tensors)

        return {'lr': lr_tensors, 'gt': gt_tensors}

def get_training_dataset(reds_root=None, vimeo_root=None, patch_size=256):
    reds = None
    vimeo = None

    if reds_root and os.path.exists(reds_root):
        print(f"[Dataset] Initializing REDS from {reds_root}...")
        reds = REDSDataset(reds_root, patch_size=patch_size, split='train')

    if vimeo_root and os.path.exists(vimeo_root):
        print(f"[Dataset] Initializing Vimeo-90K from {vimeo_root}...")
        vimeo = Vimeo90KDataset(vimeo_root, patch_size=patch_size, split='train')

    if reds:
        return reds
    elif vimeo:
        return vimeo
    else:
        raise ValueError("No valid dataset root provided (checked both REDS and Vimeo).")
