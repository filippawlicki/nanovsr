import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from models.nanovsr import NanoVSR

try:
    from dataset import get_training_dataset, REDSDataset
except ImportError:
    raise ImportError("Cannot find dataset.py. Ensure structure is correct.")

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss / x.numel()

def create_dataloader(args, phase, rank, world_size):
    if phase == 'vimeo':
        if rank == 0:
            print(f"\n[DataLoader] Initializing PHASE 1: Vimeo-90K (7 Frames)...")
        dataset = get_training_dataset(reds_root=None, vimeo_root=args.vimeo_root, patch_size=args.patch_size)
        batch_size = args.batch_size

    elif phase == 'reds':
        if rank == 0:
            print(f"\n[DataLoader] Initializing PHASE 2: REDS (Long Sequence: {args.long_num_frames} Frames)...")
        dataset = REDSDataset(args.reds_root, num_frames=args.long_num_frames, patch_size=args.patch_size,
                              split='train')
        batch_size = args.batch_size
    else:
        raise ValueError("Unknown phase")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=3,
        worker_init_fn=worker_init_fn
    )

    return loader, sampler

def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()

    TOTAL_ITERATIONS = args.total_iterations
    SWITCH_ITERATION = args.switch_iter

    if local_rank == 0:
        print(f"[Init] Training: NanoVSR")
        print(f"[Init] Schedule: Vimeo (0-{SWITCH_ITERATION}k) -> REDS ({SWITCH_ITERATION}k-{TOTAL_ITERATIONS}k)")

    nanovsr = NanoVSR(num_feat=args.num_feat, num_blocks=args.num_blocks).to(device)
    model = DDP(nanovsr, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_ITERATIONS, eta_min=1e-7)

    criterion_pixel = CharbonnierLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    current_iter = 0
    best_psnr = 0.0
    current_phase = 'vimeo'

    train_loader, train_sampler = create_dataloader(args, 'vimeo', local_rank, world_size)
    train_iter = iter(train_loader)

    while current_iter < TOTAL_ITERATIONS:
        current_iter += 1

        if current_iter == SWITCH_ITERATION:
            if local_rank == 0:
                print(f"\nSWITCHING PHASE: VIMEO -> REDS <<<")
                print(f"Clearing cache and loading new dataset...")

            del train_iter, train_loader, train_sampler
            torch.cuda.empty_cache()

            current_phase = 'reds'
            train_loader, train_sampler = create_dataloader(args, 'reds', local_rank, world_size)
            train_iter = iter(train_loader)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_sampler.set_epoch(current_iter)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = batch['lr'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_model = model(lr)
            loss_total = criterion_pixel(out_model, gt)

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if local_rank == 0 and (current_iter % 100 == 0 or current_iter == SWITCH_ITERATION):
            print(f"Iter: {current_iter}/{TOTAL_ITERATIONS} ({current_phase.upper()}) | "
                  f"Loss: {loss_total.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if current_iter % 10000 == 0 or current_iter == SWITCH_ITERATION:
                ckpt_path = Path(args.output_dir) / f"checkpoint_iter_{current_iter}.pth"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.module.state_dict(), ckpt_path)

            torch.cuda.empty_cache()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vimeo_root', type=str, required=True, help='Path to Vimeo90K')
    parser.add_argument('--reds_root', type=str, required=True, help='Path to REDS (Train)')
    parser.add_argument('--output_dir', type=str, default='output_auto_curriculum')

    parser.add_argument('--num_feat', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=256)

    parser.add_argument('--switch_iter', type=int, default=50000,
                        help='Iteration to switch from Vimeo to REDS')
    parser.add_argument('--total_iterations', type=int, default=150000, help='Total training iterations')
    parser.add_argument('--long_num_frames', type=int, default=30,
                        help='Number of frames for REDS phase (Phase 2)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
