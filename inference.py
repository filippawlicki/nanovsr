import os
import sys
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from models.nanovsr import NanoVSR

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 8,
    'scale': 4,
    'output_dir': 'inference_outputs',

    'datasets': {
        'Vid4': {
            'gt_path': 'data/Vid4/GT',
            'lq_path': 'data/Vid4/BIx4',
            'crop_border': 0,
            'test_y_channel': True,
            'center_frame_only': False,
            'flip_seq': False
        },
        'REDS4': {
            'gt_path': 'data/REDS/GT/train/train_sharp',
            'lq_path': 'data/REDS/LR/train/train_sharp_bicubic/X4',
            'clips': ['000', '011', '015', '020'],
            'crop_border': 0,
            'test_y_channel': False,
            'center_frame_only': False,
            'flip_seq': False
        },
        'Vimeo90K': {
            'gt_path': 'data/vimeo_septuplet/sequences',
            'lq_path': 'data/vimeo_septuplet_matlabLRx4/sequences',
            'file_list': 'data/vimeo_septuplet/sep_testlist.txt',
            'crop_border': 0,
            'test_y_channel': True,
            'center_frame_only': True,
            'flip_seq': True
        }
    },

    'models_to_test': [
        {
            'name': 'NanoVSR',
            'path': 'checkpoints/nanovsr_644k.pth',
            'arch_type': 'nanovsr'
        },
    ],
}


def resolve_path(path_value):
    if path_value is None:
        return None
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((script_dir / path_obj).resolve())


def img2tensor_batch(imgs_np):
    batch = np.stack(imgs_np, axis=0)
    batch = batch[..., ::-1]
    batch = batch.transpose(0, 3, 1, 2)
    batch = np.ascontiguousarray(batch)
    tensor = torch.from_numpy(batch).float()
    return tensor / 255.


def tensor2img_fast(tensor):
    img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


class BenchmarkDataset(Dataset):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.clips = []
        self._scan_dataset()

    def _scan_dataset(self):
        gt_root = resolve_path(self.config['gt_path'])
        lq_root = resolve_path(self.config['lq_path'])

        if self.name == 'Vid4':
            clip_names = sorted([d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))])
            for clip in clip_names:
                self.clips.append((os.path.join(gt_root, clip), os.path.join(lq_root, clip)))

        elif self.name == 'REDS4':
            target_clips = self.config.get('clips', [])
            for clip in target_clips:
                self.clips.append((os.path.join(gt_root, clip), os.path.join(lq_root, clip)))

        elif self.name == 'Vimeo90K':
            file_list_path = resolve_path(self.config.get('file_list'))
            if file_list_path and os.path.exists(file_list_path):
                with open(file_list_path, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        self.clips.append((os.path.join(gt_root, line), os.path.join(lq_root, line)))
            else:
                print(f"Warning: Vimeo90K file list not found at {file_list_path}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        gt_dir, lq_dir = self.clips[idx]

        gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')) + glob.glob(os.path.join(gt_dir, '*.jpg')))
        lq_paths = sorted(glob.glob(os.path.join(lq_dir, '*.png')) + glob.glob(os.path.join(lq_dir, '*.jpg')))
        min_len = min(len(gt_paths), len(lq_paths))
        lq_paths = lq_paths[:min_len]

        lq_imgs = [cv2.imread(p) for p in lq_paths]
        lq_tensor = img2tensor_batch(lq_imgs)

        return lq_tensor, os.path.basename(gt_dir)


def detect_model_config(state_dict):
    num_feat = None
    num_blocks = None
    candidates = ['feat_extract.rbr_dense.0.weight', 'conv_first.rbr_dense.0.weight']
    for key in candidates:
        if key in state_dict:
            num_feat = state_dict[key].shape[0]
            break
    if num_feat is None:
        for key in state_dict.keys():
            if any(x in key for x in ['feat_extract', 'conv_first', 'feature_extract']) and 'weight' in key:
                weight = state_dict[key]
                if weight.ndim >= 2:
                    num_feat = weight.shape[0]
                    break
    block_indices = set()
    for prefix in ['forward_net.', 'backward_net.', 'forward_trunk.', 'backward_trunk.', 'backbone.']:
        for key in state_dict.keys():
            if key.startswith(prefix):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    block_indices.add(int(parts[1]))
    if block_indices:
        num_blocks = max(block_indices) + 1
    return num_feat, num_blocks


def load_checkpoint_state_dict(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ['params_ema', 'params', 'model_state_dict']:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


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


def load_model(config_entry):
    model_name = config_entry['name']
    model_path = resolve_path(config_entry['path'])
    print(f"\nLoading {model_name} from {model_path}...")

    state_dict = None
    if os.path.exists(model_path):
        state_dict = load_checkpoint_state_dict(model_path)

    num_feat, num_blocks = detect_model_config(state_dict) if state_dict else (32, 8)
    model = NanoVSR(
        num_feat=num_feat or 32,
        num_blocks=num_blocks or 8,
        deploy=bool(config_entry.get('deploy', False))
    )

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        if config_entry.get('use_switch_to_deploy', True) and hasattr(model, 'switch_to_deploy'):
            model.switch_to_deploy()
            print(f"Switched {model_name} to deploy mode.")

    model.eval()
    return model.to(CONFIG['device'])


def run_inference():
    base_output_dir = resolve_path(CONFIG['output_dir'])

    for model_cfg in CONFIG['models_to_test']:
        model_name = model_cfg['name']
        model = load_model(model_cfg)

        for ds_name, ds_conf in CONFIG['datasets'].items():
            ds_gt_path = resolve_path(ds_conf['gt_path'])
            if not os.path.exists(ds_gt_path):
                print(f"Skipping dataset {ds_name} - not found at {ds_gt_path}")
                continue

            dataset = BenchmarkDataset(ds_name, ds_conf)
            if len(dataset) == 0:
                continue

            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=CONFIG['num_workers'],
                pin_memory=True
            )

            vis_save_dir = os.path.join(base_output_dir, model_name, ds_name)
            os.makedirs(vis_save_dir, exist_ok=True)
            flip_seq = ds_conf.get('flip_seq', False)

            pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"Inferring {model_name} on {ds_name}")

            with torch.inference_mode():
                for lq_tensor, clip_name_tuple in pbar:
                    clip_name = clip_name_tuple[0]
                    lq_tensor = lq_tensor.to(CONFIG['device'])

                    output, frame_mapping = infer_sequence(model, lq_tensor, flip_seq=flip_seq)

                    output = output.squeeze(0)

                    clip_save_dir = os.path.join(vis_save_dir, clip_name)
                    os.makedirs(clip_save_dir, exist_ok=True)

                    for idx in range(output.shape[0]):
                        sr_tensor = output[idx]
                        sr_img = tensor2img_fast(sr_tensor)

                        mapped_idx = frame_mapping[idx]
                        img_name = f"{mapped_idx:04d}.png"

                        cv2.imwrite(os.path.join(clip_save_dir, img_name), sr_img)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    run_inference()
