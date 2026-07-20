import os

import cv2
import numpy as np
import torch

from models.nanovsr import NanoVSR


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


def load_checkpoint_state_dict(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ['params_ema', 'params', 'model_state_dict']:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


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


def load_model(checkpoint_path, device='cuda', use_switch_to_deploy=True, model_name='NanoVSR'):
    print(f"\nLoading {model_name} from {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = load_checkpoint_state_dict(checkpoint_path)

    num_feat, num_blocks = detect_model_config(state_dict)
    model = NanoVSR(num_feat=num_feat or 32, num_blocks=num_blocks or 8)

    model.load_state_dict(state_dict, strict=False)
    if use_switch_to_deploy:
        model.switch_to_deploy()
        print(f"Switched {model_name} to deploy mode.")

    model.eval()
    return model.to(device)
