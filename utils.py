import cv2
import torch

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
