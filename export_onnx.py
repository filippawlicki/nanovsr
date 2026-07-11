import os
import argparse

import torch

from models.nanovsr import NanoVSR
from utils import detect_model_config, load_checkpoint_state_dict


def main():
    parser = argparse.ArgumentParser(description='NanoVSR ONNX export')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a NanoVSR checkpoint (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .onnx path (default: checkpoint name with .onnx suffix)')
    parser.add_argument('--num_frames', type=int, default=15,
                        help='Temporal window T baked into the graph (paper: 15)')
    parser.add_argument('--height', type=int, default=180, help='LR input height')
    parser.add_argument('--width', type=int, default=320, help='LR input width')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamic_spatial', action='store_true',
                        help='Mark height/width as dynamic axes (static shapes give the best TensorRT engines)')
    parser.add_argument('--no_deploy', action='store_true',
                        help='Export the multi-branch training topology instead of the fused model')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    output_path = args.output or os.path.splitext(args.checkpoint)[0] + '.onnx'

    state_dict = load_checkpoint_state_dict(args.checkpoint)
    num_feat, num_blocks = detect_model_config(state_dict)
    print(f"Detected architecture: num_feat={num_feat}, num_blocks={num_blocks}")

    model = NanoVSR(num_feat=num_feat or 32, num_blocks=num_blocks or 8)
    model.load_state_dict(state_dict, strict=True)

    if not args.no_deploy:
        model.switch_to_deploy()
        print("Reparameterized to deploy mode (fused multi-branch blocks).")
    model.eval()

    dummy_input = torch.randn(1, args.num_frames, 3, args.height, args.width)

    dynamic_axes = None
    if args.dynamic_spatial:
        dynamic_axes = {
            'lr': {0: 'batch', 3: 'height', 4: 'width'},
            'sr': {0: 'batch', 3: 'height_x4', 4: 'width_x4'},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=args.opset,
        input_names=['lr'],
        output_names=['sr'],
        dynamic_axes=dynamic_axes,
    )

    try:
        import onnx
        onnx.checker.check_model(onnx.load(output_path))
        print("ONNX model check passed.")
    except ImportError:
        print("Package 'onnx' not installed - skipping model check.")

    print(f"\nExported to {output_path}")
    print(f"Input:  lr [1, {args.num_frames}, 3, {args.height}, {args.width}]")
    print(f"Output: sr [1, {args.num_frames}, 3, {args.height * 4}, {args.width * 4}]")
    print("\nBuild a TensorRT engine (paper setup: FP16, TensorRT 10.3):")
    print(f"  trtexec --onnx={output_path} --saveEngine={os.path.splitext(output_path)[0]}.engine --fp16")


if __name__ == '__main__':
    main()
