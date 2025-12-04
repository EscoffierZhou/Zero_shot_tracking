import os
import argparse
import importlib
import sys
import torch
import torch.nn as nn

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
prj_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(prj_dir)

from lib.models.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file

# Wrapper to return tuple instead of dict
class OSTrackExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, z, x):
        # z: template, x: search
        out = self.model(template=z, search=x, ce_template_mask=None)
        return out['score_map'], out['size_map'], out['offset_map']

def parse_args():
    parser = argparse.ArgumentParser(description='Export OSTrack to ONNX')
    parser.add_argument('--config', type=str, default='vitb_256_mae_ce_32x4_ep300')
    parser.add_argument('--output', type=str, default='ostrack.onnx')
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    # We need to find the yaml file. It's in experiments/ostrack/
    yaml_file = os.path.join(prj_dir, 'experiments', 'ostrack', f'{args.config}.yaml')
    if not os.path.exists(yaml_file):
        print(f"❌ Config file not found: {yaml_file}")
        return
        
    update_config_from_file(yaml_file)
    
    # Create network
    print("Building model...")
    net = build_ostrack(cfg, training=False)
    net.cpu()
    
    # Load checkpoint
    if args.checkpoint is None:
        # Try to find it in the standard location we discovered earlier
        # F:\desktop\CV_Project\CV_TRACKING_ADVANCED\model\vitb_256_mae_32x4_ep300\OSTrack_ep0300.pth.tar
        # prj_dir is .../CV_TRACKING_ADVANCED/OSTrack-main
        # So we go up 1 level to CV_TRACKING_ADVANCED
        base_root = os.path.abspath(os.path.join(prj_dir, '..')) # CV_TRACKING_ADVANCED
        ckpt_path = os.path.join(base_root, 'model', 'vitb_256_mae_32x4_ep300', 'OSTrack_ep0300.pth.tar')
        if not os.path.exists(ckpt_path):
             ckpt_path = os.path.join(base_root, 'model', 'vitb_256_mae_ce_32x4_ep300', 'OSTrack_ep0300.pth.tar')
    else:
        ckpt_path = args.checkpoint
        
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
        
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=True)
    net.eval()
    
    # Wrap model
    model_export = OSTrackExport(net)
    
    # Dummy input
    # Config says template size 128, search size 256
    z = torch.randn(1, 3, 128, 128)
    x = torch.randn(1, 3, 256, 256)
    
    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        model_export,
        (z, x),
        args.output,
        verbose=False,
        opset_version=14, 
        input_names=["z", "x"],
        output_names=["score_map", "size_map", "offset_map"]
    )
    
    print("✅ ONNX export complete!")
    
    # Simplify if onnxsim is available
    try:
        import onnxsim
        import onnx
        print("Simplifying ONNX...")
        model_sim, check = onnxsim.simplify(args.output)
        if check:
            onnx.save(model_sim, args.output)
            print("✅ ONNX simplified")
        else:
            print("⚠️ ONNX simplification failed check")
    except ImportError:
        print("⚠️ onnxsim not installed, skipping simplification")

if __name__ == "__main__":
    main()
