#!/usr/bin/env python3
"""
export_lighttrack.py — LightTrack backbone + head -> ONNX (grayscale 1ch)

Compatible with NanoTrackTRT C++ inference (dual-engine: backbone + head).

Setup:
  git clone https://github.com/researchmm/LightTrack
  # Download LightTrackM.pth from model zoo/release

Usage:
  cd LightTrack
  python /path/to/export_lighttrack.py --checkpoint LightTrackM.pth --out_dir ./exported
"""

import argparse
import torch
import torch.nn as nn
import sys
import os


def setup_lighttrack_path(repo_dir="./LightTrack"):
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "lib"))


class GrayAdapter(nn.Module):
    """Expand 1ch grayscale to 3ch for original backbone"""
    def __init__(self, backbone_features):
        super().__init__()
        self.features = backbone_features

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.features(x)


class BackboneONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = GrayAdapter(model.features)

    def forward(self, x):
        return self.net(x)


class HeadONNX(nn.Module):
    """LightTrack head wrapper: (template_feat, search_feat) -> (cls_score, bbox_reg)"""
    def __init__(self, neck, feature_fusor, head):
        super().__init__()
        self.neck = neck
        self.feature_fusor = feature_fusor
        self.head = head

    def forward(self, zf, xf):
        if not isinstance(zf, list):
            zf = [zf]
        if not isinstance(xf, list):
            xf = [xf]
        if isinstance(self.neck, nn.Identity):
            zf, xf = zf[0], xf[0]
        else:
            zf, xf = self.neck(zf[0], xf[0])
        feat_dict = self.feature_fusor(zf, xf)
        oup = self.head(feat_dict)
        return oup['cls'], oup['reg']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="./exported")
    parser.add_argument("--template_size", type=int, default=127)
    parser.add_argument("--search_size", type=int, default=255)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_lighttrack_path(args.repo_dir)

    from lib.models.models import LightTrackM_Subnet
    from lib.utils.utils import load_pretrain

    model = LightTrackM_Subnet(
        path_name="back_04502514044521042540+cls_211000022+reg_100000111_ops_32", stride=16
    )
    model = load_pretrain(model, args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Backbone ---
    backbone = BackboneONNX(model).to(device).eval()
    dummy = torch.randn(1, 1, args.template_size, args.template_size, device=device)
    backbone_path = os.path.join(args.out_dir, "lighttrack_backbone.onnx")

    torch.onnx.export(
        backbone, dummy, backbone_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "feat_h", 3: "feat_w"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[OK] Backbone: {backbone_path}")

    with torch.no_grad():
        z = torch.randn(1, 1, args.template_size, args.template_size, device=device)
        x = torch.randn(1, 1, args.search_size, args.search_size, device=device)
        zf = backbone(z)
        xf = backbone(x)
        print(f"  template feat: {zf.shape}, search feat: {xf.shape}")

    # --- Head ---
    head = HeadONNX(nn.Identity(), model.feature_fusor, model.head).to(device).eval()
    head_path = os.path.join(args.out_dir, "lighttrack_head.onnx")

    torch.onnx.export(
        head, (zf, xf), head_path,
        input_names=["template_feat", "search_feat"],
        output_names=["cls_score", "bbox_reg"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[OK] Head: {head_path}")

    with torch.no_grad():
        c, r = head(zf, xf)
        print(f"  cls_score: {c.shape}, bbox_reg: {r.shape}")

    # --- Simplify ---
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify
        for path in [backbone_path, head_path]:
            m = onnx.load(path)
            m_sim, ok = onnx_simplify(m)
            if ok:
                onnx.save(m_sim, path)
                print(f"  [SIM] Simplified: {path}")
    except ImportError:
        print("  [SKIP] onnxsim not installed")

    print("\n=== LightTrack export done ===")
    print("Compatible with NanoTrackTRT (dual-engine mode)")
    print("\ntrtexec commands:")
    print(f"  trtexec --onnx={backbone_path} --saveEngine=lighttrack_backbone.engine "
          f"--fp16 --minShapes=input:1x1x127x127 --optShapes=input:1x1x255x255 "
          f"--maxShapes=input:1x1x255x255 --workspace=256")
    print(f"  trtexec --onnx={head_path} --saveEngine=lighttrack_head.engine "
          f"--fp16 --workspace=256")


if __name__ == "__main__":
    main()
