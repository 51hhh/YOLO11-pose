#!/usr/bin/env python3
"""
export_nanotrack_onnx.py
NanoTrackV3 -> backbone.onnx + head.onnx (grayscale 1-channel, TensorRT friendly)

Setup:
  git clone https://github.com/HonglinChu/SiamTrackers.git
  cd SiamTrackers/NanoTrack
  python setup.py build_ext --inplace
  # Download weights: models/pretrained/nanotrackv3.pth

Usage:
  python export_nanotrack_onnx.py \
    --config models/config/configv3.yaml \
    --snapshot models/pretrained/nanotrackv3.pth \
    --output_dir ./onnx_out
"""

import argparse
import os
import sys
import copy
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

from nanotrack.core.config import cfg
from nanotrack.utils.model_load import load_pretrain
from nanotrack.models.model_builder import ModelBuilder


def patch_backbone_to_grayscale(backbone):
    """Patch first Conv2d from 3ch to 1ch (weight = channel mean)"""
    first_conv = None
    first_conv_path = None
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            first_conv = module
            first_conv_path = name
            break

    if first_conv is None:
        print("[WARN] No 3-channel Conv2d found in backbone, skip patching")
        return backbone

    print(f"[INFO] Patching {first_conv_path}: Conv2d(3,{first_conv.out_channels},...) -> Conv2d(1,...)")
    new_conv = nn.Conv2d(
        1, first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=1,
        bias=(first_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    parts = first_conv_path.split('.')
    parent = backbone
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_conv
    else:
        setattr(parent, parts[-1], new_conv)

    return backbone


class HeadWrapper(nn.Module):
    """Wrap BAN head: cls output [N,2,H,W] -> [N,1,H,W] (fg only)"""
    def __init__(self, ban_head):
        super().__init__()
        self.ban_head = ban_head

    def forward(self, z_f, x_f):
        cls, loc = self.ban_head(z_f, x_f)
        cls_fg = cls[:, 1:2, :, :]
        return cls_fg, loc


class BackboneWithNeck(nn.Module):
    def __init__(self, bb, nk):
        super().__init__()
        self.backbone = bb
        self.neck = nk

    def forward(self, x):
        feat = self.backbone(x)
        if self.neck is not None:
            if isinstance(feat, (list, tuple)):
                feat = self.neck(feat[0] if len(feat) == 1 else feat)
            else:
                feat = self.neck(feat)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='models/config/configv3.yaml')
    parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth')
    parser.add_argument('--output_dir', default='./onnx_out')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg.merge_from_file(args.config)
    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Backbone: grayscale, dynamic H/W ---
    backbone = copy.deepcopy(model.backbone)
    neck = copy.deepcopy(model.neck) if hasattr(model, 'neck') else None
    backbone = patch_backbone_to_grayscale(backbone)

    backbone_net = BackboneWithNeck(backbone, neck).to(device).eval()

    dummy_input = torch.randn(1, 1, 255, 255, device=device)
    backbone_onnx = os.path.join(args.output_dir, 'nanotrack_backbone.onnx')

    torch.onnx.export(
        backbone_net, dummy_input, backbone_onnx,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {2: 'height', 3: 'width'},
            'output': {2: 'feat_h', 3: 'feat_w'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[OK] Backbone exported: {backbone_onnx}")

    with torch.no_grad():
        t127 = torch.randn(1, 1, 127, 127, device=device)
        t255 = torch.randn(1, 1, 255, 255, device=device)
        f127 = backbone_net(t127)
        f255 = backbone_net(t255)
        print(f"  template (127x127) -> feat {list(f127.shape)}")
        print(f"  search   (255x255) -> feat {list(f255.shape)}")

    # --- Head: single-channel cls ---
    head_net = HeadWrapper(model.ban_head).to(device).eval()

    zf_shape = list(f127.shape)
    xf_shape = list(f255.shape)
    dummy_zf = torch.randn(zf_shape, device=device)
    dummy_xf = torch.randn(xf_shape, device=device)

    head_onnx = os.path.join(args.output_dir, 'nanotrack_head.onnx')

    torch.onnx.export(
        head_net, (dummy_zf, dummy_xf), head_onnx,
        input_names=['template_feat', 'search_feat'],
        output_names=['cls_score', 'bbox_reg'],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[OK] Head exported: {head_onnx}")

    with torch.no_grad():
        cls_out, reg_out = head_net(dummy_zf, dummy_xf)
        print(f"  cls_score shape: {list(cls_out.shape)}")
        print(f"  bbox_reg  shape: {list(reg_out.shape)}")

    # --- Simplify (optional) ---
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify
        for path in [backbone_onnx, head_onnx]:
            m = onnx.load(path)
            m_sim, ok = onnx_simplify(m)
            if ok:
                onnx.save(m_sim, path)
                print(f"  [SIM] Simplified: {path}")
    except ImportError:
        print("  [SKIP] onnxsim not installed, skipping simplification")

    print("\n=== NanoTrack export done ===")
    print(f"Backbone: {backbone_onnx}")
    print(f"Head:     {head_onnx}")
    print("\ntrtexec commands:")
    print(f"  /usr/src/tensorrt/bin/trtexec --onnx={backbone_onnx} "
          f"--saveEngine=nanotrack_backbone_fp16.engine --fp16 "
          f"--minShapes=input:1x1x127x127 --optShapes=input:1x1x127x127 "
          f"--maxShapes=input:1x1x255x255 --workspace=256")
    print(f"  /usr/src/tensorrt/bin/trtexec --onnx={head_onnx} "
          f"--saveEngine=nanotrack_head_fp16.engine --fp16 --workspace=256")


if __name__ == '__main__':
    main()
