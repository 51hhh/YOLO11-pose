#!/usr/bin/env python3
"""
export_siamfc.py — SiamFC backbone + head -> ONNX (grayscale 1ch, no bbox regression)

Self-contained: no external repo dependency. Implements the AlexNet-based SiamFC
architecture directly.

Compatible with NanoTrackTRT C++ inference (dual-engine mode, reg_elements_=0 fallback).

Usage:
  python export_siamfc.py --checkpoint SiamFC.pth --out_dir ./exported

If no checkpoint is provided, exports with random weights (for architecture validation).
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class SiamFCBackbone(nn.Module):
    """
    AlexNet-based SiamFC backbone (native 1ch grayscale input).
    Input:  [N, 1, H, W]  (127x127 template or 255x255 search)
    Output: [N, 256, h, w] (6x6 for 127, 22x22 for 255)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, stride=2),       # 1ch input
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        return self.features(x)


class SiamFCHead(nn.Module):
    """
    Cross-correlation head (cls only, no bbox regression).
    Input:  template_feat [N,256,6,6], search_feat [N,256,22,22]
    Output: cls_score [N,1,17,17]

    Uses grouped conv for ONNX-friendly cross-correlation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, zf, xf):
        # zf: [1, 256, 6, 6]  xf: [1, 256, 22, 22]
        N, C, _, _ = zf.shape
        # Depth-wise cross-correlation via grouped convolution
        # F.conv2d with groups=C gives per-channel correlation
        # Then sum over channels
        score = F.conv2d(xf, zf, groups=N)  # [1, 256, 17, 17] for N=1
        # Sum over channel dimension to get single score map
        score = score.sum(dim=1, keepdim=True)  # [1, 1, 17, 17]
        return score


class SiamFCBackboneFromPretrained(nn.Module):
    """Load backbone from a pretrained SiamFC checkpoint (3ch -> 1ch conversion)."""
    def __init__(self, backbone_3ch):
        super().__init__()
        self.features = nn.Sequential()
        first_done = False
        for i, layer in enumerate(backbone_3ch):
            if isinstance(layer, nn.Conv2d) and not first_done:
                w = layer.weight.data  # [out, 3, kH, kW]
                new_conv = nn.Conv2d(
                    1, layer.out_channels, layer.kernel_size,
                    stride=layer.stride, padding=layer.padding, bias=layer.bias is not None
                )
                new_conv.weight.data = w.mean(dim=1, keepdim=True)
                if layer.bias is not None:
                    new_conv.bias.data = layer.bias.data
                self.features.add_module(str(i), new_conv)
                first_done = True
            else:
                self.features.add_module(str(i), layer)

    def forward(self, x):
        return self.features(x)


def load_siamfc_checkpoint(ckpt_path, backbone):
    """Attempt to load a SiamFC checkpoint into our model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]

    # Try matching keys
    model_dict = backbone.state_dict()
    matched = {}
    for k, v in ckpt.items():
        # Strip common prefixes
        for prefix in ["backbone.", "features.", "feature_extractor.", ""]:
            clean = k.replace(prefix, "", 1) if k.startswith(prefix) else None
            if clean and clean in model_dict:
                if v.shape == model_dict[clean].shape:
                    matched[clean] = v
                elif clean.endswith(".weight") and len(v.shape) == 4 and v.shape[1] == 3:
                    # 3ch -> 1ch
                    matched[clean] = v.mean(dim=1, keepdim=True)
                break

    if matched:
        model_dict.update(matched)
        backbone.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(matched)}/{len(model_dict)} parameters from checkpoint")
    else:
        print(f"  [WARN] No matching keys found in checkpoint, using random weights")

    return backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir", default="./exported")
    parser.add_argument("--template_size", type=int, default=127)
    parser.add_argument("--search_size", type=int, default=255)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build backbone ---
    backbone = SiamFCBackbone()
    if args.checkpoint:
        backbone = load_siamfc_checkpoint(args.checkpoint, backbone)
    backbone = backbone.to(device).eval()

    # --- Export backbone ---
    dummy = torch.randn(1, 1, args.template_size, args.template_size, device=device)
    backbone_path = os.path.join(args.out_dir, "siamfc_backbone.onnx")

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
    )
    print(f"[OK] Backbone: {backbone_path}")

    with torch.no_grad():
        z = torch.randn(1, 1, args.template_size, args.template_size, device=device)
        x = torch.randn(1, 1, args.search_size, args.search_size, device=device)
        zf = backbone(z)
        xf = backbone(x)
        print(f"  template feat: {zf.shape} (expect [1, 256, 6, 6])")
        print(f"  search   feat: {xf.shape} (expect [1, 256, 22, 22])")

    # --- Export head ---
    head = SiamFCHead().to(device).eval()
    head_path = os.path.join(args.out_dir, "siamfc_head.onnx")

    # ONNX requires static shape for F.conv2d groups trick
    # Export with fixed zf shape (6x6) since template is always 127x127
    torch.onnx.export(
        head, (zf, xf), head_path,
        input_names=["template_feat", "search_feat"],
        output_names=["cls_score"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[OK] Head: {head_path}")

    with torch.no_grad():
        cls = head(zf, xf)
        print(f"  cls_score: {cls.shape} (expect [1, 1, 17, 17])")
        print(f"  NO bbox_reg output (translation-only tracker)")

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

    print("\n=== SiamFC export done ===")
    print("Compatible with NanoTrackTRT (dual-engine, reg_elements_=0)")
    print("No bbox regression -> tracker keeps original target size")
    print("\ntrtexec commands:")
    print(f"  trtexec --onnx={backbone_path} --saveEngine=siamfc_backbone.engine "
          f"--fp16 --minShapes=input:1x1x127x127 --optShapes=input:1x1x255x255 "
          f"--maxShapes=input:1x1x255x255 --workspace=256")
    print(f"  trtexec --onnx={head_path} --saveEngine=siamfc_head.engine "
          f"--fp16 --workspace=256")


if __name__ == "__main__":
    main()
