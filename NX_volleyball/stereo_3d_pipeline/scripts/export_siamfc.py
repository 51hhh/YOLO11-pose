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
import os


def _load_export_deps():
    try:
        import torch
        from siamfc_export_models import SiamFCBackbone, SiamFCHead, load_siamfc_checkpoint
    except ImportError as exc:  # pragma: no cover - export environment dependent
        raise SystemExit("PyTorch is required for SiamFC export") from exc
    return torch, SiamFCBackbone, SiamFCHead, load_siamfc_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir", default="./exported")
    parser.add_argument("--template_size", type=int, default=127)
    parser.add_argument("--search_size", type=int, default=255)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    torch, backbone_cls, head_cls, load_checkpoint = _load_export_deps()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build backbone ---
    backbone = backbone_cls()
    if args.checkpoint:
        backbone = load_checkpoint(args.checkpoint, backbone)
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
    head = head_cls().to(device).eval()
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
