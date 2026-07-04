#!/usr/bin/env python3
"""
Adapt existing NanoTrack 3ch ONNX models to 1ch grayscale + dynamic shape.

Takes the pre-existing NanoTrack ONNX models and produces models compatible
with the NanoTrackTRT C++ code:
  - Backbone: 1ch input, dynamic H/W (127 or 255)
  - Head: template_feat/search_feat/cls_score/bbox_reg tensor names
"""

from __future__ import annotations

import argparse
import os


def _load_adapter_deps():
    try:
        from nanotrack_onnx_adapter import adapt_head, merge_backbones_to_dynamic
    except ImportError as exc:  # pragma: no cover - export environment dependent
        raise SystemExit("onnx is required; install with: pip install onnx") from exc
    return adapt_head, merge_backbones_to_dynamic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_template", required=True)
    parser.add_argument("--backbone_search", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out_dir", default="./adapted")
    args = parser.parse_args()

    adapt_head, merge_backbones_to_dynamic = _load_adapter_deps()
    os.makedirs(args.out_dir, exist_ok=True)

    backbone_out = os.path.join(args.out_dir, "nanotrack_backbone_1ch.onnx")
    head_out = os.path.join(args.out_dir, "nanotrack_head_adapted.onnx")

    merge_backbones_to_dynamic(args.backbone_template, args.backbone_search, backbone_out)
    adapt_head(args.head, head_out)

    print("\n========================================")
    print("=== Adaptation complete ===")
    print(f"  Backbone: {backbone_out}  (1ch, dynamic H/W)")
    print(f"  Head: {head_out}  (renamed I/O)")
    print("\n  trtexec commands:")
    print(f"  trtexec --onnx={backbone_out} --saveEngine=nanotrack_backbone.engine \\")
    print("    --fp16 --minShapes=input:1x1x127x127 --optShapes=input:1x1x255x255 \\")
    print("    --maxShapes=input:1x1x255x255 --workspace=256")
    print(f"  trtexec --onnx={head_out} --saveEngine=nanotrack_head.engine \\")
    print("    --fp16 --workspace=256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
