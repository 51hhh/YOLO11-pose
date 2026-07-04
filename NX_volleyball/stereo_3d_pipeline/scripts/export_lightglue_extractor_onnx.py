#!/usr/bin/env python3
"""Export LightGlue ALIKED/SuperPoint extractors to fixed-shape ONNX.

The realtime C++ path consumes TensorRT engines that output:
  keypoints   [1, top_k, 2]
  descriptors [1, top_k, descriptor_dim]
  scores      [1, top_k]

This script exports the real LightGlue extractor models only. Matching is still
performed by the realtime C++ direct-extractor fallback unless a separate
matcher/fused engine is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_export_deps():
    try:
        import torch
        from lightglue_extractor_models import FixedExtractor
    except ImportError as exc:  # pragma: no cover - export environment dependent
        raise SystemExit("PyTorch and lightglue are required for LightGlue extractor export") from exc
    return torch, FixedExtractor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("aliked", "superpoint"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--aliked-model", default="aliked-n16")
    args = parser.parse_args()

    if args.roi_size % 32 != 0:
        raise ValueError("roi-size must be divisible by 32")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch, fixed_extractor_cls = _load_export_deps()
    model = fixed_extractor_cls(args.backend, args.top_k, args.aliked_model, args.roi_size).eval()
    dummy = torch.rand(1, 1, args.roi_size, args.roi_size, dtype=torch.float32)
    with torch.no_grad():
        keypoints, descriptors, scores = model(dummy)
    print(
        f"{args.backend}: keypoints={tuple(keypoints.shape)} "
        f"descriptors={tuple(descriptors.shape)} scores={tuple(scores.shape)}"
    )

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=["keypoints", "descriptors", "scores"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
