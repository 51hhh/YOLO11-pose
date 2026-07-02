#!/usr/bin/env python3
"""Export the XFeat extractor network to fixed-shape ONNX.

The exported model is an extractor only:
  input:  images    [1, 1, roi_size, roi_size], float32
  output: feats     [1, 64, roi_size/8, roi_size/8], float32
  output: keypoints [1, 65, roi_size/8, roi_size/8], float32
  output: heatmap   [1, 1, roi_size/8, roi_size/8], float32

Realtime matching is done by NeuralFeatureMatcher after TensorRT inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xfeat-repo", required=True,
                        help="Path to verlab/accelerated_features checkout")
    parser.add_argument("--weights", default="",
                        help="Path to xfeat.pt; defaults to <xfeat-repo>/weights/xfeat.pt")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xfeat_repo = Path(args.xfeat_repo).resolve()
    weights = Path(args.weights).resolve() if args.weights else xfeat_repo / "weights" / "xfeat.pt"
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not xfeat_repo.exists():
        raise FileNotFoundError(f"XFeat repo not found: {xfeat_repo}")
    if not weights.exists():
        raise FileNotFoundError(f"XFeat weights not found: {weights}")
    if args.roi_size % 32 != 0:
        raise ValueError("roi-size must be divisible by 32 for fixed XFeat export")

    sys.path.insert(0, str(xfeat_repo))
    from modules.model import XFeatModel  # type: ignore

    model = XFeatModel().eval()
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state)

    dummy = torch.zeros(1, 1, args.roi_size, args.roi_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=["feats", "keypoints", "heatmap"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
