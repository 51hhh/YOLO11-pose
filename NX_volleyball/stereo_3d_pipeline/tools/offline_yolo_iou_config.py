"""CLI and gate construction for offline YOLO IoU fallback regression."""

from __future__ import annotations

import argparse
from pathlib import Path

from stereo_feature_matching.realtime_contract import BboxDisparityPriorConfig, StereoRoiPairGateConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True, help="baseline clip directory with frames.csv")
    parser.add_argument("--calib", type=Path, default=Path("NX_volleyball/calibration/stereo_calib.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--object-diameter-m", type=float, default=0.200)
    parser.add_argument("--bbox-scale", type=float, default=0.95)
    parser.add_argument("--max-disparity", type=int, default=2048)
    parser.add_argument("--pair-y-tolerance-px", type=float, default=12.0)
    parser.add_argument("--pair-max-size-ratio", type=float, default=2.0)
    parser.add_argument("--pair-min-shifted-iou", type=float, default=0.05)
    parser.add_argument("--bbox-consistency-ratio", type=float, default=0.30)
    parser.add_argument("--bbox-consistency-min-px", type=float, default=45.0)
    parser.add_argument("--bbox-penalty-scale", type=float, default=0.75)
    parser.add_argument("--fake-disparity-scales", default="0.55,1.45")
    parser.add_argument("--template-patch-radius", type=int, default=9)
    parser.add_argument("--template-search-margin-px", type=float, default=72.0)
    parser.add_argument("--template-y-tolerance-px", type=float, default=24.0)
    parser.add_argument("--template-min-score", type=float, default=0.20)
    parser.add_argument("--template-min-score-gap", type=float, default=0.010)
    parser.add_argument("--template-peak-exclusion-radius", type=int, default=12)
    parser.add_argument("--max-center-error-px", type=float, default=18.0)
    parser.add_argument("--max-y-error-px", type=float, default=8.0)
    parser.add_argument("--max-disparity-error-px", type=float, default=18.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--min-pass-rate", type=float, default=0.99)
    return parser.parse_args()


def build_pair_gate(args: argparse.Namespace) -> StereoRoiPairGateConfig:
    return StereoRoiPairGateConfig(
        max_disparity=args.max_disparity,
        epipolar_y_tolerance=args.pair_y_tolerance_px,
        max_size_ratio=args.pair_max_size_ratio,
        min_shifted_iou=args.pair_min_shifted_iou,
    )


def build_bbox_prior(args: argparse.Namespace) -> BboxDisparityPriorConfig:
    return BboxDisparityPriorConfig(
        object_diameter_m=args.object_diameter_m,
        bbox_scale=args.bbox_scale,
        consistency_ratio=args.bbox_consistency_ratio,
        consistency_min_px=args.bbox_consistency_min_px,
        penalty_scale=args.bbox_penalty_scale,
    )


def parse_fake_scales(value: str) -> list[float]:
    fake_scales = sorted(float(v.strip()) for v in value.split(",") if v.strip())
    if len(fake_scales) != 2:
        raise ValueError("--fake-disparity-scales must contain two comma-separated values")
    return fake_scales
