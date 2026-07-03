"""CLI and gate construction for offline volleyball keypoint probe."""

from __future__ import annotations

import argparse

from stereo_feature_matching.realtime_contract import FeatureValidationConfig, StereoRoiPairGateConfig

from offline_volleyball_probe_roi import ValidationThresholds, _parse_circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/left/0000.png")
    parser.add_argument("--right", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/right/0000.png")
    parser.add_argument("--calib", default="NX_volleyball/calibration/stereo_calib.yaml")
    parser.add_argument("--out", default="NX_volleyball/stereo_3d_pipeline/test_logs/offline_keypoint_probe_latest")
    parser.add_argument("--left-circle", type=_parse_circle, help="rectified left ball circle: x,y,r")
    parser.add_argument("--right-circle", type=_parse_circle, help="rectified right ball circle: x,y,r")
    parser.add_argument("--mask-margin", type=float, default=12.0, help="pixels trimmed from ball boundary for keypoints")
    parser.add_argument("--ball-diameter-m", type=float, default=0.210, help="physical volleyball diameter for ROI sanity checks")
    parser.add_argument("--edge-percentile", type=float, default=58.0, help="color-edge percentile used for keypoint sampling")
    parser.add_argument("--iou-patch-radius", type=int, default=9)
    parser.add_argument("--iou-search-radius", type=int, default=28)
    parser.add_argument("--iou-y-radius", type=int, default=2)
    parser.add_argument("--iou-min-score", type=float, default=0.58)
    parser.add_argument("--iou-reverse-tolerance-px", type=float, default=3.0)
    parser.add_argument("--iou-max-points", type=int, default=90)
    parser.add_argument("--min-valid-matches", type=int, default=8)
    parser.add_argument("--max-y-error-px", type=float, default=2.0)
    parser.add_argument("--max-disparity-mad-px", type=float, default=1.0)
    parser.add_argument("--max-disparity-range-px", type=float, default=4.0)
    parser.add_argument("--max-z-mad-m", type=float, default=0.020)
    parser.add_argument("--max-z-range-m", type=float, default=0.060)
    parser.add_argument("--max-sphere-residual-m", type=float, default=0.030)
    parser.add_argument("--max-depth-vs-center-m", type=float, default=0.140)
    parser.add_argument("--max-disparity", type=int, default=2048)
    parser.add_argument("--pair-epipolar-y-tolerance", type=float, default=12.0)
    parser.add_argument("--pair-max-size-ratio", type=float, default=2.0)
    parser.add_argument("--pair-min-shifted-iou", type=float, default=0.0)
    parser.add_argument("--min-depth-m", type=float, default=0.8)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--feature-overlap-scale", type=float, default=0.55)
    parser.add_argument("--feature-min-support", type=int, default=4)
    parser.add_argument("--feature-max-stddev-px", type=float, default=1.0)
    parser.add_argument("--feature-sphere-radius-scale", type=float, default=1.8)
    parser.add_argument("--feature-sphere-margin-m", type=float, default=0.02)
    parser.add_argument("--quiet", action="store_true", help="write files without printing the full JSON summary")
    return parser.parse_args()


def build_validation_thresholds(args: argparse.Namespace) -> ValidationThresholds:
    return ValidationThresholds(
        min_valid_matches=args.min_valid_matches,
        max_y_error_px=args.max_y_error_px,
        max_disparity_mad_px=args.max_disparity_mad_px,
        max_disparity_range_px=args.max_disparity_range_px,
        max_z_mad_m=args.max_z_mad_m,
        max_z_range_m=args.max_z_range_m,
        max_sphere_residual_m=args.max_sphere_residual_m,
        max_depth_vs_center_m=args.max_depth_vs_center_m,
    )


def build_pair_gate(args: argparse.Namespace) -> StereoRoiPairGateConfig:
    return StereoRoiPairGateConfig(
        max_disparity=args.max_disparity,
        epipolar_y_tolerance=args.pair_epipolar_y_tolerance,
        max_size_ratio=args.pair_max_size_ratio,
        min_shifted_iou=args.pair_min_shifted_iou,
    )


def build_feature_gate(args: argparse.Namespace) -> FeatureValidationConfig:
    return FeatureValidationConfig(
        min_support=args.feature_min_support,
        max_stddev_px=args.feature_max_stddev_px,
        feature_y_tolerance_px=args.max_y_error_px,
        feature_overlap_scale=args.feature_overlap_scale,
        feature_sphere_radius_m=0.5 * args.ball_diameter_m,
        feature_sphere_radius_scale=args.feature_sphere_radius_scale,
        feature_sphere_margin_m=args.feature_sphere_margin_m,
    )
