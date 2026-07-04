"""Validation helpers for offline volleyball triangulation probes."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from stereo_feature_matching.realtime_contract import (
    FeatureValidationConfig,
    SparseFeatureObservation,
    validate_sparse_feature_geometry,
)

from offline_volleyball_probe_models import BallROI, MatchResult, ValidationThresholds
from offline_volleyball_probe_roi import _roi_to_runtime_detection


def _mask_value(mask: np.ndarray, x: float, y: float) -> int:
    ix = int(round(x))
    iy = int(round(y))
    if iy < 0 or iy >= mask.shape[0] or ix < 0 or ix >= mask.shape[1]:
        return 0
    return int(mask[iy, ix])


def validate_triangulated_rows(
    rows: Sequence[Dict[str, float | int | str]],
    lroi: BallROI,
    rroi: BallROI,
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    ball_center_3d: Tuple[float, float, float],
    ball_radius_m: float,
    thresholds: ValidationThresholds,
) -> List[Dict[str, float | int | str]]:
    if not rows:
        return []

    disparities = np.array([float(row["disparity_px"]) for row in rows], dtype=np.float64)
    zs = np.array([float(row["z_m"]) for row in rows], dtype=np.float64)
    disp_median = float(np.median(disparities))
    z_median = float(np.median(zs))
    cx, cy, cz = ball_center_3d

    validated: List[Dict[str, float | int | str]] = []
    for row in rows:
        out = dict(row)
        lx = float(row["left_x"])
        ly = float(row["left_y"])
        rx = float(row["right_x"])
        ry = float(row["right_y"])
        disparity = float(row["disparity_px"])
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        z_m = float(row["z_m"])
        y_error = abs(ly - ry)

        sphere_distance = math.sqrt((x_m - cx) * (x_m - cx) + (y_m - cy) * (y_m - cy) + (z_m - cz) * (z_m - cz))
        sphere_residual = abs(sphere_distance - ball_radius_m)
        disp_dev = abs(disparity - disp_median)
        z_dev = abs(z_m - z_median)
        center_depth_delta = abs(z_m - cz)

        checks = {
            "inside_left_mask": _mask_value(lroi.mask, lx, ly) > 0,
            "inside_right_mask": _mask_value(rroi.mask, rx, ry) > 0,
            "inside_left_overlap": _mask_value(left_overlap, lx, ly) > 0,
            "inside_right_overlap": _mask_value(right_overlap, rx, ry) > 0,
            "y_error_ok": y_error <= thresholds.max_y_error_px,
            "disparity_consistent": disp_dev <= max(2.0, thresholds.max_disparity_range_px * 0.5),
            "z_consistent": z_dev <= max(0.030, thresholds.max_z_range_m * 0.5),
            "sphere_residual_ok": sphere_residual <= thresholds.max_sphere_residual_m,
            "depth_near_center": center_depth_delta <= thresholds.max_depth_vs_center_m,
        }
        fail_reasons = [name for name, passed in checks.items() if not passed]

        out.update({
            "validation_pass": int(not fail_reasons),
            "validation_fail_reasons": ";".join(fail_reasons),
            "disparity_deviation_px": disp_dev,
            "z_deviation_m": z_dev,
            "sphere_distance_m": sphere_distance,
            "sphere_residual_m": sphere_residual,
            "ball_center_depth_delta_m": center_depth_delta,
            **{name: int(passed) for name, passed in checks.items()},
        })
        validated.append(out)
    return validated


def method_validation_status(
    stats: Dict[str, float | int],
    thresholds: ValidationThresholds,
) -> Tuple[str, str]:
    reasons: List[str] = []
    valid_points = int(stats.get("validation_valid_points", 0))
    if valid_points < thresholds.min_valid_matches:
        reasons.append(f"valid_points<{thresholds.min_valid_matches}")
    if float(stats.get("y_error_max_px", 1e9)) > thresholds.max_y_error_px:
        reasons.append("y_error")
    if float(stats.get("disparity_mad_px", 1e9)) > thresholds.max_disparity_mad_px:
        reasons.append("disparity_mad")
    disparity_range = float(stats.get("disparity_max_px", -1.0)) - float(stats.get("disparity_min_px", -1.0))
    if disparity_range > thresholds.max_disparity_range_px:
        reasons.append("disparity_range")
    if float(stats.get("z_mad_m", 1e9)) > thresholds.max_z_mad_m:
        reasons.append("z_mad")
    z_range = float(stats.get("z_max_m", -1.0)) - float(stats.get("z_min_m", -1.0))
    if z_range > thresholds.max_z_range_m:
        reasons.append("z_range")
    if float(stats.get("sphere_residual_max_m", 1e9)) > thresholds.max_sphere_residual_m:
        reasons.append("sphere_residual")
    return ("pass" if not reasons else "fail", ";".join(reasons))


def runtime_feature_geometry_status(
    result: MatchResult,
    lroi: BallROI,
    rroi: BallROI,
    initial_disparity: float,
    focal_px: float,
    baseline_m: float,
    config: FeatureValidationConfig,
) -> Tuple[int, float, float]:
    if not result.matches or result.disparity <= 0.5:
        return 0, 0.0, 0.0
    xs: List[float] = []
    ys: List[float] = []
    rxs: List[float] = []
    rys: List[float] = []
    for match in result.matches:
        if match.queryIdx < 0 or match.queryIdx >= len(result.left_keypoints):
            continue
        if match.trainIdx < 0 or match.trainIdx >= len(result.right_keypoints):
            continue
        kp = result.left_keypoints[match.queryIdx]
        rkp = result.right_keypoints[match.trainIdx]
        xs.append(float(kp.pt[0]))
        ys.append(float(kp.pt[1]))
        rxs.append(float(rkp.pt[0]))
        rys.append(float(rkp.pt[1]))
    if not xs:
        return 0, 0.0, 0.0
    anchor_x = float(np.mean(xs))
    anchor_y = float(np.mean(ys))
    right_anchor_x = float(np.mean(rxs)) if rxs else None
    right_anchor_y = float(np.mean(rys)) if rys else None
    observation = SparseFeatureObservation(
        valid=True,
        disparity_px=float(result.disparity),
        anchor_left_x=anchor_x,
        anchor_left_y=anchor_y,
        anchor_right_x=right_anchor_x,
        anchor_right_y=right_anchor_y,
        stddev_px=float(max(0.0, result.std_px)),
        support=len(xs),
    )
    ok = validate_sparse_feature_geometry(
        observation,
        _roi_to_runtime_detection(lroi),
        _roi_to_runtime_detection(rroi),
        initial_disparity,
        config,
        focal_px,
        baseline_m,
    )
    return int(ok), anchor_x, anchor_y
