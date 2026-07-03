"""Triangulation and validation helpers for offline volleyball probes."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from offline_volleyball_depth_math import depth_from_disparity
from stereo_feature_matching.realtime_contract import (
    FeatureValidationConfig,
    SparseFeatureObservation,
    validate_sparse_feature_geometry,
)

from offline_volleyball_probe_roi import (
    BallROI,
    MatchResult,
    ValidationThresholds,
    _roi_to_runtime_detection,
)


def triangulate_match_rows(
    result: MatchResult,
    calib: Dict[str, np.ndarray | float | int],
    baseline_m: float,
) -> List[Dict[str, float | int | str]]:
    p1 = np.asarray(calib["P1"], dtype=np.float64)
    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])

    rows: List[Dict[str, float | int | str]] = []
    for idx, match in enumerate(result.matches):
        if match.queryIdx >= len(result.left_keypoints) or match.trainIdx >= len(result.right_keypoints):
            continue
        lpt = result.left_keypoints[match.queryIdx].pt
        rpt = result.right_keypoints[match.trainIdx].pt
        disparity = float(lpt[0] - rpt[0])
        if disparity <= 0.0:
            continue
        z_m = fx * baseline_m / disparity
        x_m = (float(lpt[0]) - cx) * z_m / fx
        y_m = (float(lpt[1]) - cy) * z_m / fy
        row: Dict[str, float | int | str] = {
            "method": result.name,
            "index": idx,
            "left_x": float(lpt[0]),
            "left_y": float(lpt[1]),
            "right_x": float(rpt[0]),
            "right_y": float(rpt[1]),
            "y_error_px": abs(float(lpt[1]) - float(rpt[1])),
            "disparity_px": disparity,
            "x_m": x_m,
            "y_m": y_m,
            "z_m": z_m,
        }
        if idx < len(result.extras):
            row.update(result.extras[idx])
        rows.append(row)
    return rows


def estimate_ball_center_3d(
    calib: Dict[str, np.ndarray | float | int],
    lroi: BallROI,
    center_disparity: float,
    baseline_m: float,
) -> Tuple[float, float, float]:
    p1 = np.asarray(calib["P1"], dtype=np.float64)
    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])
    z_m = depth_from_disparity(center_disparity, fx, baseline_m)
    x_m = (float(lroi.center[0]) - cx) * z_m / fx
    y_m = (float(lroi.center[1]) - cy) * z_m / fy
    return x_m, y_m, z_m


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


def triangulation_stats(rows: Sequence[Dict[str, float | int | str]]) -> Dict[str, float | int]:
    z = np.array([float(row["z_m"]) for row in rows if float(row["z_m"]) > 0.0], dtype=np.float64)
    d = np.array([float(row["disparity_px"]) for row in rows if float(row["disparity_px"]) > 0.0], dtype=np.float64)
    yerr = np.array([float(row.get("y_error_px", 0.0)) for row in rows], dtype=np.float64)
    sphere_residual = np.array([float(row.get("sphere_residual_m", -1.0)) for row in rows if float(row.get("sphere_residual_m", -1.0)) >= 0.0], dtype=np.float64)
    valid = np.array([int(row.get("validation_pass", 0)) for row in rows], dtype=np.int32)
    if z.size == 0 or d.size == 0:
        return {
            "triangulated_points": 0,
            "validation_valid_points": 0,
            "validation_valid_ratio": 0.0,
            "z_median_m": -1.0,
            "z_mad_m": -1.0,
            "z_min_m": -1.0,
            "z_max_m": -1.0,
            "disparity_median_px": -1.0,
            "disparity_mad_px": -1.0,
            "disparity_min_px": -1.0,
            "disparity_max_px": -1.0,
            "y_error_max_px": -1.0,
            "sphere_residual_median_m": -1.0,
            "sphere_residual_max_m": -1.0,
        }
    return {
        "triangulated_points": int(z.size),
        "validation_valid_points": int(valid.sum()) if valid.size else 0,
        "validation_valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "z_median_m": float(np.median(z)),
        "z_mad_m": float(np.median(np.abs(z - np.median(z)))),
        "z_min_m": float(np.min(z)),
        "z_max_m": float(np.max(z)),
        "disparity_median_px": float(np.median(d)),
        "disparity_mad_px": float(np.median(np.abs(d - np.median(d)))),
        "disparity_min_px": float(np.min(d)),
        "disparity_max_px": float(np.max(d)),
        "y_error_max_px": float(np.max(yerr)) if yerr.size else -1.0,
        "sphere_residual_median_m": float(np.median(sphere_residual)) if sphere_residual.size else -1.0,
        "sphere_residual_max_m": float(np.max(sphere_residual)) if sphere_residual.size else -1.0,
    }


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


def write_triangulated_points(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
