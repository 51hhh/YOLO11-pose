"""Triangulation and validation helpers for offline volleyball probes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from offline_volleyball_depth_math import depth_from_disparity
from offline_volleyball_probe_models import BallROI, MatchResult
from offline_volleyball_probe_validation import (
    method_validation_status,
    runtime_feature_geometry_status,
    validate_triangulated_rows,
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
