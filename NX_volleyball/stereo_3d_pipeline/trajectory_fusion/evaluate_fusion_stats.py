"""Shared statistics helpers for trajectory fusion evaluation."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Dict, List


CANDIDATE_DEPTH_KEYS = [
    "z_mono",
    "z_bbox_center",
    "z_bbox_left_edge",
    "z_bbox_right_edge",
    "z_circle_center",
    "z_circle_left_edge",
    "z_circle_right_edge",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
    "z_roi_corner_points",
    "z_roi_texture_points",
    "z_roi_binary_points",
    "z_roi_orb_points",
    "z_roi_brisk_points",
    "z_roi_akaze_points",
    "z_roi_sift_points",
    "z_roi_iou_region_color_patch",
    "z_roi_patch_iou_color_edge",
    "z_roi_neural_feature",
    "z_roi_center_patch",
    "z_roi_multi_point",
    "z_fallback_epipolar",
    "z_fallback_template",
    "z_fallback_feature_points",
]
LEGACY_DEPTH_KEYS = [
    "z_stereo",
    "z",
]
P0_DEPTH_KEYS = [
    "z_bbox_center",
    "z_circle_center",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
]


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _series(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [_f(row, key) for row in rows if _f(row, key) > -1e20]


def _valid_depth_series(rows: List[Dict[str, str]], key: str) -> List[float]:
    values = []
    for row in rows:
        source_key = key
        if key == "z_fallback_epipolar" and key not in row:
            source_key = "z_fallback"
        if source_key in row and _f(row, source_key, -1.0) > 0.1:
            values.append(_f(row, source_key))
    return values


def _diff(values: List[float]) -> List[float]:
    return [b - a for a, b in zip(values, values[1:])]


def _rms(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: List[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    return _median([abs(value - med) for value in values])


def _metadata_float(metadata: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _metrics(rows: List[Dict[str, str]], prefix: str = "") -> Dict[str, float]:
    x_key = f"{prefix}x" if prefix else "x"
    y_key = f"{prefix}y" if prefix else "y"
    z_key = f"{prefix}z" if prefix else "z"
    xs = _series(rows, x_key)
    ys = _series(rows, y_key)
    zs = _series(rows, z_key)
    dz = _diff(zs)
    ddz = _diff(dz)
    dddz = _diff(ddz)
    return {
        "frames": float(len(rows)),
        "x_std": pstdev(xs) if len(xs) > 1 else 0.0,
        "y_std": pstdev(ys) if len(ys) > 1 else 0.0,
        "z_mean": mean(zs) if zs else 0.0,
        "z_std": pstdev(zs) if len(zs) > 1 else 0.0,
        "z_peak_to_peak": max(zs) - min(zs) if zs else 0.0,
        "dz_rms": _rms(dz),
        "ddz_rms": _rms(ddz),
        "dddz_rms": _rms(dddz),
    }


def _group_by_track(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("track_id", "-1"), []).append(row)
    for item in grouped.values():
        item.sort(key=lambda r: (_f(r, "timestamp"), _f(r, "frame_id")))
    return grouped
