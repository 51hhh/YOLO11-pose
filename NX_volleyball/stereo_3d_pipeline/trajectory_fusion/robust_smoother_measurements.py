"""Depth measurement grouping helpers for the robust trajectory smoother."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .dataset import METHOD_COLUMNS
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _method_sigma_scale(name: str) -> float:
    if name == "mono":
        return 1.70
    if name.startswith("bbox"):
        return 1.35
    if name.startswith("circle"):
        return 1.05
    if name in {"roi_edge_centroid", "roi_radial_center", "roi_edge_pair_center"}:
        return 1.10
    if name in {"roi_multi_point", "roi_center_patch"}:
        return 1.25
    if "fallback" in name:
        return 2.00
    if name.startswith("roi_"):
        return 1.80
    return 1.50


def _method_group(name: str) -> str:
    if name == "mono":
        return "mono"
    if name.startswith("bbox"):
        return "bbox"
    if name.startswith("circle"):
        return "circle"
    if name in {"roi_edge_centroid", "roi_radial_center", "roi_edge_pair_center"}:
        return "roi_geometry"
    if name in {"roi_multi_point", "roi_center_patch"}:
        return "roi_patch"
    if "fallback" in name:
        return "fallback"
    return "roi_sparse"


def _quality_scale(row: Dict[str, float], name: str) -> float:
    support_key = f"{name}_support"
    std_key = f"{name}_std_px"
    conf_key = f"{name}_confidence"
    if name == "roi_multi_point":
        support_key = "subpixel_support"
        std_key = "subpixel_std_px"
        conf_key = "subpixel_confidence"

    scale = 1.0
    support = row.get(support_key, 0.0)
    if 0.0 < support < 4.0:
        scale *= 1.8
    std_px = row.get(std_key, -1.0)
    if std_px > 0.0:
        scale *= max(1.0, min(4.0, std_px / 1.5))
    confidence = row.get(conf_key, 0.0)
    if 0.0 < confidence < 0.4:
        scale *= 1.5

    pair_dy = abs(row.get("pair_epipolar_dy", -1.0))
    pair_tol = row.get("pair_y_tolerance", -1.0)
    if pair_dy >= 0.0 and pair_tol > 0.0 and pair_dy > pair_tol:
        scale *= 2.0
    shifted_iou = row.get("pair_shifted_iou", -1.0)
    if 0.0 <= shifted_iou < 0.2:
        scale *= 1.6
    if row.get("pair_positive_disparity", 1.0) == 0.0:
        scale *= 2.0
    return scale


def z_measurements(row: Dict[str, float], cfg: Any) -> List[Tuple[float, float, str]]:
    if not cfg.use_method_depths:
        return []
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for name, key in METHOD_COLUMNS:
        z_value = row.get(key, -1.0)
        if z_value <= 0.1:
            continue
        sigma = cfg.base_sigma_z * _method_sigma_scale(name) * _quality_scale(row, name) * max(1.0, z_value)
        grouped.setdefault(_method_group(name), []).append((z_value, sigma * sigma))

    out: List[Tuple[float, float, str]] = []
    for group, values in grouped.items():
        zs = [item[0] for item in values]
        variances = [item[1] for item in values]
        z_group = _median(zs)
        spread = _median([abs(value - z_group) for value in zs])
        variance = max(min(variances), spread * spread, (0.01 * max(1.0, z_group)) ** 2)
        out.append((z_group, variance, group))
    return out


def initial_z(row: Dict[str, float]) -> float:
    candidates = [row.get(key, -1.0) for _, key in METHOD_COLUMNS]
    valid = [value for value in candidates if value > 0.1]
    if valid:
        return _median(valid)
    return max(row.get("z", -1.0), row.get("z_stereo", -1.0), row.get("z_mono", 0.1), 0.1)
