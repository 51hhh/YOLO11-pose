#!/usr/bin/env python3
"""Evaluate trajectory stability and physics consistency."""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
from statistics import mean, pstdev
from collections import Counter
from typing import Dict, List


DEPTH_KEYS = [
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
    "z_fallback",
    "z_fallback_template",
    "z_fallback_feature_points",
    "z_stereo",
    "z",
]


def _read(path: str) -> List[Dict[str, str]]:
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _series(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [_f(row, key) for row in rows if _f(row, key) > -1e20]


def _valid_depth_series(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [_f(row, key) for row in rows if key in row and _f(row, key, -1.0) > 0.1]


def _diff(values: List[float]) -> List[float]:
    return [b - a for a, b in zip(values, values[1:])]


def _rms(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


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
    speed_z = _rms(dz)
    accel_z = _rms(ddz)
    jerk_z = _rms(dddz)
    return {
        "frames": float(len(rows)),
        "x_std": pstdev(xs) if len(xs) > 1 else 0.0,
        "y_std": pstdev(ys) if len(ys) > 1 else 0.0,
        "z_mean": mean(zs) if zs else 0.0,
        "z_std": pstdev(zs) if len(zs) > 1 else 0.0,
        "z_peak_to_peak": max(zs) - min(zs) if zs else 0.0,
        "dz_rms": speed_z,
        "ddz_rms": accel_z,
        "dddz_rms": jerk_z,
    }


def _group_by_track(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("track_id", "-1"), []).append(row)
    for item in grouped.values():
        item.sort(key=lambda r: (_f(r, "timestamp"), _f(r, "frame_id")))
    return grouped


def _print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(
        f"{name}: frames={metrics['frames']:.0f} z_mean={metrics['z_mean']:.4f} "
        f"z_std={metrics['z_std']:.4f} p2p={metrics['z_peak_to_peak']:.4f} "
        f"dz_rms={metrics['dz_rms']:.4f} ddz_rms={metrics['ddz_rms']:.4f} "
        f"jerk_rms={metrics['dddz_rms']:.4f}"
    )


def _print_depth_candidate_metrics(track_id: str, rows: List[Dict[str, str]]) -> None:
    for key in DEPTH_KEYS:
        if not rows or key not in rows[0]:
            continue
        values = _valid_depth_series(rows, key)
        if not values:
            continue
        hit_rate = len(values) / max(1, len(rows))
        print(
            f"track={track_id} {key}: valid={len(values)}/{len(rows)} "
            f"hit={hit_rate * 100:.1f}% mean={mean(values):.4f} "
            f"std={pstdev(values) if len(values) > 1 else 0.0:.4f} "
            f"p2p={(max(values) - min(values)) if values else 0.0:.4f}"
        )


def _print_sync_and_source_metrics(track_id: str, rows: List[Dict[str, str]]) -> None:
    if rows and "stereo_match_source" in rows[0]:
        sources = Counter(str(int(_f(row, "stereo_match_source", 0))) for row in rows)
        depth_sources = Counter(str(int(_f(row, "stereo_depth_source", 0))) for row in rows)
        print(
            f"track={track_id} source: match={dict(sources)} "
            f"depth={dict(depth_sources)}"
        )
    if rows and "left_circle_source" in rows[0]:
        left_sources = Counter(str(int(_f(row, "left_circle_source", 0))) for row in rows)
        right_sources = Counter(str(int(_f(row, "right_circle_source", 0))) for row in rows)
        print(
            f"track={track_id} circle_source: left={dict(left_sources)} "
            f"right={dict(right_sources)}"
        )
    for key in ("frame_counter_delta", "frame_number_delta", "timestamp_delta_us"):
        if not rows or key not in rows[0]:
            continue
        values = [_f(row, key) for row in rows]
        if not values:
            continue
        print(
            f"track={track_id} {key}: mean={mean(values):.3f} "
            f"std={pstdev(values) if len(values) > 1 else 0.0:.3f} "
            f"min={min(values):.0f} max={max(values):.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Raw or smoothed CSV")
    args = parser.parse_args()

    rows = _read(args.input)
    grouped = _group_by_track(rows)
    has_smooth = bool(rows and "smooth_z" in rows[0])
    for track_id, track_rows in grouped.items():
        raw = _metrics(track_rows)
        _print_metrics(f"track={track_id} raw", raw)
        _print_depth_candidate_metrics(track_id, track_rows)
        _print_sync_and_source_metrics(track_id, track_rows)
        if has_smooth:
            smooth = _metrics(track_rows, prefix="smooth_")
            ratio = smooth["z_std"] / raw["z_std"] if raw["z_std"] > 1e-9 else 0.0
            _print_metrics(f"track={track_id} smooth", smooth)
            print(f"track={track_id} smooth/raw z_std ratio={ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
