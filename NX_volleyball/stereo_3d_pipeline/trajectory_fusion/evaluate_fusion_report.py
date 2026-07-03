"""Machine-readable report builders for trajectory fusion evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

try:
    from .evaluate_fusion_stats import (
        CANDIDATE_DEPTH_KEYS,
        LEGACY_DEPTH_KEYS,
        P0_DEPTH_KEYS,
        _f,
        _group_by_track,
        _mad,
        _median,
        _metadata_float,
        _metrics,
        _valid_depth_series,
    )
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_fusion_stats import (
        CANDIDATE_DEPTH_KEYS,
        LEGACY_DEPTH_KEYS,
        P0_DEPTH_KEYS,
        _f,
        _group_by_track,
        _mad,
        _median,
        _metadata_float,
        _metrics,
        _valid_depth_series,
    )


def _numeric_summary(values: List[float]) -> Dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else 0.0 if values else None,
        "median": _median(values) if values else None,
        "mad": _mad(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _depth_report(rows: List[Dict[str, str]], key: str, known_z: float) -> Dict[str, float | int | None]:
    values = _valid_depth_series(rows, key)
    stats = _numeric_summary(values)
    errors = [value - known_z for value in values] if known_z > 0.0 else []
    stats.update(
        {
            "valid": len(values),
            "total": len(rows),
            "hit_rate": len(values) / max(1, len(rows)),
            "known_z_bias": mean(errors) if errors else None,
            "known_z_mad": _mad(errors) if errors else None,
        }
    )
    return stats


def _p0_median_report(rows: List[Dict[str, str]], known_z: float) -> Dict[str, float | int | None]:
    medians: List[float] = []
    for row in rows:
        values = [_f(row, key, -1.0) for key in P0_DEPTH_KEYS if key in row and _f(row, key, -1.0) > 0.1]
        if values:
            medians.append(_median(values))
    stats = _numeric_summary(medians)
    errors = [value - known_z for value in medians] if known_z > 0.0 else []
    stats.update(
        {
            "valid": len(medians),
            "total": len(rows),
            "hit_rate": len(medians) / max(1, len(rows)),
            "known_z_bias": mean(errors) if errors else None,
            "known_z_mad": _mad(errors) if errors else None,
        }
    )
    return stats


def _sync_report(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if rows and "stereo_match_source" in rows[0]:
        report["stereo_match_source"] = dict(Counter(str(int(_f(row, "stereo_match_source", 0))) for row in rows))
    if rows and "stereo_depth_source" in rows[0]:
        report["stereo_depth_source"] = dict(Counter(str(int(_f(row, "stereo_depth_source", 0))) for row in rows))
    if rows and "left_circle_source" in rows[0]:
        report["left_circle_source"] = dict(Counter(str(int(_f(row, "left_circle_source", 0))) for row in rows))
        report["right_circle_source"] = dict(Counter(str(int(_f(row, "right_circle_source", 0))) for row in rows))
    for key in ("frame_counter_delta", "frame_number_delta", "timestamp_delta_us"):
        if rows and key in rows[0]:
            report[key] = _numeric_summary([_f(row, key) for row in rows])
    return report


def _pair_report(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    pair_keys = (
        "pair_initial_disparity",
        "pair_epipolar_dy",
        "pair_y_tolerance",
        "pair_size_ratio",
        "pair_shifted_iou",
        "pair_score",
        "pair_bbox_prior_penalty",
        "pair_positive_disparity",
    )
    report: Dict[str, Any] = {}
    for key in pair_keys:
        if not rows or key not in rows[0]:
            continue
        if key == "pair_score":
            values = [_f(row, key, 0.0) for row in rows if row.get(key, "") != ""]
        else:
            values = [_f(row, key, -1.0) for row in rows if _f(row, key, -1.0) >= 0.0]
        if values:
            report[key] = _numeric_summary(values)
    return report


def build_report(rows: List[Dict[str, str]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    grouped = _group_by_track(rows)
    known_z = _metadata_float(metadata, "known_z_m", "known_z", "known_distance_m")
    has_smooth = bool(rows and "smooth_z" in rows[0])
    tracks: Dict[str, Any] = {}
    for track_id, track_rows in grouped.items():
        track_report: Dict[str, Any] = {
            "raw": _metrics(track_rows),
            "candidate_depths": {
                key: _depth_report(track_rows, key, known_z)
                for key in CANDIDATE_DEPTH_KEYS
                if track_rows and key in track_rows[0]
            },
            "legacy_depths": {
                key: _depth_report(track_rows, key, known_z)
                for key in LEGACY_DEPTH_KEYS
                if track_rows and key in track_rows[0]
            },
            "p0_median": _p0_median_report(track_rows, known_z),
            "sync": _sync_report(track_rows),
            "pair_gate": _pair_report(track_rows),
        }
        if has_smooth:
            smooth = _metrics(track_rows, prefix="smooth_")
            raw = track_report["raw"]
            smooth["raw_z_std_ratio"] = smooth["z_std"] / raw["z_std"] if raw["z_std"] > 1e-9 else 0.0
            track_report["smooth"] = smooth
        tracks[str(track_id)] = track_report
    return {
        "metadata": metadata,
        "known_z": known_z if known_z > 0.0 else None,
        "track_count": len(tracks),
        "tracks": tracks,
    }


def write_json_report(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _flatten_report(prefix: str, value: Any, rows: List[Dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_report(next_prefix, child, rows)
        return
    rows.append({"path": prefix, "value": "" if value is None else value})


def write_csv_report(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for track_id, track_report in report["tracks"].items():
        flattened: List[Dict[str, Any]] = []
        _flatten_report("", track_report, flattened)
        for row in flattened:
            rows.append({"track_id": track_id, **row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "path", "value"])
        writer.writeheader()
        writer.writerows(rows)
