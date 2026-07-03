"""Metric helpers for trajectory dataset quality checks."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

try:
    from .dataset import derive_frame_summary_path, read_csv_rows
    from .check_dataset_fields import (
        FRAME_SUMMARY_FIELDS,
        MATCH_SOURCE_NAMES,
        OPTIONAL_FRAME_SUMMARY_FIELDS,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import derive_frame_summary_path, read_csv_rows
    from check_dataset_fields import (
        FRAME_SUMMARY_FIELDS,
        MATCH_SOURCE_NAMES,
        OPTIONAL_FRAME_SUMMARY_FIELDS,
    )


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def median_value(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def mad_value(values: Sequence[float]) -> float | None:
    med = median_value(values)
    if med is None:
        return None
    return median_value([abs(value - med) for value in values])


def percentile_value(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def metadata_float(metadata: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def depth_value(row: Dict[str, str], key: str) -> object:
    if key == "z_fallback_epipolar":
        return row.get("z_fallback_epipolar", row.get("z_fallback"))
    return row.get(key)


def valid_depths(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    return [
        safe_float(depth_value(row, key), -1.0)
        for row in rows
        if safe_float(depth_value(row, key), -1.0) > 0.1
    ]


def field_set(rows: Sequence[Dict[str, str]]) -> set[str]:
    if not rows:
        return set()
    return set(rows[0].keys())


def frame_gaps(rows: Sequence[Dict[str, str]]) -> List[Dict[str, int]]:
    frames = [safe_int(row.get("frame_id"), -1) for row in rows]
    gaps: List[Dict[str, int]] = []
    for prev_frame, frame in zip(frames, frames[1:]):
        delta = frame - prev_frame
        if delta != 1:
            gaps.append({"from": prev_frame, "to": frame, "delta": delta})
    return gaps


def timing_stats(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    timestamps = [safe_float(row.get("timestamp"), math.nan) for row in rows]
    timestamps = [value for value in timestamps if not math.isnan(value)]
    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    return {
        "rows": len(rows),
        "duration_sec": duration,
        "fps_rows": len(rows) / duration if duration > 0.0 else None,
        "fps_intervals": (len(rows) - 1) / duration if duration > 0.0 and len(rows) > 1 else None,
    }


def delta_stats(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Any]:
    if not rows or key not in rows[0]:
        return {"present": False}
    values = [safe_float(row.get(key), 0.0) for row in rows]
    return {
        "present": True,
        "count": len(values),
        "nonzero": sum(1 for value in values if abs(value) > 1e-9),
        "unique": sorted(set(values))[:16],
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
    }


def depth_stats(rows: Sequence[Dict[str, str]], key: str, known_z: float | None) -> Dict[str, Any]:
    values = valid_depths(rows, key)
    med = median_value(values)
    errors = [value - known_z for value in values] if known_z is not None and known_z > 0.0 else []
    return {
        "valid": len(values),
        "total": len(rows),
        "hit_rate": len(values) / len(rows) if rows else 0.0,
        "median": med,
        "mad": mad_value(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "known_z_bias": mean(errors) if errors else None,
        "known_z_mad": mad_value(errors) if errors else None,
    }


def raw_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows or "raw_observation_valid" not in rows[0]:
        return list(rows)
    return [row for row in rows if safe_int(row.get("raw_observation_valid"), 0) == 1]


def source_breakdown(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    raw = raw_rows(rows)
    match_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    fallback_epipolar_rows: List[Dict[str, str]] = []
    fallback_by_direction: Counter[str] = Counter()

    for row in raw:
        match_source = safe_int(row.get("stereo_match_source"), 0)
        match_name = MATCH_SOURCE_NAMES.get(match_source, f"unknown_{match_source}")
        match_counts[match_name] += 1

        left_source = safe_int(row.get("left_circle_source"), -1)
        right_source = safe_int(row.get("right_circle_source"), -1)
        depth_source = safe_int(row.get("stereo_depth_source"), -1)
        combo_key = (
            f"match={match_source},left_circle={left_source},"
            f"right_circle={right_source},depth={depth_source}"
        )
        combo_counts[combo_key] += 1

        if (
            match_source in (2, 3)
            and (left_source == 3 or right_source == 3)
            and safe_float(row.get("z_fallback_epipolar", row.get("z_fallback")), -1.0) > 0.1
        ):
            fallback_epipolar_rows.append(row)
            fallback_by_direction[MATCH_SOURCE_NAMES[match_source]] += 1

    fallback_values = [
        safe_float(row.get("z_fallback_epipolar", row.get("z_fallback")), -1.0)
        for row in fallback_epipolar_rows
    ]
    return {
        "rows": len(rows),
        "raw_rows": len(raw),
        "match_source": dict(match_counts),
        "source_combinations": dict(combo_counts),
        "epipolar_fallback": {
            "valid": len(fallback_values),
            "by_direction": dict(fallback_by_direction),
            "median": median_value(fallback_values),
            "mad": mad_value(fallback_values),
            "min": min(fallback_values) if fallback_values else None,
            "max": max(fallback_values) if fallback_values else None,
        },
    }


def depth_jump_stats(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Any]:
    if not rows or (key not in rows[0] and not (key == "z_fallback_epipolar" and "z_fallback" in rows[0])):
        return {"present": False}

    by_track: Dict[str, List[tuple[int, float]]] = {}
    for row in rows:
        value = safe_float(depth_value(row, key), -1.0)
        if value <= 0.1:
            continue
        track_id = row.get("track_id") or "0"
        by_track.setdefault(track_id, []).append((safe_int(row.get("frame_id"), 0), value))

    deltas: List[float] = []
    for values in by_track.values():
        values.sort(key=lambda item: item[0])
        for (_, prev), (_, cur) in zip(values, values[1:]):
            deltas.append(abs(cur - prev))

    return {
        "present": True,
        "pairs": len(deltas),
        "median_abs_delta": median_value(deltas),
        "mad_abs_delta": mad_value(deltas),
        "p95_abs_delta": percentile_value(deltas, 95.0),
        "max_abs_delta": max(deltas) if deltas else None,
    }


def frame_summary_report(csv_path: Path) -> Dict[str, Any]:
    frame_path = derive_frame_summary_path(csv_path)
    if not frame_path.exists():
        return {"path": str(frame_path), "present": False}

    rows = read_csv_rows(frame_path)
    fields = field_set(rows)
    missing = [field for field in FRAME_SUMMARY_FIELDS if field not in fields]
    gaps = frame_gaps(rows)
    timing = timing_stats(rows)

    def sum_field(key: str) -> int:
        return sum(safe_int(row.get(key), 0) for row in rows)

    def max_field(key: str) -> int | None:
        if not rows or key not in fields:
            return None
        return max(safe_int(row.get(key), 0) for row in rows)

    totals = {
        "result_count": sum_field("result_count"),
        "raw_observation_count": sum_field("raw_observation_count"),
        "stereo_observation_count": sum_field("stereo_observation_count"),
        "direct_pair_count": sum_field("direct_pair_count"),
        "fallback_l2r_count": sum_field("fallback_l2r_count"),
        "fallback_r2l_count": sum_field("fallback_r2l_count"),
    }
    max_per_frame = {
        "result_count": max_field("result_count"),
        "raw_observation_count": max_field("raw_observation_count"),
        "stereo_observation_count": max_field("stereo_observation_count"),
        "fallback_l2r_count": max_field("fallback_l2r_count"),
        "fallback_r2l_count": max_field("fallback_r2l_count"),
    }
    for field in OPTIONAL_FRAME_SUMMARY_FIELDS:
        if field in fields:
            totals[field] = sum_field(field)
            max_per_frame[field] = max_field(field)

    return {
        "path": str(frame_path),
        "present": True,
        "rows": len(rows),
        "duration_sec": timing["duration_sec"],
        "fps_rows": timing["fps_rows"],
        "fps_intervals": timing["fps_intervals"],
        "frame_gaps": {
            "count": len(gaps),
            "first": gaps[:10],
        },
        "missing_fields": missing,
        "totals": totals,
        "max_per_frame": max_per_frame,
    }
