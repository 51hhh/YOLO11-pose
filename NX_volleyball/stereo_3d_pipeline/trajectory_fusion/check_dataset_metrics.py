"""Metric helpers for trajectory dataset quality checks."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Sequence

try:
    from .check_dataset_fields import MATCH_SOURCE_NAMES
    from .check_dataset_stats import (
        mad_value,
        median_value,
        metadata_float,
        percentile_value,
        safe_float,
        safe_int,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_dataset_fields import MATCH_SOURCE_NAMES
    from check_dataset_stats import (  # type: ignore
        mad_value,
        median_value,
        metadata_float,
        percentile_value,
        safe_float,
        safe_int,
    )


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
