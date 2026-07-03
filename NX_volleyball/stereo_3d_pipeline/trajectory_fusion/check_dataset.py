#!/usr/bin/env python3
"""Quality checks for TrajectoryRecorder CSV datasets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

try:
    from .dataset import derive_frame_summary_path, find_metadata_for_csv, read_csv_rows, read_metadata
except ImportError:  # pragma: no cover - direct script execution
    from dataset import derive_frame_summary_path, find_metadata_for_csv, read_csv_rows, read_metadata


P0_DEPTH_KEYS = (
    "z_bbox_center",
    "z_circle_center",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
)
P1_DEPTH_KEYS = (
    "z_roi_multi_point",
    "z_roi_center_patch",
)
DEPTH_KEYS = P0_DEPTH_KEYS + P1_DEPTH_KEYS
JUMP_DEPTH_KEYS = DEPTH_KEYS + ("z_fallback",)
REQUIRED_FIELDS = (
    "frame_id",
    "timestamp",
    "track_id",
    "frame_counter_delta",
    "frame_number_delta",
    "stereo_match_source",
    "pair_positive_disparity",
    *DEPTH_KEYS,
)
FRAME_SUMMARY_FIELDS = (
    "frame_id",
    "result_count",
    "raw_observation_count",
    "stereo_observation_count",
    "direct_pair_count",
    "fallback_l2r_count",
    "fallback_r2l_count",
)
MATCH_SOURCE_NAMES = {
    0: "none",
    1: "direct_pair",
    2: "fallback_l2r",
    3: "fallback_r2l",
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: Sequence[float]) -> float | None:
    med = _median(values)
    if med is None:
        return None
    return _median([abs(value - med) for value in values])


def _percentile(values: Sequence[float], pct: float) -> float | None:
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


def _metadata_float(metadata: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _valid_depths(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    return [_safe_float(row.get(key), -1.0) for row in rows if _safe_float(row.get(key), -1.0) > 0.1]


def _field_set(rows: Sequence[Dict[str, str]]) -> set[str]:
    if not rows:
        return set()
    return set(rows[0].keys())


def _frame_gaps(rows: Sequence[Dict[str, str]]) -> List[Dict[str, int]]:
    frames = [_safe_int(row.get("frame_id"), -1) for row in rows]
    gaps: List[Dict[str, int]] = []
    for prev_frame, frame in zip(frames, frames[1:]):
        delta = frame - prev_frame
        if delta != 1:
            gaps.append({"from": prev_frame, "to": frame, "delta": delta})
    return gaps


def _delta_stats(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Any]:
    if not rows or key not in rows[0]:
        return {"present": False}
    values = [_safe_float(row.get(key), 0.0) for row in rows]
    return {
        "present": True,
        "count": len(values),
        "nonzero": sum(1 for value in values if abs(value) > 1e-9),
        "unique": sorted(set(values))[:16],
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
    }


def _depth_stats(rows: Sequence[Dict[str, str]], key: str, known_z: float | None) -> Dict[str, Any]:
    values = _valid_depths(rows, key)
    med = _median(values)
    errors = [value - known_z for value in values] if known_z is not None and known_z > 0.0 else []
    return {
        "valid": len(values),
        "total": len(rows),
        "hit_rate": len(values) / len(rows) if rows else 0.0,
        "median": med,
        "mad": _mad(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "known_z_bias": mean(errors) if errors else None,
        "known_z_mad": _mad(errors) if errors else None,
    }


def _raw_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows or "raw_observation_valid" not in rows[0]:
        return list(rows)
    return [row for row in rows if _safe_int(row.get("raw_observation_valid"), 0) == 1]


def _source_breakdown(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    raw = _raw_rows(rows)
    match_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    fallback_epipolar_rows: List[Dict[str, str]] = []
    fallback_by_direction: Counter[str] = Counter()

    for row in raw:
        match_source = _safe_int(row.get("stereo_match_source"), 0)
        match_name = MATCH_SOURCE_NAMES.get(match_source, f"unknown_{match_source}")
        match_counts[match_name] += 1

        left_source = _safe_int(row.get("left_circle_source"), -1)
        right_source = _safe_int(row.get("right_circle_source"), -1)
        depth_source = _safe_int(row.get("stereo_depth_source"), -1)
        combo_key = (
            f"match={match_source},left_circle={left_source},"
            f"right_circle={right_source},depth={depth_source}"
        )
        combo_counts[combo_key] += 1

        if (
            match_source in (2, 3)
            and (left_source == 3 or right_source == 3)
            and _safe_float(row.get("z_fallback"), -1.0) > 0.1
        ):
            fallback_epipolar_rows.append(row)
            fallback_by_direction[MATCH_SOURCE_NAMES[match_source]] += 1

    fallback_values = [_safe_float(row.get("z_fallback"), -1.0) for row in fallback_epipolar_rows]
    return {
        "rows": len(rows),
        "raw_rows": len(raw),
        "match_source": dict(match_counts),
        "source_combinations": dict(combo_counts),
        "epipolar_fallback": {
            "valid": len(fallback_values),
            "by_direction": dict(fallback_by_direction),
            "median": _median(fallback_values),
            "mad": _mad(fallback_values),
            "min": min(fallback_values) if fallback_values else None,
            "max": max(fallback_values) if fallback_values else None,
        },
    }


def _depth_jump_stats(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, Any]:
    if not rows or key not in rows[0]:
        return {"present": False}

    by_track: Dict[str, List[tuple[int, float]]] = {}
    for row in rows:
        value = _safe_float(row.get(key), -1.0)
        if value <= 0.1:
            continue
        track_id = row.get("track_id") or "0"
        by_track.setdefault(track_id, []).append((_safe_int(row.get("frame_id"), 0), value))

    deltas: List[float] = []
    for values in by_track.values():
        values.sort(key=lambda item: item[0])
        for (_, prev), (_, cur) in zip(values, values[1:]):
            deltas.append(abs(cur - prev))

    return {
        "present": True,
        "pairs": len(deltas),
        "median_abs_delta": _median(deltas),
        "mad_abs_delta": _mad(deltas),
        "p95_abs_delta": _percentile(deltas, 95.0),
        "max_abs_delta": max(deltas) if deltas else None,
    }


def _frame_summary_report(csv_path: Path) -> Dict[str, Any]:
    frame_path = derive_frame_summary_path(csv_path)
    if not frame_path.exists():
        return {"path": str(frame_path), "present": False}

    rows = read_csv_rows(frame_path)
    fields = _field_set(rows)
    missing = [field for field in FRAME_SUMMARY_FIELDS if field not in fields]

    def sum_field(key: str) -> int:
        return sum(_safe_int(row.get(key), 0) for row in rows)

    def max_field(key: str) -> int | None:
        if not rows or key not in fields:
            return None
        return max(_safe_int(row.get(key), 0) for row in rows)

    return {
        "path": str(frame_path),
        "present": True,
        "rows": len(rows),
        "missing_fields": missing,
        "totals": {
            "result_count": sum_field("result_count"),
            "raw_observation_count": sum_field("raw_observation_count"),
            "stereo_observation_count": sum_field("stereo_observation_count"),
            "direct_pair_count": sum_field("direct_pair_count"),
            "fallback_l2r_count": sum_field("fallback_l2r_count"),
            "fallback_r2l_count": sum_field("fallback_r2l_count"),
        },
        "max_per_frame": {
            "result_count": max_field("result_count"),
            "raw_observation_count": max_field("raw_observation_count"),
            "stereo_observation_count": max_field("stereo_observation_count"),
            "fallback_l2r_count": max_field("fallback_l2r_count"),
            "fallback_r2l_count": max_field("fallback_r2l_count"),
        },
    }


def analyze_dataset(csv_path: str | Path, metadata_path: str | Path | None = None) -> Dict[str, Any]:
    """Return quality metrics for one trajectory CSV."""

    csv_path = Path(csv_path)
    rows = read_csv_rows(csv_path)
    metadata_file = Path(metadata_path) if metadata_path else find_metadata_for_csv(csv_path)
    metadata = read_metadata(metadata_file)
    fields = _field_set(rows)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fields]

    timestamps = [_safe_float(row.get("timestamp"), math.nan) for row in rows]
    timestamps = [value for value in timestamps if not math.isnan(value)]
    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    fps_rows = len(rows) / duration if duration > 0.0 else None
    fps_intervals = (len(rows) - 1) / duration if duration > 0.0 and len(rows) > 1 else None
    gaps = _frame_gaps(rows)
    known_z = _metadata_float(metadata, "known_z_m", "known_z", "known_distance_m")

    return {
        "csv": str(csv_path),
        "metadata": str(metadata_file) if metadata_file else None,
        "rows": len(rows),
        "duration_sec": duration,
        "fps_rows": fps_rows,
        "fps_intervals": fps_intervals,
        "missing_fields": missing_fields,
        "frame_gaps": {
            "count": len(gaps),
            "first": gaps[:10],
        },
        "watermarks": {
            "frame_counter_delta": _delta_stats(rows, "frame_counter_delta"),
            "frame_number_delta": _delta_stats(rows, "frame_number_delta"),
        },
        "source_breakdown": _source_breakdown(rows),
        "depth": {key: _depth_stats(rows, key, known_z) for key in DEPTH_KEYS},
        "depth_jump": {key: _depth_jump_stats(rows, key) for key in JUMP_DEPTH_KEYS},
        "frame_summary": _frame_summary_report(csv_path),
        "known_z": known_z,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_report(report: Dict[str, Any]) -> None:
    print(f"csv={report['csv']}")
    print(f"metadata={report['metadata']}")
    print(
        f"rows={report['rows']} duration={_fmt(report['duration_sec'], 3)}s "
        f"fps_rows={_fmt(report['fps_rows'], 3)} fps_intervals={_fmt(report['fps_intervals'], 3)}"
    )
    print(f"missing_fields={report['missing_fields']}")
    print(f"frame_gaps={report['frame_gaps']['count']} first={report['frame_gaps']['first']}")
    for key, stats in report["watermarks"].items():
        print(f"{key}: present={stats['present']} nonzero={stats.get('nonzero')} unique={stats.get('unique')}")
    source = report["source_breakdown"]
    print(f"match_source={source['match_source']}")
    print(
        "epipolar_fallback: "
        f"valid={source['epipolar_fallback']['valid']} "
        f"by_direction={source['epipolar_fallback']['by_direction']} "
        f"median={_fmt(source['epipolar_fallback']['median'])} "
        f"mad={_fmt(source['epipolar_fallback']['mad'])}"
    )
    for key in DEPTH_KEYS:
        stats = report["depth"][key]
        jumps = report["depth_jump"].get(key, {})
        print(
            f"{key}: valid={stats['valid']}/{stats['total']} "
            f"hit={stats['hit_rate'] * 100.0:.1f}% "
            f"median={_fmt(stats['median'])} mad={_fmt(stats['mad'])} "
            f"known_z_bias={_fmt(stats['known_z_bias'])} known_z_mad={_fmt(stats['known_z_mad'])} "
            f"jump_p95={_fmt(jumps.get('p95_abs_delta'))}"
        )
    frame_summary = report["frame_summary"]
    print(
        f"frame_summary: present={frame_summary['present']} "
        f"path={frame_summary['path']} rows={frame_summary.get('rows')}"
    )
    if frame_summary["present"]:
        print(f"frame_summary_totals={frame_summary['totals']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="TrajectoryRecorder CSV")
    parser.add_argument("--metadata", help="Optional metadata YAML")
    parser.add_argument("--json-out", help="Write machine-readable report")
    args = parser.parse_args()

    report = analyze_dataset(args.csv, args.metadata)
    print_report(report)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
