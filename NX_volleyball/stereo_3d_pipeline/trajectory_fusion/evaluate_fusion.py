#!/usr/bin/env python3
"""Evaluate trajectory stability and physics consistency."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from collections import Counter
from typing import Any, Dict, List

try:
    from .dataset import (
        find_metadata_for_csv,
        merge_p2_diagnostic_candidates,
        read_metadata,
        read_p2_diagnostic_candidates,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        find_metadata_for_csv,
        merge_p2_diagnostic_candidates,
        read_metadata,
        read_p2_diagnostic_candidates,
    )


CANDIDATE_DEPTH_KEYS = [
    "z_mono",
    "z_bbox_center",
    "z_bbox_left_edge",
    "z_bbox_right_edge",
    "z_circle_center",
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
    "z_roi_cuda_template_match",
    "z_roi_neural_feature",
    "z_roi_neural_xfeat",
    "z_roi_neural_superpoint",
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


def _read(path: str) -> List[Dict[str, str]]:
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    p2_diagnostic_by_frame = read_p2_diagnostic_candidates(path)
    if not p2_diagnostic_by_frame:
        return rows
    for row in rows:
        try:
            frame_id = int(float(row.get("frame_id", "-1")))
        except (TypeError, ValueError):
            continue
        merge_p2_diagnostic_candidates(row, p2_diagnostic_by_frame.get(frame_id))
    return rows


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
    for key in CANDIDATE_DEPTH_KEYS:
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
            f"mad={_mad(values):.4f} p2p={(max(values) - min(values)) if values else 0.0:.4f}"
        )
    for key in LEGACY_DEPTH_KEYS:
        if not rows or key not in rows[0]:
            continue
        values = _valid_depth_series(rows, key)
        if not values:
            continue
        print(
            f"track={track_id} legacy {key}: valid={len(values)}/{len(rows)} "
            f"mean={mean(values):.4f} std={pstdev(values) if len(values) > 1 else 0.0:.4f} "
            f"mad={_mad(values):.4f}"
        )


def _print_known_distance_metrics(track_id: str, rows: List[Dict[str, str]], metadata: Dict[str, Any]) -> None:
    known_z = _metadata_float(metadata, "known_z_m", "known_z", "known_distance_m")
    if known_z <= 0.0:
        return
    print(f"track={track_id} known_z={known_z:.4f}m")
    for key in CANDIDATE_DEPTH_KEYS + LEGACY_DEPTH_KEYS:
        if not rows or key not in rows[0]:
            continue
        values = _valid_depth_series(rows, key)
        if not values:
            continue
        errors = [value - known_z for value in values]
        print(
            f"track={track_id} {key} known_z: bias={mean(errors):+.4f}m "
            f"mad={_mad(errors):.4f}m valid={len(values)}/{len(rows)}"
        )


def _print_p0_median_metrics(track_id: str, rows: List[Dict[str, str]], metadata: Dict[str, Any]) -> None:
    medians: List[float] = []
    for row in rows:
        values = [_f(row, key, -1.0) for key in P0_DEPTH_KEYS if key in row and _f(row, key, -1.0) > 0.1]
        if values:
            medians.append(_median(values))
    if not medians:
        return
    print(
        f"track={track_id} p0_median: valid={len(medians)}/{len(rows)} "
        f"mean={mean(medians):.4f} std={pstdev(medians) if len(medians) > 1 else 0.0:.4f} "
        f"mad={_mad(medians):.4f}"
    )
    known_z = _metadata_float(metadata, "known_z_m", "known_z", "known_distance_m")
    if known_z > 0.0:
        errors = [value - known_z for value in medians]
        print(f"track={track_id} p0_median known_z: bias={mean(errors):+.4f}m mad={_mad(errors):.4f}m")


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
    for key in pair_keys:
        if not rows or key not in rows[0]:
            continue
        if key == "pair_score":
            values = [_f(row, key, 0.0) for row in rows if row.get(key, "") != ""]
        else:
            values = [_f(row, key, -1.0) for row in rows if _f(row, key, -1.0) >= 0.0]
        if not values:
            continue
        print(
            f"track={track_id} {key}: mean={mean(values):.3f} "
            f"std={pstdev(values) if len(values) > 1 else 0.0:.3f} "
            f"min={min(values):.3f} max={max(values):.3f}"
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


def _known_z_error_report(values: List[float], known_z: float) -> Dict[str, float | None]:
    errors = [value - known_z for value in values] if known_z > 0.0 else []
    return {
        "known_z_bias": mean(errors) if errors else None,
        "known_z_mad": _mad(errors) if errors else None,
    }


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
        raw_metrics = _metrics(track_rows)
        raw_metrics.update(_known_z_error_report(_series(track_rows, "z"), known_z))
        track_report: Dict[str, Any] = {
            "raw": raw_metrics,
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
            smooth.update(_known_z_error_report(_series(track_rows, "smooth_z"), known_z))
            track_report["smooth"] = smooth
        tracks[str(track_id)] = track_report
    return {
        "metadata": metadata,
        "known_z": known_z if known_z > 0.0 else None,
        "track_count": len(tracks),
        "tracks": tracks,
    }


def _write_json_report(path: str | Path, report: Dict[str, Any]) -> None:
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


def _write_csv_report(path: str | Path, report: Dict[str, Any]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Raw or smoothed CSV")
    parser.add_argument("--metadata", help="Optional metadata.yaml with weak labels")
    parser.add_argument("--json-out", help="Write machine-readable JSON report")
    parser.add_argument("--csv-out", help="Write machine-readable long CSV report")
    args = parser.parse_args()

    rows = _read(args.input)
    metadata = read_metadata(args.metadata or find_metadata_for_csv(args.input))
    report = build_report(rows, metadata)
    grouped = _group_by_track(rows)
    has_smooth = bool(rows and "smooth_z" in rows[0])
    for track_id, track_rows in grouped.items():
        raw = _metrics(track_rows)
        _print_metrics(f"track={track_id} raw", raw)
        _print_depth_candidate_metrics(track_id, track_rows)
        _print_p0_median_metrics(track_id, track_rows, metadata)
        _print_known_distance_metrics(track_id, track_rows, metadata)
        _print_sync_and_source_metrics(track_id, track_rows)
        if has_smooth:
            smooth = _metrics(track_rows, prefix="smooth_")
            ratio = smooth["z_std"] / raw["z_std"] if raw["z_std"] > 1e-9 else 0.0
            _print_metrics(f"track={track_id} smooth", smooth)
            print(f"track={track_id} smooth/raw z_std ratio={ratio:.3f}")
    if args.json_out:
        _write_json_report(args.json_out, report)
    if args.csv_out:
        _write_csv_report(args.csv_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
