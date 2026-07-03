#!/usr/bin/env python3
"""Evaluate trajectory stability and physics consistency."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from statistics import mean, pstdev
from collections import Counter
from typing import Any, Dict, List

try:
    from .dataset import find_metadata_for_csv, read_metadata
    from .evaluate_fusion_report import build_report, write_csv_report, write_json_report
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
    from dataset import find_metadata_for_csv, read_metadata
    from evaluate_fusion_report import build_report, write_csv_report, write_json_report
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


def _read(path: str) -> List[Dict[str, str]]:
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))


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
        write_json_report(args.json_out, report)
    if args.csv_out:
        write_csv_report(args.csv_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
