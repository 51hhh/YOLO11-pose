#!/usr/bin/env python3
"""Quality checks for TrajectoryRecorder CSV datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    from .dataset import find_metadata_for_csv, read_csv_rows, read_metadata
    from .check_dataset_fields import DEPTH_KEYS, JUMP_DEPTH_KEYS, REQUIRED_FIELDS
    from .check_dataset_metrics import (
        delta_stats,
        depth_jump_stats,
        depth_stats,
        field_set,
        frame_gaps,
        frame_summary_report,
        metadata_float,
        source_breakdown,
        timing_stats,
    )
    from .check_dataset_print import print_report
except ImportError:  # pragma: no cover - direct script execution
    from dataset import find_metadata_for_csv, read_csv_rows, read_metadata
    from check_dataset_fields import DEPTH_KEYS, JUMP_DEPTH_KEYS, REQUIRED_FIELDS
    from check_dataset_metrics import (
        delta_stats,
        depth_jump_stats,
        depth_stats,
        field_set,
        frame_gaps,
        frame_summary_report,
        metadata_float,
        source_breakdown,
        timing_stats,
    )
    from check_dataset_print import print_report


def analyze_dataset(csv_path: str | Path, metadata_path: str | Path | None = None) -> Dict[str, Any]:
    """Return quality metrics for one trajectory CSV."""

    csv_path = Path(csv_path)
    rows = read_csv_rows(csv_path)
    metadata_file = Path(metadata_path) if metadata_path else find_metadata_for_csv(csv_path)
    metadata = read_metadata(metadata_file)
    fields = field_set(rows)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fields]

    target_timing = timing_stats(rows)
    target_gaps = frame_gaps(rows)
    frame_summary = frame_summary_report(csv_path)
    timing_source = "frame_summary" if frame_summary["present"] else "target_csv"
    duration = frame_summary.get("duration_sec", target_timing["duration_sec"])
    fps_rows = frame_summary.get("fps_rows", target_timing["fps_rows"])
    fps_intervals = frame_summary.get("fps_intervals", target_timing["fps_intervals"])
    gaps = frame_summary.get(
        "frame_gaps",
        {
            "count": len(target_gaps),
            "first": target_gaps[:10],
        },
    )
    known_z = metadata_float(metadata, "known_z_m", "known_z", "known_distance_m")

    return {
        "csv": str(csv_path),
        "metadata": str(metadata_file) if metadata_file else None,
        "rows": len(rows),
        "duration_sec": duration,
        "fps_rows": fps_rows,
        "fps_intervals": fps_intervals,
        "timing_source": timing_source,
        "missing_fields": missing_fields,
        "frame_gaps": gaps,
        "target_frame_gaps": {
            "count": len(target_gaps),
            "first": target_gaps[:10],
        },
        "watermarks": {
            "frame_counter_delta": delta_stats(rows, "frame_counter_delta"),
            "frame_number_delta": delta_stats(rows, "frame_number_delta"),
        },
        "source_breakdown": source_breakdown(rows),
        "depth": {key: depth_stats(rows, key, known_z) for key in DEPTH_KEYS},
        "depth_jump": {key: depth_jump_stats(rows, key) for key in JUMP_DEPTH_KEYS},
        "frame_summary": frame_summary,
        "known_z": known_z,
    }


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
