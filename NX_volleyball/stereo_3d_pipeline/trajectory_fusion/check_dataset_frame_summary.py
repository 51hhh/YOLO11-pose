"""Frame sidecar summary helpers for trajectory dataset checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    from .dataset import derive_frame_summary_path, read_csv_rows
    from .check_dataset_fields import FRAME_SUMMARY_FIELDS, OPTIONAL_FRAME_SUMMARY_FIELDS
    from .check_dataset_metrics import field_set, frame_gaps, safe_int, timing_stats
except ImportError:  # pragma: no cover - direct script execution
    from dataset import derive_frame_summary_path, read_csv_rows
    from check_dataset_fields import FRAME_SUMMARY_FIELDS, OPTIONAL_FRAME_SUMMARY_FIELDS
    from check_dataset_metrics import field_set, frame_gaps, safe_int, timing_stats


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
