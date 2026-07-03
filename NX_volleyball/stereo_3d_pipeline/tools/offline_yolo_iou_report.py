"""Report helpers for offline YOLO/IoU fallback regression."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rate(values: Iterable[bool]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(1 for v in vals if v)) / float(len(vals))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    left_errors = [float(r["left_missing_center_error_px"]) for r in rows if r["left_missing_valid"]]
    right_errors = [float(r["right_missing_center_error_px"]) for r in rows if r["right_missing_valid"]]
    left_disp_errors = [float(r["left_missing_disparity_error_px"]) for r in rows if r["left_missing_valid"]]
    right_disp_errors = [float(r["right_missing_disparity_error_px"]) for r in rows if r["right_missing_valid"]]
    return {
        "frames": len(rows),
        "normal_pair_pass_rate": _rate(bool(r["normal_pair_valid"]) for r in rows),
        "fake_right_low_selected_true_rate": _rate(bool(r["fake_right_low_selected_true"]) for r in rows),
        "fake_right_high_selected_true_rate": _rate(bool(r["fake_right_high_selected_true"]) for r in rows),
        "fake_left_low_selected_true_rate": _rate(bool(r["fake_left_low_selected_true"]) for r in rows),
        "fake_left_high_selected_true_rate": _rate(bool(r["fake_left_high_selected_true"]) for r in rows),
        "right_missing_pass_rate": _rate(bool(r["right_missing_pass"]) for r in rows),
        "left_missing_pass_rate": _rate(bool(r["left_missing_pass"]) for r in rows),
        "right_missing_center_error_median_px": _percentile(right_errors, 50),
        "right_missing_center_error_p95_px": _percentile(right_errors, 95),
        "left_missing_center_error_median_px": _percentile(left_errors, 50),
        "left_missing_center_error_p95_px": _percentile(left_errors, 95),
        "right_missing_disparity_error_p95_px": _percentile(right_disp_errors, 95),
        "left_missing_disparity_error_p95_px": _percentile(left_disp_errors, 95),
        "template_elapsed_ms_median": _percentile(
            [float(r["right_missing_elapsed_ms"]) + float(r["left_missing_elapsed_ms"]) for r in rows],
            50,
        ),
        "template_elapsed_ms_p95": _percentile(
            [float(r["right_missing_elapsed_ms"]) + float(r["left_missing_elapsed_ms"]) for r in rows],
            95,
        ),
    }


def write_regression_outputs(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    clip: Path,
) -> None:
    _write_csv(out_dir / "per_frame.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics = summary["metrics"]
    report = [
        "# YOLO/IoU Fallback Regression",
        "",
        f"- clip: `{clip}`",
        f"- frames: `{metrics['frames']}`",
        f"- normal pair pass rate: `{metrics['normal_pair_pass_rate']:.3f}`",
        f"- fake right low selected true: `{metrics['fake_right_low_selected_true_rate']:.3f}`",
        f"- fake right high selected true: `{metrics['fake_right_high_selected_true_rate']:.3f}`",
        f"- fake left low selected true: `{metrics['fake_left_low_selected_true_rate']:.3f}`",
        f"- fake left high selected true: `{metrics['fake_left_high_selected_true_rate']:.3f}`",
        f"- right missing pass rate: `{metrics['right_missing_pass_rate']:.3f}`",
        f"- left missing pass rate: `{metrics['left_missing_pass_rate']:.3f}`",
        f"- right missing center p95 px: `{metrics['right_missing_center_error_p95_px']:.2f}`",
        f"- left missing center p95 px: `{metrics['left_missing_center_error_p95_px']:.2f}`",
        f"- template elapsed p95 ms: `{metrics['template_elapsed_ms_p95']:.2f}`",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
