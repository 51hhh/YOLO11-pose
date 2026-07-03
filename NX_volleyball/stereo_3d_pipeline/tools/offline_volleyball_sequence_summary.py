"""Summary helpers for the offline volleyball sequence probe."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def _num(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _motion_residual(values: Sequence[Tuple[int, float]]) -> Dict[str, float]:
    if len(values) < 3:
        return {"motion_residual_median_m": 0.0, "motion_residual_mad_m": 0.0, "motion_residual_p95_m": 0.0}
    xs = np.asarray([v[0] for v in values], dtype=np.float64)
    ys = np.asarray([v[1] for v in values], dtype=np.float64)
    xs = xs - xs[0]
    degree = 2 if len(values) >= 5 else 1
    coeff = np.polyfit(xs, ys, degree)
    pred = np.polyval(coeff, xs)
    residual = np.abs(ys - pred)
    return {
        "motion_residual_median_m": float(np.median(residual)),
        "motion_residual_mad_m": float(np.median(np.abs(residual - np.median(residual)))),
        "motion_residual_p95_m": float(np.percentile(residual, 95)),
    }


def summarize_methods(per_method_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in per_method_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for method, rows in sorted(by_method.items()):
        depths: List[float] = []
        depth_by_frame: List[Tuple[int, float]] = []
        supports: List[float] = []
        valid_points: List[float] = []
        elapsed: List[float] = []
        pass_count = 0
        runtime_ok = 0
        for row in rows:
            z = _num(row, "z_median_m", _num(row, "depth_m", 0.0))
            frame_idx = int(_num(row, "frame_index", len(depth_by_frame)))
            if z > 0.0 and math.isfinite(z):
                depths.append(z)
                depth_by_frame.append((frame_idx, z))
            supports.append(_num(row, "matches", 0.0))
            valid_points.append(_num(row, "validation_valid_points", 0.0))
            elapsed.append(_num(row, "elapsed_ms", 0.0))
            if row.get("validation_status") == "pass":
                pass_count += 1
            if int(_num(row, "runtime_feature_geometry_ok", 0.0)) == 1:
                runtime_ok += 1

        jitters = [abs(depths[i] - depths[i - 1]) for i in range(1, len(depths))]
        total = max(1, len(rows))
        summary_rows.append(
            {
                "method": method,
                "frames": len(rows),
                "validation_pass_rate": pass_count / total,
                "runtime_geometry_pass_rate": runtime_ok / total,
                "valid_points_median": _median(valid_points),
                "valid_points_min": min(valid_points) if valid_points else 0.0,
                "matches_median": _median(supports),
                "depth_median_m": _median(depths),
                "depth_mad_m": _mad(depths),
                "frame_jitter_median_m": _median(jitters),
                "frame_jitter_p95_m": _p95(jitters),
                "elapsed_median_ms": _median(elapsed),
                "elapsed_p95_ms": _p95(elapsed),
                **_motion_residual(depth_by_frame),
            }
        )
    return summary_rows


def build_markdown_report(clip: Path, frames_ok: int, frames_requested: int, summary_rows: Sequence[Dict[str, object]]) -> str:
    report = ["# Volleyball Sequence Probe", ""]
    report.append(f"- clip: `{clip}`")
    report.append(f"- frames ok: `{frames_ok}/{frames_requested}`")
    report.append("")
    report.append("| method | pass rate | valid pts med | depth med m | depth MAD m | jitter p95 m | elapsed p95 ms |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        report.append(
            "| {method} | {validation_pass_rate:.3f} | {valid_points_median:.1f} | "
            "{depth_median_m:.4f} | {depth_mad_m:.4f} | {frame_jitter_p95_m:.4f} | "
            "{elapsed_p95_ms:.2f} |".format(**row)
        )
    return "\n".join(report) + "\n"
