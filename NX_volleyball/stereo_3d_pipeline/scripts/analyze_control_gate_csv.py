#!/usr/bin/env python3
"""Summarize ungated versus realtime control-gated landing predictions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


REASONS = {
    0: "not_evaluated",
    1: "passed",
    2: "disabled",
    3: "nonfinite",
    4: "quality",
    5: "camera_bounds",
    6: "unstable",
    7: "transform_invalid",
    8: "odom_stale",
    9: "base_invalid",
}

REQUIRED = {
    "frame_id",
    "track_id",
    "pred_valid_ungated",
    "pred_x_ungated",
    "pred_y_ungated",
    "pred_t_ungated",
    "control_gate_selected",
    "control_gate_passed",
    "control_gate_reason",
    "pred_x_gated",
    "pred_y_gated",
    "pred_t_gated",
    "control_base_x",
    "control_base_y",
}


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0))
    except ValueError:
        return 0


def as_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "0") or 0)
        return value if math.isfinite(value) else 0.0
    except ValueError:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED - fields)
        if missing:
            raise SystemExit(
                "CSV does not contain control-gate audit columns: "
                + ", ".join(missing)
            )
        rows = list(reader)

    ungated = [r for r in rows if as_int(r, "pred_valid_ungated") == 1]
    selected = [r for r in rows if as_int(r, "control_gate_selected") == 1]
    passed = [r for r in rows if as_int(r, "control_gate_passed") == 1]
    reasons = Counter(as_int(r, "control_gate_reason") for r in selected)

    print("Control gate recording summary")
    print(f"csv: {args.csv_path}")
    print(f"rows: {len(rows)}")
    print(f"ungated_valid_rows: {len(ungated)}")
    print(f"gate_selected_rows: {len(selected)}")
    print(f"gate_passed_rows: {len(passed)}")
    rate = 100.0 * len(passed) / len(selected) if selected else 0.0
    print(f"gate_pass_rate_selected: {rate:.2f}%")
    print("gate_reasons:")
    for code, count in sorted(reasons.items()):
        print(f"  {code}:{REASONS.get(code, 'unknown')}={count}")

    if passed:
        first = passed[0]
        last = passed[-1]
        xs = [as_float(r, "pred_x_gated") for r in passed]
        ys = [as_float(r, "pred_y_gated") for r in passed]
        ttis = [as_float(r, "pred_t_gated") for r in passed]
        print(
            "first_gated: "
            f"frame={as_int(first, 'frame_id')} track={as_int(first, 'track_id')} "
            f"camera=({as_float(first, 'pred_x_gated'):.3f},"
            f"{as_float(first, 'pred_y_gated'):.3f}) "
            f"base=({as_float(first, 'control_base_x'):.3f},"
            f"{as_float(first, 'control_base_y'):.3f}) "
            f"tti={as_float(first, 'pred_t_gated'):.3f}s"
        )
        print(
            "last_gated: "
            f"frame={as_int(last, 'frame_id')} track={as_int(last, 'track_id')} "
            f"camera=({as_float(last, 'pred_x_gated'):.3f},"
            f"{as_float(last, 'pred_y_gated'):.3f}) "
            f"tti={as_float(last, 'pred_t_gated'):.3f}s"
        )
        print(f"gated_x_range_m: [{min(xs):.3f}, {max(xs):.3f}]")
        print(f"gated_depth_range_m: [{min(ys):.3f}, {max(ys):.3f}]")
        print(f"gated_tti_range_s: [{min(ttis):.3f}, {max(ttis):.3f}]")
    else:
        print("first_gated: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
