#!/usr/bin/env python3
"""Apply disparity zero-point correction to all z_* columns in a CSV.

Reads a recorder CSV, replaces every ``z_<method>`` column with
``z = fB / (disparity_<method> - d0)``, and writes the corrected CSV.

Usage:
    python3 apply_d0_correction.py input.csv \
        --calib ../calibration/stereo_calib.yaml \
        --offset-fit test_logs/recording_runs_20260707/disparity_offset_fit.json \
        -o input_d0corrected.csv
"""

from __future__ import annotations
import argparse, csv, math, io
from pathlib import Path
from typing import Dict, List

try:
    from .dataset import METHOD_COLUMNS
    from .reproject import load_reprojection_model, method_disparity_column
except ImportError:
    from dataset import METHOD_COLUMNS
    from reproject import load_reprojection_model, method_disparity_column


def _safe(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def correct_csv(input_path: str, model, output_path: str):
    raw = Path(input_path).read_bytes().replace(b"\x00", b"")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    if not rows:
        raise SystemExit("empty CSV")

    fieldnames = list(rows[0].keys())
    method_keys = {key for _, key in METHOD_COLUMNS}
    corrected_count = 0

    with Path(output_path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for _, key in METHOD_COLUMNS:
                if key not in row:
                    continue
                disp_col = method_disparity_column(key)
                disparity = _safe(row.get(disp_col))
                if disparity is not None and disparity > 0.1:
                    corrected = model.depth_from_disparity(disparity)
                    if corrected is not None and corrected > 0.1:
                        row[key] = str(corrected)
                        corrected_count += 1
            writer.writerow(row)

    print(f"corrected {corrected_count} depth cells, wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="TrajectoryRecorder CSV")
    parser.add_argument("--calib", required=True, help="stereo_calib.yaml")
    parser.add_argument("--offset-fit", help="disparity_offset_fit.json")
    parser.add_argument("--d0", type=float, help="Override d0 in pixels")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    model = load_reprojection_model(args.calib, args.offset_fit, args.d0)
    print(f"fB={model.fB:.3f}  d0={model.d0:.3f}px")

    correct_csv(args.input, model, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
