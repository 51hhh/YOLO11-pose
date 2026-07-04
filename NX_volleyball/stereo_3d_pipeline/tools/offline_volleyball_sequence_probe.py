#!/usr/bin/env python3
"""Run the single-frame volleyball keypoint probe over a baseline clip."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _clip_pairs(
    clip: Path,
    max_frames: int,
    stride: int,
) -> List[Tuple[str, Path, Path, Dict[str, str]]]:
    frames_csv = clip / "frames.csv"
    pairs: List[Tuple[str, Path, Path, Dict[str, str]]] = []
    if frames_csv.exists():
        with frames_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            frame_id = row.get("clip_frame_id") or row.get("pipeline_frame_id") or str(len(pairs))
            left_rel = row.get("left_image", "")
            right_rel = row.get("right_image", "")
            if not left_rel or not right_rel:
                continue
            pairs.append((frame_id, clip / left_rel, clip / right_rel, row))
    else:
        left_dir = clip / "left"
        right_dir = clip / "right"
        left_images = sorted([p for p in left_dir.iterdir() if p.is_file()])
        right_images = sorted([p for p in right_dir.iterdir() if p.is_file()])
        for idx, (left, right) in enumerate(zip(left_images, right_images)):
            pairs.append((f"{idx:06d}", left, right, {}))

    stride = max(1, stride)
    pairs = pairs[::stride]
    if max_frames > 0:
        pairs = pairs[:max_frames]
    return pairs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _manual_circle_args(row: Dict[str, str]) -> List[str]:
    try:
        if int(float(row.get("left_count", "1"))) <= 0:
            return []
        if int(float(row.get("right_count", "1"))) <= 0:
            return []
        lx = float(row["left_cx"])
        ly = float(row["left_cy"])
        lw = float(row["left_w"])
        lh = float(row["left_h"])
        rx = float(row["right_cx"])
        ry = float(row["right_cy"])
        rw = float(row["right_w"])
        rh = float(row["right_h"])
    except (KeyError, TypeError, ValueError):
        return []
    lr = 0.5 * max(lw, lh)
    rr = 0.5 * max(rw, rh)
    if lr <= 1.0 or rr <= 1.0:
        return []
    return [
        "--left-circle",
        f"{lx:.3f},{ly:.3f},{lr:.3f}",
        "--right-circle",
        f"{rx:.3f},{ry:.3f},{rr:.3f}",
    ]


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


def _summarize(per_method_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
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
        motion = _motion_residual(depth_by_frame)
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
                **motion,
            }
        )
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True, help="baseline clip dir with left/right or frames.csv")
    parser.add_argument("--calib", type=Path, default=Path("NX_volleyball/calibration/stereo_calib.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--probe", type=Path, default=Path("NX_volleyball/stereo_3d_pipeline/tools/offline_volleyball_keypoint_probe.py"))
    parser.add_argument("probe_args", nargs=argparse.REMAINDER, help="extra args passed after -- to the single-frame probe")
    args = parser.parse_args()

    extra_args = list(args.probe_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    pairs = _clip_pairs(args.clip, args.max_frames, args.stride)
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows: List[Dict[str, object]] = []
    per_method_rows: List[Dict[str, object]] = []
    for frame_index, (frame_id, left, right, frame_row) in enumerate(pairs):
        frame_out = frames_dir / f"{frame_index:06d}"
        cmd = [
            sys.executable,
            str(args.probe),
            "--left",
            str(left),
            "--right",
            str(right),
            "--calib",
            str(args.calib),
            "--out",
            str(frame_out),
            "--quiet",
            *_manual_circle_args(frame_row),
            *extra_args,
        ]
        start = time.perf_counter()
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        total_ms = (time.perf_counter() - start) * 1000.0
        ok = proc.returncode == 0
        per_frame_rows.append(
            {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "left_image": str(left),
                "right_image": str(right),
                "pair_valid": frame_row.get("pair_valid", ""),
                "pair_disparity_px": frame_row.get("pair_disparity_px", ""),
                "pair_dy_px": frame_row.get("pair_dy_px", ""),
                "pair_size_ratio": frame_row.get("pair_size_ratio", ""),
                "ok": ok,
                "elapsed_total_ms": total_ms,
                "stderr": proc.stderr.strip()[-500:],
            }
        )
        if not ok:
            continue
        for method_row in _read_csv(frame_out / "summary.csv"):
            row: Dict[str, object] = {
                "frame_index": frame_index,
                "frame_id": frame_id,
                **method_row,
            }
            per_method_rows.append(row)

    summary_rows = _summarize(per_method_rows)
    _write_csv(args.out / "frames.csv", per_frame_rows)
    _write_csv(args.out / "per_frame_methods.csv", per_method_rows)
    _write_csv(args.out / "method_summary.csv", summary_rows)
    summary = {
        "clip": str(args.clip),
        "calibration": str(args.calib),
        "frames_requested": len(pairs),
        "frames_ok": sum(1 for r in per_frame_rows if r["ok"]),
        "methods": summary_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# Volleyball Sequence Probe", ""]
    report.append(f"- clip: `{args.clip}`")
    report.append(f"- frames ok: `{summary['frames_ok']}/{summary['frames_requested']}`")
    report.append("")
    report.append("| method | pass rate | valid pts med | depth med m | depth MAD m | jitter p95 m | elapsed p95 ms |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        report.append(
            "| {method} | {validation_pass_rate:.3f} | {valid_points_median:.1f} | "
            "{depth_median_m:.4f} | {depth_mad_m:.4f} | {frame_jitter_p95_m:.4f} | "
            "{elapsed_p95_ms:.2f} |".format(**row)
        )
    (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
