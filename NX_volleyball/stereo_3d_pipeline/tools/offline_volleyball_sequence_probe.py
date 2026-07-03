#!/usr/bin/env python3
"""Run the single-frame volleyball keypoint probe over a baseline clip."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from offline_volleyball_sequence_summary import build_markdown_report, summarize_methods, write_csv


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

    summary_rows = summarize_methods(per_method_rows)
    write_csv(args.out / "frames.csv", per_frame_rows)
    write_csv(args.out / "per_frame_methods.csv", per_method_rows)
    write_csv(args.out / "method_summary.csv", summary_rows)
    summary = {
        "clip": str(args.clip),
        "calibration": str(args.calib),
        "frames_requested": len(pairs),
        "frames_ok": sum(1 for r in per_frame_rows if r["ok"]),
        "methods": summary_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(
        build_markdown_report(args.clip, summary["frames_ok"], summary["frames_requested"], summary_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
