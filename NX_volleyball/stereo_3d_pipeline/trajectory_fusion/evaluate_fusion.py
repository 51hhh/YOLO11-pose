#!/usr/bin/env python3
"""Evaluate trajectory stability and physics consistency."""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _read(path: str) -> List[Dict[str, str]]:
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _series(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [_f(row, key) for row in rows if _f(row, key) > -1e20]


def _diff(values: List[float]) -> List[float]:
    return [b - a for a, b in zip(values, values[1:])]


def _rms(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Raw or smoothed CSV")
    args = parser.parse_args()

    rows = _read(args.input)
    grouped = _group_by_track(rows)
    has_smooth = bool(rows and "smooth_z" in rows[0])
    for track_id, track_rows in grouped.items():
        raw = _metrics(track_rows)
        _print_metrics(f"track={track_id} raw", raw)
        if has_smooth:
            smooth = _metrics(track_rows, prefix="smooth_")
            ratio = smooth["z_std"] / raw["z_std"] if raw["z_std"] > 1e-9 else 0.0
            _print_metrics(f"track={track_id} smooth", smooth)
            print(f"track={track_id} smooth/raw z_std ratio={ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
