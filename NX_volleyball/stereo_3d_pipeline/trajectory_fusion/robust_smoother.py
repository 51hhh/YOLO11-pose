#!/usr/bin/env python3
"""Robust physics-aware smoother for current trajectory CSV files.

This is an offline baseline. It consumes the existing TrajectoryRecorder CSV,
so it cannot yet use bbox/circle/subpixel quality features.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from .dataset import LegacySequence, load_legacy_sequences
except ImportError:  # pragma: no cover - direct script execution
    from dataset import LegacySequence, load_legacy_sequences


@dataclass
class SmootherConfig:
    process_sigma: float = 18.0
    base_sigma_xy: float = 0.035
    base_sigma_z: float = 0.080
    mono_sigma_scale: float = 1.35
    stereo_sigma_scale: float = 1.00
    online_sigma_scale: float = 0.90
    huber_k: float = 2.5
    gravity_y: float = 9.81
    use_method_depths: bool = True


def _dt(rows: List[Dict[str, float]], index: int) -> float:
    if index == 0:
        if len(rows) > 1:
            return max(1e-4, min(0.2, rows[1]["timestamp"] - rows[0]["timestamp"]))
        return 0.01
    return max(1e-4, min(0.2, rows[index]["timestamp"] - rows[index - 1]["timestamp"]))


def _predict(state: np.ndarray, cov: np.ndarray, dt: float, cfg: SmootherConfig) -> Tuple[np.ndarray, np.ndarray]:
    f = np.eye(6, dtype=np.float64)
    f[0, 3] = dt
    f[1, 4] = dt
    f[2, 5] = dt

    control = np.array([0.0, 0.5 * cfg.gravity_y * dt * dt, 0.0, 0.0, cfg.gravity_y * dt, 0.0])
    q_pos = 0.25 * dt**4 * cfg.process_sigma**2
    q_vel = dt**2 * cfg.process_sigma**2
    q_cross = 0.5 * dt**3 * cfg.process_sigma**2
    q = np.zeros((6, 6), dtype=np.float64)
    for axis in range(3):
        q[axis, axis] = q_pos
        q[axis + 3, axis + 3] = q_vel
        q[axis, axis + 3] = q_cross
        q[axis + 3, axis] = q_cross

    return f @ state + control, f @ cov @ f.T + q


def _robust_update(
    state: np.ndarray,
    cov: np.ndarray,
    measurement: np.ndarray,
    h: np.ndarray,
    r_diag: np.ndarray,
    cfg: SmootherConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    pred = h @ state
    residual = measurement - pred
    s = h @ cov @ h.T + np.diag(r_diag)
    std = np.sqrt(np.maximum(np.diag(s), 1e-9))
    weights = np.ones_like(residual)
    for i, value in enumerate(residual):
        limit = cfg.huber_k * std[i]
        if abs(value) > limit:
            weights[i] = max(0.05, limit / abs(value))
    r_eff = r_diag / np.maximum(weights * weights, 1e-4)
    s_eff = h @ cov @ h.T + np.diag(r_eff)
    try:
        kalman_gain = cov @ h.T @ np.linalg.inv(s_eff)
    except np.linalg.LinAlgError:
        return state, cov, float(np.linalg.norm(residual))

    state = state + kalman_gain @ residual
    identity = np.eye(cov.shape[0], dtype=np.float64)
    # Joseph form is more stable for repeated robust updates.
    a = identity - kalman_gain @ h
    cov = a @ cov @ a.T + kalman_gain @ np.diag(r_eff) @ kalman_gain.T
    return state, cov, float(np.linalg.norm(residual / std))


def _position_measurement(row: Dict[str, float], cfg: SmootherConfig) -> Tuple[np.ndarray, np.ndarray]:
    z = max(row["z"], 0.1)
    confidence = max(0.05, min(1.0, row["confidence"]))
    sigma_xy = cfg.base_sigma_xy * max(1.0, z) / math.sqrt(confidence)
    sigma_z = cfg.base_sigma_z * max(1.0, z) / math.sqrt(confidence)
    return np.array([row["x"], row["y"], row["z"]], dtype=np.float64), np.array(
        [sigma_xy * sigma_xy, sigma_xy * sigma_xy, sigma_z * sigma_z],
        dtype=np.float64,
    )


def _z_measurements(row: Dict[str, float], cfg: SmootherConfig) -> List[Tuple[float, float, str]]:
    if not cfg.use_method_depths:
        return []
    out: List[Tuple[float, float, str]] = []
    confidence = max(0.05, min(1.0, row["confidence"]))
    if row["z_mono"] > 0.1:
        sigma = cfg.base_sigma_z * cfg.mono_sigma_scale * max(1.0, row["z_mono"]) / math.sqrt(confidence)
        out.append((row["z_mono"], sigma * sigma, "mono"))
    if row["z_stereo"] > 0.1:
        sigma = cfg.base_sigma_z * cfg.stereo_sigma_scale * max(1.0, row["z_stereo"]) / math.sqrt(confidence)
        out.append((row["z_stereo"], sigma * sigma, "stereo"))
    return out


def smooth_sequence(sequence: LegacySequence, cfg: SmootherConfig) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows = sequence.rows
    first = rows[0]
    state = np.array([first["x"], first["y"], first["z"], 0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.diag([0.1, 0.1, 0.2, 10.0, 10.0, 10.0]).astype(np.float64)
    h_xyz = np.zeros((3, 6), dtype=np.float64)
    h_xyz[0, 0] = 1.0
    h_xyz[1, 1] = 1.0
    h_xyz[2, 2] = 1.0
    h_z = np.zeros((1, 6), dtype=np.float64)
    h_z[0, 2] = 1.0

    output: List[Dict[str, float]] = []
    innovations: List[float] = []
    for i, row in enumerate(rows):
        state, cov = _predict(state, cov, _dt(rows, i), cfg)
        meas, r_diag = _position_measurement(row, cfg)
        state, cov, inn = _robust_update(state, cov, meas, h_xyz, r_diag, cfg)
        innovations.append(inn)

        for z_value, z_var, _name in _z_measurements(row, cfg):
            state, cov, inn = _robust_update(
                state,
                cov,
                np.array([z_value], dtype=np.float64),
                h_z,
                np.array([z_var], dtype=np.float64),
                cfg,
            )
            innovations.append(inn)

        out = dict(row)
        out.update(
            {
                "track_id": float(sequence.track_id),
                "smooth_x": float(state[0]),
                "smooth_y": float(state[1]),
                "smooth_z": float(state[2]),
                "smooth_vx": float(state[3]),
                "smooth_vy": float(state[4]),
                "smooth_vz": float(state[5]),
                "smooth_sigma_z": float(math.sqrt(max(cov[2, 2], 0.0))),
            }
        )
        output.append(out)

    z_raw = [row["z"] for row in rows if row["z"] > 0.1]
    z_smooth = [row["smooth_z"] for row in output if row["smooth_z"] > 0.1]
    metrics = {
        "track_id": float(sequence.track_id),
        "frames": float(len(rows)),
        "raw_z_std": _std(z_raw),
        "smooth_z_std": _std(z_smooth),
        "innovation_norm_mean": float(np.mean(innovations)) if innovations else 0.0,
        "innovation_norm_max": float(np.max(innovations)) if innovations else 0.0,
    }
    return output, metrics


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64)))


def write_output(path: str | Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    preferred = [
        "frame_id",
        "timestamp",
        "track_id",
        "x",
        "y",
        "z",
        "smooth_x",
        "smooth_y",
        "smooth_z",
        "smooth_vx",
        "smooth_vy",
        "smooth_vz",
        "smooth_sigma_z",
        "z_mono",
        "z_stereo",
        "depth_method",
        "confidence",
    ]
    extras = [key for key in rows[0].keys() if key not in preferred]
    fieldnames = preferred + extras
    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="TrajectoryRecorder CSV")
    parser.add_argument("-o", "--output", default="trajectory_fusion_smooth.csv")
    parser.add_argument("--no-method-depths", action="store_true", help="Do not use z_mono/z_stereo as extra z updates")
    parser.add_argument("--gravity-y", type=float, default=9.81, help="Camera-y gravity prior in m/s^2")
    args = parser.parse_args()

    cfg = SmootherConfig(use_method_depths=not args.no_method_depths, gravity_y=args.gravity_y)
    sequences = load_legacy_sequences(args.input)
    all_rows: List[Dict[str, float]] = []
    metrics: List[Dict[str, float]] = []
    for seq in sequences:
        rows, seq_metrics = smooth_sequence(seq, cfg)
        all_rows.extend(rows)
        metrics.append(seq_metrics)

    write_output(args.output, all_rows)
    for item in metrics:
        raw_std = item["raw_z_std"]
        smooth_std = item["smooth_z_std"]
        ratio = smooth_std / raw_std if raw_std > 1e-9 else 0.0
        print(
            "track={track_id:.0f} frames={frames:.0f} raw_z_std={raw:.4f} "
            "smooth_z_std={smooth:.4f} ratio={ratio:.3f} inn_mean={inn:.3f} inn_max={inn_max:.3f}".format(
                track_id=item["track_id"],
                frames=item["frames"],
                raw=raw_std,
                smooth=smooth_std,
                ratio=ratio,
                inn=item["innovation_norm_mean"],
                inn_max=item["innovation_norm_max"],
            )
        )
    print(f"wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
