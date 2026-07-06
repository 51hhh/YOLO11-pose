#!/usr/bin/env python3
"""Robust physics-aware smoother for candidate-depth trajectory CSV files.

This is an offline baseline. By default it smooths raw candidate depth fields
instead of using legacy online ``z_stereo``/``z`` as measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from .dataset import METHOD_COLUMNS, LegacySequence, load_legacy_sequences
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS, LegacySequence, load_legacy_sequences


@dataclass
class SmootherConfig:
    process_sigma: float = 18.0
    base_sigma_xy: float = 0.035
    base_sigma_z: float = 0.080
    known_z_sigma: float = 0.040
    huber_k: float = 2.5
    gravity_y: float = 9.81
    use_method_depths: bool = True
    use_online_position: bool = False
    use_static_known_z: bool = False


ZMeasurement = Tuple[float, float, str]
ZMeasurementProvider = Callable[[int, Dict[str, float]], List[ZMeasurement]]


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


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _method_sigma_scale(name: str) -> float:
    if name == "mono":
        return 1.70
    if name.startswith("bbox"):
        return 1.35
    if name.startswith("circle"):
        return 1.05
    if name in {"roi_edge_centroid", "roi_radial_center", "roi_edge_pair_center"}:
        return 1.10
    if name in {"roi_multi_point", "roi_center_patch"}:
        return 1.25
    if "fallback" in name:
        return 2.00
    if name.startswith("roi_"):
        return 1.80
    return 1.50


def _method_group(name: str) -> str:
    if name == "mono":
        return "mono"
    if name.startswith("bbox"):
        return "bbox"
    if name.startswith("circle"):
        return "circle"
    if name in {"roi_edge_centroid", "roi_radial_center", "roi_edge_pair_center"}:
        return "roi_geometry"
    if name in {"roi_multi_point", "roi_center_patch"}:
        return "roi_patch"
    if "fallback" in name:
        return "fallback"
    return "roi_sparse"


def _quality_scale(row: Dict[str, float], name: str) -> float:
    support_key = f"{name}_support"
    std_key = f"{name}_std_px"
    conf_key = f"{name}_confidence"
    if name == "roi_multi_point":
        support_key = "subpixel_support"
        std_key = "subpixel_std_px"
        conf_key = "subpixel_confidence"

    scale = 1.0
    support = row.get(support_key, 0.0)
    if 0.0 < support < 4.0:
        scale *= 1.8
    std_px = row.get(std_key, -1.0)
    if std_px > 0.0:
        scale *= max(1.0, min(4.0, std_px / 1.5))
    confidence = row.get(conf_key, 0.0)
    if 0.0 < confidence < 0.4:
        scale *= 1.5

    pair_dy = abs(row.get("pair_epipolar_dy", -1.0))
    pair_tol = row.get("pair_y_tolerance", -1.0)
    if pair_dy >= 0.0 and pair_tol > 0.0 and pair_dy > pair_tol:
        scale *= 2.0
    shifted_iou = row.get("pair_shifted_iou", -1.0)
    if 0.0 <= shifted_iou < 0.2:
        scale *= 1.6
    if row.get("pair_positive_disparity", 1.0) == 0.0:
        scale *= 2.0
    return scale


def _z_measurements(row: Dict[str, float], cfg: SmootherConfig) -> List[Tuple[float, float, str]]:
    if not cfg.use_method_depths:
        return []
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for name, key in METHOD_COLUMNS:
        z_value = row.get(key, -1.0)
        if z_value <= 0.1:
            continue
        sigma = cfg.base_sigma_z * _method_sigma_scale(name) * _quality_scale(row, name) * max(1.0, z_value)
        grouped.setdefault(_method_group(name), []).append((z_value, sigma * sigma))

    out: List[Tuple[float, float, str]] = []
    for group, values in grouped.items():
        zs = [item[0] for item in values]
        variances = [item[1] for item in values]
        z_group = _median(zs)
        spread = _median([abs(value - z_group) for value in zs])
        # Candidates in a group are highly correlated, so do not divide the
        # variance by N. Add observed disagreement as extra uncertainty.
        variance = max(min(variances), spread * spread, (0.01 * max(1.0, z_group)) ** 2)
        out.append((z_group, variance, group))
    return out


def group_correlated_z_measurements(
    values: List[Tuple[float, float, str]],
    *,
    relative_floor: float = 0.01,
) -> List[ZMeasurement]:
    """Group correlated per-method depth observations by method family.

    Multiple depth candidates produced from the same ROI and disparity prior
    are not independent measurements. This helper keeps one robust observation
    per family and preserves disagreement inside the family as extra variance.
    """

    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for z_value, variance, method_name in values:
        if z_value <= 0.1 or variance <= 0.0 or not math.isfinite(z_value) or not math.isfinite(variance):
            continue
        grouped.setdefault(_method_group(method_name), []).append((z_value, variance))

    out: List[ZMeasurement] = []
    for group, group_values in grouped.items():
        zs = [item[0] for item in group_values]
        variances = [item[1] for item in group_values]
        z_group = _median(zs)
        spread = _median([abs(value - z_group) for value in zs])
        variance = max(min(variances), spread * spread, (relative_floor * max(1.0, z_group)) ** 2)
        out.append((z_group, variance, group))
    return out


def _initial_z(row: Dict[str, float]) -> float:
    candidates = [row.get(key, -1.0) for _, key in METHOD_COLUMNS]
    valid = [value for value in candidates if value > 0.1]
    if valid:
        return _median(valid)
    return max(row.get("z", -1.0), row.get("z_stereo", -1.0), row.get("z_mono", 0.1), 0.1)


def _metadata_float(sequence: LegacySequence, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = sequence.metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _metadata_bool(sequence: LegacySequence, *keys: str) -> bool:
    for key in keys:
        value = sequence.metadata.get(key)
        if isinstance(value, bool):
            return value
        if value is not None:
            return str(value).strip().lower() in {"1", "true", "yes", "on", "static"}
    return False


def smooth_sequence(
    sequence: LegacySequence,
    cfg: SmootherConfig,
    z_measurement_provider: ZMeasurementProvider | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows = sequence.rows
    first = rows[0]
    state = np.array([first["x"], first["y"], _initial_z(first), 0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.diag([0.1, 0.1, 0.2, 10.0, 10.0, 10.0]).astype(np.float64)
    h_xyz = np.zeros((3, 6), dtype=np.float64)
    h_xyz[0, 0] = 1.0
    h_xyz[1, 1] = 1.0
    h_xyz[2, 2] = 1.0
    h_z = np.zeros((1, 6), dtype=np.float64)
    h_z[0, 2] = 1.0

    output: List[Dict[str, float]] = []
    innovations: List[float] = []
    known_z = _metadata_float(sequence, "known_z_m", "known_z", "known_distance_m")
    use_known_z = cfg.use_static_known_z and known_z > 0.0 and _metadata_bool(sequence, "static", "is_static")
    for i, row in enumerate(rows):
        state, cov = _predict(state, cov, _dt(rows, i), cfg)
        if cfg.use_online_position:
            meas, r_diag = _position_measurement(row, cfg)
            state, cov, inn = _robust_update(state, cov, meas, h_xyz, r_diag, cfg)
            innovations.append(inn)

        z_measurements = (
            z_measurement_provider(i, row)
            if z_measurement_provider is not None
            else _z_measurements(row, cfg)
        )
        for z_value, z_var, _name in z_measurements:
            state, cov, inn = _robust_update(
                state,
                cov,
                np.array([z_value], dtype=np.float64),
                h_z,
                np.array([z_var], dtype=np.float64),
                cfg,
            )
            innovations.append(inn)
        if use_known_z:
            state, cov, inn = _robust_update(
                state,
                cov,
                np.array([known_z], dtype=np.float64),
                h_z,
                np.array([cfg.known_z_sigma * cfg.known_z_sigma], dtype=np.float64),
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
    parser.add_argument("--metadata", help="Optional metadata.yaml with weak labels")
    parser.add_argument("--no-method-depths", action="store_true", help="Do not use raw candidate z_* updates")
    parser.add_argument("--use-online-position", action="store_true", help="Also use legacy online x/y/z as a position update")
    parser.add_argument(
        "--use-static-known-z",
        dest="use_static_known_z",
        action="store_true",
        help="Use static known_z metadata as a smoother update. Off by default to avoid label leakage in evaluation.",
    )
    parser.add_argument(
        "--no-static-known-z",
        dest="use_static_known_z",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_static_known_z=False)
    parser.add_argument("--gravity-y", type=float, default=9.81, help="Camera-y gravity prior in m/s^2")
    args = parser.parse_args()

    cfg = SmootherConfig(
        use_method_depths=not args.no_method_depths,
        use_online_position=args.use_online_position,
        use_static_known_z=args.use_static_known_z,
        gravity_y=args.gravity_y,
    )
    sequences = load_legacy_sequences(args.input, metadata_path=args.metadata)
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
