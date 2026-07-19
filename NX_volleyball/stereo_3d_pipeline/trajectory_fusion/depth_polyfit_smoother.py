#!/usr/bin/env python3
"""Offline batch polynomial smoother for candidate depth observations.

This baseline fits a low-order polynomial to raw candidate z_* observations.
It intentionally does not use legacy online ``z``/``z_stereo`` as
measurements, so it remains a fair comparison for reliability models.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .dataset import LegacySequence, load_legacy_sequences
    from .robust_smoother import SmootherConfig, _z_measurements, write_output
except ImportError:  # pragma: no cover - direct script execution
    from dataset import LegacySequence, load_legacy_sequences
    from robust_smoother import SmootherConfig, _z_measurements, write_output


@dataclass
class DepthPolyfitConfig:
    degree: int = 2
    huber_k: float = 2.5
    iterations: int = 5
    min_sigma: float = 0.015
    gravity_y: float = 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: List[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    return _median([abs(value - med) for value in values])


def _fit_weighted_poly(
    t_values: np.ndarray,
    z_values: np.ndarray,
    variances: np.ndarray,
    cfg: DepthPolyfitConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    degree = max(0, min(int(cfg.degree), len(z_values) - 1))
    design = np.vstack([t_values ** power for power in range(degree + 1)]).T
    base_sigma = np.sqrt(np.maximum(variances, cfg.min_sigma * cfg.min_sigma))
    robust_weights = np.ones_like(z_values, dtype=np.float64)
    coeff = np.zeros(degree + 1, dtype=np.float64)
    for _ in range(max(1, cfg.iterations)):
        weights = robust_weights / np.maximum(base_sigma, 1e-6)
        weighted_design = design * weights[:, None]
        weighted_z = z_values * weights
        coeff, *_ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)
        residuals = z_values - design @ coeff
        scale = max(1.4826 * _mad(list(residuals)), cfg.min_sigma)
        limits = cfg.huber_k * scale
        robust_weights = np.ones_like(z_values, dtype=np.float64)
        mask = np.abs(residuals) > limits
        robust_weights[mask] = np.maximum(0.05, limits / np.abs(residuals[mask]))
    residuals = z_values - design @ coeff
    sigma = max(1.4826 * _mad(list(residuals)), cfg.min_sigma)
    return coeff, residuals, sigma


def _eval_poly(coeff: np.ndarray, t_value: float) -> Tuple[float, float, float]:
    z = 0.0
    vz = 0.0
    az = 0.0
    for power, value in enumerate(coeff):
        z += float(value) * (t_value ** power)
        if power >= 1:
            vz += power * float(value) * (t_value ** (power - 1))
        if power >= 2:
            az += power * (power - 1) * float(value) * (t_value ** (power - 2))
    return z, vz, az


def smooth_sequence_polyfit(
    sequence: LegacySequence,
    cfg: DepthPolyfitConfig,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    rows = sequence.rows
    t0 = rows[0]["timestamp"]
    observations_t: List[float] = []
    observations_z: List[float] = []
    observations_var: List[float] = []
    per_frame_counts: List[int] = []
    smoother_cfg = SmootherConfig(gravity_y=cfg.gravity_y)
    for row in rows:
        measurements = _z_measurements(row, smoother_cfg)
        per_frame_counts.append(len(measurements))
        for z_value, variance, _name in measurements:
            if z_value > 0.1 and variance > 0.0 and math.isfinite(z_value) and math.isfinite(variance):
                observations_t.append(row["timestamp"] - t0)
                observations_z.append(z_value)
                observations_var.append(max(variance, cfg.min_sigma * cfg.min_sigma))

    output: List[Dict[str, float]] = []
    first = rows[0]
    if len(observations_z) < 2:
        fallback_z = _median([row.get("z_mono", -1.0) for row in rows if row.get("z_mono", -1.0) > 0.1])
        if fallback_z <= 0.1:
            fallback_z = max(first.get("z", 0.1), 0.1)
        coeff = np.array([fallback_z], dtype=np.float64)
        residuals = np.zeros(0, dtype=np.float64)
        sigma = cfg.min_sigma
        support = 0
    else:
        coeff, residuals, sigma = _fit_weighted_poly(
            np.asarray(observations_t, dtype=np.float64),
            np.asarray(observations_z, dtype=np.float64),
            np.asarray(observations_var, dtype=np.float64),
            cfg,
        )
        support = len(observations_z)

    for index, row in enumerate(rows):
        t_value = row["timestamp"] - t0
        z_value, vz, az = _eval_poly(coeff, t_value)
        out = dict(row)
        out.update(
            {
                "track_id": float(sequence.track_id),
                "smooth_x": float(first.get("x", 0.0)),
                "smooth_y": float(first.get("y", 0.0) + 0.5 * cfg.gravity_y * t_value * t_value),
                "smooth_z": float(max(z_value, 0.1)),
                "smooth_vx": 0.0,
                "smooth_vy": float(cfg.gravity_y * t_value),
                "smooth_vz": float(vz),
                "smooth_ax": 0.0,
                "smooth_ay": float(cfg.gravity_y),
                "smooth_az": float(az),
                "smooth_sigma_z": float(sigma),
                "depth_polyfit_degree": float(len(coeff) - 1),
                "depth_polyfit_support": float(support),
                "depth_polyfit_frame_support": float(per_frame_counts[index]),
            }
        )
        output.append(out)

    raw_z = [row["z"] for row in rows if row["z"] > 0.1]
    smooth_z = [row["smooth_z"] for row in output if row["smooth_z"] > 0.1]
    metrics: Dict[str, Any] = {
        "track_id": float(sequence.track_id),
        "frames": float(len(rows)),
        "rows_with_candidates": float(sum(1 for count in per_frame_counts if count > 0)),
        "observation_count": float(support),
        "degree": float(len(coeff) - 1),
        "residual_mad": _mad(list(residuals)) if len(residuals) else 0.0,
        "sigma_z": float(sigma),
        "raw_z_std": float(np.std(np.asarray(raw_z, dtype=np.float64))) if len(raw_z) > 1 else 0.0,
        "smooth_z_std": float(np.std(np.asarray(smooth_z, dtype=np.float64))) if len(smooth_z) > 1 else 0.0,
    }
    return output, metrics


def apply_depth_polyfit_smoother(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    metadata_path: str | Path | None = None,
    cfg: DepthPolyfitConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or DepthPolyfitConfig()
    all_rows: List[Dict[str, float]] = []
    sequence_reports: List[Dict[str, Any]] = []
    for sequence in load_legacy_sequences(input_csv, metadata_path=metadata_path):
        rows, metrics = smooth_sequence_polyfit(sequence, cfg)
        all_rows.extend(rows)
        sequence_reports.append(metrics)
    write_output(output_csv, all_rows)
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": len(all_rows),
        "config": cfg.__dict__,
        "sequences": sequence_reports,
    }


def _write_json(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--json-out")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--huber-k", type=float, default=2.5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--min-sigma", type=float, default=0.015)
    parser.add_argument("--gravity-y", type=float, default=0.0)
    args = parser.parse_args()

    report = apply_depth_polyfit_smoother(
        args.input,
        args.output,
        metadata_path=args.metadata,
        cfg=DepthPolyfitConfig(
            degree=args.degree,
            huber_k=args.huber_k,
            iterations=args.iterations,
            min_sigma=args.min_sigma,
            gravity_y=args.gravity_y,
        ),
    )
    if args.json_out:
        _write_json(args.json_out, report)
    print(
        "rows={rows} sequences={seqs} output={output}".format(
            rows=report["rows"],
            seqs=len(report["sequences"]),
            output=report["output_csv"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
