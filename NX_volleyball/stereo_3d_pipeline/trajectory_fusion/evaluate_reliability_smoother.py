#!/usr/bin/env python3
"""Apply ReliabilityNet as an observation model for the robust smoother.

This keeps the physical Kalman/Huber smoother as the trajectory owner. The
network only supplies per-method bias, sigma and outlier probability for raw
depth candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None

try:
    from .dataset import (
        METHOD_COLUMNS,
        METHOD_NAMES,
        apply_feature_normalizer,
        build_legacy_arrays,
        legacy_feature_names,
        load_legacy_sequences,
    )
    from .models import MeasurementReliabilityNet
    from .robust_smoother import (
        SmootherConfig,
        ZMeasurement,
        group_correlated_z_measurements,
        smooth_sequence,
        smooth_sequence_rts,
        write_output,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        METHOD_COLUMNS,
        METHOD_NAMES,
        apply_feature_normalizer,
        build_legacy_arrays,
        legacy_feature_names,
        load_legacy_sequences,
    )
    from models import MeasurementReliabilityNet
    from robust_smoother import (
        SmootherConfig,
        ZMeasurement,
        group_correlated_z_measurements,
        smooth_sequence,
        smooth_sequence_rts,
        write_output,
    )


@dataclass
class LearnedObservationConfig:
    min_sigma: float = 0.015
    max_sigma: float = 1.5
    sigma_scale: float = 1.0
    bias_scale: float = 1.0
    min_inlier_prob: float = 0.02
    relative_floor: float = 0.01


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for evaluate_reliability_smoother.py")


def _hidden_dim_from_state(state: Dict[str, Any]) -> int:
    weight = state.get("input_proj.0.weight")
    if weight is None:
        raise ValueError("checkpoint model state lacks input_proj.0.weight")
    return int(weight.shape[0])


def _load_model(checkpoint_path: str | Path, device: str) -> Tuple[MeasurementReliabilityNet, Dict[str, Any]]:
    _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    feature_names = checkpoint.get("feature_names") or legacy_feature_names()
    method_names = checkpoint.get("method_names") or METHOD_NAMES
    hidden_dim = _hidden_dim_from_state(state)
    model = MeasurementReliabilityNet(
        input_dim=len(feature_names),
        num_methods=len(method_names),
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _empty_diagnostic() -> Dict[str, float | str]:
    return {
        "reliability_smoother_valid_count": 0.0,
        "reliability_smoother_group_count": 0.0,
        "reliability_smoother_top_method": -1.0,
        "reliability_smoother_top_method_name": "",
        "reliability_smoother_top_weight": 0.0,
        "reliability_smoother_top_raw_z": 0.0,
        "reliability_smoother_top_corrected_z": 0.0,
        "reliability_smoother_top_bias": 0.0,
        "reliability_smoother_top_sigma": 0.0,
        "reliability_smoother_top_inlier_prob": 0.0,
    }


def _build_learned_observations(
    measurements: Sequence[Sequence[float]],
    valid: Sequence[Sequence[float]],
    output: Any,
    method_names: Sequence[str],
    cfg: LearnedObservationConfig,
) -> Tuple[List[List[ZMeasurement]], List[Dict[str, float | str]], Dict[str, Dict[str, float]]]:
    if torch is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("torch is required")

    log_sigma = output.log_sigma.squeeze(0).detach().cpu()
    bias = output.bias.squeeze(0).detach().cpu()
    outlier_logit = output.outlier_logit.squeeze(0).detach().cpu()
    observations_by_frame: List[List[ZMeasurement]] = []
    diagnostics: List[Dict[str, float | str]] = []
    method_stats = {
        str(name): {
            "valid": 0.0,
            "top_count": 0.0,
            "weight_sum": 0.0,
            "sigma_sum": 0.0,
            "bias_sum": 0.0,
            "abs_bias_sum": 0.0,
            "inlier_prob_sum": 0.0,
            "raw_z_sum": 0.0,
            "corrected_z_sum": 0.0,
        }
        for name in method_names
    }

    for frame_index, (measurement_row, valid_row) in enumerate(zip(measurements, valid)):
        candidates: List[ZMeasurement] = []
        diagnostic = _empty_diagnostic()
        top_method_index = -1
        top_weight = -1.0
        top_raw_z = 0.0
        top_corrected_z = 0.0
        top_bias = 0.0
        top_sigma = 0.0
        top_inlier_prob = 0.0

        for method_index, ((method_name, _key), raw_z, is_valid) in enumerate(
            zip(METHOD_COLUMNS, measurement_row, valid_row)
        ):
            if method_index >= len(method_names) or is_valid <= 0.0 or raw_z <= 0.1:
                continue
            predicted_bias = float(bias[frame_index, method_index, 0]) * cfg.bias_scale
            sigma = math.exp(float(log_sigma[frame_index, method_index, 0])) * cfg.sigma_scale
            sigma = max(cfg.min_sigma, min(cfg.max_sigma, sigma))
            outlier_prob = float(torch.sigmoid(outlier_logit[frame_index, method_index, 0]).item())
            inlier_prob = max(cfg.min_inlier_prob, min(1.0, 1.0 - outlier_prob))
            corrected_z = float(raw_z) - predicted_bias
            if not math.isfinite(corrected_z) or corrected_z <= 0.1:
                continue
            variance = (sigma * sigma) / inlier_prob
            if not math.isfinite(variance) or variance <= 0.0:
                continue

            weight = 1.0 / variance
            candidates.append((corrected_z, variance, method_name))
            stats = method_stats[str(method_names[method_index])]
            stats["valid"] += 1.0
            stats["weight_sum"] += weight
            stats["sigma_sum"] += sigma
            stats["bias_sum"] += predicted_bias
            stats["abs_bias_sum"] += abs(predicted_bias)
            stats["inlier_prob_sum"] += inlier_prob
            stats["raw_z_sum"] += float(raw_z)
            stats["corrected_z_sum"] += corrected_z
            if weight > top_weight:
                top_method_index = method_index
                top_weight = weight
                top_raw_z = float(raw_z)
                top_corrected_z = corrected_z
                top_bias = predicted_bias
                top_sigma = sigma
                top_inlier_prob = inlier_prob

        grouped = group_correlated_z_measurements(candidates, relative_floor=cfg.relative_floor)
        observations_by_frame.append(grouped)
        if top_method_index >= 0:
            diagnostic.update(
                {
                    "reliability_smoother_valid_count": float(len(candidates)),
                    "reliability_smoother_group_count": float(len(grouped)),
                    "reliability_smoother_top_method": float(top_method_index),
                    "reliability_smoother_top_method_name": str(method_names[top_method_index]),
                    "reliability_smoother_top_weight": float(top_weight),
                    "reliability_smoother_top_raw_z": top_raw_z,
                    "reliability_smoother_top_corrected_z": top_corrected_z,
                    "reliability_smoother_top_bias": top_bias,
                    "reliability_smoother_top_sigma": top_sigma,
                    "reliability_smoother_top_inlier_prob": top_inlier_prob,
                }
            )
            method_stats[str(method_names[top_method_index])]["top_count"] += 1.0
        diagnostics.append(diagnostic)

    method_summary: Dict[str, Dict[str, float]] = {}
    for name in method_names:
        stats = method_stats[str(name)]
        count = stats["valid"]
        method_summary[str(name)] = {
            "valid": count,
            "top_count": stats["top_count"],
            "top_rate": stats["top_count"] / max(1.0, float(len(measurements))),
            "mean_weight": stats["weight_sum"] / count if count > 0 else 0.0,
            "mean_sigma": stats["sigma_sum"] / count if count > 0 else 0.0,
            "mean_bias": stats["bias_sum"] / count if count > 0 else 0.0,
            "mean_abs_bias": stats["abs_bias_sum"] / count if count > 0 else 0.0,
            "mean_inlier_prob": stats["inlier_prob_sum"] / count if count > 0 else 0.0,
            "mean_raw_z": stats["raw_z_sum"] / count if count > 0 else 0.0,
            "mean_corrected_z": stats["corrected_z_sum"] / count if count > 0 else 0.0,
            "mean_corrected_minus_raw_z": (
                (stats["corrected_z_sum"] - stats["raw_z_sum"]) / count if count > 0 else 0.0
            ),
        }
    return observations_by_frame, diagnostics, method_summary


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def apply_reliability_smoother(
    input_csv: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    metadata_path: str | Path | None = None,
    device: str = "cpu",
    smoother_cfg: SmootherConfig | None = None,
    learned_cfg: LearnedObservationConfig | None = None,
    rts: bool = False,
) -> Dict[str, Any]:
    model, checkpoint = _load_model(checkpoint_path, device)
    feature_names = checkpoint.get("feature_names") or legacy_feature_names()
    method_names = list(checkpoint.get("method_names") or METHOD_NAMES)
    if list(feature_names) != legacy_feature_names():
        raise ValueError("checkpoint feature_names do not match current legacy_feature_names()")
    if method_names != list(METHOD_NAMES):
        raise ValueError("checkpoint method_names do not match current METHOD_NAMES")

    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
    sequences = load_legacy_sequences(input_csv, metadata_path=metadata_path)
    smoother_cfg = smoother_cfg or SmootherConfig()
    learned_cfg = learned_cfg or LearnedObservationConfig()
    all_rows: List[Dict[str, float]] = []
    report: Dict[str, Any] = {"sequences": []}
    smoother = smooth_sequence_rts if rts else smooth_sequence

    with torch.no_grad():  # type: ignore[union-attr]
        for sequence in sequences:
            arrays = build_legacy_arrays(sequence)
            features = apply_feature_normalizer(arrays["features"], feature_mean, feature_std)
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)  # type: ignore[union-attr]
            output = model(feature_tensor)
            observations, diagnostics, method_summary = _build_learned_observations(
                arrays["measurements"],
                arrays["valid"],
                output,
                method_names,
                learned_cfg,
            )

            def provider(index: int, _row: Dict[str, float]) -> List[ZMeasurement]:
                return observations[index]

            rows, metrics = smoother(sequence, smoother_cfg, z_measurement_provider=provider)
            for index, row in enumerate(rows):
                row.update(diagnostics[index])
            all_rows.extend(rows)

            smooth_z = [float(row["smooth_z"]) for row in rows if float(row["smooth_z"]) > 0.1]
            report["sequences"].append(
                {
                    **metrics,
                    "smooth_z_std_recomputed": _std(smooth_z),
                    "method_summary": method_summary,
                }
            )

    write_output(output_csv, all_rows)
    report["output_csv"] = str(output_csv)
    report["input_csv"] = str(input_csv)
    report["checkpoint"] = str(checkpoint_path)
    report["rows"] = len(all_rows)
    report["rts"] = rts
    report["learned_observation_config"] = learned_cfg.__dict__
    report["smoother_config"] = smoother_cfg.__dict__
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="reliability_smoother.csv")
    parser.add_argument("--metadata")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out")
    parser.add_argument("--gravity-y", type=float, default=9.81)
    parser.add_argument("--use-online-position", action="store_true")
    parser.add_argument("--rts", action="store_true", help="Run offline RTS backward smoothing after learned observations")
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
    parser.add_argument("--min-sigma", type=float, default=0.015)
    parser.add_argument("--max-sigma", type=float, default=1.5)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument("--min-inlier-prob", type=float, default=0.02)
    parser.add_argument("--relative-floor", type=float, default=0.01)
    args = parser.parse_args()

    smoother_cfg = SmootherConfig(
        use_method_depths=True,
        use_online_position=args.use_online_position,
        use_static_known_z=args.use_static_known_z,
        gravity_y=args.gravity_y,
    )
    learned_cfg = LearnedObservationConfig(
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        sigma_scale=args.sigma_scale,
        bias_scale=args.bias_scale,
        min_inlier_prob=args.min_inlier_prob,
        relative_floor=args.relative_floor,
    )
    report = apply_reliability_smoother(
        input_csv=args.input,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        metadata_path=args.metadata,
        device=args.device,
        smoother_cfg=smoother_cfg,
        learned_cfg=learned_cfg,
        rts=args.rts,
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for item in report["sequences"]:
        raw_std = float(item["raw_z_std"])
        smooth_std = float(item["smooth_z_std"])
        ratio = smooth_std / raw_std if raw_std > 1e-9 else 0.0
        print(
            "track={track_id:.0f} frames={frames:.0f} raw_z_std={raw:.4f} "
            "smooth_z_std={smooth:.4f} ratio={ratio:.3f} inn_mean={inn:.3f} inn_max={inn_max:.3f}".format(
                track_id=float(item["track_id"]),
                frames=float(item["frames"]),
                raw=raw_std,
                smooth=smooth_std,
                ratio=ratio,
                inn=float(item["innovation_norm_mean"]),
                inn_max=float(item["innovation_norm_max"]),
            )
        )
    print(f"wrote {report['rows']} rows to {report['output_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
