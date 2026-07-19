#!/usr/bin/env python3
"""Apply a trained measurement-reliability checkpoint to recorder CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None

try:
    from .dataset import (
        METHOD_NAMES,
        apply_feature_normalizer,
        build_legacy_arrays,
        legacy_feature_names,
        load_legacy_sequences,
        resolve_method_allowlist,
    )
    from .models import MeasurementReliabilityNet, reliability_weight, weighted_depth_consensus
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        METHOD_NAMES,
        apply_feature_normalizer,
        build_legacy_arrays,
        legacy_feature_names,
        load_legacy_sequences,
        resolve_method_allowlist,
    )
    from models import MeasurementReliabilityNet, reliability_weight, weighted_depth_consensus


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for evaluate_reliability_checkpoint.py")


def _hidden_dim_from_state(state: Dict[str, Any]) -> int:
    weight = state.get("input_proj.0.weight")
    if weight is None:
        raise ValueError("checkpoint model state lacks input_proj.0.weight")
    return int(weight.shape[0])


def _load_model(checkpoint_path: str | Path, device: str) -> tuple[MeasurementReliabilityNet, Dict[str, Any]]:
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


def _finite_diff(values: Sequence[float], dts: Sequence[float], valid: Sequence[bool] | None = None) -> List[float | str]:
    out: List[float | str] = []
    previous = None
    valid_values = valid if valid is not None else [True] * len(values)
    for value, dt, is_valid in zip(values, dts, valid_values):
        if not is_valid:
            out.append("")
            previous = None
            continue
        if previous is None:
            out.append(0.0)
        else:
            out.append((value - previous) / max(float(dt), 1e-4))
        previous = value
    return out


def _method_summary(
    method_names: Sequence[str],
    weights: Any,
    valid: Any,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    weights_cpu = weights.detach().cpu()
    valid_cpu = valid.detach().cpu()
    for idx, name in enumerate(method_names):
        method_valid = valid_cpu[..., idx] > 0.0
        count = int(method_valid.sum().item())
        if count <= 0:
            summary[str(name)] = {"valid": 0.0, "mean_weight": 0.0}
            continue
        method_weights = weights_cpu[..., idx, 0][method_valid]
        summary[str(name)] = {
            "valid": float(count),
            "mean_weight": float(method_weights.mean().item()),
            "max_weight": float(method_weights.max().item()),
        }
    return summary


def _write_rows(path: str | Path, rows: List[Dict[str, float]]) -> None:
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
        "reliability_valid_count",
        "reliability_top_method",
        "reliability_top_method_name",
        "reliability_top_weight",
        "z_mono",
        "z_stereo",
        "depth_method",
        "confidence",
    ]
    extras = [key for key in rows[0].keys() if key not in preferred]
    fieldnames = preferred + extras
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def apply_checkpoint(
    input_csv: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    metadata_path: str | Path | None = None,
    device: str = "cpu",
    method_names: Sequence[str] | str | None = None,
) -> Dict[str, Any]:
    model, checkpoint = _load_model(checkpoint_path, device)
    feature_names = checkpoint.get("feature_names") or legacy_feature_names()
    model_method_names = list(checkpoint.get("method_names") or METHOD_NAMES)
    if list(feature_names) != legacy_feature_names():
        raise ValueError("checkpoint feature_names do not match current legacy_feature_names()")
    if model_method_names != list(METHOD_NAMES):
        raise ValueError("checkpoint method_names do not match current METHOD_NAMES")
    checkpoint_allowlist = checkpoint.get("method_allowlist")
    requested_allowlist = resolve_method_allowlist(method_names)
    if (
        requested_allowlist is not None
        and checkpoint_allowlist is not None
        and list(requested_allowlist) != list(checkpoint_allowlist)
    ):
        raise ValueError("requested method_names do not match checkpoint method_allowlist")
    method_allowlist = requested_allowlist if requested_allowlist is not None else checkpoint_allowlist

    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
    sequences = load_legacy_sequences(input_csv, metadata_path=metadata_path)
    all_rows: List[Dict[str, float]] = []
    report: Dict[str, Any] = {"sequences": []}

    with torch.no_grad():
        for sequence in sequences:
            arrays = build_legacy_arrays(sequence, method_names=method_allowlist)
            features = apply_feature_normalizer(arrays["features"], feature_mean, feature_std)
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            measurements = torch.tensor(arrays["measurements"], dtype=torch.float32, device=device).unsqueeze(0)
            valid = torch.tensor(arrays["valid"], dtype=torch.float32, device=device).unsqueeze(0)
            output = model(feature_tensor)
            consensus = weighted_depth_consensus(measurements, valid, output, detach=True)
            weights = reliability_weight(output, valid)

            z_values = consensus.squeeze(0).squeeze(-1).detach().cpu().tolist()
            dts = [item[0] for item in arrays["dt"]]
            weights_cpu = weights.squeeze(0).detach().cpu()
            valid_cpu = valid.squeeze(0).detach().cpu()
            frame_valid = [bool((valid_cpu[idx, :] > 0.0).any()) for idx in range(len(sequence.rows))]
            vz_values = _finite_diff(z_values, dts, frame_valid)
            common_sigma = torch.exp(output.common_log_sigma).squeeze(0).squeeze(-1).detach().cpu().tolist()

            for idx, source_row in enumerate(sequence.rows):
                row = dict(source_row)
                method_weights = weights_cpu[idx, :, 0]
                method_valid = valid_cpu[idx, :] > 0.0
                has_observation = bool(method_valid.any())
                valid_weight_sum = float(method_weights[method_valid].sum().item()) if has_observation else 0.0
                consensus_sigma = valid_weight_sum ** -0.5 if valid_weight_sum > 1e-12 else 0.0
                if has_observation:
                    masked = method_weights.clone()
                    masked[~method_valid] = -1.0
                    top_index = int(torch.argmax(masked).item())
                    top_method = float(top_index)
                    top_method_name = model_method_names[top_index]
                    top_weight = float(masked[top_index].item())
                else:
                    top_method = -1.0
                    top_method_name = ""
                    top_weight = 0.0
                row.update(
                    {
                        "track_id": float(sequence.track_id),
                        "smooth_x": row.get("x", 0.0) if has_observation else "",
                        "smooth_y": row.get("y", 0.0) if has_observation else "",
                        "smooth_z": float(z_values[idx]) if has_observation else "",
                        "smooth_vx": 0.0 if has_observation else "",
                        "smooth_vy": 0.0 if has_observation else "",
                        "smooth_vz": vz_values[idx] if has_observation else "",
                        "smooth_sigma_z": float(consensus_sigma) if has_observation else "",
                        "reliability_common_sigma": float(common_sigma[idx]),
                        "reliability_valid_count": float(method_valid.sum().item()),
                        "reliability_top_method": top_method,
                        "reliability_top_method_name": top_method_name,
                        "reliability_top_weight": top_weight,
                    }
                )
                all_rows.append(row)

            report["sequences"].append(
                {
                    "track_id": sequence.track_id,
                    "frames": sequence.length,
                    "method_summary": _method_summary(model_method_names, weights, valid),
                }
            )

    _write_rows(output_csv, all_rows)
    report["output_csv"] = str(output_csv)
    report["input_csv"] = str(input_csv)
    report["checkpoint"] = str(checkpoint_path)
    report["method_allowlist"] = list(method_allowlist) if method_allowlist is not None else None
    report["rows"] = len(all_rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="reliability_consensus.csv")
    parser.add_argument("--metadata")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default=None, help="Optional method allowlist/preset; defaults to checkpoint allowlist.")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    report = apply_checkpoint(
        input_csv=args.input,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        metadata_path=args.metadata,
        device=args.device,
        method_names=args.methods,
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {report['rows']} rows to {report['output_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
