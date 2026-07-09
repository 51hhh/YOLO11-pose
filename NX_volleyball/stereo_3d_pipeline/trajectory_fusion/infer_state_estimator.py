#!/usr/bin/env python3
"""Frame-by-frame inference for the causal 3D trajectory state estimator.

This runs ``CausalKalmanNet`` exactly as it would on the robot: one ``step()``
per frame, carrying state + GRU hidden across the sequence, appending each new
observation to the causal history. It is the reference for what the real-time
C++ deployment must reproduce.

Two modes:
  - default: batch ``forward`` over the whole clip (fast, for offline eval).
  - ``--frame-by-frame``: explicit per-frame ``step`` loop, and asserts the
    output matches the batch path (train/inference consistency check).

Input CSV is reprojected with the same ``d0`` correction used in training, so
the estimator sees metric XYZ, not the raw ``z_*`` columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from .dataset import (
        build_metric_state_arrays,
        load_legacy_sequences,
        metric_feature_names,
        metric_state_method_names,
    )
    from .reproject import load_reprojection_model
    from .state_estimator import CausalKalmanNet, FilterConfig
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        build_metric_state_arrays,
        load_legacy_sequences,
        metric_feature_names,
        metric_state_method_names,
    )
    from reproject import load_reprojection_model
    from state_estimator import CausalKalmanNet, FilterConfig


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for infer_state_estimator.py")


def load_model(checkpoint_path: str, device: str) -> tuple[CausalKalmanNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fc = ckpt["filter_config"]
    cfg = FilterConfig(
        num_methods=fc["num_methods"],
        quality_dim=fc["quality_dim"],
        hidden_dim=fc["hidden_dim"],
        gru_layers=fc.get("gru_layers", 1),
        gravity=fc.get("gravity", 0.0),
        gravity_axis=fc.get("gravity_axis", 1),
        min_gain=fc.get("min_gain", 0.0),
        max_gain=fc.get("max_gain", 1.0),
        process_floor=fc.get("process_floor", 1e-3),
    )
    model = CausalKalmanNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _sequence_tensors(arrays: Dict[str, List], device: str) -> Dict[str, torch.Tensor]:
    return {
        "obs_xyz": torch.tensor(arrays["points"], dtype=torch.float32, device=device),
        "obs_valid": torch.tensor(arrays["point_valid"], dtype=torch.float32, device=device),
        "quality": torch.tensor(arrays["features"], dtype=torch.float32, device=device),
        "dt": torch.tensor([d[0] for d in arrays["dt"]], dtype=torch.float32, device=device),
    }


def run_frame_by_frame(model: CausalKalmanNet, t: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Explicit per-frame step loop, mimicking the online history queue."""

    n = t["dt"].shape[0]
    carry = model.init_state(1, t["obs_xyz"].device, t["obs_xyz"].dtype)
    states = []
    for i in range(n):
        state, _logvar, carry = model.step(
            carry,
            t["obs_xyz"][i].unsqueeze(0),
            t["obs_valid"][i].unsqueeze(0),
            t["quality"][i].unsqueeze(0),
            t["dt"][i].unsqueeze(0),
        )
        states.append(state.squeeze(0))
    return torch.stack(states, dim=0)  # [T, 6]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="state_estimator.pt from train_state_estimator.py")
    parser.add_argument("csv", help="TrajectoryRecorder CSV to filter")
    parser.add_argument("--metadata", help="metadata.yaml for the CSV")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-o", "--output", help="write filtered [x,y,z,vx,vy,vz] CSV")
    parser.add_argument("--frame-by-frame", action="store_true", help="explicit step loop + consistency assert")
    args = parser.parse_args()

    _require_torch()
    model, ckpt = load_model(args.checkpoint, args.device)
    method_allowlist = ckpt.get("method_allowlist")
    calib = ckpt.get("calib")
    offset_fit = ckpt.get("offset_fit")
    if not calib:
        raise SystemExit("checkpoint has no calib path; retrain with --calib")

    reprojection_model = load_reprojection_model(calib, offset_fit)
    sequences = load_legacy_sequences(args.csv, metadata_path=args.metadata)
    if not sequences:
        raise SystemExit("no track sequences in CSV")

    all_rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for seq in sequences:
            arrays = build_metric_state_arrays(seq, reprojection_model, method_names=method_allowlist)
            if not arrays["dt"]:
                continue
            t = _sequence_tensors(arrays, args.device)
            states_batch, _lv = model(
                t["obs_xyz"].unsqueeze(0),
                t["obs_valid"].unsqueeze(0),
                t["quality"].unsqueeze(0),
                t["dt"].unsqueeze(0),
            )
            states_batch = states_batch[0]  # [T,6]

            if args.frame_by_frame:
                states_fbf = run_frame_by_frame(model, t)
                max_diff = float((states_fbf - states_batch).abs().max())
                print(f"track={seq.track_id} frames={states_batch.shape[0]} "
                      f"max|frame_by_frame - batch|={max_diff:.2e}")
                if max_diff > 1e-4:
                    raise SystemExit(f"causal consistency broken: {max_diff}")
                states = states_fbf
            else:
                states = states_batch

            z_med = float(states[:, 2].median())
            print(f"track={seq.track_id} filtered_z median={z_med:.4f}m frames={states.shape[0]}")

            for i, row in enumerate(seq.rows):
                s = states[i]
                all_rows.append({
                    "frame_id": row.get("frame_id", i),
                    "timestamp": row.get("timestamp", 0.0),
                    "track_id": seq.track_id,
                    "filter_x": float(s[0]),
                    "filter_y": float(s[1]),
                    "filter_z": float(s[2]),
                    "filter_vx": float(s[3]),
                    "filter_vy": float(s[4]),
                    "filter_vz": float(s[5]),
                })

    if args.output and all_rows:
        fieldnames = list(all_rows[0].keys())
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
