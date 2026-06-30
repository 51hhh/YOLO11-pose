#!/usr/bin/env python3
"""Self-supervised training scaffold for measurement reliability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None

try:
    from .dataset import (
        METHOD_NAMES,
        build_legacy_arrays,
        legacy_feature_names,
        load_legacy_sequences,
        normalize_features,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        METHOD_NAMES,
        build_legacy_arrays,
        legacy_feature_names,
        load_legacy_sequences,
        normalize_features,
    )


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for train_reliability.py")


def main() -> int:
    _require_torch()
    try:
        from .losses import measurement_consistency_loss, physics_depth_loss, uncertainty_regularizer
        from .models import MeasurementReliabilityNet, weighted_depth_consensus
    except ImportError:  # pragma: no cover - direct script execution
        from losses import measurement_consistency_loss, physics_depth_loss, uncertainty_regularizer
        from models import MeasurementReliabilityNet, weighted_depth_consensus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Legacy TrajectoryRecorder CSV")
    parser.add_argument("-o", "--output", default="reliability_net.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sequences = load_legacy_sequences(args.input)
    if not sequences:
        raise SystemExit(f"no valid sequences in {args.input}")

    feature_dim = len(legacy_feature_names())
    model = MeasurementReliabilityNet(feature_dim, num_methods=len(METHOD_NAMES), hidden_dim=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    batches: List[dict] = []
    for seq in sequences:
        arrays = build_legacy_arrays(seq)
        features = normalize_features(arrays["features"])
        batches.append(
            {
                "track_id": seq.track_id,
                "features": torch.tensor(features, dtype=torch.float32, device=args.device).unsqueeze(0),
                "measurements": torch.tensor(arrays["measurements"], dtype=torch.float32, device=args.device).unsqueeze(0),
                "valid": torch.tensor(arrays["valid"], dtype=torch.float32, device=args.device).unsqueeze(0),
            }
        )

    for epoch in range(args.epochs):
        total = 0.0
        for batch in batches:
            opt.zero_grad(set_to_none=True)
            output = model(batch["features"])
            consensus = weighted_depth_consensus(batch["measurements"], batch["valid"], output, detach=True)
            learned_consensus = weighted_depth_consensus(batch["measurements"], batch["valid"], output, detach=False)
            loss_obs = measurement_consistency_loss(
                consensus,
                batch["measurements"],
                batch["valid"],
                output.log_sigma,
                output.bias,
                output.outlier_logit,
            )
            loss_phys = physics_depth_loss(learned_consensus, dt=0.01)
            loss_reg = uncertainty_regularizer(output.log_sigma, output.outlier_logit, batch["valid"])
            loss = loss_obs + 0.25 * loss_phys + loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch + 1:04d} loss={total / len(batches):.6f}")

    checkpoint = {
        "model": model.state_dict(),
        "feature_names": legacy_feature_names(),
        "method_names": METHOD_NAMES,
        "note": "Experimental reliability model. It predicts sigma/bias/outlier, not final trajectory.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
