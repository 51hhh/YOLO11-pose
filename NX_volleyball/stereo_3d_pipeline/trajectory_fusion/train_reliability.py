#!/usr/bin/env python3
"""Semi-supervised training scaffold for measurement reliability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None

try:
    from .dataset import (
        METHOD_NAMES,
        apply_feature_normalizer,
        build_legacy_arrays,
        compute_feature_normalizer,
        legacy_feature_names,
        load_legacy_sequences,
        weak_label_names,
    )
    from .manifest import DatasetClip, is_manifest_path, load_manifest
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        METHOD_NAMES,
        apply_feature_normalizer,
        build_legacy_arrays,
        compute_feature_normalizer,
        legacy_feature_names,
        load_legacy_sequences,
        weak_label_names,
    )
    from manifest import DatasetClip, is_manifest_path, load_manifest


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for train_reliability.py")


def resolve_input_clips(inputs: List[str], metadata: str | None) -> List[DatasetClip]:
    """Resolve CSV inputs or one manifest into DatasetClip objects."""

    if len(inputs) == 1 and is_manifest_path(inputs[0]):
        if metadata:
            raise SystemExit("--metadata cannot be used with a manifest")
        return load_manifest(inputs[0])
    if metadata and len(inputs) != 1:
        raise SystemExit("--metadata can only be used with a single CSV input")
    metadata_path = Path(metadata) if metadata else None
    return [
        DatasetClip(
            csv=Path(item),
            metadata=metadata_path if len(inputs) == 1 else None,
            split="train",
            name=Path(item).stem,
        )
        for item in inputs
    ]


def load_sequences_from_clips(
    clips: List[DatasetClip],
    train_split: str,
) -> Tuple[List[Tuple[DatasetClip, object]], List[Tuple[DatasetClip, object]]]:
    """Load train and held-out sequences from resolved clips."""

    train_items: List[Tuple[DatasetClip, object]] = []
    heldout_items: List[Tuple[DatasetClip, object]] = []
    for clip in clips:
        sequences = load_legacy_sequences(clip.csv, metadata_path=clip.metadata)
        target = train_items if clip.split == train_split else heldout_items
        target.extend((clip, sequence) for sequence in sequences)
    return train_items, heldout_items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="TrajectoryRecorder CSV(s), or one dataset manifest YAML/JSON")
    parser.add_argument("--metadata", help="Optional metadata.yaml with weak labels for a single CSV")
    parser.add_argument("--train-split", default="train", help="Manifest split name used for optimization")
    parser.add_argument("-o", "--output", default="reliability_net.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--known-z-weight", type=float, default=1.0)
    parser.add_argument("--known-z-range-weight", type=float, default=0.5)
    parser.add_argument("--static-jitter-weight", type=float, default=0.1)
    parser.add_argument("--bias-reg-weight", type=float, default=0.02)
    args = parser.parse_args()

    _require_torch()
    try:
        from .losses import (
            known_z_loss,
            known_z_range_loss,
            bias_regularizer,
            measurement_consistency_loss,
            physics_depth_loss,
            static_depth_jitter_loss,
            uncertainty_regularizer,
        )
        from .models import MeasurementReliabilityNet, weighted_depth_consensus
    except ImportError:  # pragma: no cover - direct script execution
        from losses import (
            known_z_loss,
            known_z_range_loss,
            bias_regularizer,
            measurement_consistency_loss,
            physics_depth_loss,
            static_depth_jitter_loss,
            uncertainty_regularizer,
        )
        from models import MeasurementReliabilityNet, weighted_depth_consensus

    clips = resolve_input_clips(args.inputs, args.metadata)
    train_items, heldout_items = load_sequences_from_clips(clips, args.train_split)
    if not train_items:
        raise SystemExit(f"no valid sequences in train split '{args.train_split}'")

    feature_dim = len(legacy_feature_names())
    model = MeasurementReliabilityNet(feature_dim, num_methods=len(METHOD_NAMES), hidden_dim=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    sequence_arrays: List[dict] = []
    all_features: List[List[float]] = []
    for clip, seq in train_items:
        arrays = build_legacy_arrays(seq)
        sequence_arrays.append({"clip": clip.name, "track_id": seq.track_id, **arrays})
        all_features.extend(arrays["features"])

    feature_mean, feature_std = compute_feature_normalizer(all_features)
    batches: List[dict] = []
    for arrays in sequence_arrays:
        features = apply_feature_normalizer(arrays["features"], feature_mean, feature_std)
        batches.append(
            {
                "track_id": arrays["track_id"],
                "features": torch.tensor(features, dtype=torch.float32, device=args.device).unsqueeze(0),
                "measurements": torch.tensor(arrays["measurements"], dtype=torch.float32, device=args.device).unsqueeze(0),
                "valid": torch.tensor(arrays["valid"], dtype=torch.float32, device=args.device).unsqueeze(0),
                "labels": torch.tensor(arrays["labels"], dtype=torch.float32, device=args.device).unsqueeze(0),
                "dt": torch.tensor(arrays["dt"], dtype=torch.float32, device=args.device).unsqueeze(0),
            }
        )
    label_index = {name: idx for idx, name in enumerate(weak_label_names())}

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
            loss_phys = physics_depth_loss(learned_consensus, batch["dt"])
            loss_reg = uncertainty_regularizer(output.log_sigma, output.outlier_logit, batch["valid"])
            loss_bias = bias_regularizer(output.bias, batch["valid"])
            labels = batch["labels"]
            loss_known_z = known_z_loss(
                learned_consensus,
                labels[..., label_index["known_z"]],
                labels[..., label_index["known_z_valid"]],
            )
            loss_known_range = known_z_range_loss(
                learned_consensus,
                labels[..., label_index["known_z_min"]],
                labels[..., label_index["known_z_max"]],
                labels[..., label_index["known_z_range_valid"]],
            )
            loss_static = static_depth_jitter_loss(
                learned_consensus,
                labels[..., label_index["static"]],
            )
            loss = loss_obs + 0.25 * loss_phys + loss_reg
            loss = loss + args.bias_reg_weight * loss_bias
            loss = loss + args.known_z_weight * loss_known_z
            loss = loss + args.known_z_range_weight * loss_known_range
            loss = loss + args.static_jitter_weight * loss_static
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch + 1:04d} loss={total / len(batches):.6f}")

    checkpoint = {
        "model": model.state_dict(),
        "feature_names": legacy_feature_names(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "method_names": METHOD_NAMES,
        "weak_label_names": weak_label_names(),
        "train_split": args.train_split,
        "source_clips": [
            {
                "csv": str(clip.csv),
                "metadata": str(clip.metadata) if clip.metadata else None,
                "split": clip.split,
                "name": clip.name,
            }
            for clip in clips
        ],
        "heldout_sequence_count": len(heldout_items),
        "note": "Experimental reliability model. It predicts sigma/bias/outlier, not final trajectory.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
