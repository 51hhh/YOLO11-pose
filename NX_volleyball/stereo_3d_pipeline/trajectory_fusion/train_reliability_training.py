"""Batch construction, optimization loop, and checkpoint metadata helpers."""

from __future__ import annotations

from typing import List, Tuple

try:
    from .dataset import (
        apply_feature_normalizer,
        build_legacy_arrays,
        compute_feature_normalizer,
        legacy_feature_names,
        weak_label_names,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        apply_feature_normalizer,
        build_legacy_arrays,
        compute_feature_normalizer,
        legacy_feature_names,
        weak_label_names,
    )


def _load_training_deps():
    try:
        from .losses import (
            known_z_loss,
            known_z_range_loss,
            measurement_consistency_loss,
            physics_depth_loss,
            static_depth_jitter_loss,
            uncertainty_regularizer,
        )
        from .models import weighted_depth_consensus
    except ImportError:  # pragma: no cover - direct script execution
        from losses import (
            known_z_loss,
            known_z_range_loss,
            measurement_consistency_loss,
            physics_depth_loss,
            static_depth_jitter_loss,
            uncertainty_regularizer,
        )
        from models import weighted_depth_consensus
    return {
        "known_z_loss": known_z_loss,
        "known_z_range_loss": known_z_range_loss,
        "measurement_consistency_loss": measurement_consistency_loss,
        "physics_depth_loss": physics_depth_loss,
        "static_depth_jitter_loss": static_depth_jitter_loss,
        "uncertainty_regularizer": uncertainty_regularizer,
        "weighted_depth_consensus": weighted_depth_consensus,
    }


def build_training_batches(
    train_items: List[Tuple[object, object]],
    device: str,
    torch_module,
) -> tuple[List[dict], list[float], list[float]]:
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
                "features": torch_module.tensor(features, dtype=torch_module.float32, device=device).unsqueeze(0),
                "measurements": torch_module.tensor(
                    arrays["measurements"], dtype=torch_module.float32, device=device
                ).unsqueeze(0),
                "valid": torch_module.tensor(arrays["valid"], dtype=torch_module.float32, device=device).unsqueeze(0),
                "labels": torch_module.tensor(arrays["labels"], dtype=torch_module.float32, device=device).unsqueeze(0),
            }
        )
    return batches, feature_mean, feature_std


def train_batches(args, model, opt, batches: List[dict], torch_module) -> None:
    deps = _load_training_deps()
    label_index = {name: idx for idx, name in enumerate(weak_label_names())}

    for epoch in range(args.epochs):
        total = 0.0
        for batch in batches:
            opt.zero_grad(set_to_none=True)
            output = model(batch["features"])
            consensus = deps["weighted_depth_consensus"](batch["measurements"], batch["valid"], output, detach=True)
            learned_consensus = deps["weighted_depth_consensus"](
                batch["measurements"], batch["valid"], output, detach=False
            )
            loss_obs = deps["measurement_consistency_loss"](
                consensus,
                batch["measurements"],
                batch["valid"],
                output.log_sigma,
                output.bias,
                output.outlier_logit,
            )
            loss_phys = deps["physics_depth_loss"](learned_consensus, dt=0.01)
            loss_reg = deps["uncertainty_regularizer"](output.log_sigma, output.outlier_logit, batch["valid"])
            labels = batch["labels"]
            loss_known_z = deps["known_z_loss"](
                learned_consensus,
                labels[..., label_index["known_z"]],
                labels[..., label_index["known_z_valid"]],
            )
            loss_known_range = deps["known_z_range_loss"](
                learned_consensus,
                labels[..., label_index["known_z_min"]],
                labels[..., label_index["known_z_max"]],
                labels[..., label_index["known_z_range_valid"]],
            )
            loss_static = deps["static_depth_jitter_loss"](
                learned_consensus,
                labels[..., label_index["static"]],
            )
            loss = loss_obs + 0.25 * loss_phys + loss_reg
            loss = loss + args.known_z_weight * loss_known_z
            loss = loss + args.known_z_range_weight * loss_known_range
            loss = loss + args.static_jitter_weight * loss_static
            loss.backward()
            torch_module.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch + 1:04d} loss={total / len(batches):.6f}")


def build_checkpoint(
    args,
    clips,
    heldout_items: List[Tuple[object, object]],
    model,
    feature_mean: list[float],
    feature_std: list[float],
    method_names,
) -> dict:
    return {
        "model": model.state_dict(),
        "feature_names": legacy_feature_names(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "method_names": method_names,
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
