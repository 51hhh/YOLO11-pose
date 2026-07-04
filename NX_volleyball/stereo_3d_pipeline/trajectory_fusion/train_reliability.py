#!/usr/bin/env python3
"""Semi-supervised training scaffold for measurement reliability."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None

try:
    from .dataset import METHOD_NAMES, legacy_feature_names
    from .train_reliability_inputs import load_sequences_from_clips, resolve_input_clips
    from .train_reliability_training import build_checkpoint, build_training_batches, train_batches
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_NAMES, legacy_feature_names
    from train_reliability_inputs import load_sequences_from_clips, resolve_input_clips
    from train_reliability_training import build_checkpoint, build_training_batches, train_batches


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for train_reliability.py")


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
    args = parser.parse_args()

    _require_torch()
    try:
        from .models import MeasurementReliabilityNet
    except ImportError:  # pragma: no cover - direct script execution
        from models import MeasurementReliabilityNet

    clips = resolve_input_clips(args.inputs, args.metadata)
    train_items, heldout_items = load_sequences_from_clips(clips, args.train_split)
    if not train_items:
        raise SystemExit(f"no valid sequences in train split '{args.train_split}'")

    feature_dim = len(legacy_feature_names())
    model = MeasurementReliabilityNet(feature_dim, num_methods=len(METHOD_NAMES), hidden_dim=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    batches, feature_mean, feature_std = build_training_batches(train_items, args.device, torch)
    train_batches(args, model, opt, batches, torch)
    checkpoint = build_checkpoint(args, clips, heldout_items, model, feature_mean, feature_std, METHOD_NAMES)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
