"""Input and manifest loading helpers for reliability training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

try:
    from .dataset import load_legacy_sequences
    from .manifest import DatasetClip, is_manifest_path, load_manifest
except ImportError:  # pragma: no cover - direct script execution
    from dataset import load_legacy_sequences
    from manifest import DatasetClip, is_manifest_path, load_manifest


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
