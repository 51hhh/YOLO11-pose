#!/usr/bin/env python3
"""Dataset helpers for trajectory fusion experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    from .dataset_io import (
        derive_frame_summary_path,
        find_metadata_for_csv,
        iter_extended_rows,
        read_csv_rows,
        read_metadata,
        safe_int as _safe_int,
    )
    from .dataset_features import build_legacy_arrays
    from .dataset_legacy_rows import parse_legacy_row
    from .dataset_normalization import (
        apply_feature_normalizer,
        compute_feature_normalizer,
        normalize_features,
    )
    from .dataset_schema import (
        METHOD_COLUMNS,
        METHOD_NAMES,
        LegacySequence,
        legacy_feature_names,
        weak_label_names,
    )
except ImportError:
    from dataset_io import (  # type: ignore
        derive_frame_summary_path,
        find_metadata_for_csv,
        iter_extended_rows,
        read_csv_rows,
        read_metadata,
        safe_int as _safe_int,
    )
    from dataset_features import build_legacy_arrays  # type: ignore
    from dataset_legacy_rows import parse_legacy_row  # type: ignore
    from dataset_normalization import (  # type: ignore
        apply_feature_normalizer,
        compute_feature_normalizer,
        normalize_features,
    )
    from dataset_schema import (  # type: ignore
        METHOD_COLUMNS,
        METHOD_NAMES,
        LegacySequence,
        legacy_feature_names,
        weak_label_names,
    )


def load_legacy_sequences(
    path: str | Path,
    min_track_len: int = 3,
    metadata_path: str | Path | None = None,
) -> List[LegacySequence]:
    """Load current trajectory recorder CSV and group rows by track_id."""

    metadata = read_metadata(metadata_path or find_metadata_for_csv(path))
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in read_csv_rows(path):
        track_id = _safe_int(row.get("track_id"), -1)
        if track_id < 0:
            continue
        parsed = parse_legacy_row(row)
        grouped.setdefault(track_id, []).append(parsed)

    sequences: List[LegacySequence] = []
    for track_id, rows in grouped.items():
        rows.sort(key=lambda r: (r["timestamp"], r["frame_id"]))
        if len(rows) >= min_track_len:
            sequences.append(LegacySequence(track_id=track_id, rows=rows, metadata=metadata))
    sequences.sort(key=lambda seq: seq.track_id)
    return sequences
