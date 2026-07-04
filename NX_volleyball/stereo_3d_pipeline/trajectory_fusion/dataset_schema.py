#!/usr/bin/env python3
"""Compatibility schema exports for trajectory fusion datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    from .dataset_feature_schema import legacy_feature_names, weak_label_names
    from .dataset_method_schema import METHOD_COLUMNS, METHOD_NAMES
except ImportError:  # pragma: no cover - direct script execution
    from dataset_feature_schema import legacy_feature_names, weak_label_names
    from dataset_method_schema import METHOD_COLUMNS, METHOD_NAMES


@dataclass
class LegacySequence:
    """One track from the current TrajectoryRecorder CSV."""

    track_id: int
    rows: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.rows)
