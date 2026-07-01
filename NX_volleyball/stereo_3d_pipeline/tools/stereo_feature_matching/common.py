"""Common data structures for local feature probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FeatureSet:
    """Sparse keypoints and optional descriptors in local image coordinates."""

    keypoints: np.ndarray
    descriptors: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    image_size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 2)
        if self.descriptors is not None:
            self.descriptors = np.asarray(self.descriptors, dtype=np.float32)
            if self.descriptors.ndim == 1:
                self.descriptors = self.descriptors.reshape(1, -1)
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)

    def subset(self, indices: np.ndarray) -> "FeatureSet":
        indices = np.asarray(indices, dtype=np.int64)
        descriptors = None if self.descriptors is None else self.descriptors[indices]
        scores = None if self.scores is None else self.scores[indices]
        return FeatureSet(self.keypoints[indices], descriptors, scores, self.image_size)

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass
class RawMatch:
    query_idx: int
    train_idx: int
    score: float


@dataclass
class TimedResult:
    left: FeatureSet
    right: FeatureSet
    matches: List[RawMatch] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
