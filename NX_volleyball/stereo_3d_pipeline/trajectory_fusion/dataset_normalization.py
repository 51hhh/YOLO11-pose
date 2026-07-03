#!/usr/bin/env python3
"""Feature normalization helpers for trajectory fusion datasets."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def compute_feature_normalizer(features: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Compute a fixed feature normalizer for training/deployment."""

    if not features:
        return [], []
    cols = len(features[0])
    means = [0.0] * cols
    for row in features:
        for i, value in enumerate(row):
            means[i] += value
    means = [value / len(features) for value in means]

    stds = [1e-6] * cols
    for row in features:
        for i, value in enumerate(row):
            diff = value - means[i]
            stds[i] += diff * diff
    stds = [(value / len(features)) ** 0.5 for value in stds]
    stds = [max(value, 1e-6) for value in stds]
    return means, stds


def apply_feature_normalizer(
    features: Sequence[Sequence[float]],
    means: Sequence[float],
    stds: Sequence[float],
) -> List[List[float]]:
    """Apply a fixed feature normalizer."""

    if not features:
        return []
    if len(means) != len(features[0]) or len(stds) != len(features[0]):
        raise ValueError("feature normalizer dimension mismatch")
    return [[(value - means[i]) / max(stds[i], 1e-6) for i, value in enumerate(row)] for row in features]


def normalize_features(features: Sequence[Sequence[float]]) -> List[List[float]]:
    """Backward-compatible helper using a normalizer fit on the input."""

    means, stds = compute_feature_normalizer(features)
    return apply_feature_normalizer(features, means, stds)
