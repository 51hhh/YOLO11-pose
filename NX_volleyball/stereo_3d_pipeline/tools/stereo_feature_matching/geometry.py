"""Descriptor matching and stereo-specific match filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .common import FeatureSet, RawMatch


@dataclass
class MatchFilterConfig:
    initial_disparity: float
    max_y_error_px: float = 2.0
    max_disp_delta_px: float = 24.0
    final_disp_gate_px: float = 2.0
    min_disparity_px: float = 0.5
    max_disparity_px: float = 512.0
    min_score: float = 0.0


@dataclass
class FilteredMatch:
    query_idx: int
    train_idx: int
    score: float
    left_xy: Tuple[float, float]
    right_xy: Tuple[float, float]
    disparity: float
    y_error: float


def _l2_normalize(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    denom = np.linalg.norm(desc, axis=1, keepdims=True)
    return desc / np.maximum(denom, 1e-8)


def match_descriptors(
    left: FeatureSet,
    right: FeatureSet,
    *,
    ratio: float = 0.80,
    mutual: bool = True,
    top_k: int = 64,
) -> List[RawMatch]:
    """Cosine nearest-neighbor matching for small ROI descriptor sets."""

    if left.descriptors is None or right.descriptors is None:
        return []
    if left.count == 0 or right.count == 0:
        return []

    dl = _l2_normalize(left.descriptors)
    dr = _l2_normalize(right.descriptors)
    sim = dl @ dr.T
    if sim.size == 0:
        return []

    order = np.argsort(-sim, axis=1)
    best = order[:, 0]
    second = order[:, 1] if right.count > 1 else order[:, 0]
    best_score = sim[np.arange(left.count), best]
    second_score = sim[np.arange(left.count), second]

    if mutual:
        right_best = np.argmax(sim, axis=0)
    else:
        right_best = np.full(right.count, -1, dtype=np.int64)

    matches: List[RawMatch] = []
    # Ratio in cosine space: require the best margin to be meaningful.
    min_margin = max(0.0, 1.0 - ratio) * 0.25
    for qi, ti in enumerate(best.tolist()):
        if mutual and int(right_best[ti]) != qi:
            continue
        if right.count > 1 and best_score[qi] - second_score[qi] < min_margin:
            continue
        matches.append(RawMatch(qi, int(ti), float(best_score[qi])))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[: max(0, int(top_k))]


def filter_feature_set_by_mask(
    features: FeatureSet,
    mask: np.ndarray,
    map_to_global,
) -> FeatureSet:
    """Keep features whose mapped global coordinates fall inside a mask."""

    if features.count == 0:
        return features
    h, w = mask.shape[:2]
    keep = []
    for idx, (x, y) in enumerate(features.keypoints):
        gx, gy = map_to_global(float(x), float(y))
        ix = int(round(gx))
        iy = int(round(gy))
        if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
            keep.append(idx)
    return features.subset(np.asarray(keep, dtype=np.int64))


def filter_stereo_matches(
    left: FeatureSet,
    right: FeatureSet,
    matches: Sequence[RawMatch],
    left_to_global,
    right_to_global,
    cfg: MatchFilterConfig,
) -> List[FilteredMatch]:
    """Apply rectified stereo gates after keypoints are mapped to full image coords."""

    out: List[FilteredMatch] = []
    for m in matches:
        if m.query_idx < 0 or m.train_idx < 0:
            continue
        if m.query_idx >= left.count or m.train_idx >= right.count:
            continue
        lx, ly = left_to_global(*left.keypoints[m.query_idx])
        rx, ry = right_to_global(*right.keypoints[m.train_idx])
        disparity = float(lx - rx)
        y_error = float(abs(ly - ry))
        if m.score < cfg.min_score:
            continue
        if y_error > cfg.max_y_error_px:
            continue
        if disparity <= cfg.min_disparity_px or disparity > cfg.max_disparity_px:
            continue
        if abs(disparity - cfg.initial_disparity) > cfg.max_disp_delta_px:
            continue
        out.append(
            FilteredMatch(
                m.query_idx,
                m.train_idx,
                float(m.score),
                (float(lx), float(ly)),
                (float(rx), float(ry)),
                disparity,
                y_error,
            )
        )

    if not out:
        return []

    disparities = np.asarray([m.disparity for m in out], dtype=np.float32)
    median = float(np.median(disparities))
    mad = float(np.median(np.abs(disparities - median)))
    gate = max(0.75, 2.5 * mad)
    inliers = [m for m in out if abs(m.disparity - median) <= gate]
    if cfg.final_disp_gate_px > 0.0 and len(inliers) >= 3:
        disparities = np.asarray([m.disparity for m in inliers], dtype=np.float32)
        median = float(np.median(disparities))
        inliers = [m for m in inliers if abs(m.disparity - median) <= cfg.final_disp_gate_px]
    inliers.sort(key=lambda m: m.score, reverse=True)
    return inliers
