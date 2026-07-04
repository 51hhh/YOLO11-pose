"""Bbox-prior scoring helpers for the realtime contract mirror."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

from .realtime_contract_pairing import evaluate_stereo_roi_pair
from .realtime_contract_types import (
    BboxDisparityPriorConfig,
    Detection,
    StereoRoiPair,
    StereoRoiPairGateConfig,
)


def estimate_bbox_disparity_px(
    det: Detection,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> float:
    if (
        det.width <= 1.0
        or baseline_m <= 0.0
        or prior.object_diameter_m <= 0.01
        or max_disparity <= 0
    ):
        return -1.0
    disp = baseline_m * det.width * prior.bbox_scale / prior.object_diameter_m
    return min(float(max_disparity), max(1.0, float(disp)))


def bbox_disparity_consistency_penalty(
    pair: StereoRoiPair,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> float:
    if pair.initial_disparity <= 0.0 or baseline_m <= 0.0:
        return 0.0

    left_expected = estimate_bbox_disparity_px(pair.left, baseline_m, prior, max_disparity)
    right_expected = estimate_bbox_disparity_px(pair.right, baseline_m, prior, max_disparity)
    if left_expected > 0.0 and right_expected > 0.0:
        expected = 0.5 * (left_expected + right_expected)
    elif left_expected > 0.0:
        expected = left_expected
    elif right_expected > 0.0:
        expected = right_expected
    else:
        return 0.0

    tolerance = max(
        max(5.0, prior.consistency_min_px),
        expected * max(0.05, prior.consistency_ratio),
    )
    excess = abs(pair.initial_disparity - expected) - tolerance
    if excess <= 0.0:
        return 0.0
    return max(0.0, prior.penalty_scale) * excess / max(1.0, tolerance)


def score_stereo_roi_pair_with_bbox_prior(
    pair: StereoRoiPair,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> float:
    return pair.score + bbox_disparity_consistency_penalty(
        pair,
        baseline_m,
        prior,
        max_disparity,
    )


def collect_scored_stereo_roi_pair_candidates(
    left_detections: Sequence[Detection],
    right_detections: Sequence[Detection],
    config: StereoRoiPairGateConfig,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> List[StereoRoiPair]:
    pairs: List[StereoRoiPair] = []
    for li, left in enumerate(left_detections):
        for ri, right in enumerate(right_detections):
            pair, _ = evaluate_stereo_roi_pair(left, right, li, ri, config)
            if pair is None:
                continue
            pairs.append(
                replace(
                    pair,
                    score=score_stereo_roi_pair_with_bbox_prior(
                        pair,
                        baseline_m,
                        prior,
                        max_disparity,
                    ),
                )
            )
    pairs.sort(key=lambda pair: pair.score)
    return pairs


def select_global_stereo_roi_pairs(
    left_detections: Sequence[Detection],
    right_detections: Sequence[Detection],
    config: StereoRoiPairGateConfig,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> List[StereoRoiPair]:
    selected: List[StereoRoiPair] = []
    left_used: set[int] = set()
    right_used: set[int] = set()
    for pair in collect_scored_stereo_roi_pair_candidates(
        left_detections,
        right_detections,
        config,
        baseline_m,
        prior,
        max_disparity,
    ):
        if pair.left_index in left_used or pair.right_index in right_used:
            continue
        selected.append(pair)
        left_used.add(pair.left_index)
        right_used.add(pair.right_index)
    return selected
