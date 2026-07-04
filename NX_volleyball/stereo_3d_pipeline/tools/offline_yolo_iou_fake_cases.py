"""Fake detection scenario evaluation for YOLO/IoU fallback regression."""

from __future__ import annotations

from typing import List, Tuple

from stereo_feature_matching.realtime_contract import (
    BboxDisparityPriorConfig,
    StereoRoiPairGateConfig,
    evaluate_stereo_roi_pair,
    score_stereo_roi_pair_with_bbox_prior,
    select_global_stereo_roi_pairs,
)

from offline_yolo_iou_pair_scenarios import (
    fake_left_detection,
    fake_right_detection,
    selected_pair_score,
)


def evaluate_fake_right_results(
    left,
    right,
    fake_scales: List[float],
    normal_pair,
    normal_reject: str,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    pair_gate: StereoRoiPairGateConfig,
    max_disparity: int,
) -> List[Tuple[bool, float, str]]:
    results: List[Tuple[bool, float, str]] = []
    for scale in fake_scales:
        fake = fake_right_detection(left, right, scale)
        fake_pair, fake_reject = evaluate_stereo_roi_pair(left, fake, 0, 1, pair_gate)
        selected = select_global_stereo_roi_pairs(
            [left],
            [fake, right],
            pair_gate,
            baseline_m,
            prior,
            max_disparity,
        )
        selected_true = any(pair.left_index == 0 and pair.right_index == 1 for pair in selected)
        if normal_pair is None:
            results.append((False, 0.0, normal_reject))
            continue
        if fake_pair is None:
            results.append((selected_true, selected_pair_score(selected, 0, 1), fake_reject))
            continue
        fake_score = score_stereo_roi_pair_with_bbox_prior(
            fake_pair, baseline_m, prior, max_disparity
        )
        results.append((selected_true, fake_score, "none"))
    return results


def evaluate_fake_left_results(
    left,
    right,
    fake_scales: List[float],
    normal_pair,
    normal_reject: str,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    pair_gate: StereoRoiPairGateConfig,
    max_disparity: int,
) -> List[Tuple[bool, float, str]]:
    results: List[Tuple[bool, float, str]] = []
    for scale in fake_scales:
        fake = fake_left_detection(left, right, scale)
        fake_pair, fake_reject = evaluate_stereo_roi_pair(fake, right, 1, 0, pair_gate)
        selected = select_global_stereo_roi_pairs(
            [fake, left],
            [right],
            pair_gate,
            baseline_m,
            prior,
            max_disparity,
        )
        selected_true = any(pair.left_index == 1 and pair.right_index == 0 for pair in selected)
        if normal_pair is None:
            results.append((False, 0.0, normal_reject))
            continue
        if fake_pair is None:
            results.append((selected_true, selected_pair_score(selected, 1, 0), fake_reject))
            continue
        fake_score = score_stereo_roi_pair_with_bbox_prior(
            fake_pair, baseline_m, prior, max_disparity
        )
        results.append((selected_true, fake_score, "none"))
    return results
