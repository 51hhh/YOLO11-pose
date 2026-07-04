"""Per-frame scenario evaluation for offline YOLO/IoU fallback regression."""

from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from stereo_feature_matching.realtime_contract import (
    BboxDisparityPriorConfig,
    StereoRoiPairGateConfig,
    bbox_disparity_consistency_penalty,
    evaluate_stereo_roi_pair,
    estimate_bbox_disparity_px,
    score_stereo_roi_pair_with_bbox_prior,
    select_global_stereo_roi_pairs,
)

from offline_yolo_iou_inputs import read_gray, row_detection
from offline_yolo_iou_pair_scenarios import (
    fake_left_detection,
    fake_right_detection,
    selected_pair_score,
)
from offline_yolo_iou_template_search import template_search_gray


def _fake_right_results(
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


def _fake_left_results(
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


def evaluate_regression_row(
    row: Dict[str, str],
    clip_dir: Path,
    args: Namespace,
    baseline_m: float,
    pair_gate: StereoRoiPairGateConfig,
    prior: BboxDisparityPriorConfig,
    fake_scales: List[float],
) -> Dict[str, object] | None:
    left = row_detection(row, "left")
    right = row_detection(row, "right")
    if left is None or right is None:
        return None

    left_img = read_gray(clip_dir / row["left_image"])
    right_img = read_gray(clip_dir / row["right_image"])
    true_disp = left.cx - right.cx

    normal_pair, normal_reject = evaluate_stereo_roi_pair(left, right, 0, 0, pair_gate)
    normal_score = (
        score_stereo_roi_pair_with_bbox_prior(normal_pair, baseline_m, prior, args.max_disparity)
        if normal_pair is not None
        else 0.0
    )
    normal_penalty = (
        bbox_disparity_consistency_penalty(normal_pair, baseline_m, prior, args.max_disparity)
        if normal_pair is not None
        else 0.0
    )

    fake_results = _fake_right_results(
        left,
        right,
        fake_scales,
        normal_pair,
        normal_reject,
        baseline_m,
        prior,
        pair_gate,
        args.max_disparity,
    )
    fake_left_results = _fake_left_results(
        left,
        right,
        fake_scales,
        normal_pair,
        normal_reject,
        baseline_m,
        prior,
        pair_gate,
        args.max_disparity,
    )

    right_expected_disp = estimate_bbox_disparity_px(
        left, baseline_m, prior, args.max_disparity
    )
    right_pred_x = left.cx - right_expected_disp
    right_found = template_search_gray(
        left_img,
        right_img,
        left,
        right_pred_x,
        left.cy,
        args.template_patch_radius,
        args.template_search_margin_px,
        args.template_y_tolerance_px,
        args.template_min_score,
        args.template_peak_exclusion_radius,
    )
    right_center_err = math.hypot(right_found.x - right.cx, right_found.y - right.cy)
    right_disp_err = abs((left.cx - right_found.x) - true_disp)
    right_y_err = abs(right_found.y - right.cy)
    right_score_gap = right_found.score - right_found.second_score
    right_pass = (
        right_found.valid
        and right_center_err <= args.max_center_error_px
        and right_y_err <= args.max_y_error_px
        and right_disp_err <= args.max_disparity_error_px
        and right_score_gap >= args.template_min_score_gap
    )

    left_expected_disp = estimate_bbox_disparity_px(
        right, baseline_m, prior, args.max_disparity
    )
    left_pred_x = right.cx + left_expected_disp
    left_found = template_search_gray(
        right_img,
        left_img,
        right,
        left_pred_x,
        right.cy,
        args.template_patch_radius,
        args.template_search_margin_px,
        args.template_y_tolerance_px,
        args.template_min_score,
        args.template_peak_exclusion_radius,
    )
    left_center_err = math.hypot(left_found.x - left.cx, left_found.y - left.cy)
    left_disp_err = abs((left_found.x - right.cx) - true_disp)
    left_y_err = abs(left_found.y - left.cy)
    left_score_gap = left_found.score - left_found.second_score
    left_pass = (
        left_found.valid
        and left_center_err <= args.max_center_error_px
        and left_y_err <= args.max_y_error_px
        and left_disp_err <= args.max_disparity_error_px
        and left_score_gap >= args.template_min_score_gap
    )

    return {
        "clip_frame_id": row.get("clip_frame_id", ""),
        "pipeline_frame_id": row.get("pipeline_frame_id", ""),
        "true_disparity_px": true_disp,
        "normal_pair_valid": normal_pair is not None,
        "normal_pair_reject": normal_reject,
        "normal_pair_score": normal_score,
        "normal_bbox_prior_penalty": normal_penalty,
        "fake_right_low_selected_true": fake_results[0][0],
        "fake_right_low_score": fake_results[0][1],
        "fake_right_low_reject": fake_results[0][2],
        "fake_right_high_selected_true": fake_results[1][0],
        "fake_right_high_score": fake_results[1][1],
        "fake_right_high_reject": fake_results[1][2],
        "fake_left_low_selected_true": fake_left_results[0][0],
        "fake_left_low_score": fake_left_results[0][1],
        "fake_left_low_reject": fake_left_results[0][2],
        "fake_left_high_selected_true": fake_left_results[1][0],
        "fake_left_high_score": fake_left_results[1][1],
        "fake_left_high_reject": fake_left_results[1][2],
        "right_missing_valid": right_found.valid,
        "right_missing_pass": right_pass,
        "right_missing_x": right_found.x,
        "right_missing_y": right_found.y,
        "right_missing_score": right_found.score,
        "right_missing_second_score": right_found.second_score,
        "right_missing_score_gap": right_score_gap,
        "right_missing_center_error_px": right_center_err,
        "right_missing_y_error_px": right_y_err,
        "right_missing_disparity_error_px": right_disp_err,
        "right_missing_elapsed_ms": right_found.elapsed_ms,
        "left_missing_valid": left_found.valid,
        "left_missing_pass": left_pass,
        "left_missing_x": left_found.x,
        "left_missing_y": left_found.y,
        "left_missing_score": left_found.score,
        "left_missing_second_score": left_found.second_score,
        "left_missing_score_gap": left_score_gap,
        "left_missing_center_error_px": left_center_err,
        "left_missing_y_error_px": left_y_err,
        "left_missing_disparity_error_px": left_disp_err,
        "left_missing_elapsed_ms": left_found.elapsed_ms,
    }
