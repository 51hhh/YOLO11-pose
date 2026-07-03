#!/usr/bin/env python3
"""Offline regression tests for YOLO ROI pairing and single-side fallback.

The tool reads a baseline clip with frames.csv, uses the recorded YOLO boxes as
ground truth proxies, then simulates degraded cases:

- normal left/right YOLO pairing;
- fake same-y/same-size right detections at wrong disparities;
- fake same-y/same-size left detections at wrong disparities;
- right missing: left ROI searches the right epipolar band;
- left missing: right ROI searches the left epipolar band.

It is intended to catch ROI/IoU baseline search drift before enabling a method
in the realtime pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from stereo_feature_matching.realtime_contract import (
    BboxDisparityPriorConfig,
    Detection,
    StereoRoiPairGateConfig,
    bbox_disparity_consistency_penalty,
    evaluate_stereo_roi_pair,
    estimate_bbox_disparity_px,
    select_global_stereo_roi_pairs,
    score_stereo_roi_pair_with_bbox_prior,
)

from offline_yolo_iou_report import build_summary, write_regression_outputs
from offline_yolo_iou_template_search import template_search_gray


def _load_baseline_from_calib(path: Path) -> float:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    try:
        node = fs.getNode("baseline")
        if node.empty():
            raise KeyError("missing calibration key: baseline")
        return float(node.real()) / 1000.0
    finally:
        fs.release()


def _read_rows(path: Path, max_frames: int) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_frames > 0:
        rows = rows[:max_frames]
    return rows


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _det(row: Dict[str, str], side: str) -> Detection | None:
    try:
        count = int(float(row[f"{side}_count"]))
    except (KeyError, ValueError):
        count = 1
    if count <= 0:
        return None
    try:
        return Detection(
            cx=float(row[f"{side}_cx"]),
            cy=float(row[f"{side}_cy"]),
            width=float(row[f"{side}_w"]),
            height=float(row[f"{side}_h"]),
            confidence=float(row.get(f"{side}_conf", 1.0)),
            class_id=int(float(row.get(f"{side}_class_id", 0))),
        )
    except (KeyError, ValueError):
        return None


def _fake_right_detection(left: Detection, true_right: Detection, scale: float) -> Detection:
    true_disp = max(1.0, left.cx - true_right.cx)
    return Detection(
        cx=left.cx - true_disp * scale,
        cy=true_right.cy,
        width=true_right.width,
        height=true_right.height,
        confidence=true_right.confidence,
        class_id=true_right.class_id,
    )


def _fake_left_detection(true_left: Detection, right: Detection, scale: float) -> Detection:
    true_disp = max(1.0, true_left.cx - right.cx)
    return Detection(
        cx=right.cx + true_disp * scale,
        cy=true_left.cy,
        width=true_left.width,
        height=true_left.height,
        confidence=true_left.confidence,
        class_id=true_left.class_id,
    )


def _selected_pair_score(selected, left_index: int, right_index: int) -> float:
    for pair in selected:
        if pair.left_index == left_index and pair.right_index == right_index:
            return pair.score
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True, help="baseline clip directory with frames.csv")
    parser.add_argument("--calib", type=Path, default=Path("NX_volleyball/calibration/stereo_calib.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--object-diameter-m", type=float, default=0.200)
    parser.add_argument("--bbox-scale", type=float, default=0.95)
    parser.add_argument("--max-disparity", type=int, default=2048)
    parser.add_argument("--pair-y-tolerance-px", type=float, default=12.0)
    parser.add_argument("--pair-max-size-ratio", type=float, default=2.0)
    parser.add_argument("--pair-min-shifted-iou", type=float, default=0.05)
    parser.add_argument("--bbox-consistency-ratio", type=float, default=0.30)
    parser.add_argument("--bbox-consistency-min-px", type=float, default=45.0)
    parser.add_argument("--bbox-penalty-scale", type=float, default=0.75)
    parser.add_argument("--fake-disparity-scales", default="0.55,1.45")
    parser.add_argument("--template-patch-radius", type=int, default=9)
    parser.add_argument("--template-search-margin-px", type=float, default=72.0)
    parser.add_argument("--template-y-tolerance-px", type=float, default=24.0)
    parser.add_argument("--template-min-score", type=float, default=0.20)
    parser.add_argument("--template-min-score-gap", type=float, default=0.010)
    parser.add_argument("--template-peak-exclusion-radius", type=int, default=12)
    parser.add_argument("--max-center-error-px", type=float, default=18.0)
    parser.add_argument("--max-y-error-px", type=float, default=8.0)
    parser.add_argument("--max-disparity-error-px", type=float, default=18.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--min-pass-rate", type=float, default=0.99)
    args = parser.parse_args()

    baseline_m = _load_baseline_from_calib(args.calib)
    pair_gate = StereoRoiPairGateConfig(
        max_disparity=args.max_disparity,
        epipolar_y_tolerance=args.pair_y_tolerance_px,
        max_size_ratio=args.pair_max_size_ratio,
        min_shifted_iou=args.pair_min_shifted_iou,
    )
    prior = BboxDisparityPriorConfig(
        object_diameter_m=args.object_diameter_m,
        bbox_scale=args.bbox_scale,
        consistency_ratio=args.bbox_consistency_ratio,
        consistency_min_px=args.bbox_consistency_min_px,
        penalty_scale=args.bbox_penalty_scale,
    )
    fake_scales = sorted(float(v.strip()) for v in args.fake_disparity_scales.split(",") if v.strip())
    if len(fake_scales) != 2:
        raise ValueError("--fake-disparity-scales must contain two comma-separated values")

    rows_in = _read_rows(args.clip / "frames.csv", args.max_frames)
    args.out.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, object]] = []

    for row in rows_in:
        left = _det(row, "left")
        right = _det(row, "right")
        if left is None or right is None:
            continue
        left_img = _read_gray(args.clip / row["left_image"])
        right_img = _read_gray(args.clip / row["right_image"])
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

        fake_results: List[Tuple[bool, float, str]] = []
        for scale in fake_scales:
            fake = _fake_right_detection(left, right, scale)
            fake_pair, fake_reject = evaluate_stereo_roi_pair(left, fake, 0, 1, pair_gate)
            selected = select_global_stereo_roi_pairs(
                [left],
                [fake, right],
                pair_gate,
                baseline_m,
                prior,
                args.max_disparity,
            )
            selected_true = any(pair.left_index == 0 and pair.right_index == 1 for pair in selected)
            if normal_pair is None:
                fake_results.append((False, 0.0, normal_reject))
                continue
            if fake_pair is None:
                fake_results.append((selected_true, _selected_pair_score(selected, 0, 1), fake_reject))
                continue
            fake_score = score_stereo_roi_pair_with_bbox_prior(
                fake_pair, baseline_m, prior, args.max_disparity
            )
            fake_results.append((selected_true, fake_score, "none"))

        fake_left_results: List[Tuple[bool, float, str]] = []
        for scale in fake_scales:
            fake = _fake_left_detection(left, right, scale)
            fake_pair, fake_reject = evaluate_stereo_roi_pair(fake, right, 1, 0, pair_gate)
            selected = select_global_stereo_roi_pairs(
                [fake, left],
                [right],
                pair_gate,
                baseline_m,
                prior,
                args.max_disparity,
            )
            selected_true = any(pair.left_index == 1 and pair.right_index == 0 for pair in selected)
            if normal_pair is None:
                fake_left_results.append((False, 0.0, normal_reject))
                continue
            if fake_pair is None:
                fake_left_results.append((selected_true, _selected_pair_score(selected, 1, 0), fake_reject))
                continue
            fake_score = score_stereo_roi_pair_with_bbox_prior(
                fake_pair, baseline_m, prior, args.max_disparity
            )
            fake_left_results.append((selected_true, fake_score, "none"))

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

        out_rows.append(
            {
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
        )

    summary = {
        "clip": str(args.clip),
        "calibration": str(args.calib),
        "baseline_m": baseline_m,
        "assignment": "global_score_one_to_one",
        "pair_gate": {
            "max_disparity": pair_gate.max_disparity,
            "epipolar_y_tolerance": pair_gate.epipolar_y_tolerance,
            "max_size_ratio": pair_gate.max_size_ratio,
            "min_shifted_iou": pair_gate.min_shifted_iou,
        },
        "bbox_prior": {
            "object_diameter_m": prior.object_diameter_m,
            "bbox_scale": prior.bbox_scale,
            "consistency_ratio": prior.consistency_ratio,
            "consistency_min_px": prior.consistency_min_px,
            "penalty_scale": prior.penalty_scale,
        },
        "thresholds": {
            "max_center_error_px": args.max_center_error_px,
            "max_y_error_px": args.max_y_error_px,
            "max_disparity_error_px": args.max_disparity_error_px,
            "template_min_score": args.template_min_score,
            "template_min_score_gap": args.template_min_score_gap,
            "template_peak_exclusion_radius": args.template_peak_exclusion_radius,
        },
        "metrics": build_summary(out_rows),
    }
    write_regression_outputs(args.out, out_rows, summary, args.clip)
    print(json.dumps(summary, indent=2))
    if args.fail_on_regression:
        metrics = summary["metrics"]
        required_rates = [
            "normal_pair_pass_rate",
            "fake_right_low_selected_true_rate",
            "fake_right_high_selected_true_rate",
            "fake_left_low_selected_true_rate",
            "fake_left_high_selected_true_rate",
            "right_missing_pass_rate",
            "left_missing_pass_rate",
        ]
        if any(float(metrics[name]) < args.min_pass_rate for name in required_rates):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
