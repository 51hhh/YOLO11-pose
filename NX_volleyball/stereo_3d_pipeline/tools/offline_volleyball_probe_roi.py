"""ROI detection and calibration helpers for offline volleyball probes."""

from __future__ import annotations

import argparse
from typing import List, Tuple

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import (
    StereoRoiPairGateConfig,
    evaluate_stereo_roi_pair,
)

from offline_volleyball_depth_math import depth_from_disparity
from offline_volleyball_calibration import load_calibration, rectify_pair
from offline_volleyball_probe_models import (
    BallROI,
    MatchResult,
    RoughBallROI,
    ValidationThresholds,
)
from offline_volleyball_roi_candidates import (
    _ball_candidates,
    _candidate_to_runtime_detection,
    _confidence_from_score,
    _rough_pair_score,
    _rough_rois_from_candidates,
    segment_ball,
)
from offline_volleyball_roi_hough import (
    _circle_roi,
    _refine_roi_pair_with_hough,
    _roi_to_runtime_detection,
)


def _parse_circle(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("circle must be x,y,r")
    try:
        x, y, r = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("circle must be numeric x,y,r") from exc
    if r <= 0.0:
        raise argparse.ArgumentTypeError("circle radius must be positive")
    return x, y, r


def detect_ball_rois(
    left: np.ndarray,
    right: np.ndarray,
    focal_px: float,
    baseline_m: float,
    mask_margin: float = 12.0,
    ball_diameter_m: float = 0.210,
    pair_gate_config: StereoRoiPairGateConfig | None = None,
    min_depth_m: float = 0.8,
    max_depth_m: float = 20.0,
) -> Tuple[BallROI, BallROI]:
    pair_gate = pair_gate_config or StereoRoiPairGateConfig()
    left_candidates = _ball_candidates(left, "left")
    right_candidates = _ball_candidates(right, "right")
    if not left_candidates or not right_candidates:
        raise RuntimeError(
            f"candidate detection failed: left={len(left_candidates)} right={len(right_candidates)}"
        )

    left_rough = _rough_rois_from_candidates(left.shape[:2], left_candidates, 16)
    right_rough = _rough_rois_from_candidates(right.shape[:2], right_candidates, 24)

    best_pair: Tuple[float, RoughBallROI, RoughBallROI] | None = None
    for lr in left_rough:
        left_det = _roi_to_runtime_detection(lr.roi, _confidence_from_score(lr.seed.score))
        for rr in right_rough:
            right_det = _roi_to_runtime_detection(rr.roi, _confidence_from_score(rr.seed.score))
            pair, _ = evaluate_stereo_roi_pair(left_det, right_det, lr.rank, rr.rank, pair_gate)
            if pair is None:
                continue
            depth = depth_from_disparity(pair.initial_disparity, focal_px, baseline_m)
            if depth < min_depth_m or depth > max_depth_m:
                continue
            score = _rough_pair_score(pair, lr, rr, baseline_m, ball_diameter_m)
            if best_pair is None or score < best_pair[0]:
                best_pair = (score, lr, rr)

    if best_pair is None:
        # Keep the old one-sided behavior as a fallback for debugging, but this
        # should be treated as low trust.
        lroi = segment_ball(left, "left")
        rroi = segment_ball(right, "right")
        return _refine_roi_pair_with_hough(
            left,
            right,
            lroi,
            rroi,
            focal_px,
            baseline_m,
            mask_margin,
            ball_diameter_m,
            pair_gate,
            min_depth_m,
            max_depth_m,
        )

    _, left_seed, right_seed = best_pair
    lroi = left_seed.roi
    rroi = right_seed.roi
    return _refine_roi_pair_with_hough(
        left,
        right,
        lroi,
        rroi,
        focal_px,
        baseline_m,
        mask_margin,
        ball_diameter_m,
        pair_gate,
        min_depth_m,
        max_depth_m,
    )


def draw_roi_debug(left: np.ndarray, right: np.ndarray, lroi: BallROI, rroi: BallROI) -> np.ndarray:
    out_l = left.copy()
    out_r = right.copy()
    for out, roi, label in [(out_l, lroi, "left"), (out_r, rroi, "right")]:
        x, y, w, h = roi.bbox
        overlay = out.copy()
        overlay[roi.mask > 0] = (0.65 * overlay[roi.mask > 0] + np.array([255, 255, 0]) * 0.35).astype(np.uint8)
        out[:] = overlay
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(round(roi.center[0])), int(round(roi.center[1]))),
                   int(round(roi.radius)), (0, 255, 255), 2)
        cv2.putText(out, f"{label}:{roi.source}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
    return np.hstack([out_l, out_r])
