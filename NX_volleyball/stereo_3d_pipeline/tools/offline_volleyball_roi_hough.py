"""Hough circle ROI refinement helpers for offline volleyball probes."""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import (
    Detection as RuntimeDetection,
    StereoRoiPairGateConfig,
    evaluate_stereo_roi_pair,
)

from offline_volleyball_depth_math import depth_from_disparity
from offline_volleyball_probe_models import BallROI


def _roi_to_runtime_detection(roi: BallROI, confidence: float = 1.0) -> RuntimeDetection:
    x, y, w, h = roi.bbox
    return RuntimeDetection(
        cx=float(roi.center[0]),
        cy=float(roi.center[1]),
        width=float(w),
        height=float(h),
        confidence=float(np.clip(confidence, 0.05, 1.0)),
        class_id=0,
    )


def _circle_roi(
    image_shape: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    source: str,
    mask_margin: float = 8.0,
) -> BallROI:
    h, w = image_shape
    cx, cy = center
    radius = float(max(4.0, radius))
    mask_radius = float(max(4.0, radius - mask_margin))

    x1 = max(0, int(math.floor(cx - radius)))
    y1 = max(0, int(math.floor(cy - radius)))
    x2 = min(w - 1, int(math.ceil(cx + radius)))
    y2 = min(h - 1, int(math.ceil(cy + radius)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(mask_radius)), 255, -1)
    return BallROI((x1, y1, x2 - x1 + 1, y2 - y1 + 1), (float(cx), float(cy)), radius, mask, source)


def _refine_roi_with_hough(
    image: np.ndarray,
    rough: BallROI,
    min_radius: int = 28,
    max_radius: int = 72,
) -> BallROI:
    """Refine one diagnostic ROI to the visible circular ball boundary."""

    options = _hough_roi_options(image, rough, min_radius, max_radius)
    if not options:
        return rough
    return options[0][0]


def _hough_roi_options(
    image: np.ndarray,
    rough: BallROI,
    min_radius: int = 28,
    max_radius: int = 72,
    mask_margin: float = 12.0,
) -> List[Tuple[BallROI, float]]:
    """Return plausible Hough circle ROIs with lower per-image score first."""

    x, y, w, h = rough.bbox
    pad = int(max(35, round(rough.radius * 0.75)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=80,
        param2=14,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    rcx, rcy = rough.center
    options: List[Tuple[BallROI, float]] = []
    for cx, cy, radius in circles[0]:
        gx = float(cx + x1)
        gy = float(cy + y1)
        gr = float(radius)
        if gy < image.shape[0] * 0.45 or gy > image.shape[0] * 0.90:
            continue
        center_dist = math.hypot(gx - rcx, gy - rcy)
        if center_dist > max(90.0, rough.radius * 1.50):
            continue
        radius_penalty = abs(gr - np.clip(rough.radius, min_radius, max_radius)) / 12.0
        score = center_dist + radius_penalty
        options.append((_circle_roi(image.shape[:2], (gx, gy), gr, "auto+hough", mask_margin), float(score)))

    options.sort(key=lambda item: item[1])
    return options


def _refine_roi_pair_with_hough(
    left: np.ndarray,
    right: np.ndarray,
    lrough: BallROI,
    rrough: BallROI,
    focal_px: float,
    baseline_m: float,
    mask_margin: float = 12.0,
    ball_diameter_m: float = 0.210,
    pair_gate_config: StereoRoiPairGateConfig | None = None,
    min_depth_m: float = 0.8,
    max_depth_m: float = 20.0,
) -> Tuple[BallROI, BallROI]:
    pair_gate = pair_gate_config or StereoRoiPairGateConfig()
    left_options = _hough_roi_options(left, lrough, mask_margin=mask_margin)
    right_options = _hough_roi_options(right, rrough, mask_margin=mask_margin)
    if not left_options or not right_options:
        return _refine_roi_with_hough(left, lrough), _refine_roi_with_hough(right, rrough)

    rough_disp = float(lrough.center[0] - rrough.center[0])
    best: Tuple[float, BallROI, BallROI] | None = None
    for lroi, lscore in left_options[:12]:
        for rroi, rscore in right_options[:12]:
            pair, _ = evaluate_stereo_roi_pair(
                _roi_to_runtime_detection(lroi),
                _roi_to_runtime_detection(rroi),
                0,
                0,
                pair_gate,
            )
            if pair is None:
                continue
            depth = depth_from_disparity(pair.initial_disparity, focal_px, baseline_m)
            if depth < min_depth_m or depth > max_depth_m:
                continue
            radius_ratio = max(lroi.radius, rroi.radius) / max(1.0, min(lroi.radius, rroi.radius))
            expected_radius = 0.5 * ball_diameter_m * pair.initial_disparity / baseline_m
            measured_radius = 0.5 * (lroi.radius + rroi.radius)
            score = (
                lscore
                + rscore
                + 2.0 * pair.epipolar_dy
                + 18.0 * abs(math.log(max(1e-3, radius_ratio)))
                + 2.6 * abs(measured_radius - expected_radius)
                + max(0.0, abs(pair.initial_disparity - rough_disp) - 80.0) / 4.0
            )
            if best is None or score < best[0]:
                best = (score, lroi, rroi)

    if best is None:
        return left_options[0][0], right_options[0][0]
    return best[1], best[2]
