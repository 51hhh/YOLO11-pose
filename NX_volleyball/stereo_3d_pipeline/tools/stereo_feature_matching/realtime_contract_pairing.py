"""Stereo ROI pairing and bbox-prior scoring for offline probes."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .realtime_contract_types import (
    Detection,
    StereoRoiPair,
    StereoRoiPairGateConfig,
)


def _rect(det: Detection, shift_x: float = 0.0) -> Tuple[float, float, float, float]:
    half_w = det.width * 0.5
    half_h = det.height * 0.5
    return (
        det.cx - half_w + shift_x,
        det.cy - half_h,
        det.cx + half_w + shift_x,
        det.cy + half_h,
    )


def _rect_area(rect: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def rect_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = _rect_area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))
    union = _rect_area(a) + _rect_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_stereo_roi_pair(
    left: Detection,
    right: Detection,
    left_index: int,
    right_index: int,
    config: StereoRoiPairGateConfig,
) -> Tuple[Optional[StereoRoiPair], str]:
    if left.class_id != right.class_id:
        return None, "class_mismatch"
    if left.width <= 1.0 or left.height <= 1.0 or right.width <= 1.0 or right.height <= 1.0:
        return None, "invalid_box"

    disparity = left.cx - right.cx
    if disparity <= 0.0:
        return None, "nonpositive_disparity"
    if disparity > float(config.max_disparity):
        return None, "over_max_disparity"

    base_y_tol = max(1.0, config.epipolar_y_tolerance)
    adaptive_y_tol = max(base_y_tol, config.adaptive_y_ratio * max(left.height, right.height))
    dy = abs(left.cy - right.cy)
    if dy > adaptive_y_tol:
        return None, "epipolar_reject"

    width_ratio = max(left.width / right.width, right.width / left.width)
    height_ratio = max(left.height / right.height, right.height / left.height)
    max_ratio = max(1.0, config.max_size_ratio)
    if width_ratio > max_ratio or height_ratio > max_ratio:
        return None, "size_reject"

    shifted_iou = rect_iou(_rect(left), _rect(right, disparity))
    if shifted_iou < max(0.0, config.min_shifted_iou):
        return None, "low_iou"

    size_cost = abs(math.log(width_ratio)) + abs(math.log(height_ratio))
    score = dy / adaptive_y_tol + size_cost - 0.25 * right.confidence
    semantic_confidence = math.sqrt(max(0.0, left.confidence * right.confidence))
    return (
        StereoRoiPair(
            left_index=left_index,
            right_index=right_index,
            left=left,
            right=right,
            initial_disparity=disparity,
            epipolar_dy=dy,
            y_tolerance=adaptive_y_tol,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            size_ratio=max(width_ratio, height_ratio),
            shifted_bbox_iou=shifted_iou,
            score=score,
            semantic_confidence=semantic_confidence,
        ),
        "none",
    )


def collect_stereo_roi_pair_candidates(
    left_detections: Sequence[Detection],
    right_detections: Sequence[Detection],
    config: StereoRoiPairGateConfig,
    max_pairs: int,
) -> List[StereoRoiPair]:
    pairs: List[StereoRoiPair] = []
    if max_pairs <= 0:
        return pairs
    for li, left in enumerate(left_detections):
        for ri, right in enumerate(right_detections):
            pair, _ = evaluate_stereo_roi_pair(left, right, li, ri, config)
            if pair is None:
                continue
            pairs.append(pair)
    pairs.sort(key=lambda pair: pair.score)
    return pairs[:max_pairs]


def find_best_stereo_roi_pair(
    left_detections: Sequence[Detection],
    right_detections: Sequence[Detection],
    config: StereoRoiPairGateConfig,
) -> Optional[StereoRoiPair]:
    best: Optional[StereoRoiPair] = None
    for li, left in enumerate(left_detections):
        for ri, right in enumerate(right_detections):
            pair, _ = evaluate_stereo_roi_pair(left, right, li, ri, config)
            if pair is None:
                continue
            if best is None or pair.score < best.score:
                best = pair
    return best
