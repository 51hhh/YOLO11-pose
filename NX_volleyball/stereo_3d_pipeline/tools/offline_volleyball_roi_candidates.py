"""Color-fragment ROI candidate helpers for offline volleyball probes."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import Detection as RuntimeDetection

from offline_volleyball_probe_models import BallCandidate, BallROI, RoughBallROI
from offline_volleyball_roi_segmentation import segment_ball


def _confidence_from_score(score: float) -> float:
    return float(np.clip(score / 12.0, 0.05, 1.0))


def _candidate_to_runtime_detection(candidate: BallCandidate) -> RuntimeDetection:
    x, y, w, h = candidate.bbox
    return RuntimeDetection(
        cx=float(candidate.center[0]),
        cy=float(candidate.center[1]),
        width=float(w),
        height=float(h),
        confidence=_confidence_from_score(candidate.score),
        class_id=0,
    )


def _ball_candidates(image: np.ndarray, side: str) -> List[BallCandidate]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    luma = lab[:, :, 0]

    yellow = (hue >= 18) & (hue <= 45) & (sat >= 90) & (val >= 45)
    blue = (hue >= 88) & (hue <= 135) & (sat >= 45) & (val >= 25)
    white = (sat <= 60) & (val >= 95) & (luma >= 85)
    color_mask = ((yellow | blue | white).astype(np.uint8)) * 255
    color_mask[: int(h * 0.45), :] = 0
    if side == "left":
        color_mask[:, : int(w * 0.38)] = 0
    else:
        color_mask[:, int(w * 0.80) :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, 8)
    candidates: List[BallCandidate] = []
    for idx in range(1, num):
        x, y, bw, bh, area = (int(v) for v in stats[idx])
        cx, cy = (float(v) for v in centroids[idx])
        if area < 60 or area > 16000:
            continue
        if bw < 8 or bh < 8 or bw > 260 or bh > 190:
            continue
        if cy < h * 0.54 or cy > h * 0.86:
            continue
        if side == "left" and cx < w * 0.52:
            continue
        if side == "right" and (cx < w * 0.18 or cx > w * 0.72):
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 0.25 or aspect > 3.2:
            continue
        extent = area / max(1.0, float(bw * bh))
        size_score = math.log1p(area)
        aspect_penalty = abs(math.log(max(0.1, aspect)))
        score = size_score + 2.2 * extent - 0.85 * aspect_penalty
        comp = np.zeros((h, w), dtype=np.uint8)
        comp[labels == idx] = 255
        candidates.append(BallCandidate(idx, (x, y, bw, bh), (cx, cy), area, score, comp))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _roi_from_group(
    image_shape: Tuple[int, int],
    seed: BallCandidate,
    candidates: Sequence[BallCandidate],
) -> BallROI:
    h, w = image_shape
    sx, sy = seed.center
    selected: List[BallCandidate] = []
    for cand in candidates:
        cx, cy = cand.center
        if math.hypot(cx - sx, cy - sy) <= 120.0:
            selected.append(cand)
    if not selected:
        selected = [seed]

    mask = np.zeros((h, w), dtype=np.uint8)
    for cand in selected:
        mask = cv2.bitwise_or(mask, cand.mask)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        x, y, bw, bh = seed.bbox
        cx, cy = seed.center
        radius = max(bw, bh) * 0.85
    else:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        radius = max(x2 - x1 + 1, y2 - y1 + 1) * 0.58

    radius = float(np.clip(radius, 32.0, 76.0))
    margin = int(max(14, round(radius * 0.22)))
    x1 = max(0, int(round(cx - radius)) - margin)
    y1 = max(0, int(round(cy - radius)) - margin)
    x2 = min(w - 1, int(round(cx + radius)) + margin)
    y2 = min(h - 1, int(round(cy + radius)) + margin)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (int(round(cx)), int(round(cy))), int(round(radius)), 255, -1)
    roi_mask[:y1, :] = 0
    roi_mask[y2 + 1 :, :] = 0
    roi_mask[:, :x1] = 0
    roi_mask[:, x2 + 1 :] = 0
    return BallROI((x1, y1, x2 - x1 + 1, y2 - y1 + 1), (cx, cy), radius, roi_mask)


def _rough_rois_from_candidates(
    image_shape: Tuple[int, int],
    candidates: Sequence[BallCandidate],
    max_candidates: int,
) -> List[RoughBallROI]:
    """Build de-duplicated ball-sized ROIs from color fragments."""

    rough: List[RoughBallROI] = []
    seen: set[Tuple[int, int, int]] = set()
    for rank, candidate in enumerate(candidates[:max_candidates]):
        roi = _roi_from_group(image_shape, candidate, candidates)
        key = (
            int(round(roi.center[0] / 6.0)),
            int(round(roi.center[1] / 6.0)),
            int(round(roi.radius / 4.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        rough.append(RoughBallROI(roi=roi, seed=candidate, rank=rank))
    return rough


def _rough_pair_score(
    pair,
    lrough: RoughBallROI,
    rrough: RoughBallROI,
    baseline_m: float,
    ball_diameter_m: float,
) -> float:
    expected_radius = 0.5 * ball_diameter_m * pair.initial_disparity / max(1e-6, baseline_m)
    measured_radius = 0.5 * (lrough.roi.radius + rrough.roi.radius)
    radius_ratio = max(lrough.roi.radius, rrough.roi.radius) / max(1.0, min(lrough.roi.radius, rrough.roi.radius))
    rank_penalty = 0.015 * float(lrough.rank + rrough.rank)
    radius_penalty = 0.025 * abs(measured_radius - expected_radius)
    ratio_penalty = 0.08 * abs(math.log(max(1e-3, radius_ratio)))
    confidence_bonus = 0.08 * (_confidence_from_score(lrough.seed.score) + _confidence_from_score(rrough.seed.score))
    return float(pair.score + rank_penalty + radius_penalty + ratio_penalty - confidence_bonus)
