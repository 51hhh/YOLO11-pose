"""ROI detection and calibration helpers for offline volleyball probes."""

from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import (
    Detection as RuntimeDetection,
    StereoRoiPairGateConfig,
    evaluate_stereo_roi_pair,
)

from offline_volleyball_depth_math import depth_from_disparity
from offline_volleyball_calibration import load_calibration, rectify_pair
from offline_volleyball_probe_models import (
    BallCandidate,
    BallROI,
    MatchResult,
    RoughBallROI,
    ValidationThresholds,
)


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


def segment_ball(image: np.ndarray, side: str) -> BallROI:
    """Segment the visible volleyball in the saved scene.

    The detector is deliberately conservative and scene-local: it prefers
    saturated yellow/blue/white regions in the lower middle of the image, then
    grows the mask around the selected connected components.
    """

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    luma = lab[:, :, 0]

    yellow = (hue >= 18) & (hue <= 45) & (sat >= 45) & (val >= 45)
    blue = (hue >= 88) & (hue <= 135) & (sat >= 35) & (val >= 25)
    white = (sat <= 80) & (val >= 85) & (luma >= 75)
    chroma = ((yellow | blue | white).astype(np.uint8)) * 255

    # Suppress ceiling lights/windows. The ball is in the lower scene.
    chroma[: int(h * 0.45), :] = 0
    if side == "left":
        chroma[:, : int(w * 0.45)] = 0
    else:
        chroma[:, int(w * 0.70) :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    chroma = cv2.morphologyEx(chroma, cv2.MORPH_OPEN, kernel)
    chroma = cv2.morphologyEx(chroma, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(chroma, 8)
    chosen: List[int] = []
    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        if area < 18 or area > 7000:
            continue
        cx, cy = centroids[idx]
        if cy < h * 0.48 or cy > h * 0.85:
            continue
        if bw > 180 or bh > 180:
            continue
        chosen.append(idx)

    if not chosen:
        raise RuntimeError(f"failed to segment ball on {side}")

    # Group nearby colored components around the strongest component.
    chosen.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    seed = chosen[0]
    seed_c = centroids[seed]
    grouped = []
    for idx in chosen:
        dist = np.linalg.norm(centroids[idx] - seed_c)
        if dist < 95.0:
            grouped.append(idx)

    mask = np.zeros((h, w), dtype=np.uint8)
    for idx in grouped:
        mask[labels == idx] = 255

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise RuntimeError(f"empty ball mask on {side}")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    r = max((x2 - x1 + 1), (y2 - y1 + 1)) * 0.62
    margin = int(max(20, round(r * 0.45)))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)

    # Add a circular ROI mask so black/white panels on the ball are retained.
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    roi_mask &= np.where(np.indices((h, w))[0] >= y1, 255, 0).astype(np.uint8)
    roi_mask &= np.where(np.indices((h, w))[0] <= y2, 255, 0).astype(np.uint8)
    roi_mask &= np.where(np.indices((h, w))[1] >= x1, 255, 0).astype(np.uint8)
    roi_mask &= np.where(np.indices((h, w))[1] <= x2, 255, 0).astype(np.uint8)

    return BallROI((x1, y1, x2 - x1 + 1, y2 - y1 + 1), (cx, cy), float(r), roi_mask)


def _ball_candidates(image: np.ndarray, side: str) -> List[BallCandidate]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    luma = lab[:, :, 0]

    # Tighter yellow threshold is important: otherwise the brown platform joins
    # the ball and dominates the component score.
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
        # Scene prior for this captured validation pair: the ball sits on the
        # lower platform. This keeps colored signs/window reflections out of
        # the offline probe without affecting the runtime pipeline.
        if side == "left" and cx < w * 0.52:
            continue
        if side == "right" and (cx < w * 0.18 or cx > w * 0.72):
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 0.25 or aspect > 3.2:
            continue
        extent = area / max(1.0, float(bw * bh))
        # Moderate-size, compact colored regions score high; huge background
        # panels/logos score low once paired with the other camera.
        size_score = math.log1p(area)
        aspect_penalty = abs(math.log(max(0.1, aspect)))
        score = size_score + 2.2 * extent - 0.85 * aspect_penalty
        comp = np.zeros((h, w), dtype=np.uint8)
        comp[labels == idx] = 255
        candidates.append(BallCandidate(idx, (x, y, bw, bh), (cx, cy), area, score, comp))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _roi_from_group(image_shape: Tuple[int, int], seed: BallCandidate, candidates: Sequence[BallCandidate]) -> BallROI:
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
    """Build de-duplicated ball-sized ROIs from color fragments.

    A single volleyball often appears as several yellow/blue/white components.
    Pairing those component bboxes directly is too brittle, so the offline
    probe first groups nearby fragments into a ball-sized ROI and only then
    applies the same left/right ROI gate as the realtime contract.
    """

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
