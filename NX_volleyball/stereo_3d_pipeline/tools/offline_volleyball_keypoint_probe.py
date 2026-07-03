#!/usr/bin/env python3
"""Offline CPU probe for volleyball stereo keypoint/depth candidates.

This is intentionally diagnostic: it uses one saved stereo pair, rectifies it
with the current calibration, segments the ball ROI, then compares descriptor
matches against a patch IoU/ZNCC epipolar search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import (
    DEPTH_CANDIDATE_PRIORITY,
    STEREO_DEPTH_SOURCE,
    FeatureValidationConfig,
    SparseFeatureObservation,
    StereoRoiPairGateConfig,
    evaluate_stereo_roi_pair,
    validate_sparse_feature_geometry,
)

from offline_volleyball_probe_roi import (
    BallROI,
    MatchResult,
    ValidationThresholds,
    _roi_to_runtime_detection,
    detect_ball_rois,
    draw_roi_debug,
    load_calibration,
    rectify_pair,
)

def _masked_keypoints(
    gray: np.ndarray,
    mask: np.ndarray,
    method: str,
    max_points: int,
) -> List[cv2.KeyPoint]:
    if method == "corner":
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_points,
            qualityLevel=0.01,
            minDistance=4,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if pts is None:
            return []
        return [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 9) for p in pts]

    if method == "edge":
        edges = cv2.Canny(gray, 45, 130)
        edges = cv2.bitwise_and(edges, mask)
        ys, xs = np.where(edges > 0)
        if xs.size == 0:
            return []
        # Prefer high-gradient, spatially spread points.
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        order = np.argsort(mag[ys, xs])[::-1]
        selected: List[Tuple[int, int]] = []
        min_dist2 = 6 * 6
        for idx in order:
            x = int(xs[idx])
            y = int(ys[idx])
            if all((x - sx) * (x - sx) + (y - sy) * (y - sy) >= min_dist2 for sx, sy in selected):
                selected.append((x, y))
                if len(selected) >= max_points:
                    break
        return [cv2.KeyPoint(float(x), float(y), 9) for x, y in selected]

    raise ValueError(method)


def _spread_points_from_score(
    score: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    min_distance: int,
) -> List[cv2.KeyPoint]:
    valid = (mask > 0) & np.isfinite(score)
    ys, xs = np.where(valid)
    if xs.size == 0:
        return []

    values = score[ys, xs]
    if values.size > max_points * 10:
        threshold = float(np.percentile(values, 65.0))
        keep = values >= threshold
        xs, ys, values = xs[keep], ys[keep], values[keep]
        if xs.size == 0:
            return []

    order = np.argsort(values)[::-1]
    selected: List[Tuple[int, int]] = []
    min_dist2 = min_distance * min_distance
    for idx in order:
        x = int(xs[idx])
        y = int(ys[idx])
        if all((x - sx) * (x - sx) + (y - sy) * (y - sy) >= min_dist2 for sx, sy in selected):
            selected.append((x, y))
            if len(selected) >= max_points:
                break
    return [cv2.KeyPoint(float(x), float(y), 9) for x, y in selected]


def _masked_color_edge_keypoints(
    image: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    percentile: float = 58.0,
) -> List[cv2.KeyPoint]:
    """Pick repeatable points on volleyball color-panel boundaries."""

    total = _color_edge_strength(image, mask)
    valid_values = total[mask > 0]
    if valid_values.size == 0:
        return []
    edge_mask = np.zeros(mask.shape, dtype=np.uint8)
    edge_mask[(mask > 0) & (total >= float(np.percentile(valid_values, percentile)))] = 255
    edge_mask = cv2.morphologyEx(
        edge_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return _spread_points_from_score(total, edge_mask, max_points, 6)


def _color_edge_strength(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2], hsv[:, :, 1]]
    total = np.zeros(image.shape[:2], dtype=np.float32)
    for ch in channels:
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        denom = float(np.percentile(mag[mask > 0], 95.0)) if np.any(mask > 0) else 1.0
        if denom < 1e-6:
            denom = 1.0
        total += np.clip(mag / denom, 0.0, 2.0)
    return total


def _color_edge_mask(image: np.ndarray, mask: np.ndarray, percentile: float = 58.0) -> np.ndarray:
    strength = _color_edge_strength(image, mask)
    values = strength[mask > 0]
    edge = np.zeros(mask.shape, dtype=np.uint8)
    if values.size == 0:
        return edge
    edge[(mask > 0) & (strength >= float(np.percentile(values, percentile)))] = 255
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return edge


def _volleyball_label_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Quantize the ball into coarse visual regions for patch IoU scoring."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    luma = lab[:, :, 0]

    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    labels[mask > 0] = 1  # dark/other ball surface
    yellow = (hue >= 18) & (hue <= 48) & (sat >= 45) & (val >= 35) & (mask > 0)
    blue = (hue >= 85) & (hue <= 140) & (sat >= 28) & (val >= 20) & (mask > 0)
    white = (sat <= 85) & (val >= 85) & (luma >= 80) & (mask > 0)
    labels[yellow] = 2
    labels[blue] = 3
    labels[white] = 4
    return labels


def _shift_mask_x(mask: np.ndarray, dx: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    if dx == 0:
        return mask.copy()
    if abs(dx) >= mask.shape[1]:
        return shifted
    if dx > 0:
        shifted[:, dx:] = mask[:, : mask.shape[1] - dx]
    else:
        sx = -dx
        shifted[:, : mask.shape[1] - sx] = mask[:, sx:]
    return shifted


def _overlap_masks_for_disparity(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    disparity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    disp = int(round(disparity))
    right_in_left = _shift_mask_x(right_mask, disp)
    left_in_right = _shift_mask_x(left_mask, -disp)
    left_overlap = cv2.bitwise_and(left_mask, right_in_left)
    right_overlap = cv2.bitwise_and(right_mask, left_in_right)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    left_overlap = cv2.erode(left_overlap, kernel)
    right_overlap = cv2.erode(right_overlap, kernel)
    return left_overlap, right_overlap


def draw_overlap_debug(
    left: np.ndarray,
    right: np.ndarray,
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
) -> np.ndarray:
    out_l = left.copy()
    out_r = right.copy()
    out_l[left_overlap > 0] = (0.60 * out_l[left_overlap > 0] + np.array([0, 255, 255]) * 0.40).astype(np.uint8)
    out_r[right_overlap > 0] = (0.60 * out_r[right_overlap > 0] + np.array([0, 255, 255]) * 0.40).astype(np.uint8)
    return np.hstack([out_l, out_r])


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aa -= float(aa.mean())
    bb -= float(bb.mean())
    denom = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    if denom < 1e-6:
        return -1.0
    return float((aa * bb).sum() / denom)


def _patch_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


def _weighted_label_iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    total = 0.0
    weight_sum = 0.0
    for label, weight in [(1, 0.20), (2, 1.00), (3, 0.85), (4, 1.15)]:
        aa = (a == label) & valid
        bb = (b == label) & valid
        union = np.logical_or(aa, bb).sum()
        if union <= 0:
            continue
        inter = np.logical_and(aa, bb).sum()
        total += weight * float(inter / union)
        weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return float(total / weight_sum)


def _appearance_patch_score(
    qfeat: np.ndarray,
    tfeat: np.ndarray,
    qlabels: np.ndarray,
    tlabels: np.ndarray,
    qedge: np.ndarray,
    tedge: np.ndarray,
    qmask: np.ndarray,
    tmask: np.ndarray,
    qx: int,
    qy: int,
    tx: int,
    ty: int,
    patch_radius: int,
) -> Tuple[float, float, float, float, float] | None:
    h, w = qmask.shape[:2]
    if qx - patch_radius < 0 or qx + patch_radius >= w or tx - patch_radius < 0 or tx + patch_radius >= w:
        return None
    if qy - patch_radius < 0 or qy + patch_radius >= h or ty - patch_radius < 0 or ty + patch_radius >= h:
        return None

    qslice = np.s_[qy - patch_radius : qy + patch_radius + 1, qx - patch_radius : qx + patch_radius + 1]
    tslice = np.s_[ty - patch_radius : ty + patch_radius + 1, tx - patch_radius : tx + patch_radius + 1]
    qmp = qmask[qslice] > 0
    tmp = tmask[tslice] > 0
    valid = qmp & tmp
    valid_ratio = float(valid.mean())
    if valid_ratio < 0.58:
        return None

    qpatch = qfeat[qslice]
    tpatch = tfeat[tslice]
    zncc = max(0.0, _zncc(qpatch, tpatch))
    label_iou = _weighted_label_iou(qlabels[qslice], tlabels[tslice], valid)
    edge_iou = _patch_iou(np.where(valid, qedge[qslice], 0), np.where(valid, tedge[tslice], 0))

    q_std = float(np.std(qpatch[valid])) if np.any(valid) else 0.0
    t_std = float(np.std(tpatch[valid])) if np.any(valid) else 0.0
    texture = float(np.clip(min(q_std, t_std) / 22.0, 0.0, 1.0))
    score = 0.44 * zncc + 0.34 * label_iou + 0.17 * edge_iou + 0.05 * texture
    return float(score), float(zncc), float(label_iou), float(edge_iou), texture


def _best_iou_region_candidate(
    qfeat: np.ndarray,
    tfeat: np.ndarray,
    qlabels: np.ndarray,
    tlabels: np.ndarray,
    qedge: np.ndarray,
    tedge: np.ndarray,
    qmask: np.ndarray,
    tmask: np.ndarray,
    toverlap: np.ndarray,
    qx: int,
    qy: int,
    initial_disp: float,
    patch_radius: int,
    search_radius: int,
    y_radius: int,
    direction: str,
) -> Tuple[float, int, int, float, float, float, float, float] | None:
    h, w = qmask.shape[:2]
    d0 = int(round(initial_disp))
    d_start = max(1, d0 - search_radius)
    d_end = min(w - 1, d0 + search_radius)

    best: Tuple[float, int, int, float, float, float, float] | None = None
    second_score = -1.0
    for dy in range(-y_radius, y_radius + 1):
        ty = qy + dy
        if ty - patch_radius < 0 or ty + patch_radius >= h:
            continue
        for disp in range(d_start, d_end + 1):
            tx = qx - disp if direction == "left_to_right" else qx + disp
            if tx - patch_radius < 0 or tx + patch_radius >= w:
                continue
            if toverlap[ty, tx] == 0:
                continue
            scored = _appearance_patch_score(
                qfeat, tfeat, qlabels, tlabels, qedge, tedge, qmask, tmask,
                qx, qy, tx, ty, patch_radius,
            )
            if scored is None:
                continue
            score, zncc, label_iou, edge_iou, texture = scored
            score -= 0.004 * abs(disp - initial_disp)
            score -= 0.018 * abs(dy)
            if best is None or score > best[0]:
                if best is not None:
                    second_score = max(second_score, best[0])
                best = (score, tx, ty, zncc, label_iou, edge_iou, texture)
            else:
                second_score = max(second_score, score)

    if best is None:
        return None
    uniqueness = best[0] - second_score if second_score >= 0.0 else 1.0
    return best + (float(uniqueness),)


def iou_region_color_patch_match(
    left_feat: np.ndarray,
    right_feat: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    left_keypoints: List[cv2.KeyPoint],
    initial_disp: float,
    name: str = "iou_region_color_patch",
    patch_radius: int = 9,
    search_radius: int = 28,
    y_radius: int = 2,
    min_score: float = 0.58,
    reverse_tolerance_px: float = 3.0,
    max_points: int = 90,
) -> MatchResult:
    points = left_keypoints[:max_points]
    candidates: List[Tuple[int, int, int, float, float, float, float, float, float]] = []

    for qi, kp in enumerate(points):
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if left_overlap[y, x] == 0:
            continue
        best = _best_iou_region_candidate(
            left_feat, right_feat,
            left_labels, right_labels,
            left_edge, right_edge,
            left_mask, right_mask,
            right_overlap,
            x, y, initial_disp,
            patch_radius, search_radius, y_radius,
            "left_to_right",
        )
        if best is None:
            continue
        score, xr, yr, zncc, label_iou, edge_iou, texture, uniqueness = best
        if score < min_score:
            continue
        if uniqueness < 0.004 and score < 0.80:
            continue

        reverse = _best_iou_region_candidate(
            right_feat, left_feat,
            right_labels, left_labels,
            right_edge, left_edge,
            right_mask, left_mask,
            left_overlap,
            xr, yr, initial_disp,
            patch_radius, search_radius, y_radius,
            "right_to_left",
        )
        if reverse is None:
            continue
        _, bx, by, _, _, _, _, _ = reverse
        if math.hypot(float(bx - x), float(by - y)) > reverse_tolerance_px:
            continue

        disp = float(x - xr)
        candidates.append((qi, xr, yr, score, disp, zncc, label_iou, edge_iou, texture))

    if not candidates:
        return MatchResult(name, points, [], [], 0, -1.0, -1.0, -1.0, 0.0)

    disparities = np.array([d for _, _, _, _, d, _, _, _, _ in candidates], dtype=np.float32)
    median = float(np.median(disparities))
    abs_dev = np.abs(disparities - median)
    mad = float(np.median(abs_dev))
    gate = max(1.2, 2.8 * mad)

    indexed_candidates = [
        cand for keep, cand in zip(abs_dev <= gate, candidates) if bool(keep)
    ]
    indexed_candidates.sort(key=lambda c: c[3], reverse=True)

    right_kps: List[cv2.KeyPoint] = []
    matches: List[cv2.DMatch] = []
    extras: List[Dict[str, float]] = []
    used_right: List[Tuple[int, int]] = []
    used_left: set[int] = set()
    inlier_disps: List[float] = []
    inlier_scores: List[float] = []

    for qi, xr, yr, score, disp, zncc, label_iou, edge_iou, texture in indexed_candidates:
        if qi in used_left:
            continue
        if any((xr - ux) * (xr - ux) + (yr - uy) * (yr - uy) < 5 * 5 for ux, uy in used_right):
            continue
        ti = len(right_kps)
        right_kps.append(cv2.KeyPoint(float(xr), float(yr), float(patch_radius * 2 + 1)))
        matches.append(cv2.DMatch(int(qi), int(ti), float(1.0 - score)))
        extras.append({
            "score": float(score),
            "disparity_px": float(disp),
            "zncc": float(zncc),
            "label_iou": float(label_iou),
            "edge_iou": float(edge_iou),
            "texture": float(texture),
        })
        used_left.add(qi)
        used_right.append((xr, yr))
        inlier_disps.append(float(disp))
        inlier_scores.append(float(score))

    if not matches:
        return MatchResult(name, points, right_kps, [], len(candidates), -1.0, -1.0, -1.0, 0.0)

    disparity = float(np.mean(inlier_disps))
    std_px = float(np.std(inlier_disps))
    confidence = float(np.clip(np.mean(inlier_scores) * min(1.0, len(matches) / 12.0) / (1.0 + std_px / 8.0), 0.0, 1.0))
    return MatchResult(name, points, right_kps, matches, len(candidates), disparity, std_px, -1.0, confidence, extras=extras)


def patch_iou_zncc_match(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_keypoints: List[cv2.KeyPoint],
    initial_disp: float,
    name: str,
    patch_radius: int = 9,
    search_radius: int = 22,
    y_radius: int = 3,
    min_score: float = 0.40,
    max_points: int = 80,
) -> MatchResult:
    h, w = left_gray.shape[:2]
    points = left_keypoints[:max_points]
    candidates: List[Tuple[int, int, int, float, float]] = []

    d0 = int(round(initial_disp))
    d_start = max(1, d0 - search_radius)
    d_end = min(w - 1, d0 + search_radius)

    for qi, kp in enumerate(points):
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x - patch_radius < 0 or x + patch_radius >= w:
            continue
        if y - patch_radius < 0 or y + patch_radius >= h:
            continue
        if left_mask[y, x] == 0:
            continue

        lpatch = left_gray[y - patch_radius : y + patch_radius + 1,
                           x - patch_radius : x + patch_radius + 1]
        lmpatch = left_mask[y - patch_radius : y + patch_radius + 1,
                            x - patch_radius : x + patch_radius + 1]
        best: Tuple[float, int, int, float, float] | None = None
        second_score = -1.0
        for dy in range(-y_radius, y_radius + 1):
            yr = y + dy
            if yr - patch_radius < 0 or yr + patch_radius >= h:
                continue
            for disp in range(d_start, d_end + 1):
                xr = x - disp
                if xr - patch_radius < 0 or xr + patch_radius >= w:
                    continue
                rpatch = right_gray[yr - patch_radius : yr + patch_radius + 1,
                                    xr - patch_radius : xr + patch_radius + 1]
                rmpatch = right_mask[yr - patch_radius : yr + patch_radius + 1,
                                     xr - patch_radius : xr + patch_radius + 1]
                zncc = _zncc(lpatch, rpatch)
                iou = _patch_iou(lmpatch, rmpatch)
                # IoU alone fails on smooth regions; ZNCC alone drifts on dark curtains.
                score = 0.62 * max(0.0, zncc) + 0.38 * iou
                if best is None or score > best[0]:
                    if best is not None:
                        second_score = max(second_score, best[0])
                    best = (score, xr, yr, zncc, iou)
                else:
                    second_score = max(second_score, score)

        if best is None:
            continue
        score, xr, yr, zncc, iou = best
        uniqueness = score - second_score if second_score >= 0 else 1.0
        if score < min_score:
            continue
        if uniqueness < 0.025 and score < 0.68:
            continue
        if right_mask[yr, xr] == 0 and iou < 0.20:
            continue
        candidates.append((qi, xr, yr, score, float(x - xr)))

    if not candidates:
        return MatchResult(name, points, [], [], 0, -1.0, -1.0, -1.0, 0.0)

    disparities = np.array([d for _, _, _, _, d in candidates], dtype=np.float32)
    median = float(np.median(disparities))
    abs_dev = np.abs(disparities - median)
    mad = float(np.median(abs_dev))
    gate = max(1.25, 2.5 * mad)
    inlier_mask = abs_dev <= gate
    indexed_candidates = [
        (idx, cand) for idx, (keep, cand) in enumerate(zip(inlier_mask, candidates)) if bool(keep)
    ]
    indexed_candidates.sort(key=lambda item: item[1][3], reverse=True)

    right_kps: List[cv2.KeyPoint] = []
    matches: List[cv2.DMatch] = []
    used_right: List[Tuple[int, int]] = []
    inlier_disps = []
    inlier_scores = []
    for _, (qi, xr, yr, score, disp) in indexed_candidates:
        if any((xr - ux) * (xr - ux) + (yr - uy) * (yr - uy) < 5 * 5 for ux, uy in used_right):
            continue
        ti = len(right_kps)
        right_kps.append(cv2.KeyPoint(float(xr), float(yr), float(patch_radius * 2 + 1)))
        matches.append(cv2.DMatch(int(qi), int(ti), float(1.0 - score)))
        used_right.append((xr, yr))
        inlier_disps.append(disp)
        inlier_scores.append(score)

    if not matches:
        return MatchResult(name, points, right_kps, [], len(candidates), -1.0, -1.0, -1.0, 0.0)

    disparity = float(np.mean(inlier_disps))
    std_px = float(np.std(inlier_disps))
    confidence = float(np.clip(np.mean(inlier_scores) * min(1.0, len(matches) / 8.0), 0.0, 1.0))
    return MatchResult(name, points, right_kps, matches, len(candidates), disparity, std_px, -1.0, confidence)


def descriptor_match(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    name: str,
    initial_disp: float,
    max_features: int = 300,
) -> MatchResult:
    if name == "orb":
        extractor = cv2.ORB_create(max_features, 1.2, 4, 12, 0, 2, cv2.ORB_HARRIS_SCORE, 17, 10)
        norm = cv2.NORM_HAMMING
    elif name == "brisk":
        extractor = cv2.BRISK_create(18, 2, 1.0)
        norm = cv2.NORM_HAMMING
    elif name == "akaze":
        extractor = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif name == "sift":
        extractor = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.015, edgeThreshold=12)
        norm = cv2.NORM_L2
    else:
        raise ValueError(name)

    lkps, ldesc = extractor.detectAndCompute(left_gray, left_mask)
    rkps, rdesc = extractor.detectAndCompute(right_gray, right_mask)
    lkps = list(lkps or [])
    rkps = list(rkps or [])
    if ldesc is None or rdesc is None or not lkps or not rkps:
        return MatchResult(name, lkps, rkps, [], 0, -1.0, -1.0, -1.0, 0.0)

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(ldesc, rdesc, k=2)
    candidates: List[cv2.DMatch] = []
    for pair in knn:
        if not pair:
            continue
        best = pair[0]
        if len(pair) > 1:
            ratio = 0.76 if norm == cv2.NORM_HAMMING else 0.72
            if best.distance >= ratio * pair[1].distance:
                continue
        lpt = lkps[best.queryIdx].pt
        rpt = rkps[best.trainIdx].pt
        disp = lpt[0] - rpt[0]
        if abs(lpt[1] - rpt[1]) > 4.0:
            continue
        if disp <= 0.0:
            continue
        if abs(disp - initial_disp) > 35.0:
            continue
        candidates.append(best)

    if not candidates:
        return MatchResult(name, lkps, rkps, [], 0, -1.0, -1.0, -1.0, 0.0)

    disps = np.array([lkps[m.queryIdx].pt[0] - rkps[m.trainIdx].pt[0] for m in candidates],
                     dtype=np.float32)
    median = float(np.median(disps))
    abs_dev = np.abs(disps - median)
    gate = max(1.5, 2.5 * float(np.median(abs_dev)))
    matches = [m for m, e in zip(candidates, abs_dev) if e <= gate]
    if not matches:
        return MatchResult(name, lkps, rkps, [], len(candidates), -1.0, -1.0, -1.0, 0.0)
    inlier_disps = np.array([lkps[m.queryIdx].pt[0] - rkps[m.trainIdx].pt[0] for m in matches],
                            dtype=np.float32)
    disparity = float(np.mean(inlier_disps))
    std_px = float(np.std(inlier_disps))
    confidence = float(np.clip(len(matches) / 12.0 / (1.0 + std_px), 0.0, 1.0))
    return MatchResult(name, lkps, rkps, matches, len(candidates), disparity, std_px, -1.0, confidence)


def depth_from_disparity(disparity: float, focal_px: float, baseline_m: float) -> float:
    if disparity <= 0.0:
        return -1.0
    return focal_px * baseline_m / disparity


def draw_matches(
    left: np.ndarray,
    right: np.ndarray,
    result: MatchResult,
    out_path: Path,
) -> None:
    canvas = np.hstack([left.copy(), right.copy()])
    xoff = left.shape[1]
    for kp in result.left_keypoints:
        cv2.circle(canvas, tuple(np.round(kp.pt).astype(int)), 4, (255, 255, 0), 1, cv2.LINE_AA)
    for kp in result.right_keypoints:
        p = (int(round(kp.pt[0] + xoff)), int(round(kp.pt[1])))
        cv2.circle(canvas, p, 4, (255, 0, 255), 1, cv2.LINE_AA)
    for m in result.matches:
        if m.queryIdx >= len(result.left_keypoints) or m.trainIdx >= len(result.right_keypoints):
            continue
        p1 = tuple(np.round(result.left_keypoints[m.queryIdx].pt).astype(int))
        p2 = (int(round(result.right_keypoints[m.trainIdx].pt[0] + xoff)),
              int(round(result.right_keypoints[m.trainIdx].pt[1])))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 5, (0, 255, 0), 1, cv2.LINE_AA)
    title = (
        f"{result.name}: Lkp={len(result.left_keypoints)} Rkp={len(result.right_keypoints)} "
        f"cand={result.candidates} match={len(result.matches)} "
        f"disp={result.disparity:.2f} std={result.std_px:.2f} z={result.depth_m:.2f}m"
    )
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
    cv2.imwrite(str(out_path), canvas)


def crop_pair(left: np.ndarray, right: np.ndarray, lroi: BallROI, rroi: BallROI, pad: int = 70) -> Tuple[np.ndarray, np.ndarray]:
    lx, ly, lw, lh = lroi.bbox
    rx, ry, rw, rh = rroi.bbox
    y1 = max(0, min(ly, ry) - pad)
    y2 = min(left.shape[0], max(ly + lh, ry + rh) + pad)
    lx1 = max(0, lx - pad)
    lx2 = min(left.shape[1], lx + lw + pad)
    rx1 = max(0, rx - pad)
    rx2 = min(right.shape[1], rx + rw + pad)
    return left[y1:y2, lx1:lx2], right[y1:y2, rx1:rx2]


def _roi_crop_bounds(shape: Tuple[int, int], roi: BallROI, pad: int) -> Tuple[int, int, int, int]:
    h, w = shape
    cx, cy = roi.center
    r = roi.radius + pad
    x1 = max(0, int(math.floor(cx - r)))
    y1 = max(0, int(math.floor(cy - r)))
    x2 = min(w, int(math.ceil(cx + r)))
    y2 = min(h, int(math.ceil(cy + r)))
    return x1, y1, x2, y2


def draw_matches_zoom(
    left: np.ndarray,
    right: np.ndarray,
    lroi: BallROI,
    rroi: BallROI,
    result: MatchResult,
    out_path: Path,
    pad: int = 22,
    scale: int = 3,
) -> None:
    lx1, ly1, lx2, ly2 = _roi_crop_bounds(left.shape[:2], lroi, pad)
    rx1, ry1, rx2, ry2 = _roi_crop_bounds(right.shape[:2], rroi, pad)
    lc = left[ly1:ly2, lx1:lx2].copy()
    rc = right[ry1:ry2, rx1:rx2].copy()
    if lc.size == 0 or rc.size == 0:
        return

    target_h = max(lc.shape[0], rc.shape[0])
    if lc.shape[0] < target_h:
        lc = cv2.copyMakeBorder(lc, 0, target_h - lc.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if rc.shape[0] < target_h:
        rc = cv2.copyMakeBorder(rc, 0, target_h - rc.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    canvas = np.hstack([lc, rc])
    xoff = lc.shape[1]

    def left_local(kp: cv2.KeyPoint) -> Tuple[int, int] | None:
        x = int(round(kp.pt[0] - lx1))
        y = int(round(kp.pt[1] - ly1))
        if 0 <= x < lc.shape[1] and 0 <= y < lc.shape[0]:
            return x, y
        return None

    def right_local(kp: cv2.KeyPoint) -> Tuple[int, int] | None:
        x = int(round(kp.pt[0] - rx1))
        y = int(round(kp.pt[1] - ry1))
        if 0 <= x < rc.shape[1] and 0 <= y < rc.shape[0]:
            return x + xoff, y
        return None

    for kp in result.left_keypoints:
        p = left_local(kp)
        if p is not None:
            cv2.circle(canvas, p, 3, (255, 255, 0), 1, cv2.LINE_AA)
    for kp in result.right_keypoints:
        p = right_local(kp)
        if p is not None:
            cv2.circle(canvas, p, 3, (255, 0, 255), 1, cv2.LINE_AA)
    for m in result.matches:
        if m.queryIdx >= len(result.left_keypoints) or m.trainIdx >= len(result.right_keypoints):
            continue
        p1 = left_local(result.left_keypoints[m.queryIdx])
        p2 = right_local(result.right_keypoints[m.trainIdx])
        if p1 is None or p2 is None:
            continue
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, (0, 255, 0), 1, cv2.LINE_AA)

    title = f"{result.name} match={len(result.matches)} disp={result.disparity:.2f} std={result.std_px:.2f}"
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 180, 255), 1)
    if scale > 1:
        canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_path), canvas)


def triangulate_match_rows(
    result: MatchResult,
    calib: Dict[str, np.ndarray | float | int],
    baseline_m: float,
) -> List[Dict[str, float | int | str]]:
    p1 = np.asarray(calib["P1"], dtype=np.float64)
    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])

    rows: List[Dict[str, float | int | str]] = []
    for idx, match in enumerate(result.matches):
        if match.queryIdx >= len(result.left_keypoints) or match.trainIdx >= len(result.right_keypoints):
            continue
        lpt = result.left_keypoints[match.queryIdx].pt
        rpt = result.right_keypoints[match.trainIdx].pt
        disparity = float(lpt[0] - rpt[0])
        if disparity <= 0.0:
            continue
        z_m = fx * baseline_m / disparity
        x_m = (float(lpt[0]) - cx) * z_m / fx
        y_m = (float(lpt[1]) - cy) * z_m / fy
        row: Dict[str, float | int | str] = {
            "method": result.name,
            "index": idx,
            "left_x": float(lpt[0]),
            "left_y": float(lpt[1]),
            "right_x": float(rpt[0]),
            "right_y": float(rpt[1]),
            "y_error_px": abs(float(lpt[1]) - float(rpt[1])),
            "disparity_px": disparity,
            "x_m": x_m,
            "y_m": y_m,
            "z_m": z_m,
        }
        if idx < len(result.extras):
            row.update(result.extras[idx])
        rows.append(row)
    return rows


def estimate_ball_center_3d(
    calib: Dict[str, np.ndarray | float | int],
    lroi: BallROI,
    center_disparity: float,
    baseline_m: float,
) -> Tuple[float, float, float]:
    p1 = np.asarray(calib["P1"], dtype=np.float64)
    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])
    z_m = depth_from_disparity(center_disparity, fx, baseline_m)
    x_m = (float(lroi.center[0]) - cx) * z_m / fx
    y_m = (float(lroi.center[1]) - cy) * z_m / fy
    return x_m, y_m, z_m


def _mask_value(mask: np.ndarray, x: float, y: float) -> int:
    ix = int(round(x))
    iy = int(round(y))
    if iy < 0 or iy >= mask.shape[0] or ix < 0 or ix >= mask.shape[1]:
        return 0
    return int(mask[iy, ix])


def validate_triangulated_rows(
    rows: Sequence[Dict[str, float | int | str]],
    lroi: BallROI,
    rroi: BallROI,
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    ball_center_3d: Tuple[float, float, float],
    ball_radius_m: float,
    thresholds: ValidationThresholds,
) -> List[Dict[str, float | int | str]]:
    if not rows:
        return []

    disparities = np.array([float(row["disparity_px"]) for row in rows], dtype=np.float64)
    zs = np.array([float(row["z_m"]) for row in rows], dtype=np.float64)
    disp_median = float(np.median(disparities))
    z_median = float(np.median(zs))
    cx, cy, cz = ball_center_3d

    validated: List[Dict[str, float | int | str]] = []
    for row in rows:
        out = dict(row)
        lx = float(row["left_x"])
        ly = float(row["left_y"])
        rx = float(row["right_x"])
        ry = float(row["right_y"])
        disparity = float(row["disparity_px"])
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        z_m = float(row["z_m"])
        y_error = abs(ly - ry)

        sphere_distance = math.sqrt((x_m - cx) * (x_m - cx) + (y_m - cy) * (y_m - cy) + (z_m - cz) * (z_m - cz))
        sphere_residual = abs(sphere_distance - ball_radius_m)
        disp_dev = abs(disparity - disp_median)
        z_dev = abs(z_m - z_median)
        center_depth_delta = abs(z_m - cz)

        checks = {
            "inside_left_mask": _mask_value(lroi.mask, lx, ly) > 0,
            "inside_right_mask": _mask_value(rroi.mask, rx, ry) > 0,
            "inside_left_overlap": _mask_value(left_overlap, lx, ly) > 0,
            "inside_right_overlap": _mask_value(right_overlap, rx, ry) > 0,
            "y_error_ok": y_error <= thresholds.max_y_error_px,
            "disparity_consistent": disp_dev <= max(2.0, thresholds.max_disparity_range_px * 0.5),
            "z_consistent": z_dev <= max(0.030, thresholds.max_z_range_m * 0.5),
            "sphere_residual_ok": sphere_residual <= thresholds.max_sphere_residual_m,
            "depth_near_center": center_depth_delta <= thresholds.max_depth_vs_center_m,
        }
        fail_reasons = [name for name, passed in checks.items() if not passed]

        out.update({
            "validation_pass": int(not fail_reasons),
            "validation_fail_reasons": ";".join(fail_reasons),
            "disparity_deviation_px": disp_dev,
            "z_deviation_m": z_dev,
            "sphere_distance_m": sphere_distance,
            "sphere_residual_m": sphere_residual,
            "ball_center_depth_delta_m": center_depth_delta,
            **{name: int(passed) for name, passed in checks.items()},
        })
        validated.append(out)
    return validated


def triangulation_stats(rows: Sequence[Dict[str, float | int | str]]) -> Dict[str, float | int]:
    z = np.array([float(row["z_m"]) for row in rows if float(row["z_m"]) > 0.0], dtype=np.float64)
    d = np.array([float(row["disparity_px"]) for row in rows if float(row["disparity_px"]) > 0.0], dtype=np.float64)
    yerr = np.array([float(row.get("y_error_px", 0.0)) for row in rows], dtype=np.float64)
    sphere_residual = np.array([float(row.get("sphere_residual_m", -1.0)) for row in rows if float(row.get("sphere_residual_m", -1.0)) >= 0.0], dtype=np.float64)
    valid = np.array([int(row.get("validation_pass", 0)) for row in rows], dtype=np.int32)
    if z.size == 0 or d.size == 0:
        return {
            "triangulated_points": 0,
            "validation_valid_points": 0,
            "validation_valid_ratio": 0.0,
            "z_median_m": -1.0,
            "z_mad_m": -1.0,
            "z_min_m": -1.0,
            "z_max_m": -1.0,
            "disparity_median_px": -1.0,
            "disparity_mad_px": -1.0,
            "disparity_min_px": -1.0,
            "disparity_max_px": -1.0,
            "y_error_max_px": -1.0,
            "sphere_residual_median_m": -1.0,
            "sphere_residual_max_m": -1.0,
        }
    return {
        "triangulated_points": int(z.size),
        "validation_valid_points": int(valid.sum()) if valid.size else 0,
        "validation_valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "z_median_m": float(np.median(z)),
        "z_mad_m": float(np.median(np.abs(z - np.median(z)))),
        "z_min_m": float(np.min(z)),
        "z_max_m": float(np.max(z)),
        "disparity_median_px": float(np.median(d)),
        "disparity_mad_px": float(np.median(np.abs(d - np.median(d)))),
        "disparity_min_px": float(np.min(d)),
        "disparity_max_px": float(np.max(d)),
        "y_error_max_px": float(np.max(yerr)) if yerr.size else -1.0,
        "sphere_residual_median_m": float(np.median(sphere_residual)) if sphere_residual.size else -1.0,
        "sphere_residual_max_m": float(np.max(sphere_residual)) if sphere_residual.size else -1.0,
    }


def method_validation_status(
    stats: Dict[str, float | int],
    thresholds: ValidationThresholds,
) -> Tuple[str, str]:
    reasons: List[str] = []
    valid_points = int(stats.get("validation_valid_points", 0))
    if valid_points < thresholds.min_valid_matches:
        reasons.append(f"valid_points<{thresholds.min_valid_matches}")
    if float(stats.get("y_error_max_px", 1e9)) > thresholds.max_y_error_px:
        reasons.append("y_error")
    if float(stats.get("disparity_mad_px", 1e9)) > thresholds.max_disparity_mad_px:
        reasons.append("disparity_mad")
    disparity_range = float(stats.get("disparity_max_px", -1.0)) - float(stats.get("disparity_min_px", -1.0))
    if disparity_range > thresholds.max_disparity_range_px:
        reasons.append("disparity_range")
    if float(stats.get("z_mad_m", 1e9)) > thresholds.max_z_mad_m:
        reasons.append("z_mad")
    z_range = float(stats.get("z_max_m", -1.0)) - float(stats.get("z_min_m", -1.0))
    if z_range > thresholds.max_z_range_m:
        reasons.append("z_range")
    if float(stats.get("sphere_residual_max_m", 1e9)) > thresholds.max_sphere_residual_m:
        reasons.append("sphere_residual")
    return ("pass" if not reasons else "fail", ";".join(reasons))


def runtime_feature_geometry_status(
    result: MatchResult,
    lroi: BallROI,
    rroi: BallROI,
    initial_disparity: float,
    focal_px: float,
    baseline_m: float,
    config: FeatureValidationConfig,
) -> Tuple[int, float, float]:
    if not result.matches or result.disparity <= 0.5:
        return 0, 0.0, 0.0
    xs: List[float] = []
    ys: List[float] = []
    rxs: List[float] = []
    rys: List[float] = []
    for match in result.matches:
        if match.queryIdx < 0 or match.queryIdx >= len(result.left_keypoints):
            continue
        if match.trainIdx < 0 or match.trainIdx >= len(result.right_keypoints):
            continue
        kp = result.left_keypoints[match.queryIdx]
        rkp = result.right_keypoints[match.trainIdx]
        xs.append(float(kp.pt[0]))
        ys.append(float(kp.pt[1]))
        rxs.append(float(rkp.pt[0]))
        rys.append(float(rkp.pt[1]))
    if not xs:
        return 0, 0.0, 0.0
    anchor_x = float(np.mean(xs))
    anchor_y = float(np.mean(ys))
    right_anchor_x = float(np.mean(rxs)) if rxs else None
    right_anchor_y = float(np.mean(rys)) if rys else None
    observation = SparseFeatureObservation(
        valid=True,
        disparity_px=float(result.disparity),
        anchor_left_x=anchor_x,
        anchor_left_y=anchor_y,
        anchor_right_x=right_anchor_x,
        anchor_right_y=right_anchor_y,
        stddev_px=float(max(0.0, result.std_px)),
        support=len(xs),
    )
    ok = validate_sparse_feature_geometry(
        observation,
        _roi_to_runtime_detection(lroi),
        _roi_to_runtime_detection(rroi),
        initial_disparity,
        config,
        focal_px,
        baseline_m,
    )
    return int(ok), anchor_x, anchor_y


def write_triangulated_points(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/left/0000.png")
    parser.add_argument("--right", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/right/0000.png")
    parser.add_argument("--calib", default="NX_volleyball/calibration/stereo_calib.yaml")
    parser.add_argument("--out", default="NX_volleyball/stereo_3d_pipeline/test_logs/offline_keypoint_probe_latest")
    parser.add_argument("--left-circle", type=_parse_circle, help="rectified left ball circle: x,y,r")
    parser.add_argument("--right-circle", type=_parse_circle, help="rectified right ball circle: x,y,r")
    parser.add_argument("--mask-margin", type=float, default=12.0, help="pixels trimmed from ball boundary for keypoints")
    parser.add_argument("--ball-diameter-m", type=float, default=0.210, help="physical volleyball diameter for ROI sanity checks")
    parser.add_argument("--edge-percentile", type=float, default=58.0, help="color-edge percentile used for keypoint sampling")
    parser.add_argument("--iou-patch-radius", type=int, default=9)
    parser.add_argument("--iou-search-radius", type=int, default=28)
    parser.add_argument("--iou-y-radius", type=int, default=2)
    parser.add_argument("--iou-min-score", type=float, default=0.58)
    parser.add_argument("--iou-reverse-tolerance-px", type=float, default=3.0)
    parser.add_argument("--iou-max-points", type=int, default=90)
    parser.add_argument("--min-valid-matches", type=int, default=8)
    parser.add_argument("--max-y-error-px", type=float, default=2.0)
    parser.add_argument("--max-disparity-mad-px", type=float, default=1.0)
    parser.add_argument("--max-disparity-range-px", type=float, default=4.0)
    parser.add_argument("--max-z-mad-m", type=float, default=0.020)
    parser.add_argument("--max-z-range-m", type=float, default=0.060)
    parser.add_argument("--max-sphere-residual-m", type=float, default=0.030)
    parser.add_argument("--max-depth-vs-center-m", type=float, default=0.140)
    parser.add_argument("--max-disparity", type=int, default=2048)
    parser.add_argument("--pair-epipolar-y-tolerance", type=float, default=12.0)
    parser.add_argument("--pair-max-size-ratio", type=float, default=2.0)
    parser.add_argument("--pair-min-shifted-iou", type=float, default=0.0)
    parser.add_argument("--min-depth-m", type=float, default=0.8)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--feature-overlap-scale", type=float, default=0.55)
    parser.add_argument("--feature-min-support", type=int, default=4)
    parser.add_argument("--feature-max-stddev-px", type=float, default=1.0)
    parser.add_argument("--feature-sphere-radius-scale", type=float, default=1.8)
    parser.add_argument("--feature-sphere-margin-m", type=float, default=0.02)
    parser.add_argument("--quiet", action="store_true", help="write files without printing the full JSON summary")
    args = parser.parse_args()
    thresholds = ValidationThresholds(
        min_valid_matches=args.min_valid_matches,
        max_y_error_px=args.max_y_error_px,
        max_disparity_mad_px=args.max_disparity_mad_px,
        max_disparity_range_px=args.max_disparity_range_px,
        max_z_mad_m=args.max_z_mad_m,
        max_z_range_m=args.max_z_range_m,
        max_sphere_residual_m=args.max_sphere_residual_m,
        max_depth_vs_center_m=args.max_depth_vs_center_m,
    )
    pair_gate = StereoRoiPairGateConfig(
        max_disparity=args.max_disparity,
        epipolar_y_tolerance=args.pair_epipolar_y_tolerance,
        max_size_ratio=args.pair_max_size_ratio,
        min_shifted_iou=args.pair_min_shifted_iou,
    )
    feature_gate = FeatureValidationConfig(
        min_support=args.feature_min_support,
        max_stddev_px=args.feature_max_stddev_px,
        feature_y_tolerance_px=args.max_y_error_px,
        feature_overlap_scale=args.feature_overlap_scale,
        feature_sphere_radius_m=0.5 * args.ball_diameter_m,
        feature_sphere_radius_scale=args.feature_sphere_radius_scale,
        feature_sphere_margin_m=args.feature_sphere_margin_m,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    left = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError("failed to load input images")

    calib = load_calibration(Path(args.calib))
    left_rect, right_rect = rectify_pair(left, right, calib)
    cv2.imwrite(str(out_dir / "left_rect_color.png"), left_rect)
    cv2.imwrite(str(out_dir / "right_rect_color.png"), right_rect)

    focal_px = float(np.asarray(calib["P1"])[0, 0])
    baseline_m = float(calib["baseline_m"])
    if (args.left_circle is None) != (args.right_circle is None):
        raise ValueError("--left-circle and --right-circle must be provided together")
    if args.left_circle is not None and args.right_circle is not None:
        lx, ly, lr = args.left_circle
        rx, ry, rr = args.right_circle
        lroi = _circle_roi(left_rect.shape[:2], (lx, ly), lr, "manual", args.mask_margin)
        rroi = _circle_roi(right_rect.shape[:2], (rx, ry), rr, "manual", args.mask_margin)
    else:
        lroi, rroi = detect_ball_rois(
            left_rect,
            right_rect,
            focal_px,
            baseline_m,
            args.mask_margin,
            args.ball_diameter_m,
            pair_gate,
            args.min_depth_m,
            args.max_depth_m,
        )
    initial_disp = float(lroi.center[0] - rroi.center[0])
    initial_depth = depth_from_disparity(initial_disp, focal_px, baseline_m)
    runtime_pair, runtime_pair_reject = evaluate_stereo_roi_pair(
        _roi_to_runtime_detection(lroi),
        _roi_to_runtime_detection(rroi),
        0,
        0,
        pair_gate,
    )

    roi_debug = draw_roi_debug(left_rect, right_rect, lroi, rroi)
    cv2.imwrite(str(out_dir / "rectified_roi_debug.png"), roi_debug)

    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast for feature extraction/matching only.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_eq = clahe.apply(left_gray)
    right_eq = clahe.apply(right_gray)
    left_lab = cv2.cvtColor(left_rect, cv2.COLOR_BGR2LAB)
    right_lab = cv2.cvtColor(right_rect, cv2.COLOR_BGR2LAB)
    left_lab[:, :, 0] = left_eq
    right_lab[:, :, 0] = right_eq

    left_labels = _volleyball_label_map(left_rect, lroi.mask)
    right_labels = _volleyball_label_map(right_rect, rroi.mask)
    left_edge_color = _color_edge_mask(left_rect, lroi.mask, args.edge_percentile)
    right_edge_color = _color_edge_mask(right_rect, rroi.mask, args.edge_percentile)
    left_overlap, right_overlap = _overlap_masks_for_disparity(lroi.mask, rroi.mask, initial_disp)
    cv2.imwrite(str(out_dir / "iou_overlap_debug.png"),
                draw_overlap_debug(left_rect, right_rect, left_overlap, right_overlap))

    results: List[MatchResult] = []
    method_elapsed_ms: Dict[str, float] = {}
    for name in ["orb", "brisk", "akaze", "sift"]:
        method_start = time.perf_counter()
        res = descriptor_match(left_eq, right_eq, lroi.mask, rroi.mask, name, initial_disp)
        res.depth_m = depth_from_disparity(res.disparity, focal_px, baseline_m)
        method_elapsed_ms[res.name] = (time.perf_counter() - method_start) * 1000.0
        results.append(res)

    corner_kps = _masked_keypoints(left_eq, lroi.mask, "corner", 80)
    edge_kps = _masked_keypoints(left_eq, lroi.mask, "edge", 80)
    color_edge_kps = _masked_color_edge_keypoints(left_rect, lroi.mask, 80, args.edge_percentile)
    iou_region_kps = _masked_color_edge_keypoints(left_rect, left_overlap, args.iou_max_points, args.edge_percentile)
    method_start = time.perf_counter()
    iou_res = iou_region_color_patch_match(
        left_lab, right_lab,
        left_labels, right_labels,
        left_edge_color, right_edge_color,
        lroi.mask, rroi.mask,
        left_overlap, right_overlap,
        iou_region_kps,
        initial_disp,
        patch_radius=args.iou_patch_radius,
        search_radius=args.iou_search_radius,
        y_radius=args.iou_y_radius,
        min_score=args.iou_min_score,
        reverse_tolerance_px=args.iou_reverse_tolerance_px,
        max_points=args.iou_max_points,
    )
    iou_res.depth_m = depth_from_disparity(iou_res.disparity, focal_px, baseline_m)
    method_elapsed_ms[iou_res.name] = (time.perf_counter() - method_start) * 1000.0
    results.append(iou_res)

    for name, points, lfeat, rfeat, min_score in [
        ("patch_iou_zncc_corner", corner_kps, left_eq, right_eq, 0.40),
        ("patch_iou_zncc_edge", edge_kps, left_eq, right_eq, 0.40),
        ("patch_iou_color_edge", color_edge_kps, left_lab, right_lab, 0.44),
    ]:
        method_start = time.perf_counter()
        res = patch_iou_zncc_match(
            lfeat, rfeat, lroi.mask, rroi.mask, points, initial_disp, name, min_score=min_score
        )
        res.depth_m = depth_from_disparity(res.disparity, focal_px, baseline_m)
        method_elapsed_ms[res.name] = (time.perf_counter() - method_start) * 1000.0
        results.append(res)

    for res in results:
        draw_matches(left_rect, right_rect, res, out_dir / f"{res.name}_matches.png")
        draw_matches_zoom(left_rect, right_rect, lroi, rroi, res, out_dir / f"{res.name}_matches_zoom.png")

    triangulated_rows_by_method: Dict[str, List[Dict[str, float | int | str]]] = {}
    all_triangulated_rows: List[Dict[str, float | int | str]] = []
    ball_center_3d = estimate_ball_center_3d(calib, lroi, initial_disp, baseline_m)
    ball_radius_m = 0.5 * args.ball_diameter_m
    for res in results:
        tri_rows = triangulate_match_rows(res, calib, baseline_m)
        tri_rows = validate_triangulated_rows(
            tri_rows,
            lroi,
            rroi,
            left_overlap,
            right_overlap,
            ball_center_3d,
            ball_radius_m,
            thresholds,
        )
        triangulated_rows_by_method[res.name] = tri_rows
        all_triangulated_rows.extend(tri_rows)
        write_triangulated_points(out_dir / f"{res.name}_triangulated_points.csv", tri_rows)
    write_triangulated_points(out_dir / "triangulated_points.csv", all_triangulated_rows)

    rows = []
    for res in results:
        tri_stats = triangulation_stats(triangulated_rows_by_method.get(res.name, []))
        validation_status, validation_fail_reasons = method_validation_status(tri_stats, thresholds)
        runtime_feature_ok, runtime_anchor_x, runtime_anchor_y = (
            runtime_feature_geometry_status(
                res,
                lroi,
                rroi,
                initial_disp,
                focal_px,
                baseline_m,
                feature_gate,
            )
        )
        rows.append({
            "method": res.name,
            "left_keypoints": len(res.left_keypoints),
            "right_keypoints": len(res.right_keypoints),
            "candidates": res.candidates,
            "matches": len(res.matches),
            "disparity_px": res.disparity,
            "std_px": res.std_px,
            "depth_m": res.depth_m,
            "confidence": res.confidence,
            "elapsed_ms": method_elapsed_ms.get(res.name, 0.0),
            "validation_status": validation_status,
            "validation_fail_reasons": validation_fail_reasons,
            "runtime_feature_geometry_ok": runtime_feature_ok,
            "runtime_feature_anchor_x": runtime_anchor_x,
            "runtime_feature_anchor_y": runtime_anchor_y,
            **tri_stats,
        })

    summary = {
        "left_image": args.left,
        "right_image": args.right,
        "calibration": args.calib,
        "focal_px": focal_px,
        "baseline_m": baseline_m,
        "ball_diameter_m": args.ball_diameter_m,
        "ball_center_3d_m": {
            "x": ball_center_3d[0],
            "y": ball_center_3d[1],
            "z": ball_center_3d[2],
            "source": "roi_center_disparity",
        },
        "validation_thresholds": {
            "min_valid_matches": thresholds.min_valid_matches,
            "max_y_error_px": thresholds.max_y_error_px,
            "max_disparity_mad_px": thresholds.max_disparity_mad_px,
            "max_disparity_range_px": thresholds.max_disparity_range_px,
            "max_z_mad_m": thresholds.max_z_mad_m,
            "max_z_range_m": thresholds.max_z_range_m,
            "max_sphere_residual_m": thresholds.max_sphere_residual_m,
            "max_depth_vs_center_m": thresholds.max_depth_vs_center_m,
        },
        "runtime_contract": {
            "roi_pair_gate": {
                "max_disparity": pair_gate.max_disparity,
                "epipolar_y_tolerance": pair_gate.epipolar_y_tolerance,
                "max_size_ratio": pair_gate.max_size_ratio,
                "adaptive_y_ratio": pair_gate.adaptive_y_ratio,
                "min_shifted_iou": pair_gate.min_shifted_iou,
                "min_depth_m": args.min_depth_m,
                "max_depth_m": args.max_depth_m,
            },
            "roi_pair": {
                "valid": runtime_pair is not None,
                "reject_reason": runtime_pair_reject,
                "initial_disparity_px": runtime_pair.initial_disparity if runtime_pair else initial_disp,
                "epipolar_dy_px": runtime_pair.epipolar_dy if runtime_pair else abs(lroi.center[1] - rroi.center[1]),
                "shifted_bbox_iou": runtime_pair.shifted_bbox_iou if runtime_pair else 0.0,
                "score": runtime_pair.score if runtime_pair else None,
            },
            "depth_candidate_priority": list(DEPTH_CANDIDATE_PRIORITY),
            "stereo_depth_source": STEREO_DEPTH_SOURCE,
            "feature_validation_gate": {
                "min_support": feature_gate.min_support,
                "max_stddev_px": feature_gate.max_stddev_px,
                "feature_y_tolerance_px": feature_gate.feature_y_tolerance_px,
                "feature_y_slope": feature_gate.feature_y_slope,
                "feature_y_offset_px": feature_gate.feature_y_offset_px,
                "feature_overlap_scale": feature_gate.feature_overlap_scale,
                "feature_sphere_radius_m": feature_gate.feature_sphere_radius_m,
                "feature_sphere_radius_scale": feature_gate.feature_sphere_radius_scale,
                "feature_sphere_margin_m": feature_gate.feature_sphere_margin_m,
            },
        },
        "matcher_params": {
            "edge_percentile": args.edge_percentile,
            "iou_patch_radius": args.iou_patch_radius,
            "iou_search_radius": args.iou_search_radius,
            "iou_y_radius": args.iou_y_radius,
            "iou_min_score": args.iou_min_score,
            "iou_reverse_tolerance_px": args.iou_reverse_tolerance_px,
            "iou_max_points": args.iou_max_points,
        },
        "initial_disparity_px": initial_disp,
        "initial_depth_m": initial_depth,
        "left_roi": {"bbox": lroi.bbox, "center": lroi.center, "radius": lroi.radius, "source": lroi.source},
        "right_roi": {"bbox": rroi.bbox, "center": rroi.center, "radius": rroi.radius, "source": rroi.source},
        "results": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Contact sheet, cropped around the ball for quick visual inspection.
    sheets = []
    for res in results:
        img = cv2.imread(str(out_dir / f"{res.name}_matches.png"))
        if img is not None:
            sheets.append(img)
    if sheets:
        target_w = sheets[0].shape[1]
        normalized = []
        for img in sheets:
            if img.shape[1] != target_w:
                scale = target_w / img.shape[1]
                img = cv2.resize(img, (target_w, max(1, round(img.shape[0] * scale))))
            normalized.append(img)
        cv2.imwrite(str(out_dir / "contact_sheet.png"), cv2.vconcat(normalized))

    zoom_sheets = []
    for res in results:
        img = cv2.imread(str(out_dir / f"{res.name}_matches_zoom.png"))
        if img is not None:
            zoom_sheets.append(img)
    if zoom_sheets:
        target_w = zoom_sheets[0].shape[1]
        normalized = []
        for img in zoom_sheets:
            if img.shape[1] != target_w:
                scale = target_w / img.shape[1]
                img = cv2.resize(img, (target_w, max(1, round(img.shape[0] * scale))))
            normalized.append(img)
        cv2.imwrite(str(out_dir / "zoom_contact_sheet.png"), cv2.vconcat(normalized))

    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
