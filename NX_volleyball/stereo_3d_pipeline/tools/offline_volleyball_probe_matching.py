"""Keypoint, patch, and descriptor matching helpers for offline volleyball probes."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from offline_volleyball_color_matching import (
    _color_edge_mask,
    _masked_color_edge_keypoints,
    _volleyball_label_map,
    iou_region_color_patch_match,
)
from offline_volleyball_descriptor_matching import descriptor_match
from offline_volleyball_patch_scores import _patch_iou, _zncc
from offline_volleyball_probe_roi import MatchResult


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


from offline_volleyball_probe_triangulation import (
    depth_from_disparity,
    estimate_ball_center_3d,
    method_validation_status,
    runtime_feature_geometry_status,
    triangulate_match_rows,
    triangulation_stats,
    validate_triangulated_rows,
    write_triangulated_points,
)
