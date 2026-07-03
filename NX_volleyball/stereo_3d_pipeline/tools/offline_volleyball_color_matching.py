"""Color-edge and label-IoU matching helpers for offline volleyball probes."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from offline_volleyball_color_features import (
    _color_edge_mask,
    _masked_color_edge_keypoints,
    _volleyball_label_map,
    _weighted_label_iou,
)
from offline_volleyball_patch_scores import _patch_iou, _zncc
from offline_volleyball_probe_roi import MatchResult


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
