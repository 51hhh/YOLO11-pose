"""Appearance patch scoring for offline volleyball color matching."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from offline_volleyball_color_features import _weighted_label_iou
from offline_volleyball_patch_scores import _patch_iou, _zncc


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
