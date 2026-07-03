"""Visualization helpers for offline volleyball probes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from offline_volleyball_probe_roi import BallROI, MatchResult


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
