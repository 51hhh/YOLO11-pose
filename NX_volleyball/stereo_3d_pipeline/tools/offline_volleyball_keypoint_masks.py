"""Masked keypoint and overlap-mask helpers for offline volleyball probes."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


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
