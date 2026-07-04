"""Legacy one-sided color segmentation for offline volleyball ROI probes."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from offline_volleyball_probe_models import BallROI


def segment_ball(image: np.ndarray, side: str) -> BallROI:
    """Segment the visible volleyball in the saved scene."""

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    luma = lab[:, :, 0]

    yellow = (hue >= 18) & (hue <= 45) & (sat >= 45) & (val >= 45)
    blue = (hue >= 88) & (hue <= 135) & (sat >= 35) & (val >= 25)
    white = (sat <= 80) & (val >= 85) & (luma >= 75)
    chroma = ((yellow | blue | white).astype(np.uint8)) * 255

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

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    roi_mask &= np.where(np.indices((h, w))[0] >= y1, 255, 0).astype(np.uint8)
    roi_mask &= np.where(np.indices((h, w))[0] <= y2, 255, 0).astype(np.uint8)
    roi_mask &= np.where(np.indices((h, w))[1] >= x1, 255, 0).astype(np.uint8)
    roi_mask &= np.where(np.indices((h, w))[1] <= x2, 255, 0).astype(np.uint8)

    return BallROI((x1, y1, x2 - x1 + 1, y2 - y1 + 1), (cx, cy), float(r), roi_mask)
