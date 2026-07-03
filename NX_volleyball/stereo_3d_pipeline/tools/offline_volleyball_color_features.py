"""Color-edge and label-map helpers for offline volleyball probes."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from offline_volleyball_patch_scores import _patch_iou


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
    labels[mask > 0] = 1
    yellow = (hue >= 18) & (hue <= 48) & (sat >= 45) & (val >= 35) & (mask > 0)
    blue = (hue >= 85) & (hue <= 140) & (sat >= 28) & (val >= 20) & (mask > 0)
    white = (sat <= 85) & (val >= 85) & (luma >= 80) & (mask > 0)
    labels[yellow] = 2
    labels[blue] = 3
    labels[white] = 4
    return labels


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
