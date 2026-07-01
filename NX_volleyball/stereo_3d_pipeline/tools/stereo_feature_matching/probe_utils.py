"""Small reusable helpers for offline stereo feature probes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .geometry import FilteredMatch


@dataclass
class CropTransform:
    x1: int
    y1: int
    width: int
    height: int
    output_size: int

    def to_global(self, x: float, y: float) -> Tuple[float, float]:
        return (
            self.x1 + x * (self.width / float(self.output_size)),
            self.y1 + y * (self.height / float(self.output_size)),
        )


def crop_square(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    pad: int,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray, CropTransform]:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    cx = x + bw * 0.5
    cy = y + bh * 0.5
    side = int(round(max(bw, bh) + 2 * pad))
    side = max(side, 16)
    x1 = int(round(cx - side * 0.5))
    y1 = int(round(cy - side * 0.5))
    x1 = max(0, min(x1, max(0, w - side)))
    y1 = max(0, min(y1, max(0, h - side)))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        raise RuntimeError("empty ROI crop")
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return resized, resized_mask, CropTransform(x1, y1, x2 - x1, y2 - y1, output_size)


def write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inside_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    ix = int(round(x))
    iy = int(round(y))
    return 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0


def filter_matches_by_roi_masks(
    matches: List[FilteredMatch],
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> List[FilteredMatch]:
    return [
        m for m in matches
        if inside_mask(left_mask, m.left_xy[0], m.left_xy[1]) and
           inside_mask(right_mask, m.right_xy[0], m.right_xy[1])
    ]
