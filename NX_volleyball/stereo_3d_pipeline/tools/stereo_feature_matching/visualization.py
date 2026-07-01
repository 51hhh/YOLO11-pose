"""Conversion helpers between neural matches and the existing probe output."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .common import FeatureSet
from .geometry import FilteredMatch


def cv_keypoints_from_global(features: FeatureSet, map_to_global, size: float = 7.0):
    out = []
    for x, y in features.keypoints:
        gx, gy = map_to_global(float(x), float(y))
        out.append(cv2.KeyPoint(float(gx), float(gy), size))
    return out


def cv_matches_from_filtered(matches: Sequence[FilteredMatch]):
    return [
        cv2.DMatch(int(m.query_idx), int(m.train_idx), float(max(0.0, 1.0 - m.score)))
        for m in matches
    ]


def draw_crop_debug(left_crop: np.ndarray, right_crop: np.ndarray, path) -> None:
    h = max(left_crop.shape[0], right_crop.shape[0])
    def pad(img):
        if img.shape[0] == h:
            return img
        bottom = h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, bottom, 0, 0, cv2.BORDER_CONSTANT)
    cv2.imwrite(str(path), np.hstack([pad(left_crop), pad(right_crop)]))
