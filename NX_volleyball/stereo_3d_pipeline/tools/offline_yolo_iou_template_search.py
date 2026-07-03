"""Template-search fallback helpers for offline YOLO/IoU regression."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from stereo_feature_matching.realtime_contract import Detection


@dataclass
class TemplateSearchResult:
    valid: bool = False
    x: float = 0.0
    y: float = 0.0
    score: float = -2.0
    second_score: float = -2.0
    elapsed_ms: float = 0.0


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch_f = patch.astype(np.float32)
    return (patch_f - float(patch_f.mean())) / (float(patch_f.std()) + 1e-6)


def template_search_gray(
    source: np.ndarray,
    target: np.ndarray,
    source_det: Detection,
    predicted_cx: float,
    predicted_cy: float,
    patch_radius: int,
    search_margin_px: float,
    y_tolerance_px: float,
    min_score: float,
    peak_exclusion_radius: int,
) -> TemplateSearchResult:
    start = time.perf_counter()
    result = TemplateSearchResult()
    h, w = target.shape[:2]
    pr = int(max(3, patch_radius))
    sx = int(round(source_det.cx))
    sy = int(round(source_det.cy))
    if sx - pr < 0 or sx + pr >= source.shape[1] or sy - pr < 0 or sy + pr >= source.shape[0]:
        return result

    source_patch = _normalize_patch(source[sy - pr : sy + pr + 1, sx - pr : sx + pr + 1])
    x1 = max(pr, int(math.floor(predicted_cx - search_margin_px)))
    x2 = min(w - pr - 1, int(math.ceil(predicted_cx + search_margin_px)))
    y1 = max(pr, int(math.floor(predicted_cy - y_tolerance_px)))
    y2 = min(h - pr - 1, int(math.ceil(predicted_cy + y_tolerance_px)))
    if x1 >= x2 or y1 >= y2:
        return result

    best_score = -2.0
    second_score = -2.0
    best_xy: Tuple[int, int] | None = None
    step = 2 if (x2 - x1) * (y2 - y1) > 7000 else 1
    for y in range(y1, y2 + 1, step):
        for x in range(x1, x2 + 1, step):
            target_patch = _normalize_patch(target[y - pr : y + pr + 1, x - pr : x + pr + 1])
            score = float(np.mean(source_patch * target_patch))
            if score > best_score:
                second_score = best_score
                best_score = score
                best_xy = (x, y)
            elif score > second_score:
                second_score = score

    if best_xy is None:
        return result

    bx, by = best_xy
    for y in range(max(y1, by - step), min(y2, by + step) + 1):
        for x in range(max(x1, bx - step), min(x2, bx + step) + 1):
            target_patch = _normalize_patch(target[y - pr : y + pr + 1, x - pr : x + pr + 1])
            score = float(np.mean(source_patch * target_patch))
            if score > best_score:
                second_score = best_score
                best_score = score
                bx, by = x, y
            elif score > second_score and (x != bx or y != by):
                second_score = score

    # Adjacent pixels near the best response do not indicate a distinct
    # competing match, so measure ambiguity outside a small exclusion radius.
    second_score = -2.0
    exclusion = max(1, int(peak_exclusion_radius))
    for y in range(y1, y2 + 1, step):
        for x in range(x1, x2 + 1, step):
            if math.hypot(float(x - bx), float(y - by)) < exclusion:
                continue
            target_patch = _normalize_patch(target[y - pr : y + pr + 1, x - pr : x + pr + 1])
            score = float(np.mean(source_patch * target_patch))
            if score > second_score:
                second_score = score

    result.elapsed_ms = (time.perf_counter() - start) * 1000.0
    result.x = float(bx)
    result.y = float(by)
    result.score = best_score
    result.second_score = second_score
    result.valid = best_score >= min_score
    return result
