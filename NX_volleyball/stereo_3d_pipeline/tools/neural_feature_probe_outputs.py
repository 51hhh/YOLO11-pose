"""Output helpers for the offline neural feature probe."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

import offline_volleyball_keypoint_probe as probe
from stereo_feature_matching.visualization import (
    cv_keypoints_from_global,
    cv_matches_from_filtered,
)


def build_match_result(
    name: str,
    left_features,
    right_features,
    filtered,
    attempted: int,
    depth_m: float,
    timings_ms: Dict[str, float],
    left_to_global: Callable[[float, float], Tuple[float, float]],
    right_to_global: Callable[[float, float], Tuple[float, float]],
) -> probe.MatchResult:
    disparities = np.asarray([match.disparity for match in filtered], dtype=np.float32)
    if disparities.size:
        disparity = float(np.median(disparities))
        std_px = float(np.std(disparities))
        confidence = float(np.clip(len(filtered) / 12.0 / (1.0 + std_px), 0.0, 1.0))
    else:
        disparity = -1.0
        std_px = -1.0
        confidence = 0.0

    notes = ",".join(f"{key}={value:.3f}ms" for key, value in timings_ms.items())
    return probe.MatchResult(
        name=name,
        left_keypoints=cv_keypoints_from_global(left_features, left_to_global),
        right_keypoints=cv_keypoints_from_global(right_features, right_to_global),
        matches=cv_matches_from_filtered(filtered),
        candidates=attempted,
        disparity=disparity,
        std_px=std_px,
        depth_m=depth_m if disparity > 0.0 else -1.0,
        confidence=confidence,
        notes=notes,
    )


def write_zoom_contact_sheet(out_dir: Path, results: List[probe.MatchResult]) -> None:
    zoom_sheets = []
    for res in results:
        img = cv2.imread(str(out_dir / f"{res.name}_matches_zoom.png"))
        if img is not None:
            zoom_sheets.append(img)
    if not zoom_sheets:
        return

    target_w = zoom_sheets[0].shape[1]
    normalized = []
    for img in zoom_sheets:
        if img.shape[1] != target_w:
            scale = target_w / img.shape[1]
            img = cv2.resize(img, (target_w, max(1, round(img.shape[0] * scale))))
        normalized.append(img)
    cv2.imwrite(str(out_dir / "zoom_contact_sheet.png"), cv2.vconcat(normalized))
