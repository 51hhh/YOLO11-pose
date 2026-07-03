"""Traditional descriptor matching helpers for offline volleyball probes."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from offline_volleyball_probe_roi import MatchResult


def descriptor_match(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    name: str,
    initial_disp: float,
    max_features: int = 300,
) -> MatchResult:
    if name == "orb":
        extractor = cv2.ORB_create(max_features, 1.2, 4, 12, 0, 2, cv2.ORB_HARRIS_SCORE, 17, 10)
        norm = cv2.NORM_HAMMING
    elif name == "brisk":
        extractor = cv2.BRISK_create(18, 2, 1.0)
        norm = cv2.NORM_HAMMING
    elif name == "akaze":
        extractor = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif name == "sift":
        extractor = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.015, edgeThreshold=12)
        norm = cv2.NORM_L2
    else:
        raise ValueError(name)

    lkps, ldesc = extractor.detectAndCompute(left_gray, left_mask)
    rkps, rdesc = extractor.detectAndCompute(right_gray, right_mask)
    lkps = list(lkps or [])
    rkps = list(rkps or [])
    if ldesc is None or rdesc is None or not lkps or not rkps:
        return MatchResult(name, lkps, rkps, [], 0, -1.0, -1.0, -1.0, 0.0)

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(ldesc, rdesc, k=2)
    candidates: List[cv2.DMatch] = []
    for pair in knn:
        if not pair:
            continue
        best = pair[0]
        if len(pair) > 1:
            ratio = 0.76 if norm == cv2.NORM_HAMMING else 0.72
            if best.distance >= ratio * pair[1].distance:
                continue
        lpt = lkps[best.queryIdx].pt
        rpt = rkps[best.trainIdx].pt
        disp = lpt[0] - rpt[0]
        if abs(lpt[1] - rpt[1]) > 4.0:
            continue
        if disp <= 0.0:
            continue
        if abs(disp - initial_disp) > 35.0:
            continue
        candidates.append(best)

    if not candidates:
        return MatchResult(name, lkps, rkps, [], 0, -1.0, -1.0, -1.0, 0.0)

    disps = np.array([lkps[m.queryIdx].pt[0] - rkps[m.trainIdx].pt[0] for m in candidates],
                     dtype=np.float32)
    median = float(np.median(disps))
    abs_dev = np.abs(disps - median)
    gate = max(1.5, 2.5 * float(np.median(abs_dev)))
    matches = [m for m, e in zip(candidates, abs_dev) if e <= gate]
    if not matches:
        return MatchResult(name, lkps, rkps, [], len(candidates), -1.0, -1.0, -1.0, 0.0)
    inlier_disps = np.array([lkps[m.queryIdx].pt[0] - rkps[m.trainIdx].pt[0] for m in matches],
                            dtype=np.float32)
    disparity = float(np.mean(inlier_disps))
    std_px = float(np.std(inlier_disps))
    confidence = float(np.clip(len(matches) / 12.0 / (1.0 + std_px), 0.0, 1.0))
    return MatchResult(name, lkps, rkps, matches, len(candidates), disparity, std_px, -1.0, confidence)
