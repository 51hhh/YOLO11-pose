"""Keypoint, patch, and descriptor matching helpers for offline volleyball probes."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from offline_volleyball_probe_roi import MatchResult


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
        # Prefer high-gradient, spatially spread points.
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
    labels[mask > 0] = 1  # dark/other ball surface
    yellow = (hue >= 18) & (hue <= 48) & (sat >= 45) & (val >= 35) & (mask > 0)
    blue = (hue >= 85) & (hue <= 140) & (sat >= 28) & (val >= 20) & (mask > 0)
    white = (sat <= 85) & (val >= 85) & (luma >= 80) & (mask > 0)
    labels[yellow] = 2
    labels[blue] = 3
    labels[white] = 4
    return labels


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


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aa -= float(aa.mean())
    bb -= float(bb.mean())
    denom = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    if denom < 1e-6:
        return -1.0
    return float((aa * bb).sum() / denom)


def _patch_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


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


def iou_region_color_patch_match(
    left_feat: np.ndarray,
    right_feat: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    left_keypoints: List[cv2.KeyPoint],
    initial_disp: float,
    name: str = "iou_region_color_patch",
    patch_radius: int = 9,
    search_radius: int = 28,
    y_radius: int = 2,
    min_score: float = 0.58,
    reverse_tolerance_px: float = 3.0,
    max_points: int = 90,
) -> MatchResult:
    points = left_keypoints[:max_points]
    candidates: List[Tuple[int, int, int, float, float, float, float, float, float]] = []

    for qi, kp in enumerate(points):
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if left_overlap[y, x] == 0:
            continue
        best = _best_iou_region_candidate(
            left_feat, right_feat,
            left_labels, right_labels,
            left_edge, right_edge,
            left_mask, right_mask,
            right_overlap,
            x, y, initial_disp,
            patch_radius, search_radius, y_radius,
            "left_to_right",
        )
        if best is None:
            continue
        score, xr, yr, zncc, label_iou, edge_iou, texture, uniqueness = best
        if score < min_score:
            continue
        if uniqueness < 0.004 and score < 0.80:
            continue

        reverse = _best_iou_region_candidate(
            right_feat, left_feat,
            right_labels, left_labels,
            right_edge, left_edge,
            right_mask, left_mask,
            left_overlap,
            xr, yr, initial_disp,
            patch_radius, search_radius, y_radius,
            "right_to_left",
        )
        if reverse is None:
            continue
        _, bx, by, _, _, _, _, _ = reverse
        if math.hypot(float(bx - x), float(by - y)) > reverse_tolerance_px:
            continue

        disp = float(x - xr)
        candidates.append((qi, xr, yr, score, disp, zncc, label_iou, edge_iou, texture))

    if not candidates:
        return MatchResult(name, points, [], [], 0, -1.0, -1.0, -1.0, 0.0)

    disparities = np.array([d for _, _, _, _, d, _, _, _, _ in candidates], dtype=np.float32)
    median = float(np.median(disparities))
    abs_dev = np.abs(disparities - median)
    mad = float(np.median(abs_dev))
    gate = max(1.2, 2.8 * mad)

    indexed_candidates = [
        cand for keep, cand in zip(abs_dev <= gate, candidates) if bool(keep)
    ]
    indexed_candidates.sort(key=lambda c: c[3], reverse=True)

    right_kps: List[cv2.KeyPoint] = []
    matches: List[cv2.DMatch] = []
    extras: List[Dict[str, float]] = []
    used_right: List[Tuple[int, int]] = []
    used_left: set[int] = set()
    inlier_disps: List[float] = []
    inlier_scores: List[float] = []

    for qi, xr, yr, score, disp, zncc, label_iou, edge_iou, texture in indexed_candidates:
        if qi in used_left:
            continue
        if any((xr - ux) * (xr - ux) + (yr - uy) * (yr - uy) < 5 * 5 for ux, uy in used_right):
            continue
        ti = len(right_kps)
        right_kps.append(cv2.KeyPoint(float(xr), float(yr), float(patch_radius * 2 + 1)))
        matches.append(cv2.DMatch(int(qi), int(ti), float(1.0 - score)))
        extras.append({
            "score": float(score),
            "disparity_px": float(disp),
            "zncc": float(zncc),
            "label_iou": float(label_iou),
            "edge_iou": float(edge_iou),
            "texture": float(texture),
        })
        used_left.add(qi)
        used_right.append((xr, yr))
        inlier_disps.append(float(disp))
        inlier_scores.append(float(score))

    if not matches:
        return MatchResult(name, points, right_kps, [], len(candidates), -1.0, -1.0, -1.0, 0.0)

    disparity = float(np.mean(inlier_disps))
    std_px = float(np.std(inlier_disps))
    confidence = float(np.clip(np.mean(inlier_scores) * min(1.0, len(matches) / 12.0) / (1.0 + std_px / 8.0), 0.0, 1.0))
    return MatchResult(name, points, right_kps, matches, len(candidates), disparity, std_px, -1.0, confidence, extras=extras)


def patch_iou_zncc_match(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_keypoints: List[cv2.KeyPoint],
    initial_disp: float,
    name: str,
    patch_radius: int = 9,
    search_radius: int = 22,
    y_radius: int = 3,
    min_score: float = 0.40,
    max_points: int = 80,
) -> MatchResult:
    h, w = left_gray.shape[:2]
    points = left_keypoints[:max_points]
    candidates: List[Tuple[int, int, int, float, float]] = []

    d0 = int(round(initial_disp))
    d_start = max(1, d0 - search_radius)
    d_end = min(w - 1, d0 + search_radius)

    for qi, kp in enumerate(points):
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if x - patch_radius < 0 or x + patch_radius >= w:
            continue
        if y - patch_radius < 0 or y + patch_radius >= h:
            continue
        if left_mask[y, x] == 0:
            continue

        lpatch = left_gray[y - patch_radius : y + patch_radius + 1,
                           x - patch_radius : x + patch_radius + 1]
        lmpatch = left_mask[y - patch_radius : y + patch_radius + 1,
                            x - patch_radius : x + patch_radius + 1]
        best: Tuple[float, int, int, float, float] | None = None
        second_score = -1.0
        for dy in range(-y_radius, y_radius + 1):
            yr = y + dy
            if yr - patch_radius < 0 or yr + patch_radius >= h:
                continue
            for disp in range(d_start, d_end + 1):
                xr = x - disp
                if xr - patch_radius < 0 or xr + patch_radius >= w:
                    continue
                rpatch = right_gray[yr - patch_radius : yr + patch_radius + 1,
                                    xr - patch_radius : xr + patch_radius + 1]
                rmpatch = right_mask[yr - patch_radius : yr + patch_radius + 1,
                                     xr - patch_radius : xr + patch_radius + 1]
                zncc = _zncc(lpatch, rpatch)
                iou = _patch_iou(lmpatch, rmpatch)
                # IoU alone fails on smooth regions; ZNCC alone drifts on dark curtains.
                score = 0.62 * max(0.0, zncc) + 0.38 * iou
                if best is None or score > best[0]:
                    if best is not None:
                        second_score = max(second_score, best[0])
                    best = (score, xr, yr, zncc, iou)
                else:
                    second_score = max(second_score, score)

        if best is None:
            continue
        score, xr, yr, zncc, iou = best
        uniqueness = score - second_score if second_score >= 0 else 1.0
        if score < min_score:
            continue
        if uniqueness < 0.025 and score < 0.68:
            continue
        if right_mask[yr, xr] == 0 and iou < 0.20:
            continue
        candidates.append((qi, xr, yr, score, float(x - xr)))

    if not candidates:
        return MatchResult(name, points, [], [], 0, -1.0, -1.0, -1.0, 0.0)

    disparities = np.array([d for _, _, _, _, d in candidates], dtype=np.float32)
    median = float(np.median(disparities))
    abs_dev = np.abs(disparities - median)
    mad = float(np.median(abs_dev))
    gate = max(1.25, 2.5 * mad)
    inlier_mask = abs_dev <= gate
    indexed_candidates = [
        (idx, cand) for idx, (keep, cand) in enumerate(zip(inlier_mask, candidates)) if bool(keep)
    ]
    indexed_candidates.sort(key=lambda item: item[1][3], reverse=True)

    right_kps: List[cv2.KeyPoint] = []
    matches: List[cv2.DMatch] = []
    used_right: List[Tuple[int, int]] = []
    inlier_disps = []
    inlier_scores = []
    for _, (qi, xr, yr, score, disp) in indexed_candidates:
        if any((xr - ux) * (xr - ux) + (yr - uy) * (yr - uy) < 5 * 5 for ux, uy in used_right):
            continue
        ti = len(right_kps)
        right_kps.append(cv2.KeyPoint(float(xr), float(yr), float(patch_radius * 2 + 1)))
        matches.append(cv2.DMatch(int(qi), int(ti), float(1.0 - score)))
        used_right.append((xr, yr))
        inlier_disps.append(disp)
        inlier_scores.append(score)

    if not matches:
        return MatchResult(name, points, right_kps, [], len(candidates), -1.0, -1.0, -1.0, 0.0)

    disparity = float(np.mean(inlier_disps))
    std_px = float(np.std(inlier_disps))
    confidence = float(np.clip(np.mean(inlier_scores) * min(1.0, len(matches) / 8.0), 0.0, 1.0))
    return MatchResult(name, points, right_kps, matches, len(candidates), disparity, std_px, -1.0, confidence)


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


from offline_volleyball_probe_triangulation import (
    depth_from_disparity,
    estimate_ball_center_3d,
    method_validation_status,
    runtime_feature_geometry_status,
    triangulate_match_rows,
    triangulation_stats,
    validate_triangulated_rows,
    write_triangulated_points,
)
