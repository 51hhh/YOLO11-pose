"""Feature image preparation and method execution for keypoint probes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from offline_volleyball_probe_roi import BallROI, MatchResult
from offline_volleyball_probe_triangulation import depth_from_disparity
from offline_volleyball_probe_matching import (
    _color_edge_mask,
    _masked_color_edge_keypoints,
    _masked_keypoints,
    _overlap_masks_for_disparity,
    _volleyball_label_map,
    descriptor_match,
    draw_overlap_debug,
    iou_region_color_patch_match,
    patch_iou_zncc_match,
)


def prepare_keypoint_feature_inputs(
    left_rect,
    right_rect,
    lroi: BallROI,
    rroi: BallROI,
    initial_disp: float,
    args,
    out_dir: Path,
) -> Tuple[Dict[str, object], object, object]:
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast for feature extraction/matching only.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_eq = clahe.apply(left_gray)
    right_eq = clahe.apply(right_gray)
    left_lab = cv2.cvtColor(left_rect, cv2.COLOR_BGR2LAB)
    right_lab = cv2.cvtColor(right_rect, cv2.COLOR_BGR2LAB)
    left_lab[:, :, 0] = left_eq
    right_lab[:, :, 0] = right_eq

    left_labels = _volleyball_label_map(left_rect, lroi.mask)
    right_labels = _volleyball_label_map(right_rect, rroi.mask)
    left_edge_color = _color_edge_mask(left_rect, lroi.mask, args.edge_percentile)
    right_edge_color = _color_edge_mask(right_rect, rroi.mask, args.edge_percentile)
    left_overlap, right_overlap = _overlap_masks_for_disparity(lroi.mask, rroi.mask, initial_disp)
    cv2.imwrite(
        str(out_dir / "iou_overlap_debug.png"),
        draw_overlap_debug(left_rect, right_rect, left_overlap, right_overlap),
    )

    features: Dict[str, object] = {
        "left_rect": left_rect,
        "left_eq": left_eq,
        "right_eq": right_eq,
        "left_lab": left_lab,
        "right_lab": right_lab,
        "left_labels": left_labels,
        "right_labels": right_labels,
        "left_edge_color": left_edge_color,
        "right_edge_color": right_edge_color,
    }
    return features, left_overlap, right_overlap


def run_keypoint_probe_methods(
    features: Dict[str, object],
    lroi: BallROI,
    rroi: BallROI,
    left_overlap,
    right_overlap,
    initial_disp: float,
    focal_px: float,
    baseline_m: float,
    args,
) -> Tuple[List[MatchResult], Dict[str, float]]:
    left_eq = features["left_eq"]
    right_eq = features["right_eq"]
    left_lab = features["left_lab"]
    right_lab = features["right_lab"]

    results: List[MatchResult] = []
    method_elapsed_ms: Dict[str, float] = {}
    for name in ["orb", "brisk", "akaze", "sift"]:
        method_start = time.perf_counter()
        res = descriptor_match(left_eq, right_eq, lroi.mask, rroi.mask, name, initial_disp)
        res.depth_m = depth_from_disparity(res.disparity, focal_px, baseline_m)
        method_elapsed_ms[res.name] = (time.perf_counter() - method_start) * 1000.0
        results.append(res)

    corner_kps = _masked_keypoints(left_eq, lroi.mask, "corner", 80)
    edge_kps = _masked_keypoints(left_eq, lroi.mask, "edge", 80)
    color_edge_kps = _masked_color_edge_keypoints(features["left_rect"], lroi.mask, 80, args.edge_percentile)
    iou_region_kps = _masked_color_edge_keypoints(
        features["left_rect"], left_overlap, args.iou_max_points, args.edge_percentile
    )
    method_start = time.perf_counter()
    iou_res = iou_region_color_patch_match(
        left_lab,
        right_lab,
        features["left_labels"],
        features["right_labels"],
        features["left_edge_color"],
        features["right_edge_color"],
        lroi.mask,
        rroi.mask,
        left_overlap,
        right_overlap,
        iou_region_kps,
        initial_disp,
        patch_radius=args.iou_patch_radius,
        search_radius=args.iou_search_radius,
        y_radius=args.iou_y_radius,
        min_score=args.iou_min_score,
        reverse_tolerance_px=args.iou_reverse_tolerance_px,
        max_points=args.iou_max_points,
    )
    iou_res.depth_m = depth_from_disparity(iou_res.disparity, focal_px, baseline_m)
    method_elapsed_ms[iou_res.name] = (time.perf_counter() - method_start) * 1000.0
    results.append(iou_res)

    for name, points, lfeat, rfeat, min_score in [
        ("patch_iou_zncc_corner", corner_kps, left_eq, right_eq, 0.40),
        ("patch_iou_zncc_edge", edge_kps, left_eq, right_eq, 0.40),
        ("patch_iou_color_edge", color_edge_kps, left_lab, right_lab, 0.44),
    ]:
        method_start = time.perf_counter()
        res = patch_iou_zncc_match(
            lfeat, rfeat, lroi.mask, rroi.mask, points, initial_disp, name, min_score=min_score
        )
        res.depth_m = depth_from_disparity(res.disparity, focal_px, baseline_m)
        method_elapsed_ms[res.name] = (time.perf_counter() - method_start) * 1000.0
        results.append(res)

    return results, method_elapsed_ms
