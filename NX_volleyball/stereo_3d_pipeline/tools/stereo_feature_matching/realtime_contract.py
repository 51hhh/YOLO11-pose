"""Runtime stereo matching contract mirrored by offline CPU probes.

This module intentionally mirrors the lightweight C++ contract in
src/stereo/depth_match_contract.*.  It should not contain feature extractor
implementation details; offline probes use it to apply the same ROI pairing,
candidate naming, and priority semantics as the realtime pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEPTH_CANDIDATE_PRIORITY: Tuple[str, ...] = (
    "roi_multi_point",
    "fallback_feature_points",
    "fallback_template",
    "fallback_epipolar",
    "circle_center",
    "roi_center_patch",
    "roi_iou_region_color_patch",
    "roi_patch_iou_color_edge",
    "roi_neural_feature",
    "roi_corner_points",
    "roi_texture_points",
    "roi_binary_points",
    "roi_orb_points",
    "roi_brisk_points",
    "roi_akaze_points",
    "roi_sift_points",
    "roi_radial_center",
    "roi_edge_pair_center",
    "roi_edge_centroid",
    "circle_left_edge",
    "circle_right_edge",
    "bbox_center",
    "bbox_edges",
    "bbox_left_edge",
    "bbox_right_edge",
)


STEREO_DEPTH_SOURCE: Dict[str, int] = {
    "circle_center": 1,
    "circle_left_edge": 1,
    "circle_right_edge": 1,
    "fallback_epipolar": 1,
    "roi_multi_point": 2,
    "bbox_center": 3,
    "roi_center_patch": 4,
    "roi_edge_centroid": 5,
    "bbox_left_edge": 6,
    "bbox_right_edge": 6,
    "bbox_edges": 6,
    "fallback_template": 7,
    "roi_radial_center": 8,
    "roi_edge_pair_center": 9,
    "roi_corner_points": 10,
    "roi_texture_points": 11,
    "fallback_feature_points": 12,
    "roi_binary_points": 13,
    "roi_orb_points": 14,
    "roi_brisk_points": 15,
    "roi_akaze_points": 16,
    "roi_sift_points": 17,
    "roi_iou_region_color_patch": 18,
    "roi_patch_iou_color_edge": 19,
    "roi_neural_feature": 20,
}


@dataclass(frozen=True)
class Detection:
    cx: float
    cy: float
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 0


@dataclass(frozen=True)
class StereoRoiPairGateConfig:
    max_disparity: int = 2048
    epipolar_y_tolerance: float = 12.0
    max_size_ratio: float = 2.0
    adaptive_y_ratio: float = 0.35
    min_shifted_iou: float = 0.0


@dataclass(frozen=True)
class BboxDisparityPriorConfig:
    object_diameter_m: float = 0.200
    bbox_scale: float = 0.95
    consistency_ratio: float = 0.30
    consistency_min_px: float = 45.0
    penalty_scale: float = 0.75


@dataclass(frozen=True)
class StereoRoiPair:
    left_index: int
    right_index: int
    left: Detection
    right: Detection
    initial_disparity: float
    epipolar_dy: float
    y_tolerance: float
    width_ratio: float
    height_ratio: float
    size_ratio: float
    shifted_bbox_iou: float
    score: float
    semantic_confidence: float


@dataclass(frozen=True)
class DepthCandidateObservation:
    method: str
    disparity_px: float
    depth_m: float
    confidence: float = 0.0
    fusion_confidence: float = 1.0
    stddev_px: float = -1.0
    support: int = 0
    anchor_left_x: float = 0.0
    anchor_left_y: float = 0.0

    @property
    def stereo_depth_source(self) -> int:
        return STEREO_DEPTH_SOURCE.get(self.method, 0)

    @property
    def usable(self) -> bool:
        return (
            self.stereo_depth_source > 0
            and math.isfinite(self.disparity_px)
            and math.isfinite(self.depth_m)
            and self.disparity_px > 0.0
            and self.depth_m > 0.0
        )


@dataclass(frozen=True)
class FeatureValidationConfig:
    min_support: int = 4
    max_stddev_px: float = 1.0
    feature_y_tolerance_px: float = 2.0
    feature_y_slope: float = 0.0
    feature_y_offset_px: float = 0.0
    feature_overlap_scale: float = 0.55
    feature_sphere_radius_m: float = 0.10
    feature_sphere_radius_scale: float = 1.8
    feature_sphere_margin_m: float = 0.02


@dataclass(frozen=True)
class SparseFeatureObservation:
    valid: bool
    disparity_px: float
    anchor_left_x: float
    anchor_left_y: float
    anchor_right_x: Optional[float] = None
    anchor_right_y: Optional[float] = None
    stddev_px: float = 0.0
    support: int = 0


def _rect(det: Detection, shift_x: float = 0.0) -> Tuple[float, float, float, float]:
    half_w = det.width * 0.5
    half_h = det.height * 0.5
    return (
        det.cx - half_w + shift_x,
        det.cy - half_h,
        det.cx + half_w + shift_x,
        det.cy + half_h,
    )


def _rect_area(rect: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def rect_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = _rect_area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))
    union = _rect_area(a) + _rect_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_stereo_roi_pair(
    left: Detection,
    right: Detection,
    left_index: int,
    right_index: int,
    config: StereoRoiPairGateConfig,
) -> Tuple[Optional[StereoRoiPair], str]:
    if left.class_id != right.class_id:
        return None, "class_mismatch"
    if left.width <= 1.0 or left.height <= 1.0 or right.width <= 1.0 or right.height <= 1.0:
        return None, "invalid_box"

    disparity = left.cx - right.cx
    if disparity <= 0.0:
        return None, "nonpositive_disparity"
    if disparity > float(config.max_disparity):
        return None, "over_max_disparity"

    base_y_tol = max(1.0, config.epipolar_y_tolerance)
    adaptive_y_tol = max(base_y_tol, config.adaptive_y_ratio * max(left.height, right.height))
    dy = abs(left.cy - right.cy)
    if dy > adaptive_y_tol:
        return None, "epipolar_reject"

    width_ratio = max(left.width / right.width, right.width / left.width)
    height_ratio = max(left.height / right.height, right.height / left.height)
    max_ratio = max(1.0, config.max_size_ratio)
    if width_ratio > max_ratio or height_ratio > max_ratio:
        return None, "size_reject"

    shifted_iou = rect_iou(_rect(left), _rect(right, disparity))
    if shifted_iou < max(0.0, config.min_shifted_iou):
        return None, "low_iou"

    size_cost = abs(math.log(width_ratio)) + abs(math.log(height_ratio))
    score = dy / adaptive_y_tol + size_cost - 0.25 * right.confidence
    semantic_confidence = math.sqrt(max(0.0, left.confidence * right.confidence))
    return (
        StereoRoiPair(
            left_index=left_index,
            right_index=right_index,
            left=left,
            right=right,
            initial_disparity=disparity,
            epipolar_dy=dy,
            y_tolerance=adaptive_y_tol,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            size_ratio=max(width_ratio, height_ratio),
            shifted_bbox_iou=shifted_iou,
            score=score,
            semantic_confidence=semantic_confidence,
        ),
        "none",
    )


def collect_stereo_roi_pair_candidates(
    left_detections: Sequence[Detection],
    right_detections: Sequence[Detection],
    config: StereoRoiPairGateConfig,
    max_pairs: int,
) -> List[StereoRoiPair]:
    pairs: List[StereoRoiPair] = []
    if max_pairs <= 0:
        return pairs
    for li, left in enumerate(left_detections):
        for ri, right in enumerate(right_detections):
            pair, _ = evaluate_stereo_roi_pair(left, right, li, ri, config)
            if pair is None:
                continue
            pairs.append(pair)
    pairs.sort(key=lambda pair: pair.score)
    return pairs[:max_pairs]


def find_best_stereo_roi_pair(
    left_detections: Sequence[Detection],
    right_detections: Sequence[Detection],
    config: StereoRoiPairGateConfig,
) -> Optional[StereoRoiPair]:
    best: Optional[StereoRoiPair] = None
    for li, left in enumerate(left_detections):
        for ri, right in enumerate(right_detections):
            pair, _ = evaluate_stereo_roi_pair(left, right, li, ri, config)
            if pair is None:
                continue
            if best is None or pair.score < best.score:
                best = pair
    return best


def estimate_bbox_disparity_px(
    det: Detection,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> float:
    if (
        det.width <= 1.0
        or baseline_m <= 0.0
        or prior.object_diameter_m <= 0.01
        or max_disparity <= 0
    ):
        return -1.0
    disp = baseline_m * det.width * prior.bbox_scale / prior.object_diameter_m
    return min(float(max_disparity), max(1.0, float(disp)))


def bbox_disparity_consistency_penalty(
    pair: StereoRoiPair,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> float:
    if pair.initial_disparity <= 0.0 or baseline_m <= 0.0:
        return 0.0

    left_expected = estimate_bbox_disparity_px(pair.left, baseline_m, prior, max_disparity)
    right_expected = estimate_bbox_disparity_px(pair.right, baseline_m, prior, max_disparity)
    if left_expected > 0.0 and right_expected > 0.0:
        expected = 0.5 * (left_expected + right_expected)
    elif left_expected > 0.0:
        expected = left_expected
    elif right_expected > 0.0:
        expected = right_expected
    else:
        return 0.0

    tolerance = max(
        max(5.0, prior.consistency_min_px),
        expected * max(0.05, prior.consistency_ratio),
    )
    excess = abs(pair.initial_disparity - expected) - tolerance
    if excess <= 0.0:
        return 0.0
    return max(0.0, prior.penalty_scale) * excess / max(1.0, tolerance)


def score_stereo_roi_pair_with_bbox_prior(
    pair: StereoRoiPair,
    baseline_m: float,
    prior: BboxDisparityPriorConfig,
    max_disparity: int = 2048,
) -> float:
    return pair.score + bbox_disparity_consistency_penalty(
        pair,
        baseline_m,
        prior,
        max_disparity,
    )


def select_first_usable_depth_candidate(
    candidates: Iterable[DepthCandidateObservation],
) -> Optional[DepthCandidateObservation]:
    priority = {method: idx for idx, method in enumerate(DEPTH_CANDIDATE_PRIORITY)}
    ordered = sorted(candidates, key=lambda item: priority.get(item.method, 10_000))
    for candidate in ordered:
        if candidate.usable:
            return candidate
    return None


def point_inside_detection_ellipse(
    det: Detection,
    x: float,
    y: float,
    scale: float,
) -> bool:
    if det.width <= 1.0 or det.height <= 1.0:
        return False
    rx = max(1.0, det.width * scale)
    ry = max(1.0, det.height * scale)
    nx = (x - det.cx) / rx
    ny = (y - det.cy) / ry
    return nx * nx + ny * ny <= 1.0


def validate_sparse_feature_geometry(
    observation: SparseFeatureObservation,
    left_det: Detection,
    right_det: Detection,
    initial_disparity: float,
    config: FeatureValidationConfig,
    focal_px: float,
    baseline_m: float,
) -> bool:
    if (
        not observation.valid
        or observation.disparity_px <= 0.5
        or initial_disparity <= 0.5
        or focal_px <= 1e-3
        or baseline_m <= 1e-6
    ):
        return False
    if observation.support < max(1, config.min_support):
        return False
    if observation.stddev_px > max(0.05, config.max_stddev_px):
        return False

    left_x = observation.anchor_left_x
    left_y = observation.anchor_left_y
    expected_y = config.feature_y_offset_px + config.feature_y_slope * (left_x - left_det.cx)
    has_right_anchor = (
        observation.anchor_right_x is not None
        and observation.anchor_right_y is not None
        and math.isfinite(observation.anchor_right_x)
        and math.isfinite(observation.anchor_right_y)
    )
    right_x = observation.anchor_right_x if has_right_anchor else left_x - observation.disparity_px
    right_y = observation.anchor_right_y if has_right_anchor else left_y - expected_y
    if abs((left_y - right_y) - expected_y) > float(
        min(8.0, max(0.5, config.feature_y_tolerance_px))
    ):
        return False

    scale = float(min(0.90, max(0.35, config.feature_overlap_scale)))
    projection_scale = min(0.98, scale + 0.12)
    if not point_inside_detection_ellipse(left_det, left_x, left_y, scale):
        return False
    if not point_inside_detection_ellipse(right_det, right_x, right_y, scale):
        return False
    if not point_inside_detection_ellipse(
        right_det, left_x - initial_disparity, left_y, projection_scale
    ):
        return False
    if not point_inside_detection_ellipse(
        left_det, right_x + initial_disparity, right_y, projection_scale
    ):
        return False

    if config.feature_sphere_radius_m > 0.0:
        fb = focal_px * baseline_m
        center_z = fb / initial_disparity
        z = fb / observation.disparity_px
        if not math.isfinite(center_z) or not math.isfinite(z):
            return False
        dx = (left_x - left_det.cx) * z / focal_px
        dy = (left_y - left_det.cy) * z / focal_px
        dz = z - center_z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        max_distance = (
            config.feature_sphere_radius_m * max(1.0, config.feature_sphere_radius_scale)
            + max(0.0, config.feature_sphere_margin_m)
        )
        if distance > max_distance:
            return False
    return True
