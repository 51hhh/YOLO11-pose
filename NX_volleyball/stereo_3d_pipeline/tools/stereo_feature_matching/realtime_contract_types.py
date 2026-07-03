"""Shared realtime contract data types for offline stereo probes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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
