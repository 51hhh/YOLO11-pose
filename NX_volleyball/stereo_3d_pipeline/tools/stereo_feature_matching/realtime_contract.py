"""Compatibility facade for the realtime stereo matching contract mirror."""

from __future__ import annotations

from .realtime_contract_depth import select_first_usable_depth_candidate
from .realtime_contract_bbox_prior import (
    bbox_disparity_consistency_penalty,
    collect_scored_stereo_roi_pair_candidates,
    estimate_bbox_disparity_px,
    score_stereo_roi_pair_with_bbox_prior,
    select_global_stereo_roi_pairs,
)
from .realtime_contract_pairing import (
    collect_stereo_roi_pair_candidates,
    evaluate_stereo_roi_pair,
    find_best_stereo_roi_pair,
    rect_iou,
)
from .realtime_contract_types import (
    DEPTH_CANDIDATE_PRIORITY,
    STEREO_DEPTH_SOURCE,
    BboxDisparityPriorConfig,
    DepthCandidateObservation,
    Detection,
    FeatureValidationConfig,
    SparseFeatureObservation,
    StereoRoiPair,
    StereoRoiPairGateConfig,
)
from .realtime_contract_validation import (
    point_inside_detection_ellipse,
    validate_sparse_feature_geometry,
)


__all__ = [
    "DEPTH_CANDIDATE_PRIORITY",
    "STEREO_DEPTH_SOURCE",
    "BboxDisparityPriorConfig",
    "DepthCandidateObservation",
    "Detection",
    "FeatureValidationConfig",
    "SparseFeatureObservation",
    "StereoRoiPair",
    "StereoRoiPairGateConfig",
    "bbox_disparity_consistency_penalty",
    "collect_scored_stereo_roi_pair_candidates",
    "collect_stereo_roi_pair_candidates",
    "estimate_bbox_disparity_px",
    "evaluate_stereo_roi_pair",
    "find_best_stereo_roi_pair",
    "point_inside_detection_ellipse",
    "rect_iou",
    "score_stereo_roi_pair_with_bbox_prior",
    "select_first_usable_depth_candidate",
    "select_global_stereo_roi_pairs",
    "validate_sparse_feature_geometry",
]
