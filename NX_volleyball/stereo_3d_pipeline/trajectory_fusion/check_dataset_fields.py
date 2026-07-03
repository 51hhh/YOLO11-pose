"""Field groups used by trajectory dataset quality checks."""

from __future__ import annotations


P0_DEPTH_KEYS = (
    "z_bbox_center",
    "z_circle_center",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
)
P1_DEPTH_KEYS = (
    "z_roi_multi_point",
    "z_roi_center_patch",
)
DEPTH_KEYS = P0_DEPTH_KEYS + P1_DEPTH_KEYS
JUMP_DEPTH_KEYS = DEPTH_KEYS + ("z_fallback_epipolar",)
REQUIRED_FIELDS = (
    "frame_id",
    "timestamp",
    "track_id",
    "frame_counter_delta",
    "frame_number_delta",
    "stereo_match_source",
    "pair_positive_disparity",
    *DEPTH_KEYS,
)
FRAME_SUMMARY_FIELDS = (
    "frame_id",
    "result_count",
    "raw_observation_count",
    "stereo_observation_count",
    "direct_pair_count",
    "fallback_l2r_count",
    "fallback_r2l_count",
)
OPTIONAL_FRAME_SUMMARY_FIELDS = (
    "p2_candidate_observed_count",
    "p2_candidate_valid_count",
    "p2_feature_valid_count",
    "p2_cuda_valid_count",
    "p2_neural_valid_count",
)
MATCH_SOURCE_NAMES = {
    0: "none",
    1: "direct_pair",
    2: "fallback_l2r",
    3: "fallback_r2l",
}
