"""Depth candidate method columns used by trajectory-fusion datasets."""

from __future__ import annotations


METHOD_COLUMNS = (
    ("mono", "z_mono"),
    ("bbox_center", "z_bbox_center"),
    ("bbox_left_edge", "z_bbox_left_edge"),
    ("bbox_right_edge", "z_bbox_right_edge"),
    ("circle_center", "z_circle_center"),
    ("circle_left_edge", "z_circle_left_edge"),
    ("circle_right_edge", "z_circle_right_edge"),
    ("roi_edge_centroid", "z_roi_edge_centroid"),
    ("roi_radial_center", "z_roi_radial_center"),
    ("roi_edge_pair_center", "z_roi_edge_pair_center"),
    ("roi_corner_points", "z_roi_corner_points"),
    ("roi_texture_points", "z_roi_texture_points"),
    ("roi_binary_points", "z_roi_binary_points"),
    ("roi_orb_points", "z_roi_orb_points"),
    ("roi_brisk_points", "z_roi_brisk_points"),
    ("roi_akaze_points", "z_roi_akaze_points"),
    ("roi_sift_points", "z_roi_sift_points"),
    ("roi_iou_region_color_patch", "z_roi_iou_region_color_patch"),
    ("roi_patch_iou_color_edge", "z_roi_patch_iou_color_edge"),
    ("roi_cuda_template_match", "z_roi_cuda_template_match"),
    ("roi_cuda_stereo_bm", "z_roi_cuda_stereo_bm"),
    ("roi_cuda_stereo_sgm", "z_roi_cuda_stereo_sgm"),
    ("roi_neural_feature", "z_roi_neural_feature"),
    ("roi_center_patch", "z_roi_center_patch"),
    ("roi_multi_point", "z_roi_multi_point"),
    ("epipolar_fallback", "z_fallback_epipolar"),
    ("fallback_template", "z_fallback_template"),
    ("fallback_feature_points", "z_fallback_feature_points"),
)
METHOD_NAMES = tuple(name for name, _ in METHOD_COLUMNS)
