"""Offline/online depth method mapping table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodMapping:
    offline_method: str
    online_method: str
    z_col: str | None
    disparity_col: str | None
    support_col: str | None = None
    std_col: str | None = None
    confidence_col: str | None = None


MAPPINGS: tuple[MethodMapping, ...] = (
    MethodMapping(
        "iou_region_color_patch",
        "roi_iou_region_color_patch",
        "z_roi_iou_region_color_patch",
        "disparity_roi_iou_region_color_patch",
        "roi_iou_region_color_patch_support",
        "roi_iou_region_color_patch_std_px",
        "roi_iou_region_color_patch_confidence",
    ),
    MethodMapping(
        "patch_iou_zncc_corner",
        "roi_corner_points",
        "z_roi_corner_points",
        "disparity_roi_corner_points",
        "roi_corner_points_support",
        "roi_corner_points_std_px",
        "roi_corner_points_confidence",
    ),
    MethodMapping(
        "patch_iou_color_edge",
        "roi_patch_iou_color_edge",
        "z_roi_patch_iou_color_edge",
        "disparity_roi_patch_iou_color_edge",
        "roi_patch_iou_color_edge_support",
        "roi_patch_iou_color_edge_std_px",
        "roi_patch_iou_color_edge_confidence",
    ),
    MethodMapping(
        "patch_iou_zncc_edge",
        "missing_online_patch_iou_zncc_edge",
        None,
        None,
    ),
    MethodMapping(
        "orb",
        "roi_orb_points",
        "z_roi_orb_points",
        "disparity_roi_orb_points",
        "roi_orb_points_support",
        "roi_orb_points_std_px",
        "roi_orb_points_confidence",
    ),
    MethodMapping(
        "brisk",
        "roi_brisk_points",
        "z_roi_brisk_points",
        "disparity_roi_brisk_points",
        "roi_brisk_points_support",
        "roi_brisk_points_std_px",
        "roi_brisk_points_confidence",
    ),
    MethodMapping(
        "akaze",
        "roi_akaze_points",
        "z_roi_akaze_points",
        "disparity_roi_akaze_points",
        "roi_akaze_points_support",
        "roi_akaze_points_std_px",
        "roi_akaze_points_confidence",
    ),
    MethodMapping(
        "sift",
        "roi_sift_points",
        "z_roi_sift_points",
        "disparity_roi_sift_points",
        "roi_sift_points_support",
        "roi_sift_points_std_px",
        "roi_sift_points_confidence",
    ),
    MethodMapping(
        "xfeat",
        "roi_neural_feature",
        "z_roi_neural_feature",
        "disparity_roi_neural_feature",
        "roi_neural_feature_support",
        "roi_neural_feature_std_px",
        "roi_neural_feature_confidence",
    ),
    MethodMapping(
        "aliked",
        "roi_neural_feature",
        "z_roi_neural_feature",
        "disparity_roi_neural_feature",
        "roi_neural_feature_support",
        "roi_neural_feature_std_px",
        "roi_neural_feature_confidence",
    ),
    MethodMapping(
        "superpoint_lightglue",
        "roi_neural_feature",
        "z_roi_neural_feature",
        "disparity_roi_neural_feature",
        "roi_neural_feature_support",
        "roi_neural_feature_std_px",
        "roi_neural_feature_confidence",
    ),
)
