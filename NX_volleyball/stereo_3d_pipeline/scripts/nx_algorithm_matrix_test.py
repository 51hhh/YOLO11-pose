#!/usr/bin/env python3
"""Run isolated NX realtime algorithm cases from pipeline_dual_yolo_roi.yaml.

Each case disables every dual-YOLO depth mode first, then enables only the
candidate under test. Geometry needed as an internal seed may still run inside
that algorithm, but unrelated depth candidates stay disabled.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


MODE_KEYS = {
    "bbox_pair",
    "bbox_edges",
    "circle_center",
    "circle_edges",
    "roi_edge_centroid",
    "roi_radial_center",
    "roi_edge_pair_center",
    "roi_corner_points",
    "roi_texture_points",
    "roi_binary_points",
    "roi_orb_points",
    "roi_brisk_points",
    "roi_akaze_points",
    "roi_sift_points",
    "roi_iou_region_color_patch",
    "roi_patch_iou_color_edge",
    "roi_cuda_template_match",
    "roi_cuda_stereo_bm",
    "roi_cuda_stereo_sgm",
    "roi_ring_edge_profile",
    "roi_vpi_template_match",
    "roi_vpi_stereo_disparity",
    "roi_vpi_harris_lk",
    "roi_vpi_orb",
    "roi_cuda_gftt_lk",
    "roi_cuda_sift",
    "roi_libsgm",
    "roi_cuda_hough_circle",
    "roi_center_patch",
    "roi_subpixel",
    "epipolar_fallback",
    "fallback_template",
    "fallback_feature_points",
}


@dataclass
class Case:
    name: str
    modes: dict[str, bool] = field(default_factory=dict)
    candidate_fields: tuple[str, ...] = ()
    support_field: str | None = None
    subpixel_enabled: bool | None = None
    yaml_scalars: dict[str, str] = field(default_factory=dict)
    neural_backend: str | None = None
    neural_engine: str | None = None
    roi_size: int = 224
    top_k: int = 128
    descriptor_dim: int = 64
    neural_min_matches: int = 8
    neural_max_y_error_px: float = 2.0
    neural_max_disp_delta_px: float = 32.0
    neural_final_disp_gate_px: float = 2.0
    neural_min_score: float = 0.0
    algo_stage_override: str | None = None
    note: str = ""


CASES = (
    Case(
        "bbox_pair",
        {"bbox_pair": True},
        ("z_bbox_center",),
        note="YOLO bbox-center disparity only; no ROI keypoints",
    ),
    Case(
        "circle_center",
        {"circle_center": True},
        ("z_circle_center",),
        note="circle-fit center disparity only",
    ),
    Case(
        "roi_edge_centroid",
        {"roi_edge_centroid": True},
        ("z_roi_edge_centroid",),
        note="CUDA ROI edge centroid only",
    ),
    Case(
        "roi_radial_center",
        {"roi_radial_center": True},
        ("z_roi_radial_center",),
        note="CUDA radial center only",
    ),
    Case(
        "roi_edge_pair_center",
        {"roi_edge_pair_center": True},
        ("z_roi_edge_pair_center",),
        note="CUDA paired-edge center only",
    ),
    Case(
        "roi_center_patch",
        {"roi_center_patch": True},
        ("z_roi_center_patch",),
        support_field="subpixel_support",
        note="CUDA center-patch ZNCC only",
    ),
    Case(
        "roi_subpixel",
        {"roi_subpixel": True},
        ("z_roi_multi_point",),
        support_field="subpixel_support",
        subpixel_enabled=True,
        note="CUDA multi-point subpixel only",
    ),
    Case(
        "opencv_cuda_orb",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        note="true OpenCV CUDA ORB + CUDA matcher",
    ),
    Case(
        "opencv_cpu_brisk",
        {"roi_brisk_points": True},
        ("z_roi_brisk_points",),
        support_field="roi_brisk_points_support",
        note="true OpenCV CPU BRISK",
    ),
    Case(
        "opencv_cpu_akaze",
        {"roi_akaze_points": True},
        ("z_roi_akaze_points",),
        support_field="roi_akaze_points_support",
        note="true OpenCV CPU AKAZE",
    ),
    Case(
        "opencv_cpu_sift",
        {"roi_sift_points": True},
        ("z_roi_sift_points",),
        support_field="roi_sift_points_support",
        note="true OpenCV CPU SIFT",
    ),
    Case(
        "iou_region_color_patch",
        {"roi_iou_region_color_patch": True},
        ("z_roi_iou_region_color_patch",),
        support_field="roi_iou_region_color_patch_support",
        note="CUDA BGR color IoU/patch candidate",
    ),
    Case(
        "patch_iou_color_edge",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        note="CUDA BGR color-edge patch candidate",
    ),
    Case(
        "opencv_cuda_template_match",
        {"roi_cuda_template_match": True},
        ("z_roi_cuda_template_match",),
        support_field="roi_cuda_template_match_support",
        note="OpenCV CUDA TemplateMatching small-ROI epipolar candidate",
    ),
    Case(
        "opencv_cuda_stereo_bm",
        {"roi_cuda_stereo_bm": True},
        ("z_roi_cuda_stereo_bm",),
        support_field="roi_cuda_stereo_bm_support",
        note="OpenCV CUDA StereoBM small-ROI dense disparity candidate",
    ),
    Case(
        "opencv_cuda_stereo_sgm",
        {"roi_cuda_stereo_sgm": True},
        ("z_roi_cuda_stereo_sgm",),
        support_field="roi_cuda_stereo_sgm_support",
        note="OpenCV CUDA StereoSGM small-ROI dense disparity candidate",
    ),
    Case(
        "neural_xfeat",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_128.engine",
        roi_size=128,
        top_k=64,
        descriptor_dim=64,
        note="TensorRT XFeat 128 extractor; C++ postprocess/mutual-NN",
    ),
    Case(
        "neural_aliked",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="aliked",
        neural_engine="aliked_extractor_224_top128.engine",
        descriptor_dim=128,
    ),
    Case(
        "neural_superpoint_lightglue",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top128.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=256,
        note="TensorRT SuperPoint extractor; direct descriptor matching fallback",
    ),
)

APPROX_CASES = (
    Case(
        "approx_corner_points",
        {"roi_corner_points": True},
        ("z_roi_corner_points",),
        support_field="roi_corner_points_support",
        note="custom CUDA sparse-lite corner; not OpenCV ORB/BRISK/AKAZE/SIFT",
    ),
    Case(
        "approx_texture_points",
        {"roi_texture_points": True},
        ("z_roi_texture_points",),
        support_field="roi_texture_points_support",
        note="custom CUDA sparse-lite texture; diagnostic only",
    ),
    Case(
        "approx_binary_points",
        {"roi_binary_points": True},
        ("z_roi_binary_points",),
        support_field="roi_binary_points_support",
        note="custom CUDA sparse-lite binary; diagnostic only",
    ),
)


ORB_FAST_SWEEP = {
    "subpixel_patch_radius": "3",
    "subpixel_search_radius_px": "4",
    "subpixel_max_points": "8",
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.08",
    "subpixel_max_disp_delta_px": "6.0",
    "subpixel_max_stddev_px": "2.5",
    "feature_y_tolerance_px": "3.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.70",
    "feature_mad_scale": "3.5",
    "feature_ransac_gate_px": "2.0",
}

ORB_WIDE_Y_SWEEP = {
    "subpixel_patch_radius": "5",
    "subpixel_search_radius_px": "8",
    "subpixel_max_points": "12",
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.08",
    "subpixel_max_disp_delta_px": "10.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "8.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.90",
    "feature_mad_scale": "4.0",
    "feature_ransac_gate_px": "3.0",
    "feature_sphere_radius_scale": "2.8",
}

COLOR_OFFLINE_TUNED = {
    "subpixel_patch_radius": "9",
    "subpixel_search_radius_px": "12",
    "subpixel_min_points": "4",
    "subpixel_min_confidence": "0.54",
    "subpixel_max_disp_delta_px": "6.0",
    "subpixel_max_stddev_px": "1.5",
    "feature_y_tolerance_px": "1.0",
    "feature_reverse_check_px": "1.0",
    "feature_overlap_scale": "0.55",
    "feature_mad_scale": "2.5",
    "feature_ransac_gate_px": "0.75",
}

COLOR_WIDE_SEARCH = {
    "subpixel_patch_radius": "9",
    "subpixel_search_radius_px": "24",
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.25",
    "subpixel_max_disp_delta_px": "18.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "4.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.85",
    "feature_mad_scale": "4.0",
    "feature_ransac_gate_px": "3.0",
    "feature_sphere_radius_scale": "2.5",
}

TEMPLATE_PATCH9_SWEEP = {
    "subpixel_patch_radius": "9",
    "subpixel_search_radius_px": "16",
    "subpixel_min_confidence": "0.35",
    "subpixel_max_disp_delta_px": "10.0",
    "feature_y_tolerance_px": "2.0",
    "feature_overlap_scale": "0.70",
    "feature_sphere_radius_scale": "2.2",
}

DENSE_PATCH9_SWEEP = {
    "subpixel_patch_radius": "9",
    "subpixel_min_points": "4",
    "subpixel_min_confidence": "0.12",
    "subpixel_max_disp_delta_px": "18.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "3.0",
    "feature_overlap_scale": "0.80",
    "feature_sphere_radius_scale": "2.5",
}

P2_DIAGNOSTIC_ONLY = {
    "p2_feature_job_scaffold_enabled": "true",
    "p2_realtime_lane_decision_enabled": "false",
    "p2_diagnostic_lane_decision_enabled": "true",
    "p2_selective_trigger": "false",
    "p2_diagnostic_stride": "1",
    "p2_diagnostic_max_in_flight": "1",
    "p2_diagnostic_deadline_ms": "50.0",
}

P2_SELECTIVE_PAIR_QUALITY = {
    "p2_feature_job_scaffold_enabled": "true",
    "p2_selective_trigger": "true",
    "p2_trigger_on_fallback": "false",
    "p2_trigger_on_direct_pair": "false",
    "p2_trigger_on_host_gray": "false",
    "p2_trigger_on_bgr": "false",
    "p2_trigger_on_pair_quality": "true",
    "p2_trigger_on_no_valid_direct_pair": "true",
    "p2_pair_quality_min_shifted_iou": "0.99",
    "p2_pair_quality_max_epipolar_dy": "0.10",
    "p2_pair_quality_min_confidence": "0.99",
}

P2_SELECTIVE_NO_TRIGGER = {
    "p2_feature_job_scaffold_enabled": "true",
    "p2_selective_trigger": "true",
    "p2_trigger_on_fallback": "false",
    "p2_trigger_on_direct_pair": "false",
    "p2_trigger_on_host_gray": "false",
    "p2_trigger_on_bgr": "false",
    "p2_trigger_on_pair_quality": "true",
    "p2_trigger_on_no_valid_direct_pair": "false",
    "p2_pair_quality_min_shifted_iou": "0.0",
    "p2_pair_quality_max_epipolar_dy": "0.0",
    "p2_pair_quality_min_confidence": "0.0",
}


RELAXED_CASES = (
    Case(
        "realtime_gpu_bundle",
        {
            "bbox_pair": True,
            "circle_center": True,
            "roi_edge_centroid": True,
            "roi_radial_center": True,
            "roi_edge_pair_center": True,
            "roi_center_patch": True,
            "roi_subpixel": True,
        },
        (
            "z_bbox_center",
            "z_circle_center",
            "z_roi_edge_centroid",
            "z_roi_radial_center",
            "z_roi_edge_pair_center",
            "z_roi_center_patch",
            "z_roi_multi_point",
        ),
        support_field="subpixel_support",
        subpixel_enabled=True,
        note="diagnostic only: all 100fps-capable CUDA depth candidates enabled together",
    ),
    Case(
        "opencv_cuda_orb_relaxed",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "8.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "6.0",
            "feature_reverse_check_px": "-1.0",
            "feature_overlap_scale": "0.90",
            "feature_mad_scale": "4.0",
            "feature_ransac_gate_px": "3.0",
        },
        note="diagnostic only: true OpenCV CUDA ORB with relaxed gates",
    ),
    Case(
        "opencv_cuda_orb_fast48",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        yaml_scalars=ORB_FAST_SWEEP,
        note="P2 sweep: true OpenCV CUDA ORB, smaller ROI/search and 48-point cap",
    ),
    Case(
        "opencv_cuda_orb_diagnostic_only",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        yaml_scalars={**ORB_FAST_SWEEP, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticOpenCVCudaORB",
        note="diagnostic lane only: OpenCV CUDA ORB runs from independent GPU snapshot; no CSV candidate writeback",
    ),
    Case(
        "opencv_cuda_orb_wide_y",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        yaml_scalars=ORB_WIDE_Y_SWEEP,
        note="P2 sweep: true OpenCV CUDA ORB with wide y/overlap gates",
    ),
    Case(
        "iou_region_color_patch_relaxed",
        {"roi_iou_region_color_patch": True},
        ("z_roi_iou_region_color_patch",),
        support_field="roi_iou_region_color_patch_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "8.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "6.0",
            "feature_reverse_check_px": "-1.0",
            "feature_overlap_scale": "0.90",
            "feature_mad_scale": "4.0",
            "feature_ransac_gate_px": "3.0",
        },
        note="diagnostic only: CUDA color IoU/patch with relaxed gates",
    ),
    Case(
        "iou_region_color_patch_offline_tuned",
        {"roi_iou_region_color_patch": True},
        ("z_roi_iou_region_color_patch",),
        support_field="roi_iou_region_color_patch_support",
        yaml_scalars=COLOR_OFFLINE_TUNED,
        note="P2 sweep: CUDA color IoU/patch with offline best patch/y settings",
    ),
    Case(
        "iou_region_color_patch_wide_search",
        {"roi_iou_region_color_patch": True},
        ("z_roi_iou_region_color_patch",),
        support_field="roi_iou_region_color_patch_support",
        yaml_scalars=COLOR_WIDE_SEARCH,
        note="P2 sweep: CUDA color IoU/patch with wider search and looser gates",
    ),
    Case(
        "patch_iou_color_edge_relaxed",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "8.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "6.0",
            "feature_reverse_check_px": "-1.0",
            "feature_overlap_scale": "0.90",
            "feature_mad_scale": "4.0",
            "feature_ransac_gate_px": "3.0",
        },
        note="diagnostic only: CUDA color-edge patch with relaxed gates",
    ),
    Case(
        "patch_iou_color_edge_offline_tuned",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        yaml_scalars=COLOR_OFFLINE_TUNED,
        note="P2 sweep: CUDA color-edge patch with offline best patch/y settings",
    ),
    Case(
        "patch_iou_color_edge_wide_search",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        yaml_scalars=COLOR_WIDE_SEARCH,
        note="P2 sweep: CUDA color-edge patch with wider search and looser gates",
    ),
    Case(
        "patch_iou_color_edge_selective_pair_quality",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        yaml_scalars={**COLOR_WIDE_SEARCH, **P2_SELECTIVE_PAIR_QUALITY},
        note="P2 selective-trigger smoke: force pair quality triggers for color-edge patch",
    ),
    Case(
        "patch_iou_color_edge_selective_no_trigger",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        yaml_scalars={**COLOR_WIDE_SEARCH, **P2_SELECTIVE_NO_TRIGGER},
        note="P2 selective-trigger smoke: no trigger should skip color-edge patch",
    ),
    Case(
        "opencv_cuda_template_match_relaxed",
        {"roi_cuda_template_match": True},
        ("z_roi_cuda_template_match",),
        support_field="roi_cuda_template_match_support",
        yaml_scalars={
            "subpixel_patch_radius": "7",
            "subpixel_search_radius_px": "16",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "16.0",
            "feature_y_tolerance_px": "4.0",
            "feature_overlap_scale": "0.85",
            "feature_sphere_radius_scale": "2.5",
        },
        note="diagnostic only: OpenCV CUDA TemplateMatching with wider search/gates",
    ),
    Case(
        "opencv_cuda_template_match_patch9",
        {"roi_cuda_template_match": True},
        ("z_roi_cuda_template_match",),
        support_field="roi_cuda_template_match_support",
        yaml_scalars=TEMPLATE_PATCH9_SWEEP,
        note="P2 sweep: OpenCV CUDA TemplateMatching patch radius 9",
    ),
    Case(
        "opencv_cuda_template_match_diagnostic_only",
        {"roi_cuda_template_match": True},
        ("z_roi_cuda_template_match",),
        support_field="roi_cuda_template_match_support",
        yaml_scalars={**TEMPLATE_PATCH9_SWEEP, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticCudaTemplate",
        note="diagnostic lane only: OpenCV CUDA TemplateMatching runs from independent GPU snapshot; no CSV candidate writeback",
    ),
    Case(
        "opencv_cuda_stereo_bm_relaxed",
        {"roi_cuda_stereo_bm": True},
        ("z_roi_cuda_stereo_bm",),
        support_field="roi_cuda_stereo_bm_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "16.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "4.0",
            "feature_overlap_scale": "0.85",
            "feature_sphere_radius_scale": "2.5",
        },
        note="diagnostic only: OpenCV CUDA StereoBM with relaxed gates",
    ),
    Case(
        "opencv_cuda_stereo_bm_patch9",
        {"roi_cuda_stereo_bm": True},
        ("z_roi_cuda_stereo_bm",),
        support_field="roi_cuda_stereo_bm_support",
        yaml_scalars=DENSE_PATCH9_SWEEP,
        note="P2 sweep: OpenCV CUDA StereoBM larger census/block window",
    ),
    Case(
        "opencv_cuda_stereo_bm_diagnostic_only",
        {"roi_cuda_stereo_bm": True},
        ("z_roi_cuda_stereo_bm",),
        support_field="roi_cuda_stereo_bm_support",
        yaml_scalars={**DENSE_PATCH9_SWEEP, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticCudaStereoBM",
        note="diagnostic lane only: OpenCV CUDA StereoBM runs from independent GPU snapshot; no CSV candidate writeback",
    ),
    Case(
        "opencv_cuda_stereo_sgm_relaxed",
        {"roi_cuda_stereo_sgm": True},
        ("z_roi_cuda_stereo_sgm",),
        support_field="roi_cuda_stereo_sgm_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "16.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "4.0",
            "feature_overlap_scale": "0.85",
            "feature_sphere_radius_scale": "2.5",
        },
        note="diagnostic only: OpenCV CUDA StereoSGM with relaxed gates",
    ),
    Case(
        "opencv_cuda_stereo_sgm_patch9",
        {"roi_cuda_stereo_sgm": True},
        ("z_roi_cuda_stereo_sgm",),
        support_field="roi_cuda_stereo_sgm_support",
        yaml_scalars=DENSE_PATCH9_SWEEP,
        note="P2 sweep: OpenCV CUDA StereoSGM larger aggregation window",
    ),
    Case(
        "opencv_cuda_stereo_sgm_diagnostic_only",
        {"roi_cuda_stereo_sgm": True},
        ("z_roi_cuda_stereo_sgm",),
        support_field="roi_cuda_stereo_sgm_support",
        yaml_scalars={**DENSE_PATCH9_SWEEP, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticCudaStereoSGM",
        note="diagnostic lane only: OpenCV CUDA StereoSGM runs from independent GPU snapshot; no CSV candidate writeback",
    ),
    Case(
        "cuda_ring_edge_profile_diagnostic_only",
        {"roi_ring_edge_profile": True},
        yaml_scalars={**COLOR_WIDE_SEARCH, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticCudaRingEdgeProfile",
        note="diagnostic lane only: custom CUDA ring/edge profile matcher; no CSV candidate writeback",
    ),
    Case(
        "opencv_cuda_gftt_lk_diagnostic_only",
        {"roi_cuda_gftt_lk": True},
        yaml_scalars={**ORB_WIDE_Y_SWEEP, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticOpenCVCudaGfttLk",
        note="diagnostic lane only: OpenCV CUDA GFTT/Harris + SparsePyrLK; no CSV candidate writeback",
    ),
    Case(
        "cuda_hough_circle_diagnostic_only",
        {"roi_cuda_hough_circle": True},
        yaml_scalars={**COLOR_WIDE_SEARCH, **P2_DIAGNOSTIC_ONLY},
        algo_stage_override="Stage2_P2FeatureJobDiagnosticCudaHoughCircle",
        note="diagnostic lane only: OpenCV CUDA Hough circle ROI refinement; no CSV candidate writeback",
    ),
    Case(
        "vpi_template_match_diagnostic_only",
        {"roi_vpi_template_match": True},
        yaml_scalars=P2_DIAGNOSTIC_ONLY,
        algo_stage_override="Stage2_P2FeatureJobDiagnosticVpiTemplate",
        note="diagnostic lane only: VPI CUDA TemplateMatching; no CSV candidate writeback",
    ),
    Case(
        "vpi_stereo_disparity_diagnostic_only",
        {"roi_vpi_stereo_disparity": True},
        yaml_scalars=P2_DIAGNOSTIC_ONLY,
        algo_stage_override="Stage2_P2FeatureJobDiagnosticVpiStereo",
        note="diagnostic lane only: VPI CUDA StereoDisparity; no CSV candidate writeback",
    ),
    Case(
        "vpi_harris_lk_diagnostic_only",
        {"roi_vpi_harris_lk": True},
        yaml_scalars=P2_DIAGNOSTIC_ONLY,
        algo_stage_override="Stage2_P2FeatureJobDiagnosticVpiHarrisLk",
        note="diagnostic lane only: VPI CUDA Harris + Pyramidal LK; no CSV candidate writeback",
    ),
    Case(
        "vpi_orb_diagnostic_only",
        {"roi_vpi_orb": True},
        yaml_scalars=P2_DIAGNOSTIC_ONLY,
        algo_stage_override="Stage2_P2FeatureJobDiagnosticVpiOrb",
        note="diagnostic lane only: VPI CUDA ORB + BruteForceMatcher; no CSV candidate writeback",
    ),
    Case(
        "cuda_sift_diagnostic_only",
        {"roi_cuda_sift": True},
        yaml_scalars=P2_DIAGNOSTIC_ONLY,
        algo_stage_override="Stage2_P2FeatureJobDiagnosticCudaSift",
        note="diagnostic lane only: CUDA-SIFT backend availability check",
    ),
    Case(
        "fixstars_libsgm_diagnostic_only",
        {"roi_libsgm": True},
        yaml_scalars=P2_DIAGNOSTIC_ONLY,
        algo_stage_override="Stage2_P2FeatureJobDiagnosticFixstarsLibSgm",
        note="diagnostic lane only: Fixstars libSGM CUDA SGM; requires third_party/libSGM or LIBSGM_ROOT",
    ),
    Case(
        "neural_xfeat_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_128.engine",
        roi_size=128,
        top_k=64,
        descriptor_dim=64,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT XFeat with relaxed gates",
    ),
    Case(
        "neural_xfeat_96_top32",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_96.engine",
        roi_size=96,
        top_k=32,
        descriptor_dim=64,
        neural_min_matches=4,
        note="P2 sweep: TensorRT XFeat 96 extractor top32",
    ),
    Case(
        "neural_xfeat_96_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_96.engine",
        roi_size=96,
        top_k=64,
        descriptor_dim=64,
        neural_min_matches=4,
        note="P2 sweep: TensorRT XFeat 96 extractor top64",
    ),
    Case(
        "neural_xfeat_128_top32",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_128.engine",
        roi_size=128,
        top_k=32,
        descriptor_dim=64,
        neural_min_matches=4,
        note="P2 sweep: TensorRT XFeat 128 extractor top32",
    ),
    Case(
        "neural_xfeat_128_top96",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_128.engine",
        roi_size=128,
        top_k=96,
        descriptor_dim=64,
        note="P2 sweep: TensorRT XFeat 128 extractor top96",
    ),
    Case(
        "neural_xfeat_160",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_160.engine",
        roi_size=160,
        top_k=96,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 160 extractor",
    ),
    Case(
        "neural_xfeat_160_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_160.engine",
        roi_size=160,
        top_k=64,
        descriptor_dim=64,
        neural_min_matches=4,
        note="P2 sweep: TensorRT XFeat 160 extractor top64",
    ),
    Case(
        "neural_xfeat_224",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 224 extractor",
    ),
    Case(
        "neural_xfeat_224_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=64,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 224 extractor top64",
    ),
    Case(
        "neural_xfeat_224_top32",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=32,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 224 extractor top32",
    ),
    Case(
        "neural_xfeat_224_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=64,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT XFeat 224 extractor with relaxed gates",
    ),
    Case(
        "neural_superpoint_lightglue_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top128.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=256,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT SuperPoint extractor with relaxed gates",
    ),
    Case(
        "neural_superpoint_128_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_128_top64.engine",
        roi_size=128,
        top_k=64,
        descriptor_dim=256,
        neural_min_matches=4,
        note="P2 sweep: TensorRT SuperPoint fixed extractor 128/top64",
    ),
    Case(
        "neural_superpoint_160_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_160_top64.engine",
        roi_size=160,
        top_k=64,
        descriptor_dim=256,
        neural_min_matches=4,
        note="P2 sweep: TensorRT SuperPoint fixed extractor 160/top64",
    ),
    Case(
        "neural_superpoint_224_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top64.engine",
        roi_size=224,
        top_k=64,
        descriptor_dim=256,
        neural_min_matches=4,
        note="P2 sweep: TensorRT SuperPoint fixed extractor 224/top64",
    ),
    Case(
        "neural_aliked_160_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="aliked",
        neural_engine="aliked_extractor_160_top64.engine",
        roi_size=160,
        top_k=64,
        descriptor_dim=128,
        neural_min_matches=4,
        note="P2 sweep: ALIKED fixed extractor if a TensorRT engine is available",
    ),
    Case(
        "neural_aliked_224_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="aliked",
        neural_engine="aliked_extractor_224_top64.engine",
        roi_size=224,
        top_k=64,
        descriptor_dim=128,
        neural_min_matches=4,
        note="P2 sweep: ALIKED fixed extractor 224/top64 if engine exists",
    ),
)


def set_yaml_bool(text: str, key: str, value: bool) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)(true|false)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{'true' if value else 'false'}{match.group(3)}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing bool key: {key}")
    return new


def set_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        suffix = match.group(3)
        if suffix.startswith("#"):
            suffix = " " + suffix
        return f"{match.group(1)}{value}{suffix}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing scalar key: {key}")
    return new


def set_depth_mode(text: str, key: str, value: bool) -> str:
    if key not in MODE_KEYS:
        raise RuntimeError(f"unknown depth mode: {key}")
    return set_yaml_bool(text, key, value)


def disable_all_depth_modes(text: str) -> str:
    for key in sorted(MODE_KEYS):
        text = set_depth_mode(text, key, False)
    return text


def set_neural_enabled(text: str, value: bool) -> str:
    pattern = re.compile(
        r"(^neural_feature_matching:\n(?:^[ \t].*\n)*?^[ \t]*enabled:\s*)(true|false)(.*)$",
        re.M,
    )
    replacement = rf"\g<1>{'true' if value else 'false'}\3"
    text, _ = pattern.subn(replacement, text, count=1)
    return text


def render_neural_block(case: Case, neural_model_dir: Path) -> str:
    use_lightglue = str(case.neural_backend == "superpoint_lightglue").lower()
    extractor_engine_path = ""
    if case.neural_engine:
        extractor_engine_path = str(neural_model_dir / case.neural_engine)
    return f"""neural_feature_matching:
  enabled: true
  backend: "{case.neural_backend}"
  extractor_engine_path: "{extractor_engine_path}"
  matcher_engine_path: ""
  fused_engine_path: ""
  roi_size: {case.roi_size}
  top_k: {case.top_k}
  descriptor_dim: {case.descriptor_dim}
  min_matches: {case.neural_min_matches}
  max_y_error_px: {case.neural_max_y_error_px}
  max_disp_delta_px: {case.neural_max_disp_delta_px}
  final_disp_gate_px: {case.neural_final_disp_gate_px}
  min_score: {case.neural_min_score}
  use_lightglue: {use_lightglue}
"""


def upsert_neural_block(text: str, block: str) -> str:
    pattern = re.compile(r"^neural_feature_matching:\n(?:^[ \t].*\n?)*", re.M)
    new, count = pattern.subn(block.rstrip() + "\n", text, count=1)
    if count:
        return new
    return text.rstrip() + "\n\n" + block


def prepare_config(
    base: str,
    case: Case,
    out_dir: Path,
    config_dir: Path,
    neural_model_dir: Path,
) -> Path:
    text = base
    text = re.sub(r"(\nros2:\n\s*)enable:\s*true", r"\1enable: false", text, count=1)
    text = disable_all_depth_modes(text)
    text = set_yaml_bool(text, "subpixel_enabled", False)
    text = set_yaml_bool(text, "fallback_epipolar_search", False)
    text = set_neural_enabled(text, False)
    text = re.sub(
        r'output_path:\s*"dual_yolo_observation_data\.csv"',
        f'output_path: "{out_dir / (case.name + ".csv")}"',
        text,
        count=1,
    )
    for mode, value in case.modes.items():
        text = set_depth_mode(text, mode, value)
    if case.subpixel_enabled is not None:
        text = set_yaml_bool(text, "subpixel_enabled", case.subpixel_enabled)
    for key, value in case.yaml_scalars.items():
        text = set_yaml_scalar(text, key, value)
    if "P2FeatureJobDiagnostic" in (case.algo_stage_override or ""):
        diag_csv = out_dir / f"{case.name}.p2_diagnostic.csv"
        diag_artifacts = out_dir / f"{case.name}.p2_artifacts"
        text = set_yaml_scalar(text, "p2_diagnostic_results_enabled", "true")
        text = set_yaml_scalar(
            text,
            "p2_diagnostic_results_path",
            f'"{diag_csv}"',
        )
        text = set_yaml_scalar(text, "p2_diagnostic_artifacts_enabled", "true")
        text = set_yaml_scalar(
            text,
            "p2_diagnostic_artifacts_dir",
            f'"{diag_artifacts}"',
        )
        text = set_yaml_scalar(text, "p2_diagnostic_artifacts_max", "20")
    if case.neural_backend:
        text = upsert_neural_block(text, render_neural_block(case, neural_model_dir))
    cfg = config_dir / f"{case.name}.yaml"
    cfg.write_text(text)
    return cfg


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return v


def count_frame_rows(frames_path: Path) -> int:
    if not frames_path.exists():
        return 0
    with frames_path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def profile_stage_for_case(case: Case) -> str:
    if case.algo_stage_override:
        return case.algo_stage_override
    if case.neural_backend:
        return "Stage2_NeuralFeatureMatch"
    if case.modes.get("roi_orb_points"):
        return "Stage2_OpenCVCudaORB"
    if case.modes.get("roi_brisk_points"):
        return "Stage2_CPUFeatureOpenCVBRISK"
    if case.modes.get("roi_akaze_points"):
        return "Stage2_CPUFeatureOpenCVAKAZE"
    if case.modes.get("roi_sift_points"):
        return "Stage2_CPUFeatureOpenCVSIFT"
    if case.modes.get("roi_cuda_template_match"):
        return "Stage2_OpenCVCudaTemplateMatch"
    if case.modes.get("roi_cuda_stereo_bm"):
        return "Stage2_OpenCVCudaStereoBM"
    if case.modes.get("roi_cuda_stereo_sgm"):
        return "Stage2_OpenCVCudaStereoSGM"
    if case.modes.get("roi_subpixel"):
        return "Stage2_SubpixelMatch"
    return "Stage2_DualYoloGpuCandidates"


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def row_int(row: dict[str, str], key: str) -> int:
    return parse_int(row.get(key))


def classify_case_result(row: dict[str, str]) -> str:
    status = row.get("status", "")
    if status.startswith("skipped"):
        return "skipped_missing_dependency"
    if status == "failed":
        return "pipeline_failed"
    if row_int(row, "p2_diag_rows") > 0:
        if row_int(row, "p2_diag_over_deadline") > 0:
            return "diagnostic_over_deadline"
        if row_int(row, "p2_diag_valid") > 0:
            return "diagnostic_ok"
        return "diagnostic_ran_but_no_valid_candidate"

    stale_or_expired = (
        row_int(row, "async_drop_stale_count") +
        row_int(row, "async_drop_stale_ready_count") +
        row_int(row, "async_drop_expired_pending_count") +
        row_int(row, "stage2_drop_stale_roi_count")
    )
    infrastructure_drop = (
        row_int(row, "async_drop_pending_count") +
        row_int(row, "async_drop_no_buffer_count") +
        row_int(row, "async_submit_drop_count")
    )
    if row_int(row, "async_over_deadline_count") > 0 or stale_or_expired > 0:
        return "late_or_deadline_dropped"
    if infrastructure_drop > 0:
        return "async_queue_or_buffer_drop"
    if (
        row_int(row, "cpu_fallback_count") > 0
        or row_int(row, "async_need_host_gray_count") > 0
        or row_int(row, "async_host_gray_submit_count") > 0
    ):
        return "realtime_path_used_cpu_or_host_gray"
    if row_int(row, "async_no_detections_count") > 0 and row_int(row, "target_rows") == 0:
        return "no_detections"
    if row_int(row, "target_rows") == 0 and row_int(row, "async_accepted_count") == 0:
        return "no_accepted_results"
    if row_int(row, "candidate_rows") > 0 and row_int(row, "candidate_valid") == 0:
        return "ran_but_no_valid_candidate"
    if row_int(row, "candidate_valid") > 0:
        return "ok"
    return "needs_log_review"


def should_debug_case(row: dict[str, str]) -> bool:
    return classify_case_result(row) not in {
        "ok",
        "skipped_missing_dependency",
    }


def summarize_candidate_csv(case: Case, csv_path: Path, frames_path: Path) -> dict[str, str]:
    empty = {
        "candidate_rows": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "target_rows": "",
    }
    frame_total = count_frame_rows(frames_path)
    if not case.candidate_fields or not csv_path.exists():
        return empty

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    target_total = len(rows)
    total = frame_total if frame_total > 0 else target_total
    if target_total == 0 and total == 0:
        return {**empty, "candidate_rows": "0", "target_rows": "0"}

    valid_depths: list[float] = []
    field_counts: dict[str, int] = {field: 0 for field in case.candidate_fields}
    supports: list[float] = []
    for row in rows:
        row_values = []
        for field in case.candidate_fields:
            value = parse_float(row.get(field))
            if value is not None and value > 0.0:
                field_counts[field] += 1
                row_values.append(value)
        if row_values:
            valid_depths.append(row_values[0])
            if case.support_field:
                support = parse_float(row.get(case.support_field))
                if support is not None and support >= 0.0:
                    supports.append(support)

    valid = len(valid_depths)
    if valid_depths:
        med = median(valid_depths)
        mad = median(abs(v - med) for v in valid_depths)
        med_s = f"{med:.4f}"
        mad_s = f"{mad:.4f}"
    else:
        med_s = ""
        mad_s = ""
    support_s = f"{median(supports):.1f}" if supports else ""
    field_valids = ";".join(f"{k}={v}/{total}" for k, v in field_counts.items())
    return {
        "candidate_rows": str(total),
        "candidate_valid": str(valid),
        "candidate_rate": f"{valid / total:.3f}" if total else "",
        "candidate_median_m": med_s,
        "candidate_mad_m": mad_s,
        "support_median": support_s,
        "field_valids": field_valids,
        "target_rows": str(target_total),
    }


def summarize_diagnostic_csv(csv_path: Path) -> dict[str, str]:
    empty = {
        "p2_diag_rows": "",
        "p2_diag_valid": "",
        "p2_diag_invalid": "",
        "p2_diag_rate": "",
        "p2_diag_over_deadline": "",
        "p2_diag_median_m": "",
        "p2_diag_mad_m": "",
        "p2_diag_support_median": "",
        "p2_diag_attempted_median": "",
        "p2_diag_debug_match_rows": "",
        "p2_diag_debug_matches_median": "",
        "p2_diag_artifacts": "",
        "p2_diag_artifact_dirs": "",
        "p2_diag_status_counts": "",
        "p2_diag_mode_counts": "",
        "p2_diag_path": "",
    }
    if not csv_path.exists():
        return empty

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    if total == 0:
        return {**empty, "p2_diag_rows": "0", "p2_diag_path": str(csv_path)}

    valid_depths: list[float] = []
    supports: list[float] = []
    attempted_values: list[float] = []
    positive_debug_match_values: list[float] = []
    artifact_paths: list[str] = []
    status_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    valid = 0
    over_deadline = 0
    rows_with_debug_matches = 0
    for row in rows:
        status = row.get("status", "")
        mode = row.get("mode", "")
        status_counts[status] = status_counts.get(status, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if row.get("valid") == "1":
            valid += 1
            z_m = parse_float(row.get("z_m"))
            if z_m is not None and z_m > 0.0:
                valid_depths.append(z_m)
        if row.get("over_deadline") == "1":
            over_deadline += 1
        support = parse_float(row.get("support"))
        if support is not None and support >= 0.0:
            supports.append(support)
        attempted = parse_float(row.get("attempted"))
        if attempted is not None and attempted >= 0.0:
            attempted_values.append(attempted)
        debug_matches = parse_float(row.get("debug_match_count"))
        if debug_matches is not None and debug_matches >= 0.0:
            if debug_matches > 0.0:
                rows_with_debug_matches += 1
                positive_debug_match_values.append(debug_matches)
        artifact_path = row.get("artifact_path", "")
        if artifact_path:
            artifact_paths.append(artifact_path)

    if valid_depths:
        med = median(valid_depths)
        mad = median(abs(v - med) for v in valid_depths)
        med_s = f"{med:.4f}"
        mad_s = f"{mad:.4f}"
    else:
        med_s = ""
        mad_s = ""

    return {
        "p2_diag_rows": str(total),
        "p2_diag_valid": str(valid),
        "p2_diag_invalid": str(total - valid),
        "p2_diag_rate": f"{valid / total:.3f}" if total else "",
        "p2_diag_over_deadline": str(over_deadline),
        "p2_diag_median_m": med_s,
        "p2_diag_mad_m": mad_s,
        "p2_diag_support_median": f"{median(supports):.1f}" if supports else "",
        "p2_diag_attempted_median": f"{median(attempted_values):.1f}" if attempted_values else "",
        "p2_diag_debug_match_rows": str(rows_with_debug_matches),
        "p2_diag_debug_matches_median": (
            f"{median(positive_debug_match_values):.1f}"
            if positive_debug_match_values else ""
        ),
        "p2_diag_artifacts": str(len(artifact_paths)),
        "p2_diag_artifact_dirs": ";".join(
            sorted({str(Path(path).parent) for path in artifact_paths})
        ),
        "p2_diag_status_counts": ";".join(
            f"{k}={v}" for k, v in sorted(status_counts.items())
        ),
        "p2_diag_mode_counts": ";".join(
            f"{k}={v}" for k, v in sorted(mode_counts.items())
        ),
        "p2_diag_path": str(csv_path),
    }


def parse_log(case: Case, log: str, rc: int, log_path: Path) -> dict[str, str]:
    fps_matches = re.findall(r"\[ROI\] FPS:\s*([0-9.]+).*?stale_drop=([0-9]+)", log)

    def stage(name: str) -> tuple[str, str, str, str, str, str, str, str]:
        matches = re.findall(
            rf"^{re.escape(name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)"
            rf"(?:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+))?",
            log,
            re.M,
        )
        if not matches:
            return ("", "", "", "", "", "", "", "")
        match = matches[-1]
        return (
            match[0], match[1], match[2], match[3],
            match[4] or "", match[5] or "", match[6] or "", match[7] or "",
        )

    stage_gpu = stage("Stage2_DualYoloGpuCandidates")
    stage_match = stage("Stage2_DualYoloMatch")
    subpixel = stage("Stage2_SubpixelMatch")
    neural = stage("Stage2_NeuralFeatureMatch")
    opencv_cuda_orb = stage("Stage2_OpenCVCudaORB")
    opencv_cuda_template = stage("Stage2_OpenCVCudaTemplateMatch")
    opencv_cuda_bm = stage("Stage2_OpenCVCudaStereoBM")
    opencv_cuda_sgm = stage("Stage2_OpenCVCudaStereoSGM")
    cpu_opencv = stage("Stage2_CPUFeatureOpenCV")
    cpu_opencv_orb = stage("Stage2_CPUFeatureOpenCVORB")
    cpu_opencv_brisk = stage("Stage2_CPUFeatureOpenCVBRISK")
    cpu_opencv_akaze = stage("Stage2_CPUFeatureOpenCVAKAZE")
    cpu_opencv_sift = stage("Stage2_CPUFeatureOpenCVSIFT")
    cpu_fallback = stage("Stage2_CPUFallbackSearch")
    algo_stage_name = profile_stage_for_case(case)
    algo_stage = stage(algo_stage_name)
    async_worker = stage("Stage2_AsyncRoiWorker")
    async_over_deadline = stage("Stage2_AsyncRoiOverDeadline")
    async_drop_stale = stage("Stage2_AsyncRoiDropStaleResult")
    async_drop_stale_ready = stage("Stage2_AsyncRoiDropStaleReady")
    async_drop_expired_pending = stage("Stage2_AsyncRoiDropExpiredPending")
    async_drop_pending = stage("Stage2_AsyncRoiDropPending")
    async_drop_no_buffer = stage("Stage2_AsyncRoiDropNoBuffer")
    async_submit_drop = stage("Stage2_AsyncRoiSubmitDrop")
    async_no_detections = stage("Stage2_AsyncRoiNoDetections")
    async_submitted = stage("Stage2_AsyncRoiSubmitted")
    async_accepted = stage("Stage2_AsyncRoiAccepted")
    async_accepted_reused = stage("Stage2_AsyncRoiAcceptedReusedSlot")
    async_frame_cb_skipped = stage("Stage2_AsyncRoiFrameCallbackSkippedReusedSlot")
    async_need_host_gray = stage("Stage2_AsyncRoiNeedHostGray")
    async_need_bgr = stage("Stage2_AsyncRoiNeedBgr")
    async_skip_host_gray_selective = stage("Stage2_AsyncRoiSkipHostGraySelective")
    async_skip_bgr_selective = stage("Stage2_AsyncRoiSkipBgrSelective")
    async_host_gray_submit = stage("Stage2_AsyncRoiHostGrayD2HSubmit")
    async_gray_submit = stage("Stage2_AsyncRoiGrayD2DSubmit")
    async_bgr_submit = stage("Stage2_AsyncRoiBgrD2DSubmit")
    async_copy_wait = stage("Stage2_AsyncRoiCopyWait")
    async_slot_copy_wait = stage("Stage2_AsyncRoiSlotCopyWait")
    async_worker_busy = stage("Stage2_AsyncRoiWorkerBusy")
    stage2_drop_stale_roi = stage("Stage2_DropStaleROI")
    p2_configured = stage("Stage2_P2FeatureJobConfigured")
    p2_realtime_requested = stage("Stage2_P2FeatureJobRealtimeRequested")
    p2_realtime_not_attempted = stage("Stage2_P2FeatureJobRealtimeNotAttempted")
    p2_not_triggered = stage("Stage2_P2FeatureJobNotTriggered")
    p2_skip_selective = stage("Stage2_P2FeatureJobSkipSelective")
    p2_trigger_pair_low_iou = stage("Stage2_P2FeatureJobTriggerPairLowIou")
    p2_trigger_pair_epipolar_dy = stage("Stage2_P2FeatureJobTriggerPairEpipolarDy")
    p2_trigger_pair_low_conf = stage("Stage2_P2FeatureJobTriggerPairLowConf")
    p2_trigger_no_valid_pair = stage("Stage2_P2FeatureJobTriggerNoValidPair")
    p2_inline_skipped_selective = stage("Stage2_P2FeatureJobInlineSkippedSelective")
    error_lines = [line for line in log.splitlines() if "[ERROR]" in line or "[WARN ]" in line]
    neural_unbound = (
        "does not bind/use NeuralFeatureMatcher outputs yet" in log
        or "tensor_binding_not_implemented" in log
    )
    pipeline_failed = "Pipeline init failed" in log or rc not in (0, 124, 143)
    status = "failed" if pipeline_failed else ("ran_timeout" if rc == 124 else "ran")
    return {
        "case": case.name,
        "status": status,
        "return_code": str(rc),
        "fps_last": fps_matches[-1][0] if fps_matches else "",
        "stale_drop_last": fps_matches[-1][1] if fps_matches else "",
        "gpu_candidates_avg_ms": stage_gpu[0],
        "gpu_candidates_max_ms": stage_gpu[2],
        "dual_yolo_match_avg_ms": stage_match[0],
        "dual_yolo_match_max_ms": stage_match[2],
        "subpixel_avg_ms": subpixel[0],
        "subpixel_max_ms": subpixel[2],
        "subpixel_p95_ms": subpixel[6],
        "subpixel_p99_ms": subpixel[7],
        "neural_avg_ms": neural[0],
        "neural_max_ms": neural[2],
        "neural_p95_ms": neural[6],
        "neural_p99_ms": neural[7],
        "algo_stage": algo_stage_name,
        "algo_avg_ms": algo_stage[0],
        "algo_max_ms": algo_stage[2],
        "algo_p95_ms": algo_stage[6],
        "algo_p99_ms": algo_stage[7],
        "algo_count": algo_stage[3],
        "opencv_cuda_orb_avg_ms": opencv_cuda_orb[0],
        "opencv_cuda_orb_max_ms": opencv_cuda_orb[2],
        "opencv_cuda_orb_p95_ms": opencv_cuda_orb[6],
        "opencv_cuda_orb_p99_ms": opencv_cuda_orb[7],
        "opencv_cuda_template_avg_ms": opencv_cuda_template[0],
        "opencv_cuda_template_max_ms": opencv_cuda_template[2],
        "opencv_cuda_template_p95_ms": opencv_cuda_template[6],
        "opencv_cuda_template_p99_ms": opencv_cuda_template[7],
        "opencv_cuda_stereo_bm_avg_ms": opencv_cuda_bm[0],
        "opencv_cuda_stereo_bm_max_ms": opencv_cuda_bm[2],
        "opencv_cuda_stereo_bm_p95_ms": opencv_cuda_bm[6],
        "opencv_cuda_stereo_bm_p99_ms": opencv_cuda_bm[7],
        "opencv_cuda_stereo_sgm_avg_ms": opencv_cuda_sgm[0],
        "opencv_cuda_stereo_sgm_max_ms": opencv_cuda_sgm[2],
        "opencv_cuda_stereo_sgm_p95_ms": opencv_cuda_sgm[6],
        "opencv_cuda_stereo_sgm_p99_ms": opencv_cuda_sgm[7],
        "cpu_opencv_avg_ms": cpu_opencv[0],
        "cpu_opencv_max_ms": cpu_opencv[2],
        "cpu_opencv_orb_avg_ms": cpu_opencv_orb[0],
        "cpu_opencv_brisk_avg_ms": cpu_opencv_brisk[0],
        "cpu_opencv_akaze_avg_ms": cpu_opencv_akaze[0],
        "cpu_opencv_sift_avg_ms": cpu_opencv_sift[0],
        "cpu_fallback_avg_ms": cpu_fallback[0],
        "cpu_fallback_max_ms": cpu_fallback[2],
        "cpu_fallback_count": cpu_fallback[3],
        "async_worker_avg_ms": async_worker[0],
        "async_worker_max_ms": async_worker[2],
        "async_worker_p95_ms": async_worker[6],
        "async_worker_p99_ms": async_worker[7],
        "async_worker_count": async_worker[3],
        "async_over_deadline_count": async_over_deadline[3],
        "async_drop_stale_count": async_drop_stale[3],
        "async_drop_stale_ready_count": async_drop_stale_ready[3],
        "async_drop_expired_pending_count": async_drop_expired_pending[3],
        "async_drop_pending_count": async_drop_pending[3],
        "async_drop_no_buffer_count": async_drop_no_buffer[3],
        "async_submit_drop_count": async_submit_drop[3],
        "async_no_detections_count": async_no_detections[3],
        "async_submitted_count": async_submitted[3],
        "async_accepted_count": async_accepted[3],
        "async_accepted_reused_count": async_accepted_reused[3],
        "async_frame_callback_skipped_count": async_frame_cb_skipped[3],
        "async_need_host_gray_count": async_need_host_gray[3],
        "async_need_bgr_count": async_need_bgr[3],
        "async_skip_host_gray_selective_count": async_skip_host_gray_selective[3],
        "async_skip_bgr_selective_count": async_skip_bgr_selective[3],
        "async_host_gray_submit_avg_ms": async_host_gray_submit[0],
        "async_host_gray_submit_count": async_host_gray_submit[3],
        "async_gray_submit_avg_ms": async_gray_submit[0],
        "async_bgr_submit_avg_ms": async_bgr_submit[0],
        "async_copy_wait_avg_ms": async_copy_wait[0],
        "async_slot_copy_wait_avg_ms": async_slot_copy_wait[0],
        "async_worker_busy_count": async_worker_busy[3],
        "stage2_drop_stale_roi_count": stage2_drop_stale_roi[3],
        "p2_configured_count": p2_configured[3],
        "p2_realtime_requested_count": p2_realtime_requested[3],
        "p2_realtime_not_attempted_count": p2_realtime_not_attempted[3],
        "p2_not_triggered_count": p2_not_triggered[3],
        "p2_skip_selective_count": p2_skip_selective[3],
        "p2_trigger_pair_low_iou_count": p2_trigger_pair_low_iou[3],
        "p2_trigger_pair_epipolar_dy_count": p2_trigger_pair_epipolar_dy[3],
        "p2_trigger_pair_low_conf_count": p2_trigger_pair_low_conf[3],
        "p2_trigger_no_valid_pair_count": p2_trigger_no_valid_pair[3],
        "p2_inline_skipped_selective_count": p2_inline_skipped_selective[3],
        "candidate_rows": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "target_rows": "",
        "p2_diag_rows": "",
        "p2_diag_valid": "",
        "p2_diag_invalid": "",
        "p2_diag_rate": "",
        "p2_diag_over_deadline": "",
        "p2_diag_median_m": "",
        "p2_diag_mad_m": "",
        "p2_diag_support_median": "",
        "p2_diag_attempted_median": "",
        "p2_diag_status_counts": "",
        "p2_diag_mode_counts": "",
        "p2_diag_path": "",
        "p2_diag_debug_match_rows": "",
        "p2_diag_debug_matches_median": "",
        "p2_diag_artifacts": "",
        "p2_diag_artifact_dirs": "",
        "diagnosis": "",
        "debug_feature_dir": "",
        "debug_realtime_dir": "",
        "debug_feature_rc": "",
        "debug_realtime_rc": "",
        "neural_stub_or_unbound": "yes" if neural_unbound else "no",
        "log": str(log_path),
        "note": case.note,
        "last_error_or_warn": error_lines[-1][-220:] if error_lines else "",
    }


def skipped_row(case: Case, reason: str, log_path: Path) -> dict[str, str]:
    log_path.write_text(reason + "\n")
    return {
        "case": case.name,
        "status": "skipped_missing_engine",
        "return_code": "",
        "fps_last": "",
        "stale_drop_last": "",
        "gpu_candidates_avg_ms": "",
        "gpu_candidates_max_ms": "",
        "dual_yolo_match_avg_ms": "",
        "dual_yolo_match_max_ms": "",
        "subpixel_avg_ms": "",
        "subpixel_max_ms": "",
        "subpixel_p95_ms": "",
        "subpixel_p99_ms": "",
        "neural_avg_ms": "",
        "neural_max_ms": "",
        "neural_p95_ms": "",
        "neural_p99_ms": "",
        "algo_stage": profile_stage_for_case(case),
        "algo_avg_ms": "",
        "algo_max_ms": "",
        "algo_p95_ms": "",
        "algo_p99_ms": "",
        "algo_count": "",
        "opencv_cuda_orb_avg_ms": "",
        "opencv_cuda_orb_max_ms": "",
        "opencv_cuda_orb_p95_ms": "",
        "opencv_cuda_orb_p99_ms": "",
        "opencv_cuda_template_avg_ms": "",
        "opencv_cuda_template_max_ms": "",
        "opencv_cuda_template_p95_ms": "",
        "opencv_cuda_template_p99_ms": "",
        "opencv_cuda_stereo_bm_avg_ms": "",
        "opencv_cuda_stereo_bm_max_ms": "",
        "opencv_cuda_stereo_bm_p95_ms": "",
        "opencv_cuda_stereo_bm_p99_ms": "",
        "opencv_cuda_stereo_sgm_avg_ms": "",
        "opencv_cuda_stereo_sgm_max_ms": "",
        "opencv_cuda_stereo_sgm_p95_ms": "",
        "opencv_cuda_stereo_sgm_p99_ms": "",
        "cpu_opencv_avg_ms": "",
        "cpu_opencv_max_ms": "",
        "cpu_opencv_orb_avg_ms": "",
        "cpu_opencv_brisk_avg_ms": "",
        "cpu_opencv_akaze_avg_ms": "",
        "cpu_opencv_sift_avg_ms": "",
        "cpu_fallback_avg_ms": "",
        "cpu_fallback_max_ms": "",
        "cpu_fallback_count": "",
        "async_worker_avg_ms": "",
        "async_worker_max_ms": "",
        "async_worker_p95_ms": "",
        "async_worker_p99_ms": "",
        "async_worker_count": "",
        "async_over_deadline_count": "",
        "async_drop_stale_count": "",
        "async_drop_stale_ready_count": "",
        "async_drop_expired_pending_count": "",
        "async_drop_pending_count": "",
        "async_drop_no_buffer_count": "",
        "async_submit_drop_count": "",
        "async_no_detections_count": "",
        "async_submitted_count": "",
        "async_accepted_count": "",
        "async_accepted_reused_count": "",
        "async_frame_callback_skipped_count": "",
        "async_need_host_gray_count": "",
        "async_need_bgr_count": "",
        "async_skip_host_gray_selective_count": "",
        "async_skip_bgr_selective_count": "",
        "async_host_gray_submit_avg_ms": "",
        "async_host_gray_submit_count": "",
        "async_gray_submit_avg_ms": "",
        "async_bgr_submit_avg_ms": "",
        "async_copy_wait_avg_ms": "",
        "async_slot_copy_wait_avg_ms": "",
        "async_worker_busy_count": "",
        "stage2_drop_stale_roi_count": "",
        "p2_configured_count": "",
        "p2_realtime_requested_count": "",
        "p2_realtime_not_attempted_count": "",
        "p2_not_triggered_count": "",
        "p2_skip_selective_count": "",
        "p2_trigger_pair_low_iou_count": "",
        "p2_trigger_pair_epipolar_dy_count": "",
        "p2_trigger_pair_low_conf_count": "",
        "p2_trigger_no_valid_pair_count": "",
        "p2_inline_skipped_selective_count": "",
        "candidate_rows": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "target_rows": "",
        "p2_diag_rows": "",
        "p2_diag_valid": "",
        "p2_diag_invalid": "",
        "p2_diag_rate": "",
        "p2_diag_over_deadline": "",
        "p2_diag_median_m": "",
        "p2_diag_mad_m": "",
        "p2_diag_support_median": "",
        "p2_diag_attempted_median": "",
        "p2_diag_status_counts": "",
        "p2_diag_mode_counts": "",
        "p2_diag_path": "",
        "p2_diag_debug_match_rows": "",
        "p2_diag_debug_matches_median": "",
        "p2_diag_artifacts": "",
        "p2_diag_artifact_dirs": "",
        "diagnosis": "skipped_missing_dependency",
        "debug_feature_dir": "",
        "debug_realtime_dir": "",
        "debug_feature_rc": "",
        "debug_realtime_rc": "",
        "neural_stub_or_unbound": "no",
        "log": str(log_path),
        "note": case.note,
        "last_error_or_warn": reason,
    }


def run_case(project: Path, binary: Path, duration_sec: int, case: Case, cfg: Path, log_dir: Path) -> dict[str, str]:
    log_path = log_dir / f"{case.name}.log"
    cmd = ["timeout", str(duration_sec), str(binary), "--config", str(cfg)]
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=str(project), stdout=log_file, stderr=subprocess.STDOUT)
    row = parse_log(case, log_path.read_text(errors="replace"), proc.returncode, log_path)
    out_dir = cfg.parent.parent
    row.update(summarize_candidate_csv(
        case,
        out_dir / f"{case.name}.csv",
        out_dir / f"{case.name}.frames.csv",
    ))
    row.update(summarize_diagnostic_csv(
        out_dir / f"{case.name}.p2_diagnostic.csv",
    ))
    row["diagnosis"] = classify_case_result(row)
    return row


def run_debug_captures(
    project: Path,
    binary: Path,
    case: Case,
    cfg: Path,
    out_dir: Path,
    log_dir: Path,
    realtime_sec: int,
) -> dict[str, str]:
    debug_root = out_dir / "debug" / case.name
    feature_dir = debug_root / "feature_matches"
    realtime_dir = debug_root / "realtime_zoom"
    p2_artifact_dir = debug_root / "p2_artifacts"
    feature_dir.mkdir(parents=True, exist_ok=True)
    realtime_dir.mkdir(parents=True, exist_ok=True)
    p2_artifact_dir.mkdir(parents=True, exist_ok=True)

    debug_cfg = debug_root / f"{case.name}.debug.yaml"
    text = cfg.read_text()
    if "P2FeatureJobDiagnostic" in (case.algo_stage_override or ""):
        text = set_yaml_scalar(text, "p2_diagnostic_results_enabled", "true")
        text = set_yaml_scalar(
            text,
            "p2_diagnostic_results_path",
            f'"{debug_root / (case.name + ".p2_diagnostic.csv")}"',
        )
    text = set_yaml_scalar(text, "p2_diagnostic_artifacts_enabled", "true")
    text = set_yaml_scalar(
        text,
        "p2_diagnostic_artifacts_dir",
        f'"{p2_artifact_dir}"',
    )
    text = set_yaml_scalar(text, "p2_diagnostic_artifacts_max", "20")
    debug_cfg.write_text(text)

    feature_log = log_dir / f"{case.name}.debug_feature_matches.log"
    feature_cmd = [
        "timeout", "12", str(binary),
        "--config", str(debug_cfg),
        "--debug-feature-matches",
        "--debug-feature-matches-dir", str(feature_dir),
    ]
    with feature_log.open("w") as log_file:
        feature_proc = subprocess.run(
            feature_cmd, cwd=str(project),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    realtime_log = log_dir / f"{case.name}.debug_realtime_dump.log"
    realtime_cmd = [
        "timeout", str(realtime_sec), str(binary),
        "--config", str(debug_cfg),
        "--debug-realtime-dump",
        "--debug-realtime-dump-dir", str(realtime_dir),
        "--debug-realtime-dump-stride", "1",
        "--debug-realtime-dump-max", "20",
    ]
    with realtime_log.open("w") as log_file:
        realtime_proc = subprocess.run(
            realtime_cmd, cwd=str(project),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    return {
        "debug_feature_dir": str(feature_dir),
        "debug_realtime_dir": str(realtime_dir),
        "debug_p2_artifact_dir": str(p2_artifact_dir),
        "debug_feature_rc": str(feature_proc.returncode),
        "debug_realtime_rc": str(realtime_proc.returncode),
    }


def write_reports(out_dir: Path, rows: list[dict[str, str]], duration_sec: int, project: Path) -> None:
    summary_csv = out_dir / "summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# NX algorithm matrix report",
        "",
        f"- project: {project}",
        f"- duration per case: {duration_sec}s",
        "- mode: isolated true algorithms; all dual-YOLO depth modes are disabled before each case enables its own candidate",
        "- `algo_*` is the profiler stage selected for this case; `async_worker_*` is full async Stage2. `p95` is the main tail-latency gate.",
        "- `late_or_deadline_dropped` means the worker finished late or the result was discarded after the next-frame deadline; CUDA/CPU work is not killed mid-kernel.",
        "- `realtime_path_used_cpu_or_host_gray` means the supposedly realtime P2 run triggered CPU fallback or host gray D2H and must be rerun with those paths disabled.",
        "- `debug_dirs` is populated only when `--debug-on-failure` or `--debug-all` is used.",
        "- `debug/<case>/realtime_zoom` is a realtime status view: bbox/circle/field overlay only, not an algorithm-specific feature-match overlay.",
        "- `debug/<case>/feature_matches` currently runs the legacy CPU sparse/OpenCV debug path; it must not be used as proof of OpenCV CUDA, VPI, TensorRT, libSGM, or custom CUDA P2 internal matches.",
        "- Diagnostic-only P2 cases write `<case>.p2_diagnostic.csv` plus `<case>.p2_artifacts/` overlays from backend-exposed debug matches. With `--debug-on-failure`, the rerun also writes `debug/<case>/p2_artifacts/`; realtime status zoom remains bbox/circle only.",
        "",
        "| case | diagnosis | status | fps | algo_stage | algo avg/p95/max | worker avg/p95/max | over_deadline | stale/expired | queue_drop | candidate_valid/frames | diag_valid/rows | diag_over_deadline | debug match rows/median | artifacts | p2 triggers | selective skip | rate | median/MAD | support | accepted | frame_cb_skip | host_gray | debug_dirs | note | last error/warn |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        has_diag_rows = bool(row.get("p2_diag_rows"))
        rate_value = (
            row.get("p2_diag_rate", "")
            if has_diag_rows else row.get("candidate_rate", "")
        )
        median_value = (
            row.get("p2_diag_median_m", "")
            if has_diag_rows else row.get("candidate_median_m", "")
        )
        mad_value = (
            row.get("p2_diag_mad_m", "")
            if has_diag_rows else row.get("candidate_mad_m", "")
        )
        support_value = (
            row.get("p2_diag_support_median", "")
            if has_diag_rows else row.get("support_median", "")
        )
        stale_expired = (
            row_int(row, "async_drop_stale_count") +
            row_int(row, "async_drop_stale_ready_count") +
            row_int(row, "async_drop_expired_pending_count") +
            row_int(row, "stage2_drop_stale_roi_count")
        )
        queue_drop = (
            row_int(row, "async_drop_pending_count") +
            row_int(row, "async_drop_no_buffer_count") +
            row_int(row, "async_submit_drop_count")
        )
        debug_dirs = ""
        debug_parts = []
        if row.get("debug_feature_dir"):
            debug_parts.append(f'feature={row.get("debug_feature_dir", "")}')
        if row.get("debug_realtime_dir"):
            debug_parts.append(f'realtime={row.get("debug_realtime_dir", "")}')
        if row.get("debug_p2_artifact_dir"):
            debug_parts.append(f'p2_debug={row.get("debug_p2_artifact_dir", "")}')
        if row.get("p2_diag_artifact_dirs"):
            debug_parts.append(f'p2_artifacts={row.get("p2_diag_artifact_dirs", "")}')
        if debug_parts:
            debug_dirs = " ".join(debug_parts)
        p2_triggers = []
        for key, label in (
            ("p2_trigger_pair_low_iou_count", "low_iou"),
            ("p2_trigger_pair_epipolar_dy_count", "dy"),
            ("p2_trigger_pair_low_conf_count", "low_conf"),
            ("p2_trigger_no_valid_pair_count", "no_pair"),
            ("p2_skip_selective_count", "skip"),
        ):
            count = row_int(row, key)
            if count:
                p2_triggers.append(f"{label}={count}")
        selective_skips = []
        for key, label in (
            ("p2_inline_skipped_selective_count", "inline"),
            ("async_skip_bgr_selective_count", "bgr"),
            ("async_skip_host_gray_selective_count", "host"),
        ):
            count = row_int(row, key)
            if count:
                selective_skips.append(f"{label}={count}")
        values = [
            row.get("case", ""),
            row.get("diagnosis", ""),
            row.get("status", ""),
            row.get("fps_last", ""),
            row.get("algo_stage", ""),
            f'{row.get("algo_avg_ms", "")}/{row.get("algo_p95_ms", "")}/{row.get("algo_max_ms", "")}',
            f'{row.get("async_worker_avg_ms", "")}/{row.get("async_worker_p95_ms", "")}/{row.get("async_worker_max_ms", "")}',
            row.get("async_over_deadline_count", ""),
            str(stale_expired) if stale_expired else "",
            str(queue_drop) if queue_drop else "",
            f'{row.get("candidate_valid", "")}/{row.get("candidate_rows", "")}'
            if row.get("candidate_rows") else "",
            f'{row.get("p2_diag_valid", "")}/{row.get("p2_diag_rows", "")}'
            if row.get("p2_diag_rows") else "",
            row.get("p2_diag_over_deadline", ""),
            f'{row.get("p2_diag_debug_match_rows", "")}/{row.get("p2_diag_debug_matches_median", "")}'
            if row.get("p2_diag_debug_match_rows") else "",
            row.get("p2_diag_artifacts", ""),
            ";".join(p2_triggers),
            ";".join(selective_skips),
            rate_value,
            f"{median_value}/{mad_value}",
            support_value,
            row.get("async_accepted_count", ""),
            row.get("async_frame_callback_skipped_count", ""),
            f'{row.get("async_need_host_gray_count", "")}/{row.get("async_host_gray_submit_count", "")}',
            debug_dirs,
            row.get("note", ""),
            row.get("last_error_or_warn", "").replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_static_findings(project: Path, out_dir: Path) -> None:
    files = [
        path
        for path in (project / "src").rglob("*")
        if path.suffix in {".cpp", ".cu", ".h", ".hpp"}
    ]
    combined = "\n".join(path.read_text(errors="ignore") for path in files)
    findings = {
        "neural_matchGpuRoi_stub": "tensor_binding_not_implemented" in combined,
        "realtime_iou_region_color_patch_symbol": "iou_region_color_patch" in combined,
        "realtime_patch_iou_color_edge_symbol": "patch_iou_color_edge" in combined,
        "realtime_cuda_template_match_symbol": "roi_cuda_template_match" in combined,
        "realtime_cuda_stereo_bm_symbol": "roi_cuda_stereo_bm" in combined,
        "realtime_cuda_stereo_sgm_symbol": "roi_cuda_stereo_sgm" in combined,
        "realtime_sift_symbol": re.search(r"\bsift\b", combined, re.I) is not None,
    }
    with (out_dir / "static_findings.txt").open("w") as f:
        for key, value in findings.items():
            f.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/nvidia/NX_volleyball/stereo_3d_pipeline"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/codex_algo_tests"))
    parser.add_argument("--duration-sec", type=int, default=8)
    parser.add_argument(
        "--include-approx",
        action="store_true",
        help="also run custom sparse-lite diagnostic cases; these are not true OpenCV feature algorithms",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="also run relaxed-gate diagnostic cases; not production-quality results",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case names to run after filtering; default runs all formal cases",
    )
    parser.add_argument(
        "--neural-model-dir",
        type=Path,
        default=Path("/home/nvidia/NX_volleyball/stereo_3d_pipeline/models/neural"),
    )
    parser.add_argument(
        "--debug-on-failure",
        action="store_true",
        help=(
            "after a failed/no-valid/deadline case, capture legacy CPU "
            "feature debug images and a short realtime status zoom dump"
        ),
    )
    parser.add_argument(
        "--debug-all",
        action="store_true",
        help=(
            "capture debug outputs for every selected case; use only for "
            "diagnostic reruns, not formal FPS admission"
        ),
    )
    parser.add_argument(
        "--debug-realtime-sec",
        type=int,
        default=5,
        help="seconds for --debug-on-failure realtime zoom dump",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project
    out_dir = args.out.expanduser().resolve()
    base_config = project / "config/pipeline_dual_yolo_roi.yaml"
    binary = project / "build/stereo_pipeline"
    config_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = base_config.read_text()

    rows: list[dict[str, str]] = []
    cases = list(CASES)
    if args.include_approx:
        cases.extend(APPROX_CASES)
    if args.include_experimental:
        cases.extend(RELAXED_CASES)
    if args.cases.strip():
        requested = {
            name.strip()
            for name in args.cases.split(",")
            if name.strip()
        }
        known = {case.name for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"unknown case(s): {', '.join(missing)}")
        cases = [case for case in cases if case.name in requested]

    for case in cases:
        if case.neural_engine:
            engine_path = args.neural_model_dir / case.neural_engine
            if not engine_path.exists():
                reason = f"missing neural TensorRT engine: {engine_path}"
                print(f"[SKIP] {case.name}: {reason}", flush=True)
                rows.append(skipped_row(case, reason, log_dir / f"{case.name}.log"))
                continue
        cfg = prepare_config(base, case, out_dir, config_dir, args.neural_model_dir)
        print(f"[RUN] {case.name}", flush=True)
        row = run_case(project, binary, args.duration_sec, case, cfg, log_dir)
        if args.debug_all or (args.debug_on_failure and should_debug_case(row)):
            print(f"[DEBUG] {case.name}: {row['diagnosis']}", flush=True)
            row.update(run_debug_captures(
                project, binary, case, cfg, out_dir, log_dir,
                args.debug_realtime_sec,
            ))
        rows.append(row)

    write_reports(out_dir, rows, args.duration_sec, project)
    write_static_findings(project, out_dir)
    print(out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
