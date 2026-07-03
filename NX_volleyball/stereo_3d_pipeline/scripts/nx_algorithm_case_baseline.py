"""Baseline and approximate NX algorithm matrix cases."""

from __future__ import annotations

from nx_algorithm_case_types import Case


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
