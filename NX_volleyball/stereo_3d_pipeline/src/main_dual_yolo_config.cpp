#include "main_dual_yolo_config.h"

#include <string>

void loadDualYoloConfig(const YAML::Node& dual, stereo3d::PipelineConfig& cfg) {
    if (!dual) return;

    if (dual["enabled"]) cfg.dual_yolo.enabled = dual["enabled"].as<bool>();
    if (dual["right_engine_path"])
        cfg.dual_yolo.right_engine_file = dual["right_engine_path"].as<std::string>();
    if (dual["right_engine_file"])
        cfg.dual_yolo.right_engine_file = dual["right_engine_file"].as<std::string>();
    if (dual["right_input_format"])
        cfg.dual_yolo.right_input_format = dual["right_input_format"].as<std::string>();
    cfg.dual_yolo.right_use_dla = cfg.use_dla;
    if (dual["right_use_dla"])
        cfg.dual_yolo.right_use_dla = dual["right_use_dla"].as<bool>();
    if (dual["right_dla_core"])
        cfg.dual_yolo.right_dla_core = dual["right_dla_core"].as<int>();
    if (dual["use_for_depth"])
        cfg.dual_yolo.use_for_depth = dual["use_for_depth"].as<bool>();
    if (dual["fallback_to_roi_match"])
        cfg.dual_yolo.fallback_to_roi_match = dual["fallback_to_roi_match"].as<bool>();
    if (dual["gpu_candidate_refine"])
        cfg.dual_yolo.gpu_candidate_refine = dual["gpu_candidate_refine"].as<bool>();
    if (dual["fallback_epipolar_search"]) {
        cfg.dual_yolo.fallback_epipolar_search = dual["fallback_epipolar_search"].as<bool>();
        cfg.dual_yolo.depth_epipolar_fallback = cfg.dual_yolo.fallback_epipolar_search;
    }
    if (dual["center_refine"])
        cfg.dual_yolo.center_refine = dual["center_refine"].as<bool>();
    if (dual["roi_denoise"])
        cfg.dual_yolo.roi_denoise = dual["roi_denoise"].as<bool>();
    if (dual["log_matches"])
        cfg.dual_yolo.log_matches = dual["log_matches"].as<bool>();
    if (dual["depth_solver"])
        cfg.dual_yolo.depth_solver = dual["depth_solver"].as<std::string>();
    if (dual["subpixel_enabled"]) {
        cfg.dual_yolo.subpixel_enabled = dual["subpixel_enabled"].as<bool>();
        cfg.dual_yolo.depth_roi_subpixel = cfg.dual_yolo.subpixel_enabled;
    }
    if (auto modes = dual["depth_modes"]) {
        if (modes["bbox_pair"])
            cfg.dual_yolo.depth_bbox_pair = modes["bbox_pair"].as<bool>();
        if (modes["bbox_edges"])
            cfg.dual_yolo.depth_bbox_edges = modes["bbox_edges"].as<bool>();
        if (modes["circle_center"])
            cfg.dual_yolo.depth_circle_center = modes["circle_center"].as<bool>();
        if (modes["circle_edges"])
            cfg.dual_yolo.depth_circle_edges = modes["circle_edges"].as<bool>();
        if (modes["roi_edge_centroid"])
            cfg.dual_yolo.depth_roi_edge_centroid = modes["roi_edge_centroid"].as<bool>();
        if (modes["roi_radial_center"])
            cfg.dual_yolo.depth_roi_radial_center = modes["roi_radial_center"].as<bool>();
        if (modes["roi_edge_pair_center"])
            cfg.dual_yolo.depth_roi_edge_pair_center = modes["roi_edge_pair_center"].as<bool>();
        if (modes["roi_corner_points"])
            cfg.dual_yolo.depth_roi_corner_points = modes["roi_corner_points"].as<bool>();
        if (modes["roi_texture_points"])
            cfg.dual_yolo.depth_roi_texture_points = modes["roi_texture_points"].as<bool>();
        if (modes["roi_binary_points"])
            cfg.dual_yolo.depth_roi_binary_points = modes["roi_binary_points"].as<bool>();
        if (modes["roi_orb_points"])
            cfg.dual_yolo.depth_roi_orb_points = modes["roi_orb_points"].as<bool>();
        if (modes["roi_brisk_points"])
            cfg.dual_yolo.depth_roi_brisk_points = modes["roi_brisk_points"].as<bool>();
        if (modes["roi_akaze_points"])
            cfg.dual_yolo.depth_roi_akaze_points = modes["roi_akaze_points"].as<bool>();
        if (modes["roi_sift_points"])
            cfg.dual_yolo.depth_roi_sift_points = modes["roi_sift_points"].as<bool>();
        if (modes["roi_iou_region_color_patch"])
            cfg.dual_yolo.depth_roi_iou_region_color_patch =
                modes["roi_iou_region_color_patch"].as<bool>();
        if (modes["roi_patch_iou_color_edge"])
            cfg.dual_yolo.depth_roi_patch_iou_color_edge =
                modes["roi_patch_iou_color_edge"].as<bool>();
        if (modes["roi_cuda_template_match"])
            cfg.dual_yolo.depth_roi_cuda_template_match =
                modes["roi_cuda_template_match"].as<bool>();
        if (modes["roi_cuda_stereo_bm"])
            cfg.dual_yolo.depth_roi_cuda_stereo_bm =
                modes["roi_cuda_stereo_bm"].as<bool>();
        if (modes["roi_cuda_stereo_sgm"])
            cfg.dual_yolo.depth_roi_cuda_stereo_sgm =
                modes["roi_cuda_stereo_sgm"].as<bool>();
        if (modes["roi_ring_edge_profile"])
            cfg.dual_yolo.depth_roi_ring_edge_profile =
                modes["roi_ring_edge_profile"].as<bool>();
        if (modes["roi_vpi_template_match"])
            cfg.dual_yolo.depth_roi_vpi_template_match =
                modes["roi_vpi_template_match"].as<bool>();
        if (modes["roi_vpi_stereo_disparity"])
            cfg.dual_yolo.depth_roi_vpi_stereo_disparity =
                modes["roi_vpi_stereo_disparity"].as<bool>();
        if (modes["roi_vpi_harris_lk"])
            cfg.dual_yolo.depth_roi_vpi_harris_lk =
                modes["roi_vpi_harris_lk"].as<bool>();
        if (modes["roi_vpi_orb"])
            cfg.dual_yolo.depth_roi_vpi_orb =
                modes["roi_vpi_orb"].as<bool>();
        if (modes["roi_cuda_gftt_lk"])
            cfg.dual_yolo.depth_roi_cuda_gftt_lk =
                modes["roi_cuda_gftt_lk"].as<bool>();
        if (modes["roi_cuda_sift"])
            cfg.dual_yolo.depth_roi_cuda_sift =
                modes["roi_cuda_sift"].as<bool>();
        if (modes["roi_libsgm"])
            cfg.dual_yolo.depth_roi_libsgm =
                modes["roi_libsgm"].as<bool>();
        if (modes["roi_cuda_hough_circle"])
            cfg.dual_yolo.depth_roi_cuda_hough_circle =
                modes["roi_cuda_hough_circle"].as<bool>();
        if (modes["roi_center_patch"])
            cfg.dual_yolo.depth_roi_center_patch = modes["roi_center_patch"].as<bool>();
        if (modes["roi_subpixel"])
            cfg.dual_yolo.depth_roi_subpixel = modes["roi_subpixel"].as<bool>();
        if (modes["epipolar_fallback"])
            cfg.dual_yolo.depth_epipolar_fallback = modes["epipolar_fallback"].as<bool>();
        if (modes["fallback_template"])
            cfg.dual_yolo.depth_fallback_template = modes["fallback_template"].as<bool>();
        if (modes["fallback_feature_points"])
            cfg.dual_yolo.depth_fallback_feature_points =
                modes["fallback_feature_points"].as<bool>();
    }
    if (dual["subpixel_patch_radius"])
        cfg.dual_yolo.subpixel_patch_radius = dual["subpixel_patch_radius"].as<int>();
    if (dual["subpixel_search_radius_px"])
        cfg.dual_yolo.subpixel_search_radius_px = dual["subpixel_search_radius_px"].as<int>();
    if (dual["subpixel_max_points"])
        cfg.dual_yolo.subpixel_max_points = dual["subpixel_max_points"].as<int>();
    if (dual["subpixel_min_points"])
        cfg.dual_yolo.subpixel_min_points = dual["subpixel_min_points"].as<int>();
    if (dual["subpixel_min_confidence"])
        cfg.dual_yolo.subpixel_min_confidence = dual["subpixel_min_confidence"].as<float>();
    if (dual["subpixel_max_disp_delta_px"])
        cfg.dual_yolo.subpixel_max_disp_delta_px =
            dual["subpixel_max_disp_delta_px"].as<float>();
    if (dual["subpixel_max_disp_delta_ratio"])
        cfg.dual_yolo.subpixel_max_disp_delta_ratio =
            dual["subpixel_max_disp_delta_ratio"].as<float>();
    if (dual["subpixel_max_depth_delta_m"])
        cfg.dual_yolo.subpixel_max_depth_delta_m =
            dual["subpixel_max_depth_delta_m"].as<float>();
    if (dual["subpixel_max_stddev_px"])
        cfg.dual_yolo.subpixel_max_stddev_px = dual["subpixel_max_stddev_px"].as<float>();
    if (dual["subpixel_time_budget_ms"])
        cfg.dual_yolo.subpixel_time_budget_ms = dual["subpixel_time_budget_ms"].as<float>();
    if (dual["epipolar_y_tolerance"])
        cfg.dual_yolo.epipolar_y_tolerance = dual["epipolar_y_tolerance"].as<float>();
    if (dual["feature_y_tolerance_px"])
        cfg.dual_yolo.feature_y_tolerance_px = dual["feature_y_tolerance_px"].as<float>();
    if (dual["feature_y_slope"])
        cfg.dual_yolo.feature_y_slope = dual["feature_y_slope"].as<float>();
    if (dual["feature_y_offset_px"])
        cfg.dual_yolo.feature_y_offset_px = dual["feature_y_offset_px"].as<float>();
    if (dual["feature_reverse_check_px"])
        cfg.dual_yolo.feature_reverse_check_px = dual["feature_reverse_check_px"].as<float>();
    if (dual["feature_overlap_scale"])
        cfg.dual_yolo.feature_overlap_scale = dual["feature_overlap_scale"].as<float>();
    if (dual["feature_mad_scale"])
        cfg.dual_yolo.feature_mad_scale = dual["feature_mad_scale"].as<float>();
    if (dual["feature_ransac_gate_px"])
        cfg.dual_yolo.feature_ransac_gate_px = dual["feature_ransac_gate_px"].as<float>();
    if (dual["feature_sphere_radius_scale"])
        cfg.dual_yolo.feature_sphere_radius_scale =
            dual["feature_sphere_radius_scale"].as<float>();
    if (dual["feature_sphere_margin_m"])
        cfg.dual_yolo.feature_sphere_margin_m = dual["feature_sphere_margin_m"].as<float>();
    if (dual["feature_normalize_large_roi"])
        cfg.dual_yolo.feature_normalize_large_roi =
            dual["feature_normalize_large_roi"].as<bool>();
    if (dual["feature_normalized_diameter_px"])
        cfg.dual_yolo.feature_normalized_diameter_px =
            dual["feature_normalized_diameter_px"].as<int>();
    if (dual["feature_normalize_min_diameter_px"])
        cfg.dual_yolo.feature_normalize_min_diameter_px =
            dual["feature_normalize_min_diameter_px"].as<float>();
    if (dual["feature_normalize_margin_scale"])
        cfg.dual_yolo.feature_normalize_margin_scale =
            dual["feature_normalize_margin_scale"].as<float>();
    if (dual["feature_precompute_roi_maps"])
        cfg.dual_yolo.feature_precompute_roi_maps =
            dual["feature_precompute_roi_maps"].as<bool>();
    if (dual["max_size_ratio"])
        cfg.dual_yolo.max_size_ratio = dual["max_size_ratio"].as<float>();
    if (dual["min_shifted_iou"])
        cfg.dual_yolo.min_shifted_iou = dual["min_shifted_iou"].as<float>();
    if (dual["bbox_disparity_consistency_ratio"])
        cfg.dual_yolo.bbox_disparity_consistency_ratio =
            dual["bbox_disparity_consistency_ratio"].as<float>();
    if (dual["bbox_disparity_consistency_min_px"])
        cfg.dual_yolo.bbox_disparity_consistency_min_px =
            dual["bbox_disparity_consistency_min_px"].as<float>();
    if (dual["bbox_disparity_penalty_scale"])
        cfg.dual_yolo.bbox_disparity_penalty_scale =
            dual["bbox_disparity_penalty_scale"].as<float>();
    if (dual["fallback_search_margin_px"])
        cfg.dual_yolo.fallback_search_margin_px = dual["fallback_search_margin_px"].as<int>();
    if (dual["fallback_max_width_px"])
        cfg.dual_yolo.fallback_max_width_px = dual["fallback_max_width_px"].as<int>();
    if (dual["circle_max_roi_pixels"])
        cfg.dual_yolo.circle_max_roi_pixels = dual["circle_max_roi_pixels"].as<int>();
}
