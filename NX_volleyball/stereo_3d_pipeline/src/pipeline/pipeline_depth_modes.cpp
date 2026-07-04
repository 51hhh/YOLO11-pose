#include "pipeline_depth_modes.h"

#include <algorithm>
#include <cctype>

namespace stereo3d {

bool isBGRFormat(std::string fmt) {
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return fmt == "bgr";
}

bool isROISubpixelDepthSolver(std::string solver) {
    std::transform(solver.begin(), solver.end(), solver.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return solver == "roi_subpixel_match" ||
           solver == "subpixel" ||
           solver == "multi_point";
}

bool dualYoloBBoxDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_bbox_pair;
}

bool dualYoloBBoxEdgesDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_bbox_edges;
}

bool dualYoloCircleDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_circle_center && cfg.center_refine;
}

bool dualYoloCircleEdgesDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_circle_edges && cfg.center_refine;
}

bool dualYoloROIEdgeCentroidDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_edge_centroid && cfg.center_refine;
}

bool dualYoloROIRadialCenterDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_radial_center;
}

bool dualYoloROIEdgePairCenterDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_edge_pair_center;
}

bool dualYoloROICornerPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_corner_points;
}

bool dualYoloROITexturePointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_texture_points;
}

bool dualYoloROIBinaryPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_binary_points;
}

bool dualYoloROIORBPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_orb_points;
}

bool dualYoloROIBRISKPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_brisk_points;
}

bool dualYoloROIAKAZEPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_akaze_points;
}

bool dualYoloROISIFTPointsDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_sift_points;
}

bool dualYoloROIIoURegionColorPatchDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_iou_region_color_patch;
}

bool dualYoloROIPatchIoUColorEdgeDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_patch_iou_color_edge;
}

bool dualYoloROICudaTemplateMatchDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_cuda_template_match;
}

bool dualYoloROICudaStereoBMDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_cuda_stereo_bm;
}

bool dualYoloROICudaStereoSGMDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_cuda_stereo_sgm;
}

bool dualYoloROIRingEdgeProfileDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_ring_edge_profile;
}

bool dualYoloROICenterPatchDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_center_patch && cfg.center_refine;
}

bool dualYoloSubpixelDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_roi_subpixel &&
           cfg.subpixel_enabled &&
           isROISubpixelDepthSolver(cfg.depth_solver);
}

bool dualYoloEpipolarFallbackEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_epipolar_fallback && cfg.fallback_epipolar_search;
}

bool dualYoloFallbackTemplateEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_fallback_template && cfg.fallback_epipolar_search;
}

bool dualYoloFallbackFeaturePointsEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.depth_fallback_feature_points && cfg.fallback_epipolar_search;
}

bool dualYoloOpenCVCpuDescriptorDepthEnabled(
    const PipelineConfig::DualYoloConfig& cfg) {
    return dualYoloROIBRISKPointsDepthEnabled(cfg) ||
           dualYoloROIAKAZEPointsDepthEnabled(cfg) ||
           dualYoloROISIFTPointsDepthEnabled(cfg);
}

bool dualYoloCpuFallbackSearchEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return dualYoloEpipolarFallbackEnabled(cfg) ||
           dualYoloFallbackTemplateEnabled(cfg) ||
           dualYoloFallbackFeaturePointsEnabled(cfg);
}

bool dualYoloNeedsCircleSeedRefine(const PipelineConfig::DualYoloConfig& cfg) {
    return cfg.center_refine &&
           (cfg.depth_circle_center ||
            cfg.depth_circle_edges ||
            cfg.depth_roi_edge_centroid ||
            cfg.depth_roi_center_patch ||
            dualYoloSubpixelDepthEnabled(cfg) ||
            dualYoloEpipolarFallbackEnabled(cfg) ||
            dualYoloFallbackTemplateEnabled(cfg) ||
            dualYoloFallbackFeaturePointsEnabled(cfg));
}

bool dualYoloAnyDepthModeEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return dualYoloBBoxDepthEnabled(cfg) ||
           dualYoloBBoxEdgesDepthEnabled(cfg) ||
           dualYoloCircleDepthEnabled(cfg) ||
           dualYoloCircleEdgesDepthEnabled(cfg) ||
           dualYoloROIEdgeCentroidDepthEnabled(cfg) ||
           dualYoloROIRadialCenterDepthEnabled(cfg) ||
           dualYoloROIEdgePairCenterDepthEnabled(cfg) ||
           dualYoloROICornerPointsDepthEnabled(cfg) ||
           dualYoloROITexturePointsDepthEnabled(cfg) ||
           dualYoloROIBinaryPointsDepthEnabled(cfg) ||
           dualYoloROIORBPointsDepthEnabled(cfg) ||
           dualYoloROIBRISKPointsDepthEnabled(cfg) ||
           dualYoloROIAKAZEPointsDepthEnabled(cfg) ||
           dualYoloROISIFTPointsDepthEnabled(cfg) ||
           dualYoloROIIoURegionColorPatchDepthEnabled(cfg) ||
           dualYoloROIPatchIoUColorEdgeDepthEnabled(cfg) ||
           dualYoloROICudaTemplateMatchDepthEnabled(cfg) ||
           dualYoloROICudaStereoBMDepthEnabled(cfg) ||
           dualYoloROICudaStereoSGMDepthEnabled(cfg) ||
           dualYoloROIRingEdgeProfileDepthEnabled(cfg) ||
           cfg.depth_roi_vpi_template_match ||
           cfg.depth_roi_vpi_stereo_disparity ||
           cfg.depth_roi_vpi_harris_lk ||
           cfg.depth_roi_vpi_orb ||
           cfg.depth_roi_cuda_gftt_lk ||
           cfg.depth_roi_cuda_sift ||
           cfg.depth_roi_libsgm ||
           cfg.depth_roi_cuda_hough_circle ||
           dualYoloROICenterPatchDepthEnabled(cfg) ||
           dualYoloSubpixelDepthEnabled(cfg) ||
           dualYoloEpipolarFallbackEnabled(cfg) ||
           dualYoloFallbackTemplateEnabled(cfg) ||
           dualYoloFallbackFeaturePointsEnabled(cfg);
}

bool dualYoloNeedsHostImages(const PipelineConfig::DualYoloConfig& cfg) {
    if (cfg.gpu_candidate_refine) {
        return dualYoloOpenCVCpuDescriptorDepthEnabled(cfg) ||
               dualYoloCpuFallbackSearchEnabled(cfg);
    }
    return dualYoloNeedsCircleSeedRefine(cfg) ||
           dualYoloROIRadialCenterDepthEnabled(cfg) ||
           dualYoloROIEdgePairCenterDepthEnabled(cfg) ||
           dualYoloROICornerPointsDepthEnabled(cfg) ||
           dualYoloROITexturePointsDepthEnabled(cfg) ||
           dualYoloROIBinaryPointsDepthEnabled(cfg) ||
           dualYoloROIORBPointsDepthEnabled(cfg) ||
           dualYoloROIBRISKPointsDepthEnabled(cfg) ||
           dualYoloROIAKAZEPointsDepthEnabled(cfg) ||
           dualYoloROISIFTPointsDepthEnabled(cfg) ||
           dualYoloROIIoURegionColorPatchDepthEnabled(cfg) ||
           dualYoloROIPatchIoUColorEdgeDepthEnabled(cfg) ||
           dualYoloSubpixelDepthEnabled(cfg) ||
           dualYoloEpipolarFallbackEnabled(cfg) ||
           dualYoloFallbackTemplateEnabled(cfg) ||
           dualYoloFallbackFeaturePointsEnabled(cfg);
}

ROIFeatureMatchConfig makeROIFeatureMatchConfig(
    const PipelineConfig::DualYoloConfig& cfg,
    const HybridDepthConfig& depth_cfg) {
    ROIFeatureMatchConfig out;
    out.roi_denoise = cfg.roi_denoise;
    out.circle_max_roi_pixels = cfg.circle_max_roi_pixels;
    out.subpixel_patch_radius = cfg.subpixel_patch_radius;
    out.subpixel_search_radius_px = cfg.subpixel_search_radius_px;
    out.subpixel_max_points = cfg.subpixel_max_points;
    out.subpixel_min_points = cfg.subpixel_min_points;
    out.subpixel_min_confidence = cfg.subpixel_min_confidence;
    out.subpixel_max_disp_delta_px = cfg.subpixel_max_disp_delta_px;
    out.subpixel_max_disp_delta_ratio = cfg.subpixel_max_disp_delta_ratio;
    out.subpixel_max_depth_delta_m = cfg.subpixel_max_depth_delta_m;
    out.subpixel_max_stddev_px = cfg.subpixel_max_stddev_px;
    out.epipolar_y_tolerance = cfg.epipolar_y_tolerance;
    out.feature_y_tolerance_px = cfg.feature_y_tolerance_px;
    out.feature_y_slope = cfg.feature_y_slope;
    out.feature_y_offset_px = cfg.feature_y_offset_px;
    out.feature_reverse_check_px = cfg.feature_reverse_check_px;
    out.feature_overlap_scale = cfg.feature_overlap_scale;
    out.feature_mad_scale = cfg.feature_mad_scale;
    out.feature_ransac_gate_px = cfg.feature_ransac_gate_px;
    out.feature_sphere_radius_m = std::max(0.0f, depth_cfg.object_diameter * 0.5f);
    out.feature_sphere_radius_scale = cfg.feature_sphere_radius_scale;
    out.feature_sphere_margin_m = cfg.feature_sphere_margin_m;
    out.feature_normalize_large_roi = cfg.feature_normalize_large_roi;
    out.feature_normalized_diameter_px = cfg.feature_normalized_diameter_px;
    out.feature_normalize_min_diameter_px = cfg.feature_normalize_min_diameter_px;
    out.feature_normalize_margin_scale = cfg.feature_normalize_margin_scale;
    out.feature_precompute_roi_maps = cfg.feature_precompute_roi_maps;
    return out;
}

ROICircleSearchConfig makeROICircleSearchConfig(
    const PipelineConfig::DualYoloConfig& cfg) {
    ROICircleSearchConfig out;
    out.denoise = cfg.roi_denoise;
    out.max_roi_pixels = cfg.circle_max_roi_pixels;
    out.fallback_search_margin_px = cfg.fallback_search_margin_px;
    out.fallback_max_width_px = cfg.fallback_max_width_px;
    return out;
}

StereoRoiPairGateConfig makeStereoRoiPairGateConfig(
    const PipelineConfig& config) {
    StereoRoiPairGateConfig gate;
    gate.max_disparity = config.max_disparity;
    gate.epipolar_y_tolerance = config.dual_yolo.epipolar_y_tolerance;
    gate.max_size_ratio = config.dual_yolo.max_size_ratio;
    gate.adaptive_y_ratio = 0.35f;
    gate.min_shifted_iou = config.dual_yolo.min_shifted_iou;
    return gate;
}

}  // namespace stereo3d
