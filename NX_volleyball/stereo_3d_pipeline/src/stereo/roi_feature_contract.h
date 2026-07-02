#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_CONTRACT_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_CONTRACT_H_

namespace stereo3d {

struct ROIFeatureMatchConfig {
    bool roi_denoise = true;
    int circle_max_roi_pixels = 18000;
    int subpixel_patch_radius = 5;
    int subpixel_search_radius_px = 8;
    int subpixel_max_points = 12;
    int subpixel_min_points = 4;
    float subpixel_min_confidence = 0.25f;
    float subpixel_max_disp_delta_px = 2.0f;
    float subpixel_max_disp_delta_ratio = 0.03f;
    float subpixel_max_depth_delta_m = 0.5f;
    float subpixel_max_stddev_px = 1.0f;
    float epipolar_y_tolerance = 12.0f;
    float feature_y_tolerance_px = 2.0f;
    float feature_y_slope = 0.0f;
    float feature_y_offset_px = 0.0f;
    float feature_reverse_check_px = 1.0f;
    float feature_overlap_scale = 0.55f;
    float feature_mad_scale = 2.5f;
    float feature_ransac_gate_px = 0.75f;
    float feature_sphere_radius_m = 0.10f;
    float feature_sphere_radius_scale = 1.8f;
    float feature_sphere_margin_m = 0.02f;
    bool feature_normalize_large_roi = true;
    int feature_normalized_diameter_px = 96;
    float feature_normalize_min_diameter_px = 128.0f;
    float feature_normalize_margin_scale = 0.62f;
    bool feature_precompute_roi_maps = true;
};

enum class SparseFeatureMode {
    CORNER,
    TEXTURE,
    BINARY
};

enum class OpenCVFeatureMode {
    ORB,
    BRISK,
    AKAZE,
    SIFT
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_CONTRACT_H_
