#ifndef STEREO_3D_PIPELINE_DEPTH_CANDIDATE_BUILDER_H_
#define STEREO_3D_PIPELINE_DEPTH_CANDIDATE_BUILDER_H_

#include "depth_match_contract.h"
#include "roi_feature_result.h"
#include "roi_geometry_cpu.h"
#include "roi_patch_match_cpu.h"
#include "pipeline/detection_types.h"

#include <vector>

namespace stereo3d {

struct DepthCandidateBuilderInput {
    Detection left_detection;
    CircleFit2D left_circle;
    PointMeasure2D left_edge_centroid_measure;
    PointMeasure2D left_radial_measure;
    PointMeasure2D left_edge_pair_measure;

    bool subpixel_valid = false;
    SubpixelDisparityResult subpixel_result;
    float z_subpixel = -1.0f;

    SparseFeatureDisparityResult fallback_feature_result;
    float z_fallback_feature_points = -1.0f;

    bool fallback_template_depth_valid = false;
    bool epipolar_fallback_depth_valid = false;
    bool circle_candidate_valid = false;
    float circle_disparity = -1.0f;
    float z_circle_raw = -1.0f;
    float circle_confidence = 0.0f;
    // Deprecated: sphere left/right silhouettes are not shared physical points.
    float disparity_circle_left_edge = -1.0f;
    float z_circle_left_edge = -1.0f;
    float disparity_circle_right_edge = -1.0f;
    float z_circle_right_edge = -1.0f;

    bool center_patch_valid = false;
    SubpixelDisparityResult center_patch_result;
    float z_roi_center_patch = -1.0f;

    SparseFeatureDisparityResult iou_region_color_patch_result;
    float z_roi_iou_region_color_patch = -1.0f;
    SparseFeatureDisparityResult patch_iou_color_edge_result;
    float z_roi_patch_iou_color_edge = -1.0f;
    SparseFeatureDisparityResult cuda_template_match_result;
    float z_roi_cuda_template_match = -1.0f;
    SparseFeatureDisparityResult cuda_stereo_bm_result;
    float z_roi_cuda_stereo_bm = -1.0f;
    SparseFeatureDisparityResult cuda_stereo_sgm_result;
    float z_roi_cuda_stereo_sgm = -1.0f;
    SparseFeatureDisparityResult ring_edge_profile_result;
    float z_roi_ring_edge_profile = -1.0f;
    SparseFeatureDisparityResult neural_feature_result;
    float z_roi_neural_feature = -1.0f;
    SparseFeatureDisparityResult corner_points_result;
    float z_roi_corner_points = -1.0f;
    SparseFeatureDisparityResult texture_points_result;
    float z_roi_texture_points = -1.0f;
    SparseFeatureDisparityResult binary_points_result;
    float z_roi_binary_points = -1.0f;
    SparseFeatureDisparityResult orb_points_result;
    float z_roi_orb_points = -1.0f;
    SparseFeatureDisparityResult brisk_points_result;
    float z_roi_brisk_points = -1.0f;
    SparseFeatureDisparityResult akaze_points_result;
    float z_roi_akaze_points = -1.0f;
    SparseFeatureDisparityResult sift_points_result;
    float z_roi_sift_points = -1.0f;

    float disparity_roi_radial_center = -1.0f;
    float z_roi_radial_center = -1.0f;
    float disparity_roi_edge_pair_center = -1.0f;
    float z_roi_edge_pair_center = -1.0f;
    float disparity_roi_edge_centroid = -1.0f;
    float z_roi_edge_centroid = -1.0f;

    float yolo_disparity = -1.0f;
    float z_yolo = -1.0f;
    float disparity_bbox_edge_final = -1.0f;
    float z_bbox_edge_final = -1.0f;
    float disparity_bbox_left_edge = -1.0f;
    float z_bbox_left_edge = -1.0f;
    float disparity_bbox_right_edge = -1.0f;
    float z_bbox_right_edge = -1.0f;
};

struct DepthCandidateBuildResult {
    std::vector<DepthCandidateObservation> candidates;
    DepthCandidateSelection selection;
};

DepthCandidateBuildResult buildDepthCandidateObservations(
    const DepthCandidateBuilderInput& input);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_DEPTH_CANDIDATE_BUILDER_H_
