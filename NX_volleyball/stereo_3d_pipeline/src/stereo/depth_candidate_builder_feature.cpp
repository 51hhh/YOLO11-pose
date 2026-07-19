#include "depth_candidate_builder_feature.h"

#include <algorithm>

namespace stereo3d {
namespace {

float confidenceBlend(float base, float scale, float confidence) {
    return std::clamp(base + scale * confidence, 0.0f, 1.0f);
}

void addCandidate(std::vector<DepthCandidateObservation>* candidates,
                  DepthCandidateMethod method,
                  float disparity,
                  float depth_m,
                  float confidence,
                  float fusion_confidence,
                  float stddev_px,
                  int support,
                  float anchor_x,
                  float anchor_y) {
    candidates->push_back(makeDepthCandidateObservation(
        method,
        disparity,
        depth_m,
        confidence,
        fusion_confidence,
        stddev_px,
        support,
        anchor_x,
        anchor_y));
}

}  // namespace

void appendFeatureDepthCandidates(
    const DepthCandidateBuilderInput& in,
    float circle_anchor_x,
    float circle_anchor_y,
    std::vector<DepthCandidateObservation>* candidates) {
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_MULTI_POINT,
        in.subpixel_valid ? in.subpixel_result.disparity : -1.0f,
        in.z_subpixel,
        in.subpixel_result.confidence,
        confidenceBlend(0.70f, 0.30f, in.subpixel_result.confidence),
        in.subpixel_result.stddev,
        in.subpixel_result.support,
        circle_anchor_x,
        circle_anchor_y);
    addCandidate(
        candidates,
        DepthCandidateMethod::FALLBACK_FEATURE_POINTS,
        in.fallback_feature_result.valid
            ? in.fallback_feature_result.disparity
            : -1.0f,
        in.z_fallback_feature_points,
        in.fallback_feature_result.confidence,
        confidenceBlend(0.55f, 0.30f, in.fallback_feature_result.confidence),
        in.fallback_feature_result.stddev,
        in.fallback_feature_result.support,
        in.fallback_feature_result.anchor_cx,
        in.fallback_feature_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::FALLBACK_TEMPLATE,
        in.fallback_template_depth_valid ? in.circle_disparity : -1.0f,
        in.fallback_template_depth_valid ? in.z_circle_raw : -1.0f,
        in.circle_confidence,
        std::max(0.45f, in.circle_confidence),
        -1.0f,
        0,
        circle_anchor_x,
        circle_anchor_y);
    addCandidate(
        candidates,
        in.epipolar_fallback_depth_valid
            ? DepthCandidateMethod::FALLBACK_EPIPOLAR
            : DepthCandidateMethod::CIRCLE_CENTER,
        (in.epipolar_fallback_depth_valid || in.circle_candidate_valid)
            ? in.circle_disparity
            : -1.0f,
        (in.epipolar_fallback_depth_valid || in.circle_candidate_valid)
            ? in.z_circle_raw
            : -1.0f,
        in.circle_confidence,
        1.0f,
        -1.0f,
        0,
        circle_anchor_x,
        circle_anchor_y);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_CENTER_PATCH,
        in.center_patch_valid ? in.center_patch_result.disparity : -1.0f,
        in.z_roi_center_patch,
        in.center_patch_result.confidence,
        confidenceBlend(0.60f, 0.25f, in.center_patch_result.confidence),
        in.center_patch_result.stddev,
        in.center_patch_result.support,
        circle_anchor_x,
        circle_anchor_y);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_IOU_REGION_COLOR_PATCH,
        in.iou_region_color_patch_result.valid
            ? in.iou_region_color_patch_result.disparity
            : -1.0f,
        in.z_roi_iou_region_color_patch,
        in.iou_region_color_patch_result.confidence,
        confidenceBlend(0.58f, 0.30f,
                        in.iou_region_color_patch_result.confidence),
        in.iou_region_color_patch_result.stddev,
        in.iou_region_color_patch_result.support,
        in.iou_region_color_patch_result.anchor_cx,
        in.iou_region_color_patch_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_PATCH_IOU_COLOR_EDGE,
        in.patch_iou_color_edge_result.valid
            ? in.patch_iou_color_edge_result.disparity
            : -1.0f,
        in.z_roi_patch_iou_color_edge,
        in.patch_iou_color_edge_result.confidence,
        confidenceBlend(0.56f, 0.30f,
                        in.patch_iou_color_edge_result.confidence),
        in.patch_iou_color_edge_result.stddev,
        in.patch_iou_color_edge_result.support,
        in.patch_iou_color_edge_result.anchor_cx,
        in.patch_iou_color_edge_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_CUDA_TEMPLATE_MATCH,
        in.cuda_template_match_result.valid
            ? in.cuda_template_match_result.disparity
            : -1.0f,
        in.z_roi_cuda_template_match,
        in.cuda_template_match_result.confidence,
        confidenceBlend(0.58f, 0.30f,
                        in.cuda_template_match_result.confidence),
        in.cuda_template_match_result.stddev,
        in.cuda_template_match_result.support,
        in.cuda_template_match_result.anchor_cx,
        in.cuda_template_match_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_CUDA_STEREO_BM,
        in.cuda_stereo_bm_result.valid
            ? in.cuda_stereo_bm_result.disparity
            : -1.0f,
        in.z_roi_cuda_stereo_bm,
        in.cuda_stereo_bm_result.confidence,
        confidenceBlend(0.54f, 0.28f,
                        in.cuda_stereo_bm_result.confidence),
        in.cuda_stereo_bm_result.stddev,
        in.cuda_stereo_bm_result.support,
        in.cuda_stereo_bm_result.anchor_cx,
        in.cuda_stereo_bm_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_CUDA_STEREO_SGM,
        in.cuda_stereo_sgm_result.valid
            ? in.cuda_stereo_sgm_result.disparity
            : -1.0f,
        in.z_roi_cuda_stereo_sgm,
        in.cuda_stereo_sgm_result.confidence,
        confidenceBlend(0.56f, 0.30f,
                        in.cuda_stereo_sgm_result.confidence),
        in.cuda_stereo_sgm_result.stddev,
        in.cuda_stereo_sgm_result.support,
        in.cuda_stereo_sgm_result.anchor_cx,
        in.cuda_stereo_sgm_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_RING_EDGE_PROFILE,
        in.ring_edge_profile_result.valid
            ? in.ring_edge_profile_result.disparity
            : -1.0f,
        in.z_roi_ring_edge_profile,
        in.ring_edge_profile_result.confidence,
        confidenceBlend(0.56f, 0.30f,
                        in.ring_edge_profile_result.confidence),
        in.ring_edge_profile_result.stddev,
        in.ring_edge_profile_result.support,
        in.ring_edge_profile_result.anchor_cx,
        in.ring_edge_profile_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_NEURAL_FEATURE,
        in.neural_feature_result.valid ? in.neural_feature_result.disparity
                                       : -1.0f,
        in.z_roi_neural_feature,
        in.neural_feature_result.confidence,
        confidenceBlend(0.56f, 0.32f, in.neural_feature_result.confidence),
        in.neural_feature_result.stddev,
        in.neural_feature_result.support,
        in.neural_feature_result.anchor_cx,
        in.neural_feature_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_CORNER_POINTS,
        in.corner_points_result.valid ? in.corner_points_result.disparity
                                      : -1.0f,
        in.z_roi_corner_points,
        in.corner_points_result.confidence,
        confidenceBlend(0.55f, 0.30f, in.corner_points_result.confidence),
        in.corner_points_result.stddev,
        in.corner_points_result.support,
        in.corner_points_result.anchor_cx,
        in.corner_points_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_TEXTURE_POINTS,
        in.texture_points_result.valid ? in.texture_points_result.disparity
                                       : -1.0f,
        in.z_roi_texture_points,
        in.texture_points_result.confidence,
        confidenceBlend(0.52f, 0.28f, in.texture_points_result.confidence),
        in.texture_points_result.stddev,
        in.texture_points_result.support,
        in.texture_points_result.anchor_cx,
        in.texture_points_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_BINARY_POINTS,
        in.binary_points_result.valid ? in.binary_points_result.disparity
                                      : -1.0f,
        in.z_roi_binary_points,
        in.binary_points_result.confidence,
        confidenceBlend(0.55f, 0.30f, in.binary_points_result.confidence),
        in.binary_points_result.stddev,
        in.binary_points_result.support,
        in.binary_points_result.anchor_cx,
        in.binary_points_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_ORB_POINTS,
        in.orb_points_result.valid ? in.orb_points_result.disparity : -1.0f,
        in.z_roi_orb_points,
        in.orb_points_result.confidence,
        confidenceBlend(0.54f, 0.30f, in.orb_points_result.confidence),
        in.orb_points_result.stddev,
        in.orb_points_result.support,
        in.orb_points_result.anchor_cx,
        in.orb_points_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_BRISK_POINTS,
        in.brisk_points_result.valid ? in.brisk_points_result.disparity
                                     : -1.0f,
        in.z_roi_brisk_points,
        in.brisk_points_result.confidence,
        confidenceBlend(0.53f, 0.30f, in.brisk_points_result.confidence),
        in.brisk_points_result.stddev,
        in.brisk_points_result.support,
        in.brisk_points_result.anchor_cx,
        in.brisk_points_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_AKAZE_POINTS,
        in.akaze_points_result.valid ? in.akaze_points_result.disparity
                                     : -1.0f,
        in.z_roi_akaze_points,
        in.akaze_points_result.confidence,
        confidenceBlend(0.52f, 0.30f, in.akaze_points_result.confidence),
        in.akaze_points_result.stddev,
        in.akaze_points_result.support,
        in.akaze_points_result.anchor_cx,
        in.akaze_points_result.anchor_cy);
    addCandidate(
        candidates,
        DepthCandidateMethod::ROI_SIFT_POINTS,
        in.sift_points_result.valid ? in.sift_points_result.disparity
                                    : -1.0f,
        in.z_roi_sift_points,
        in.sift_points_result.confidence,
        confidenceBlend(0.52f, 0.30f, in.sift_points_result.confidence),
        in.sift_points_result.stddev,
        in.sift_points_result.support,
        in.sift_points_result.anchor_cx,
        in.sift_points_result.anchor_cy);
}

}  // namespace stereo3d
