#include "depth_candidate_builder.h"

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

DepthCandidateBuildResult buildDepthCandidateObservations(
    const DepthCandidateBuilderInput& in) {
    DepthCandidateBuildResult out;
    out.candidates.reserve(24);

    const float circle_anchor_x = in.left_circle.valid
        ? in.left_circle.cx
        : in.left_detection.cx;
    const float circle_anchor_y = in.left_circle.valid
        ? in.left_circle.cy
        : in.left_detection.cy;
    const float circle_left_edge_anchor_x = in.left_circle.valid
        ? in.left_circle.cx - in.left_circle.radius
        : in.left_detection.cx - 0.5f * in.left_detection.width;
    const float circle_right_edge_anchor_x = in.left_circle.valid
        ? in.left_circle.cx + in.left_circle.radius
        : in.left_detection.cx + 0.5f * in.left_detection.width;
    const float bbox_left_edge_anchor_x =
        in.left_detection.cx - 0.5f * in.left_detection.width;
    const float bbox_right_edge_anchor_x =
        in.left_detection.cx + 0.5f * in.left_detection.width;

    addCandidate(
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
        &out.candidates,
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
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::ROI_RADIAL_CENTER,
        in.disparity_roi_radial_center,
        in.z_roi_radial_center,
        in.left_radial_measure.confidence,
        0.62f,
        -1.0f,
        1,
        in.left_radial_measure.cx,
        in.left_radial_measure.cy);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::ROI_EDGE_PAIR_CENTER,
        in.disparity_roi_edge_pair_center,
        in.z_roi_edge_pair_center,
        in.left_edge_pair_measure.confidence,
        0.58f,
        -1.0f,
        1,
        in.left_edge_pair_measure.cx,
        in.left_edge_pair_measure.cy);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::ROI_EDGE_CENTROID,
        in.disparity_roi_edge_centroid,
        in.z_roi_edge_centroid,
        in.left_edge_centroid_measure.confidence,
        0.60f,
        -1.0f,
        1,
        in.left_edge_centroid_measure.cx,
        in.left_edge_centroid_measure.cy);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::CIRCLE_LEFT_EDGE,
        in.disparity_circle_left_edge,
        in.z_circle_left_edge,
        in.circle_confidence,
        std::max(0.45f, in.circle_confidence),
        -1.0f,
        1,
        circle_left_edge_anchor_x,
        circle_anchor_y);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::CIRCLE_RIGHT_EDGE,
        in.disparity_circle_right_edge,
        in.z_circle_right_edge,
        in.circle_confidence,
        std::max(0.45f, in.circle_confidence),
        -1.0f,
        1,
        circle_right_edge_anchor_x,
        circle_anchor_y);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::BBOX_CENTER,
        (in.z_yolo > 0.0f) ? in.yolo_disparity : -1.0f,
        in.z_yolo,
        0.65f,
        0.65f,
        -1.0f,
        1,
        in.left_detection.cx,
        in.left_detection.cy);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::BBOX_EDGES,
        in.disparity_bbox_edge_final,
        in.z_bbox_edge_final,
        0.55f,
        0.55f,
        -1.0f,
        1,
        in.left_detection.cx,
        in.left_detection.cy);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::BBOX_LEFT_EDGE,
        in.disparity_bbox_left_edge,
        in.z_bbox_left_edge,
        0.50f,
        0.50f,
        -1.0f,
        1,
        bbox_left_edge_anchor_x,
        in.left_detection.cy);
    addCandidate(
        &out.candidates,
        DepthCandidateMethod::BBOX_RIGHT_EDGE,
        in.disparity_bbox_right_edge,
        in.z_bbox_right_edge,
        0.50f,
        0.50f,
        -1.0f,
        1,
        bbox_right_edge_anchor_x,
        in.left_detection.cy);

    out.selection = selectFirstUsableDepthCandidate(out.candidates);
    return out;
}

}  // namespace stereo3d
