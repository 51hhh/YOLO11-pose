#include "depth_candidate_builder.h"
#include "depth_candidate_builder_feature.h"

#include <algorithm>

namespace stereo3d {

namespace {

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

    appendFeatureDepthCandidates(
        in,
        circle_anchor_x,
        circle_anchor_y,
        &out.candidates);
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

    out.selection = selectLegacyDepthOutputCandidate(out.candidates);
    return out;
}

}  // namespace stereo3d
