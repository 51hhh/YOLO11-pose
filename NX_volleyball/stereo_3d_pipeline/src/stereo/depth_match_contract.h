#ifndef STEREO_3D_PIPELINE_DEPTH_MATCH_CONTRACT_H_
#define STEREO_3D_PIPELINE_DEPTH_MATCH_CONTRACT_H_

#include "pipeline/detection_types.h"

#include <cstddef>
#include <vector>

namespace stereo3d {

enum class DepthCandidateMethod {
    BBOX_CENTER,
    BBOX_LEFT_EDGE,
    BBOX_RIGHT_EDGE,
    BBOX_EDGES,
    CIRCLE_CENTER,
    CIRCLE_LEFT_EDGE,
    CIRCLE_RIGHT_EDGE,
    ROI_EDGE_CENTROID,
    ROI_RADIAL_CENTER,
    ROI_EDGE_PAIR_CENTER,
    ROI_CORNER_POINTS,
    ROI_TEXTURE_POINTS,
    ROI_BINARY_POINTS,
    ROI_ORB_POINTS,
    ROI_BRISK_POINTS,
    ROI_AKAZE_POINTS,
    ROI_SIFT_POINTS,
    ROI_IOU_REGION_COLOR_PATCH,
    ROI_PATCH_IOU_COLOR_EDGE,
    ROI_CUDA_TEMPLATE_MATCH,
    ROI_CUDA_STEREO_BM,
    ROI_CUDA_STEREO_SGM,
    ROI_RING_EDGE_PROFILE,
    ROI_NEURAL_FEATURE,
    ROI_CENTER_PATCH,
    ROI_MULTI_POINT,
    FALLBACK_EPIPOLAR,
    FALLBACK_TEMPLATE,
    FALLBACK_FEATURE_POINTS
};

enum class DepthCandidateStatus {
    NOT_RUN,
    OK,
    INVALID_INPUT,
    REJECTED,
    TIMEOUT,
    UNSUPPORTED_BACKEND,
    BACKEND_ERROR
};

struct DepthCandidateObservation {
    DepthCandidateMethod method = DepthCandidateMethod::BBOX_CENTER;
    DepthCandidateStatus status = DepthCandidateStatus::NOT_RUN;
    float disparity_px = -1.0f;
    float depth_m = -1.0f;
    float confidence = 0.0f;
    float fusion_confidence = 1.0f;
    float stddev_px = -1.0f;
    int support = 0;
    int stereo_depth_source = 0;
    float anchor_left_x = 0.0f;
    float anchor_left_y = 0.0f;
};

struct DepthCandidateSelection {
    bool valid = false;
    DepthCandidateObservation observation;
};

enum class StereoRoiPairRejectReason {
    NONE,
    CLASS_MISMATCH,
    INVALID_BOX,
    NONPOSITIVE_DISPARITY,
    OVER_MAX_DISPARITY,
    EPIPOLAR_REJECT,
    SIZE_REJECT,
    LOW_IOU
};

struct StereoRoiPairGateConfig {
    int max_disparity = 2048;
    float epipolar_y_tolerance = 12.0f;
    float max_size_ratio = 2.0f;
    float adaptive_y_ratio = 0.35f;
    float min_shifted_iou = 0.0f;
};

struct StereoRoiPair {
    int left_index = -1;
    int right_index = -1;
    Detection left;
    Detection right;
    float initial_disparity = -1.0f;
    float epipolar_dy = -1.0f;
    float y_tolerance = -1.0f;
    float width_ratio = -1.0f;
    float height_ratio = -1.0f;
    float size_ratio = -1.0f;
    float shifted_bbox_iou = 0.0f;
    float score = 0.0f;
    float semantic_confidence = 0.0f;
};

struct StereoRoiPairStats {
    int class_mismatch = 0;
    int invalid_box = 0;
    int nonpositive_disparity = 0;
    int over_max_disparity = 0;
    int epipolar_reject = 0;
    int size_reject = 0;
    int low_iou = 0;
};

const char* depthCandidateMethodName(DepthCandidateMethod method);
const char* depthCandidateStatusName(DepthCandidateStatus status);
const char* stereoRoiPairRejectReasonName(StereoRoiPairRejectReason reason);
int stereoDepthSourceForMethod(DepthCandidateMethod method);

DepthCandidateObservation makeDepthCandidateObservation(
    DepthCandidateMethod method,
    float disparity_px,
    float depth_m,
    float confidence = 0.0f,
    float fusion_confidence = 1.0f,
    float stddev_px = -1.0f,
    int support = 0,
    float anchor_left_x = 0.0f,
    float anchor_left_y = 0.0f);

bool isUsableDepthCandidate(const DepthCandidateObservation& candidate);
bool isLegacyDepthOutputCandidate(const DepthCandidateObservation& candidate);
DepthCandidateSelection selectLegacyDepthOutputCandidate(
    const std::vector<DepthCandidateObservation>& candidates);

bool evaluateStereoRoiPair(
    const Detection& left,
    const Detection& right,
    int left_index,
    int right_index,
    const StereoRoiPairGateConfig& config,
    StereoRoiPair* pair,
    StereoRoiPairRejectReason* reason);

void accumulateStereoRoiPairReject(
    StereoRoiPairStats* stats,
    StereoRoiPairRejectReason reason);

std::vector<StereoRoiPair> collectStereoRoiPairCandidates(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const StereoRoiPairGateConfig& config,
    std::size_t max_pairs,
    StereoRoiPairStats* stats = nullptr);

bool findBestStereoRoiPair(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const StereoRoiPairGateConfig& config,
    StereoRoiPair* best_pair,
    StereoRoiPairStats* stats = nullptr);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_DEPTH_MATCH_CONTRACT_H_
