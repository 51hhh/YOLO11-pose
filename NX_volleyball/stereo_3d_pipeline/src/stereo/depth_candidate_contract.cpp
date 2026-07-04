#include "depth_match_contract.h"

#include <cmath>

namespace stereo3d {

const char* depthCandidateMethodName(DepthCandidateMethod method) {
    switch (method) {
    case DepthCandidateMethod::BBOX_CENTER: return "bbox_center";
    case DepthCandidateMethod::BBOX_LEFT_EDGE: return "bbox_left_edge";
    case DepthCandidateMethod::BBOX_RIGHT_EDGE: return "bbox_right_edge";
    case DepthCandidateMethod::BBOX_EDGES: return "bbox_edges";
    case DepthCandidateMethod::CIRCLE_CENTER: return "circle_center";
    case DepthCandidateMethod::CIRCLE_LEFT_EDGE: return "circle_left_edge";
    case DepthCandidateMethod::CIRCLE_RIGHT_EDGE: return "circle_right_edge";
    case DepthCandidateMethod::ROI_EDGE_CENTROID: return "roi_edge_centroid";
    case DepthCandidateMethod::ROI_RADIAL_CENTER: return "roi_radial_center";
    case DepthCandidateMethod::ROI_EDGE_PAIR_CENTER: return "roi_edge_pair_center";
    case DepthCandidateMethod::ROI_CORNER_POINTS: return "roi_corner_points";
    case DepthCandidateMethod::ROI_TEXTURE_POINTS: return "roi_texture_points";
    case DepthCandidateMethod::ROI_BINARY_POINTS: return "roi_binary_points";
    case DepthCandidateMethod::ROI_ORB_POINTS: return "roi_orb_points";
    case DepthCandidateMethod::ROI_BRISK_POINTS: return "roi_brisk_points";
    case DepthCandidateMethod::ROI_AKAZE_POINTS: return "roi_akaze_points";
    case DepthCandidateMethod::ROI_SIFT_POINTS: return "roi_sift_points";
    case DepthCandidateMethod::ROI_IOU_REGION_COLOR_PATCH: return "roi_iou_region_color_patch";
    case DepthCandidateMethod::ROI_PATCH_IOU_COLOR_EDGE: return "roi_patch_iou_color_edge";
    case DepthCandidateMethod::ROI_CUDA_TEMPLATE_MATCH: return "roi_cuda_template_match";
    case DepthCandidateMethod::ROI_CUDA_STEREO_BM: return "roi_cuda_stereo_bm";
    case DepthCandidateMethod::ROI_CUDA_STEREO_SGM: return "roi_cuda_stereo_sgm";
    case DepthCandidateMethod::ROI_NEURAL_FEATURE: return "roi_neural_feature";
    case DepthCandidateMethod::ROI_CENTER_PATCH: return "roi_center_patch";
    case DepthCandidateMethod::ROI_MULTI_POINT: return "roi_multi_point";
    case DepthCandidateMethod::FALLBACK_EPIPOLAR: return "fallback_epipolar";
    case DepthCandidateMethod::FALLBACK_TEMPLATE: return "fallback_template";
    case DepthCandidateMethod::FALLBACK_FEATURE_POINTS: return "fallback_feature_points";
    }
    return "unknown";
}

int stereoDepthSourceForMethod(DepthCandidateMethod method) {
    switch (method) {
    case DepthCandidateMethod::CIRCLE_CENTER:
    case DepthCandidateMethod::FALLBACK_EPIPOLAR:
        return 1;
    case DepthCandidateMethod::ROI_MULTI_POINT:
        return 2;
    case DepthCandidateMethod::BBOX_CENTER:
        return 3;
    case DepthCandidateMethod::ROI_CENTER_PATCH:
        return 4;
    case DepthCandidateMethod::ROI_EDGE_CENTROID:
        return 5;
    case DepthCandidateMethod::BBOX_LEFT_EDGE:
    case DepthCandidateMethod::BBOX_RIGHT_EDGE:
    case DepthCandidateMethod::BBOX_EDGES:
        return 6;
    case DepthCandidateMethod::FALLBACK_TEMPLATE:
        return 7;
    case DepthCandidateMethod::ROI_RADIAL_CENTER:
        return 8;
    case DepthCandidateMethod::ROI_EDGE_PAIR_CENTER:
        return 9;
    case DepthCandidateMethod::ROI_CORNER_POINTS:
        return 10;
    case DepthCandidateMethod::ROI_TEXTURE_POINTS:
        return 11;
    case DepthCandidateMethod::FALLBACK_FEATURE_POINTS:
        return 12;
    case DepthCandidateMethod::ROI_BINARY_POINTS:
        return 13;
    case DepthCandidateMethod::ROI_ORB_POINTS:
        return 14;
    case DepthCandidateMethod::ROI_BRISK_POINTS:
        return 15;
    case DepthCandidateMethod::ROI_AKAZE_POINTS:
        return 16;
    case DepthCandidateMethod::ROI_SIFT_POINTS:
        return 17;
    case DepthCandidateMethod::ROI_IOU_REGION_COLOR_PATCH:
        return 18;
    case DepthCandidateMethod::ROI_PATCH_IOU_COLOR_EDGE:
        return 19;
    case DepthCandidateMethod::ROI_CUDA_TEMPLATE_MATCH:
        return 21;
    case DepthCandidateMethod::ROI_CUDA_STEREO_BM:
        return 22;
    case DepthCandidateMethod::ROI_CUDA_STEREO_SGM:
        return 23;
    case DepthCandidateMethod::ROI_NEURAL_FEATURE:
        return 20;
    case DepthCandidateMethod::CIRCLE_LEFT_EDGE:
    case DepthCandidateMethod::CIRCLE_RIGHT_EDGE:
        return 1;
    }
    return 0;
}

DepthCandidateObservation makeDepthCandidateObservation(
    DepthCandidateMethod method,
    float disparity_px,
    float depth_m,
    float confidence,
    float fusion_confidence,
    float stddev_px,
    int support,
    float anchor_left_x,
    float anchor_left_y) {
    DepthCandidateObservation candidate;
    candidate.method = method;
    candidate.status = (std::isfinite(disparity_px) &&
                        std::isfinite(depth_m) &&
                        disparity_px > 0.0f &&
                        depth_m > 0.0f)
        ? DepthCandidateStatus::OK
        : DepthCandidateStatus::REJECTED;
    candidate.disparity_px = disparity_px;
    candidate.depth_m = depth_m;
    candidate.confidence = confidence;
    candidate.fusion_confidence = fusion_confidence;
    candidate.stddev_px = stddev_px;
    candidate.support = support;
    candidate.stereo_depth_source = stereoDepthSourceForMethod(method);
    candidate.anchor_left_x = anchor_left_x;
    candidate.anchor_left_y = anchor_left_y;
    return candidate;
}

bool isUsableDepthCandidate(const DepthCandidateObservation& candidate) {
    return candidate.status == DepthCandidateStatus::OK &&
           candidate.stereo_depth_source > 0 &&
           std::isfinite(candidate.disparity_px) &&
           std::isfinite(candidate.depth_m) &&
           candidate.disparity_px > 0.0f &&
           candidate.depth_m > 0.0f;
}

bool isLegacyDepthOutputCandidate(const DepthCandidateObservation& candidate) {
    if (!isUsableDepthCandidate(candidate)) return false;
    switch (candidate.method) {
    case DepthCandidateMethod::CIRCLE_CENTER:
    case DepthCandidateMethod::FALLBACK_EPIPOLAR:
    case DepthCandidateMethod::ROI_RADIAL_CENTER:
    case DepthCandidateMethod::ROI_EDGE_PAIR_CENTER:
    case DepthCandidateMethod::ROI_EDGE_CENTROID:
    case DepthCandidateMethod::CIRCLE_LEFT_EDGE:
    case DepthCandidateMethod::CIRCLE_RIGHT_EDGE:
    case DepthCandidateMethod::BBOX_CENTER:
    case DepthCandidateMethod::BBOX_EDGES:
    case DepthCandidateMethod::BBOX_LEFT_EDGE:
    case DepthCandidateMethod::BBOX_RIGHT_EDGE:
    case DepthCandidateMethod::FALLBACK_TEMPLATE:
    case DepthCandidateMethod::FALLBACK_FEATURE_POINTS:
        return true;
    case DepthCandidateMethod::ROI_CENTER_PATCH:
    case DepthCandidateMethod::ROI_MULTI_POINT:
    case DepthCandidateMethod::ROI_CORNER_POINTS:
    case DepthCandidateMethod::ROI_TEXTURE_POINTS:
    case DepthCandidateMethod::ROI_BINARY_POINTS:
    case DepthCandidateMethod::ROI_ORB_POINTS:
    case DepthCandidateMethod::ROI_BRISK_POINTS:
    case DepthCandidateMethod::ROI_AKAZE_POINTS:
    case DepthCandidateMethod::ROI_SIFT_POINTS:
    case DepthCandidateMethod::ROI_IOU_REGION_COLOR_PATCH:
    case DepthCandidateMethod::ROI_PATCH_IOU_COLOR_EDGE:
    case DepthCandidateMethod::ROI_CUDA_TEMPLATE_MATCH:
    case DepthCandidateMethod::ROI_CUDA_STEREO_BM:
    case DepthCandidateMethod::ROI_CUDA_STEREO_SGM:
    case DepthCandidateMethod::ROI_NEURAL_FEATURE:
        return false;
    }
    return false;
}

DepthCandidateSelection selectLegacyDepthOutputCandidate(
    const std::vector<DepthCandidateObservation>& candidates) {
    for (const auto& candidate : candidates) {
        if (!isLegacyDepthOutputCandidate(candidate)) continue;
        DepthCandidateSelection selection;
        selection.valid = true;
        selection.observation = candidate;
        return selection;
    }
    return DepthCandidateSelection{};
}

const char* depthCandidateStatusName(DepthCandidateStatus status) {
    switch (status) {
    case DepthCandidateStatus::NOT_RUN: return "not_run";
    case DepthCandidateStatus::OK: return "ok";
    case DepthCandidateStatus::INVALID_INPUT: return "invalid_input";
    case DepthCandidateStatus::REJECTED: return "rejected";
    case DepthCandidateStatus::TIMEOUT: return "timeout";
    case DepthCandidateStatus::UNSUPPORTED_BACKEND: return "unsupported_backend";
    case DepthCandidateStatus::BACKEND_ERROR: return "backend_error";
    }
    return "unknown";
}

}  // namespace stereo3d
