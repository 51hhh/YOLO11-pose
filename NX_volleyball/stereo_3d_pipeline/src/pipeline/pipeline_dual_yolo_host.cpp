/**
 * @file pipeline_dual_yolo_host.cpp
 * @brief Host image dependency checks for dual-YOLO ROI stage2.
 */

#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "pipeline_roi_match_helpers.h"

#include <limits>
#include <vector>

namespace stereo3d {

bool Pipeline::roiStage2NeedsHostImages(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections) const {
    if (left_detections.empty() && right_detections.empty()) {
        return false;
    }

    const bool has_stereo_detections =
        !left_detections.empty() && !right_detections.empty();
    const bool opencv_descriptor_cpu_possible =
        dualYoloOpenCVCpuDescriptorDepthEnabled(config_.dual_yolo);

    if (config_.dual_yolo.gpu_candidate_refine) {
        return (has_stereo_detections && opencv_descriptor_cpu_possible) ||
               roiStage2FallbackMayNeedHostImages(left_detections,
                                                   right_detections);
    }

    return (has_stereo_detections &&
            dualYoloNeedsHostImages(config_.dual_yolo)) ||
           roiStage2FallbackMayNeedHostImages(left_detections,
                                               right_detections);
}

bool Pipeline::roiStage2FallbackMayNeedHostImages(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections) const {
    if (!dualYoloCpuFallbackSearchEnabled(config_.dual_yolo)) {
        return false;
    }
    if (left_detections.empty() && right_detections.empty()) {
        return false;
    }
    if (left_detections.empty() || right_detections.empty()) {
        return true;
    }
    if (left_detections.size() != 1 || right_detections.size() != 1) {
        return true;
    }

    const StereoRoiPairGateConfig roi_pair_gate =
        makeStereoRoiPairGateConfig(config_);
    std::vector<bool> right_used(right_detections.size(), false);
    int matched_left = 0;
    for (size_t li = 0; li < left_detections.size(); ++li) {
        int best_idx = -1;
        float best_score = std::numeric_limits<float>::max();
        for (size_t ri = 0; ri < right_detections.size(); ++ri) {
            if (right_used[ri]) continue;
            StereoRoiPair candidate_pair;
            if (!evaluateStereoRoiPair(left_detections[li],
                                       right_detections[ri],
                                       static_cast<int>(li),
                                       static_cast<int>(ri),
                                       roi_pair_gate,
                                       &candidate_pair,
                                       nullptr)) {
                continue;
            }
            if (candidate_pair.score < best_score) {
                best_score = candidate_pair.score;
                best_idx = static_cast<int>(ri);
            }
        }
        if (best_idx >= 0) {
            right_used[static_cast<size_t>(best_idx)] = true;
            ++matched_left;
        }
    }

    if (matched_left < static_cast<int>(left_detections.size())) {
        return true;
    }
    for (bool used : right_used) {
        if (!used) return true;
    }
    return false;
}

}  // namespace stereo3d
