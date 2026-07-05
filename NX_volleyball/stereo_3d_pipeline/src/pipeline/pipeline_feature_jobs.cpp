#include "pipeline_feature_jobs.h"
#include "../stereo/depth_match_contract.h"

#include <algorithm>
#include <string>

namespace stereo3d {

namespace {

bool diagnosticStrideHit(int frame_id, int stride) {
    if (stride <= 0 || frame_id < 0) {
        return false;
    }
    return frame_id % stride == 0;
}

void appendTriggerName(std::string* out, uint32_t triggers,
                       uint32_t bit, const char* name) {
    if ((triggers & bit) == 0u) {
        return;
    }
    if (!out->empty()) {
        out->append("|");
    }
    out->append(name);
}

uint32_t diagnosticSidecarDepthModeMask(uint32_t mask) {
    return mask & (P2_DEPTH_MODE_VPI_TEMPLATE |
                   P2_DEPTH_MODE_VPI_ORB |
                   P2_DEPTH_MODE_CUDA_GFTT_LK |
                   P2_DEPTH_MODE_NEURAL_FEATURE);
}

}  // namespace

uint32_t dualYoloP2DepthModeMask(const PipelineConfig::DualYoloConfig& cfg) {
    uint32_t mask = P2_DEPTH_MODE_NONE;
    if (cfg.depth_roi_corner_points) {
        mask |= P2_DEPTH_MODE_CORNER_POINTS;
    }
    if (cfg.depth_roi_texture_points) {
        mask |= P2_DEPTH_MODE_TEXTURE_POINTS;
    }
    if (cfg.depth_roi_binary_points) {
        mask |= P2_DEPTH_MODE_BINARY_POINTS;
    }
    if (cfg.depth_roi_orb_points) {
        mask |= P2_DEPTH_MODE_ORB_POINTS;
    }
    if (cfg.depth_roi_brisk_points) {
        mask |= P2_DEPTH_MODE_BRISK_POINTS;
    }
    if (cfg.depth_roi_akaze_points) {
        mask |= P2_DEPTH_MODE_AKAZE_POINTS;
    }
    if (cfg.depth_roi_sift_points) {
        mask |= P2_DEPTH_MODE_SIFT_POINTS;
    }
    if (cfg.depth_roi_iou_region_color_patch) {
        mask |= P2_DEPTH_MODE_IOU_COLOR_PATCH;
    }
    if (cfg.depth_roi_patch_iou_color_edge) {
        mask |= P2_DEPTH_MODE_PATCH_IOU_EDGE;
    }
    if (cfg.depth_roi_cuda_template_match) {
        mask |= P2_DEPTH_MODE_CUDA_TEMPLATE;
    }
    if (cfg.depth_roi_cuda_stereo_bm) {
        mask |= P2_DEPTH_MODE_CUDA_STEREO_BM;
    }
    if (cfg.depth_roi_cuda_stereo_sgm) {
        mask |= P2_DEPTH_MODE_CUDA_STEREO_SGM;
    }
    if (cfg.depth_roi_ring_edge_profile) {
        mask |= P2_DEPTH_MODE_RING_EDGE_PROFILE;
    }
    if (cfg.depth_roi_vpi_template_match) {
        mask |= P2_DEPTH_MODE_VPI_TEMPLATE;
    }
    if (cfg.depth_roi_vpi_stereo_disparity) {
        mask |= P2_DEPTH_MODE_VPI_STEREO;
    }
    if (cfg.depth_roi_vpi_harris_lk) {
        mask |= P2_DEPTH_MODE_VPI_HARRIS_LK;
    }
    if (cfg.depth_roi_vpi_orb) {
        mask |= P2_DEPTH_MODE_VPI_ORB;
    }
    if (cfg.depth_roi_cuda_gftt_lk) {
        mask |= P2_DEPTH_MODE_CUDA_GFTT_LK;
    }
    if (cfg.depth_roi_cuda_sift) {
        mask |= P2_DEPTH_MODE_CUDA_SIFT;
    }
    if (cfg.depth_roi_libsgm) {
        mask |= P2_DEPTH_MODE_LIBSGM;
    }
    if (cfg.depth_roi_cuda_hough_circle) {
        mask |= P2_DEPTH_MODE_CUDA_HOUGH_CIRCLE;
    }
    if (cfg.depth_fallback_feature_points) {
        mask |= P2_DEPTH_MODE_FALLBACK_FEATURE_POINTS;
    }
    return mask;
}

bool dualYoloP2DepthModesEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    return dualYoloP2DepthModeMask(cfg) != P2_DEPTH_MODE_NONE;
}

uint32_t p2FeatureDepthModeMask(const PipelineConfig& cfg) {
    uint32_t mask = dualYoloP2DepthModeMask(cfg.dual_yolo);
    if (cfg.neural_features.enabled) {
        mask |= P2_DEPTH_MODE_NEURAL_FEATURE;
    }
    return mask;
}

bool pipelineP2DepthModesEnabled(const PipelineConfig& cfg) {
    return p2FeatureDepthModeMask(cfg) != P2_DEPTH_MODE_NONE;
}

P2FeatureJobPolicy makeP2FeatureJobPolicy(const PipelineConfig& cfg) {
    P2FeatureJobPolicy out;
    out.depth_mode_mask = p2FeatureDepthModeMask(cfg);
    out.p2_depth_modes_enabled = out.depth_mode_mask != P2_DEPTH_MODE_NONE;
    out.split_feature_jobs = cfg.p2_feature_job_scaffold_enabled;
    out.realtime_lane_enabled = cfg.p2_realtime_lane_decision_enabled;
    out.diagnostic_lane_enabled = cfg.p2_diagnostic_lane_decision_enabled;
    out.selective_trigger = cfg.p2_selective_trigger;
    out.trigger_on_fallback = cfg.p2_trigger_on_fallback;
    out.trigger_on_direct_pair = cfg.p2_trigger_on_direct_pair;
    out.trigger_on_host_gray = cfg.p2_trigger_on_host_gray;
    out.trigger_on_bgr = cfg.p2_trigger_on_bgr;
    out.trigger_on_pair_quality = cfg.p2_trigger_on_pair_quality;
    out.trigger_on_no_valid_direct_pair =
        cfg.p2_trigger_on_no_valid_direct_pair;
    out.pair_quality_min_shifted_iou =
        std::max(0.0f, cfg.p2_pair_quality_min_shifted_iou);
    out.pair_quality_max_epipolar_dy =
        std::max(0.0f, cfg.p2_pair_quality_max_epipolar_dy);
    out.pair_quality_min_confidence =
        std::max(0.0f, cfg.p2_pair_quality_min_confidence);
    out.pair_gate_max_disparity = std::max(1, cfg.max_disparity);
    out.pair_gate_epipolar_y_tolerance =
        std::max(1.0f, cfg.dual_yolo.epipolar_y_tolerance);
    out.pair_gate_max_size_ratio =
        std::max(1.0f, cfg.dual_yolo.max_size_ratio);
    out.pair_gate_min_shifted_iou =
        std::max(0.0f, cfg.dual_yolo.min_shifted_iou);
    out.diagnostic_stride = std::max(1, cfg.p2_diagnostic_stride);
    out.diagnostic_max_in_flight = std::max(1, cfg.p2_diagnostic_max_in_flight);
    out.realtime_deadline_ms = std::max(1.0f, cfg.p2_realtime_deadline_ms);
    out.diagnostic_deadline_ms =
        std::max(out.realtime_deadline_ms, cfg.p2_diagnostic_deadline_ms);
    return out;
}

P2FeatureJobDecision decideP2FeatureJobs(
    const P2FeatureJobPolicy& policy,
    int frame_id,
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    bool needs_host_gray,
    bool needs_bgr) {
    P2FeatureJobDecision out;
    out.p2_depth_modes_enabled = policy.p2_depth_modes_enabled;
    out.depth_mode_mask = policy.depth_mode_mask;
    out.split_feature_jobs = policy.split_feature_jobs;
    out.frame_id = frame_id;
    out.left_count = static_cast<int>(left_detections.size());
    out.right_count = static_cast<int>(right_detections.size());
    if (!policy.p2_depth_modes_enabled) {
        out.realtime_skip_reasons |= P2_SKIP_NO_DEPTH_MODE;
        out.diagnostic_skip_reasons |= P2_SKIP_NO_DEPTH_MODE;
        return out;
    }

    uint32_t triggers = P2_TRIGGER_CONFIGURED;
    const bool direct_pair_possible =
        !left_detections.empty() && !right_detections.empty();
    const bool fallback_possible =
        left_detections.empty() != right_detections.empty();

    if (!policy.selective_trigger) {
        triggers |= P2_TRIGGER_ALWAYS;
    }
    if (direct_pair_possible && policy.trigger_on_direct_pair) {
        triggers |= P2_TRIGGER_DIRECT_PAIR;
    }
    if (fallback_possible && policy.trigger_on_fallback) {
        triggers |= P2_TRIGGER_FALLBACK_POSSIBLE;
    }
    if (needs_host_gray && policy.trigger_on_host_gray) {
        triggers |= P2_TRIGGER_HOST_GRAY;
    }
    if (needs_bgr && policy.trigger_on_bgr) {
        triggers |= P2_TRIGGER_BGR;
    }
    if (direct_pair_possible) {
        StereoRoiPairGateConfig pair_gate;
        pair_gate.max_disparity = policy.pair_gate_max_disparity;
        pair_gate.epipolar_y_tolerance = policy.pair_gate_epipolar_y_tolerance;
        pair_gate.max_size_ratio = policy.pair_gate_max_size_ratio;
        pair_gate.min_shifted_iou = policy.pair_gate_min_shifted_iou;

        std::vector<StereoRoiPair> pairs =
            collectStereoRoiPairCandidates(left_detections,
                                           right_detections,
                                           pair_gate,
                                           left_detections.size() *
                                               right_detections.size());
        out.valid_direct_pair_count = static_cast<int>(pairs.size());
        if (pairs.empty()) {
            if (policy.trigger_on_no_valid_direct_pair) {
                triggers |= P2_TRIGGER_NO_VALID_DIRECT_PAIR;
            }
        } else if (policy.trigger_on_pair_quality) {
            const StereoRoiPair& best = pairs.front();
            if (policy.pair_quality_min_shifted_iou > 0.0f &&
                best.shifted_bbox_iou < policy.pair_quality_min_shifted_iou) {
                triggers |= P2_TRIGGER_PAIR_LOW_IOU;
            }
            if (policy.pair_quality_max_epipolar_dy > 0.0f &&
                best.epipolar_dy > policy.pair_quality_max_epipolar_dy) {
                triggers |= P2_TRIGGER_PAIR_EPIPOLAR_DY;
            }
            if (policy.pair_quality_min_confidence > 0.0f &&
                best.semantic_confidence <
                    policy.pair_quality_min_confidence) {
                triggers |= P2_TRIGGER_PAIR_LOW_CONFIDENCE;
            }
        }
    }

    const bool selected =
        !policy.selective_trigger || triggers != P2_TRIGGER_CONFIGURED;
    if (policy.realtime_lane_enabled && selected) {
        out.realtime_requested = true;
        out.realtime_triggers = triggers;
    } else if (!policy.realtime_lane_enabled) {
        out.realtime_skip_reasons |= P2_SKIP_REALTIME_LANE_DISABLED;
    } else if (!selected) {
        out.realtime_skip_reasons |= P2_SKIP_SELECTIVE_NOT_TRIGGERED;
        if (!direct_pair_possible) {
            out.realtime_skip_reasons |= P2_SKIP_NO_STEREO_DETECTIONS;
        }
    }

    if (policy.diagnostic_lane_enabled &&
        diagnosticStrideHit(frame_id, policy.diagnostic_stride)) {
        out.diagnostic_requested = true;
        out.diagnostic_triggers = P2_TRIGGER_CONFIGURED |
                                  P2_TRIGGER_DIAGNOSTIC_STRIDE;
    } else if (!policy.diagnostic_lane_enabled) {
        out.diagnostic_skip_reasons |= P2_SKIP_DIAGNOSTIC_LANE_DISABLED;
    } else {
        out.diagnostic_skip_reasons |= P2_SKIP_DIAGNOSTIC_STRIDE_MISS;
    }
    return out;
}

std::vector<P2FeatureJobDescriptor> buildP2FeatureJobDescriptors(
    const P2FeatureJobPolicy& policy,
    const P2FeatureJobDecision& decision,
    bool needs_host_gray,
    bool needs_bgr) {
    std::vector<P2FeatureJobDescriptor> out;
    if (!decision.p2_depth_modes_enabled || !decision.split_feature_jobs) {
        return out;
    }
    if (decision.realtime_requested) {
        P2FeatureJobDescriptor job;
        job.lane = P2FeatureJobLane::REALTIME;
        job.depth_mode_mask = decision.depth_mode_mask;
        job.frame_id = decision.frame_id;
        job.left_count = decision.left_count;
        job.right_count = decision.right_count;
        job.triggers = decision.realtime_triggers;
        job.deadline_ms = policy.realtime_deadline_ms;
        job.needs_host_gray = needs_host_gray;
        job.needs_bgr = needs_bgr;
        out.push_back(job);
    }
    if (decision.diagnostic_requested) {
        const uint32_t diagnostic_mask = policy.realtime_lane_enabled
            ? diagnosticSidecarDepthModeMask(decision.depth_mode_mask)
            : decision.depth_mode_mask;
        if (diagnostic_mask == P2_DEPTH_MODE_NONE) {
            return out;
        }
        P2FeatureJobDescriptor job;
        job.lane = P2FeatureJobLane::DIAGNOSTIC;
        job.depth_mode_mask = diagnostic_mask;
        job.frame_id = decision.frame_id;
        job.left_count = decision.left_count;
        job.right_count = decision.right_count;
        job.triggers = decision.diagnostic_triggers;
        job.deadline_ms = policy.diagnostic_deadline_ms;
        job.needs_host_gray = needs_host_gray;
        job.needs_bgr = needs_bgr;
        out.push_back(job);
    }
    return out;
}

const char* p2FeatureJobLaneName(P2FeatureJobLane lane) {
    switch (lane) {
    case P2FeatureJobLane::REALTIME:
        return "realtime";
    case P2FeatureJobLane::DIAGNOSTIC:
        return "diagnostic";
    }
    return "unknown";
}

std::string p2FeatureJobTriggerString(uint32_t triggers) {
    std::string out;
    appendTriggerName(&out, triggers, P2_TRIGGER_CONFIGURED, "configured");
    appendTriggerName(&out, triggers, P2_TRIGGER_ALWAYS, "always");
    appendTriggerName(&out, triggers, P2_TRIGGER_DIRECT_PAIR, "direct_pair");
    appendTriggerName(&out, triggers, P2_TRIGGER_FALLBACK_POSSIBLE, "fallback");
    appendTriggerName(&out, triggers, P2_TRIGGER_HOST_GRAY, "host_gray");
    appendTriggerName(&out, triggers, P2_TRIGGER_BGR, "bgr");
    appendTriggerName(&out, triggers, P2_TRIGGER_DIAGNOSTIC_STRIDE,
                      "diagnostic_stride");
    appendTriggerName(&out, triggers, P2_TRIGGER_PAIR_LOW_IOU,
                      "pair_low_iou");
    appendTriggerName(&out, triggers, P2_TRIGGER_PAIR_EPIPOLAR_DY,
                      "pair_epipolar_dy");
    appendTriggerName(&out, triggers, P2_TRIGGER_PAIR_LOW_CONFIDENCE,
                      "pair_low_confidence");
    appendTriggerName(&out, triggers, P2_TRIGGER_NO_VALID_DIRECT_PAIR,
                      "no_valid_direct_pair");
    if (out.empty()) {
        return "none";
    }
    return out;
}

std::string p2FeatureJobSkipReasonString(uint32_t reasons) {
    std::string out;
    appendTriggerName(&out, reasons, P2_SKIP_NO_DEPTH_MODE, "no_depth_mode");
    appendTriggerName(&out, reasons, P2_SKIP_REALTIME_LANE_DISABLED,
                      "realtime_lane_disabled");
    appendTriggerName(&out, reasons, P2_SKIP_DIAGNOSTIC_LANE_DISABLED,
                      "diagnostic_lane_disabled");
    appendTriggerName(&out, reasons, P2_SKIP_SELECTIVE_NOT_TRIGGERED,
                      "selective_not_triggered");
    appendTriggerName(&out, reasons, P2_SKIP_DIAGNOSTIC_STRIDE_MISS,
                      "diagnostic_stride_miss");
    appendTriggerName(&out, reasons, P2_SKIP_NO_STEREO_DETECTIONS,
                      "no_stereo_detections");
    if (out.empty()) {
        return "none";
    }
    return out;
}

}  // namespace stereo3d
