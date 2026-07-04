#include "pipeline_feature_jobs.h"

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

    const bool selected =
        !policy.selective_trigger || triggers != P2_TRIGGER_CONFIGURED;
    if (policy.realtime_lane_enabled && selected) {
        out.realtime_requested = true;
        out.realtime_triggers = triggers;
    }

    if (policy.diagnostic_lane_enabled &&
        diagnosticStrideHit(frame_id, policy.diagnostic_stride)) {
        out.diagnostic_requested = true;
        out.diagnostic_triggers = P2_TRIGGER_CONFIGURED |
                                  P2_TRIGGER_DIAGNOSTIC_STRIDE;
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
        P2FeatureJobDescriptor job;
        job.lane = P2FeatureJobLane::DIAGNOSTIC;
        job.depth_mode_mask = decision.depth_mode_mask;
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
    if (out.empty()) {
        return "none";
    }
    return out;
}

}  // namespace stereo3d
