#ifndef STEREO_3D_PIPELINE_PIPELINE_FEATURE_JOBS_H_
#define STEREO_3D_PIPELINE_PIPELINE_FEATURE_JOBS_H_

#include "detection_types.h"
#include "pipeline_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace stereo3d {

enum class P2FeatureJobLane {
    REALTIME,
    DIAGNOSTIC,
};

enum P2FeatureJobTrigger : uint32_t {
    P2_TRIGGER_NONE = 0u,
    P2_TRIGGER_CONFIGURED = 1u << 0,
    P2_TRIGGER_ALWAYS = 1u << 1,
    P2_TRIGGER_DIRECT_PAIR = 1u << 2,
    P2_TRIGGER_FALLBACK_POSSIBLE = 1u << 3,
    P2_TRIGGER_HOST_GRAY = 1u << 4,
    P2_TRIGGER_BGR = 1u << 5,
    P2_TRIGGER_DIAGNOSTIC_STRIDE = 1u << 6,
    P2_TRIGGER_PAIR_LOW_IOU = 1u << 7,
    P2_TRIGGER_PAIR_EPIPOLAR_DY = 1u << 8,
    P2_TRIGGER_PAIR_LOW_CONFIDENCE = 1u << 9,
    P2_TRIGGER_NO_VALID_DIRECT_PAIR = 1u << 10,
};

enum P2FeatureJobSkipReason : uint32_t {
    P2_SKIP_NONE = 0u,
    P2_SKIP_NO_DEPTH_MODE = 1u << 0,
    P2_SKIP_REALTIME_LANE_DISABLED = 1u << 1,
    P2_SKIP_DIAGNOSTIC_LANE_DISABLED = 1u << 2,
    P2_SKIP_SELECTIVE_NOT_TRIGGERED = 1u << 3,
    P2_SKIP_DIAGNOSTIC_STRIDE_MISS = 1u << 4,
    P2_SKIP_NO_STEREO_DETECTIONS = 1u << 5,
};

enum P2FeatureDepthMode : uint32_t {
    P2_DEPTH_MODE_NONE = 0u,
    P2_DEPTH_MODE_CORNER_POINTS = 1u << 0,
    P2_DEPTH_MODE_TEXTURE_POINTS = 1u << 1,
    P2_DEPTH_MODE_BINARY_POINTS = 1u << 2,
    P2_DEPTH_MODE_ORB_POINTS = 1u << 3,
    P2_DEPTH_MODE_BRISK_POINTS = 1u << 4,
    P2_DEPTH_MODE_AKAZE_POINTS = 1u << 5,
    P2_DEPTH_MODE_SIFT_POINTS = 1u << 6,
    P2_DEPTH_MODE_IOU_COLOR_PATCH = 1u << 7,
    P2_DEPTH_MODE_PATCH_IOU_EDGE = 1u << 8,
    P2_DEPTH_MODE_CUDA_TEMPLATE = 1u << 9,
    P2_DEPTH_MODE_CUDA_STEREO_BM = 1u << 10,
    P2_DEPTH_MODE_CUDA_STEREO_SGM = 1u << 11,
    P2_DEPTH_MODE_NEURAL_FEATURE = 1u << 12,
    P2_DEPTH_MODE_FALLBACK_FEATURE_POINTS = 1u << 13,
    P2_DEPTH_MODE_RING_EDGE_PROFILE = 1u << 14,
};

struct P2FeatureJobPolicy {
    bool p2_depth_modes_enabled = false;
    uint32_t depth_mode_mask = P2_DEPTH_MODE_NONE;
    bool split_feature_jobs = false;
    bool realtime_lane_enabled = true;
    bool diagnostic_lane_enabled = false;
    bool selective_trigger = false;
    bool trigger_on_fallback = true;
    bool trigger_on_direct_pair = false;
    bool trigger_on_host_gray = false;
    bool trigger_on_bgr = false;
    bool trigger_on_pair_quality = false;
    bool trigger_on_no_valid_direct_pair = false;
    float pair_quality_min_shifted_iou = 0.0f;
    float pair_quality_max_epipolar_dy = 0.0f;
    float pair_quality_min_confidence = 0.0f;
    int pair_gate_max_disparity = 2048;
    float pair_gate_epipolar_y_tolerance = 12.0f;
    float pair_gate_max_size_ratio = 2.0f;
    float pair_gate_min_shifted_iou = 0.0f;
    int diagnostic_stride = 10;
    int diagnostic_max_in_flight = 1;
    float realtime_deadline_ms = 10.0f;
    float diagnostic_deadline_ms = 50.0f;
};

struct P2FeatureJobDecision {
    bool p2_depth_modes_enabled = false;
    uint32_t depth_mode_mask = P2_DEPTH_MODE_NONE;
    bool split_feature_jobs = false;
    bool realtime_requested = false;
    bool diagnostic_requested = false;
    uint32_t realtime_triggers = P2_TRIGGER_NONE;
    uint32_t diagnostic_triggers = P2_TRIGGER_NONE;
    uint32_t realtime_skip_reasons = P2_SKIP_NONE;
    uint32_t diagnostic_skip_reasons = P2_SKIP_NONE;
    int frame_id = -1;
    int left_count = 0;
    int right_count = 0;
    int valid_direct_pair_count = 0;
};

struct P2FeatureJobDescriptor {
    P2FeatureJobLane lane = P2FeatureJobLane::REALTIME;
    uint32_t depth_mode_mask = P2_DEPTH_MODE_NONE;
    int frame_id = -1;
    int left_count = 0;
    int right_count = 0;
    uint32_t triggers = P2_TRIGGER_NONE;
    float deadline_ms = 10.0f;
    bool needs_host_gray = false;
    bool needs_bgr = false;
};

uint32_t dualYoloP2DepthModeMask(const PipelineConfig::DualYoloConfig& cfg);
bool dualYoloP2DepthModesEnabled(const PipelineConfig::DualYoloConfig& cfg);
uint32_t p2FeatureDepthModeMask(const PipelineConfig& cfg);
bool pipelineP2DepthModesEnabled(const PipelineConfig& cfg);
P2FeatureJobPolicy makeP2FeatureJobPolicy(const PipelineConfig& cfg);
P2FeatureJobDecision decideP2FeatureJobs(
    const P2FeatureJobPolicy& policy,
    int frame_id,
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    bool needs_host_gray,
    bool needs_bgr);
std::vector<P2FeatureJobDescriptor> buildP2FeatureJobDescriptors(
    const P2FeatureJobPolicy& policy,
    const P2FeatureJobDecision& decision,
    bool needs_host_gray,
    bool needs_bgr);

const char* p2FeatureJobLaneName(P2FeatureJobLane lane);
std::string p2FeatureJobTriggerString(uint32_t triggers);
std::string p2FeatureJobSkipReasonString(uint32_t reasons);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_FEATURE_JOBS_H_
