#include "main_runtime_overrides.h"

#include "utils/logger.h"

#include <algorithm>
#include <cctype>

void applyBaselineClipOverrides(
    const MainCliOptions& cli,
    stereo3d::PipelineConfig& pipeline_cfg,
    stereo3d::BaselineClipRecorderConfig& baseline_cfg) {
    if (cli.baseline_clip_cli) baseline_cfg.enabled = true;
    if (!cli.baseline_out_override.empty()) {
        baseline_cfg.output_dir = cli.baseline_out_override;
    }
    if (cli.baseline_duration_override > 0.0) {
        baseline_cfg.duration_sec = cli.baseline_duration_override;
    }
    if (cli.baseline_frames_override > 0) {
        baseline_cfg.frame_limit = cli.baseline_frames_override;
    }
    if (cli.baseline_clips_override > 0) {
        baseline_cfg.clip_count = cli.baseline_clips_override;
    }
    if (cli.baseline_gap_override >= 0.0) {
        baseline_cfg.clip_gap_sec = cli.baseline_gap_override;
    }
    if (!cli.baseline_format_override.empty()) {
        baseline_cfg.image_format = cli.baseline_format_override;
    }
    if (!cli.baseline_image_mode_override.empty()) {
        baseline_cfg.image_mode = cli.baseline_image_mode_override;
    }
    if (cli.baseline_start_immediately) {
        baseline_cfg.require_left_detection = false;
        baseline_cfg.require_right_detection = false;
        baseline_cfg.require_pair_gate = false;
    }
    baseline_cfg.trigger_hz = pipeline_cfg.trigger_freq_hz;
    if (cli.debug_feature_matches) baseline_cfg.enabled = false;

    if (!baseline_cfg.enabled) return;

    std::string baseline_mode = baseline_cfg.image_mode;
    std::transform(baseline_mode.begin(), baseline_mode.end(), baseline_mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if ((baseline_mode == "bgr" || baseline_mode == "both") &&
        pipeline_cfg.detector_input_format != "bgr") {
        LOG_WARN("Baseline image_mode=%s requested but detector.input_format=%s; "
                 "BGR images are only valid when the color pipeline is enabled",
                 baseline_cfg.image_mode.c_str(),
                 pipeline_cfg.detector_input_format.c_str());
    }

    pipeline_cfg.detection_only = true;
    pipeline_cfg.disparity_strategy = stereo3d::DisparityStrategy::ROI_ONLY;
    pipeline_cfg.tracker.enabled = false;
    pipeline_cfg.neural_features.enabled = false;
    pipeline_cfg.dual_yolo.use_for_depth = false;
    pipeline_cfg.dual_yolo.fallback_to_roi_match = false;
    pipeline_cfg.dual_yolo.log_matches = false;
    LOG_INFO("Baseline clip mode enabled: detection-only, stereo/depth/tracker disabled");
    if (!pipeline_cfg.dual_yolo.enabled) {
        LOG_WARN("Baseline clip mode has dual_yolo.enabled=false; right detections will be empty");
    }
}

void applyRealtimeDebugDumpOverrides(
    const MainCliOptions& cli,
    RealtimeDebugDumpConfig& realtime_dump_cfg) {
    if (cli.debug_realtime_dump_cli) realtime_dump_cfg.enabled = true;
    if (!cli.debug_realtime_dump_dir_override.empty()) {
        realtime_dump_cfg.output_dir = cli.debug_realtime_dump_dir_override;
    }
    if (cli.debug_realtime_dump_stride_override >= 0) {
        realtime_dump_cfg.stride = cli.debug_realtime_dump_stride_override;
    }
    if (cli.debug_realtime_dump_max_override >= 0) {
        realtime_dump_cfg.max_frames = cli.debug_realtime_dump_max_override;
    }
    realtime_dump_cfg.stride = std::max(0, realtime_dump_cfg.stride);
    realtime_dump_cfg.max_frames = std::max(0, realtime_dump_cfg.max_frames);
    realtime_dump_cfg.max_queue = std::max(1, realtime_dump_cfg.max_queue);
}
