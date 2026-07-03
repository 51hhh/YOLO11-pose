/**
 * @file main_cli_options.h
 * @brief Command-line options for stereo_3d_pipeline.
 */

#ifndef STEREO_3D_PIPELINE_MAIN_CLI_OPTIONS_H_
#define STEREO_3D_PIPELINE_MAIN_CLI_OPTIONS_H_

#include <string>

struct MainCliOptions {
    std::string config_path = "config/pipeline.yaml";
    bool enable_display = false;
    bool debug_feature_matches = false;
    std::string debug_feature_matches_dir = "test_logs/feature_match_debug";
    std::string recording_out_override;
    bool baseline_clip_cli = false;
    std::string baseline_out_override;
    double baseline_duration_override = -1.0;
    int baseline_frames_override = 0;
    int baseline_clips_override = 0;
    double baseline_gap_override = -1.0;
    std::string baseline_format_override;
    std::string baseline_image_mode_override;
    bool baseline_start_immediately = false;
    bool debug_realtime_dump_cli = false;
    std::string debug_realtime_dump_dir_override;
    int debug_realtime_dump_stride_override = -1;
    int debug_realtime_dump_max_override = -1;
    bool should_exit = false;
    int exit_code = 0;
};

MainCliOptions parseMainCliOptions(int argc, char* argv[]);

#endif  // STEREO_3D_PIPELINE_MAIN_CLI_OPTIONS_H_
