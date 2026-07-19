/**
 * @file stereo_depth_viewer_args.h
 * @brief Command-line options for stereo_depth_viewer.
 */

#ifndef STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_ARGS_H_
#define STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_ARGS_H_

#include <string>

struct StereoDepthViewerArgs {
    std::string calib_path;
    std::string config_path;
    int max_disp = 256;
    int win_size = 5;
    bool free_run = false;
    bool headless = false;
    int headless_frames = 30;
    bool diagnose = false;
    bool swap_lr = false;
    double pwm_freq = 15.0;
    bool no_pwm = false;
    std::string crestereo_path;
    std::string hitnet_path;
    bool should_exit = false;
    int exit_code = 0;
};

StereoDepthViewerArgs parseStereoDepthViewerArgs(int argc, char** argv);

#endif  // STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_ARGS_H_
