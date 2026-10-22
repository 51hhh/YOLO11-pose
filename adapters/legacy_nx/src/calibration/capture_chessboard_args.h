/**
 * @file capture_chessboard_args.h
 * @brief Command-line options for capture_chessboard.
 */

#ifndef STEREO_3D_PIPELINE_CAPTURE_CHESSBOARD_ARGS_H_
#define STEREO_3D_PIPELINE_CAPTURE_CHESSBOARD_ARGS_H_

#include <string>

struct CaptureChessboardArgs {
    std::string output_dir = "calibration_images";
    int exposure_us = 9867;
    float gain_db = 11.9906f;
    bool free_run = false;
    bool no_pwm = false;
    bool headless = false;
    int auto_count = 0;
    int cam_width = 1440;
    int cam_height = 1080;
    int left_index = 0;
    int right_index = 1;
    int image_node_num = 3;
    std::string serial_left;
    std::string serial_right;
};

bool parseCaptureChessboardArgs(int argc,
                                char* argv[],
                                CaptureChessboardArgs& args);

bool validateCaptureChessboardArgs(const CaptureChessboardArgs& args);

#endif  // STEREO_3D_PIPELINE_CAPTURE_CHESSBOARD_ARGS_H_
