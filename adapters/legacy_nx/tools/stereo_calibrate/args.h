/**
 * @file args.h
 * @brief Command-line options for the stereo_calibrate tool.
 */

#ifndef STEREO_3D_PIPELINE_STEREO_CALIBRATE_ARGS_H_
#define STEREO_3D_PIPELINE_STEREO_CALIBRATE_ARGS_H_

#include <cstdio>
#include <string>

struct StereoCalibrateArgs {
    float square_size = 0.0f;
    std::string images_dir = "calibration_images";
    std::string output = "stereo_calib.yaml";
    int board_w = 5;
    int board_h = 8;
    bool no_vis = false;
    bool fix_intrinsics = true;
    bool use_sb = false;
    bool use_exhaustive = false;
    int jobs = 0;
    bool help_requested = false;
};

bool parseStereoCalibrateArgs(int argc,
                              char* argv[],
                              StereoCalibrateArgs& args);
bool validateStereoCalibrateArgs(const StereoCalibrateArgs& args);
bool validateStereoCalibrateOutputPath(const std::string& output);
void printStereoCalibrateUsage(const char* argv0,
                               std::FILE* stream = stdout);

#endif  // STEREO_3D_PIPELINE_STEREO_CALIBRATE_ARGS_H_
