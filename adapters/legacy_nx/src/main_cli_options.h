/**
 * @file main_cli_options.h
 * @brief Command-line options for stereo_3d_pipeline.
 */

#ifndef STEREO_3D_PIPELINE_MAIN_CLI_OPTIONS_H_
#define STEREO_3D_PIPELINE_MAIN_CLI_OPTIONS_H_

#include <string>

struct MainCliOptions {
    std::string config_path = "config/base_pipeline.yaml";
    bool enable_display = false;
    bool should_exit = false;
    int exit_code = 0;
};

MainCliOptions parseMainCliOptions(int argc, char* argv[]);

#endif  // STEREO_3D_PIPELINE_MAIN_CLI_OPTIONS_H_
