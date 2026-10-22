/**
 * @file main_cli_options.cpp
 * @brief Command-line options for stereo_3d_pipeline.
 */

#include "main_cli_options.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

const char* kUsage =
    "Usage: %s [--config <path>] [--visualize] "
    "[--debug-feature-matches] [--debug-feature-matches-dir <dir>]\n";

bool hasValue(int i, int argc, char* argv[]) {
    return i + 1 < argc && argv[i + 1][0] != '-';
}

void printHelp(const char* argv0) {
    printf(kUsage, argv0);
    printf("  --config, -c                  Pipeline configuration YAML\n");
    printf("  --visualize, -v               Show detection + distance overlay window\n");
    printf("  --debug-feature-matches       Capture one stereo pair and export ROI feature-match images\n");
    printf("  --debug-feature-matches-dir   Output directory for feature-match images\n");
}

void failMissingValue(MainCliOptions& options,
                      const char* argv0,
                      const std::string& arg) {
    fprintf(stderr, "Error: %s requires a value.\n", arg.c_str());
    fprintf(stderr, kUsage, argv0);
    options.should_exit = true;
    options.exit_code = 1;
}

}  // namespace

MainCliOptions parseMainCliOptions(int argc, char* argv[]) {
    MainCliOptions options;
    std::vector<std::string> unknown_args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && hasValue(i, argc, argv)) {
            options.config_path = argv[++i];
        } else if (arg == "--config" || arg == "-c") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--visualize" || arg == "--display" || arg == "-v") {
            options.enable_display = true;
        } else if (arg == "--debug-feature-matches") {
            options.debug_feature_matches = true;
        } else if (arg == "--debug-feature-matches-dir" &&
                   hasValue(i, argc, argv)) {
            options.debug_feature_matches = true;
            options.debug_feature_matches_dir = argv[++i];
        } else if (arg == "--debug-feature-matches-dir") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            options.should_exit = true;
            options.exit_code = 0;
            return options;
        } else {
            unknown_args.push_back(arg);
        }
    }

    if (!unknown_args.empty()) {
        fprintf(stderr, "Error: unknown option(s):");
        for (const auto& opt : unknown_args) {
            fprintf(stderr, " %s", opt.c_str());
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "Hint: use --help to see supported options.\n");
        options.should_exit = true;
        options.exit_code = 1;
    }
    return options;
}
