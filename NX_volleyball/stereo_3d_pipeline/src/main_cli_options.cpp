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
    "[--debug-feature-matches] [--debug-feature-matches-dir <dir>] "
    "[--debug-realtime-dump] [--debug-realtime-dump-dir <dir>] "
    "[--debug-realtime-dump-stride <n>] [--debug-realtime-dump-max <n>] "
    "[--recording-out <csv>] "
    "[--record-baseline-clip] [--baseline-out <dir>] "
    "[--baseline-duration <sec>] [--baseline-frames <n>] "
    "[--baseline-clips <n>] [--baseline-gap <sec>] "
    "[--baseline-format <png|pgm>] [--baseline-image-mode <gray|bgr|both>] "
    "[--baseline-start-immediately]\n";

bool hasValue(int i, int argc, char* argv[]) {
    return i + 1 < argc && argv[i + 1][0] != '-';
}

void printHelp(const char* argv0) {
    printf(kUsage, argv0);
    printf("  --config, -c                  Pipeline configuration YAML\n");
    printf("  --visualize, -v               Show detection + distance overlay window\n");
    printf("  --debug-feature-matches       Capture one stereo pair and export ROI feature-match images\n");
    printf("  --debug-feature-matches-dir   Output directory for feature-match images\n");
    printf("  --debug-realtime-dump         Low-rate realtime zoom PNG/JSON dump (background writer)\n");
    printf("  --debug-realtime-dump-dir     Output directory for realtime debug dump\n");
    printf("  --debug-realtime-dump-stride  Dump every N frames; 0 disables periodic dumps\n");
    printf("  --debug-realtime-dump-max     Stop dumping after N frames; 0 means unlimited\n");
    printf("  --recording-out <csv>         Override trajectory recorder CSV output path\n");
    printf("  --record-baseline-clip        Record one fixed-length left/right image sequence + CSV after ball detection\n");
    printf("  --baseline-out                Output root directory for baseline clips\n");
    printf("  --baseline-duration           Clip duration in seconds, converted by trigger frequency\n");
    printf("  --baseline-frames             Exact number of frames to record\n");
    printf("  --baseline-clips              Number of clips to record\n");
    printf("  --baseline-gap                Gap between clips in seconds\n");
    printf("  --baseline-format             Lossless image format: png or pgm\n");
    printf("  --baseline-image-mode         Image mode: gray, bgr, or both\n");
    printf("  --baseline-start-immediately  Record without waiting for detections\n");
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
        } else if (arg == "--debug-realtime-dump") {
            options.debug_realtime_dump_cli = true;
        } else if (arg == "--debug-realtime-dump-dir" &&
                   hasValue(i, argc, argv)) {
            options.debug_realtime_dump_cli = true;
            options.debug_realtime_dump_dir_override = argv[++i];
        } else if (arg == "--debug-realtime-dump-dir") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--debug-realtime-dump-stride" &&
                   hasValue(i, argc, argv)) {
            options.debug_realtime_dump_cli = true;
            options.debug_realtime_dump_stride_override = std::stoi(argv[++i]);
        } else if (arg == "--debug-realtime-dump-stride") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--debug-realtime-dump-max" &&
                   hasValue(i, argc, argv)) {
            options.debug_realtime_dump_cli = true;
            options.debug_realtime_dump_max_override = std::stoi(argv[++i]);
        } else if (arg == "--debug-realtime-dump-max") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--recording-out" && hasValue(i, argc, argv)) {
            options.recording_out_override = argv[++i];
        } else if (arg == "--recording-out") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--record-baseline-clip") {
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-out" && hasValue(i, argc, argv)) {
            options.baseline_out_override = argv[++i];
        } else if (arg == "--baseline-out") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-duration" && hasValue(i, argc, argv)) {
            options.baseline_duration_override = std::stod(argv[++i]);
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-duration") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-frames" && hasValue(i, argc, argv)) {
            options.baseline_frames_override = std::stoi(argv[++i]);
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-frames") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-clips" && hasValue(i, argc, argv)) {
            options.baseline_clips_override = std::stoi(argv[++i]);
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-clips") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-gap" && hasValue(i, argc, argv)) {
            options.baseline_gap_override = std::stod(argv[++i]);
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-gap") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-format" && hasValue(i, argc, argv)) {
            options.baseline_format_override = argv[++i];
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-format") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-image-mode" &&
                   hasValue(i, argc, argv)) {
            options.baseline_image_mode_override = argv[++i];
            options.baseline_clip_cli = true;
        } else if (arg == "--baseline-image-mode") {
            failMissingValue(options, argv[0], arg);
            return options;
        } else if (arg == "--baseline-start-immediately") {
            options.baseline_start_immediately = true;
            options.baseline_clip_cli = true;
        } else if (arg == "--visualizels") {
            fprintf(stderr, "Warning: unknown option '--visualizels', treating as '--visualize'.\n");
            options.enable_display = true;
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
