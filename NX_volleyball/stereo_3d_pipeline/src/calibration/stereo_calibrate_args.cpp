/**
 * @file stereo_calibrate_args.cpp
 * @brief Command-line options for the stereo_calibrate tool.
 */

#include "stereo_calibrate_args.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool parseIntValue(const char* text, int& out) {
    if (!text || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parseFloatValue(const char* text, float& out) {
    if (!text || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

const char* requireValue(int& i, int argc, char* argv[],
                         const std::string& arg) {
    if (i + 1 >= argc) {
        fprintf(stderr, "[ERROR] %s requires a value\n", arg.c_str());
        std::exit(1);
    }
    return argv[++i];
}

}  // namespace

StereoCalibrateArgs parseStereoCalibrateArgs(int argc, char* argv[]) {
    StereoCalibrateArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--square-size") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseFloatValue(value, a.square_size)) {
                fprintf(stderr, "[ERROR] Invalid square size: %s\n", value);
                std::exit(1);
            }
        } else if (arg == "-d" || arg == "--images-dir") {
            a.images_dir = requireValue(i, argc, argv, arg);
        } else if (arg == "-o" || arg == "--output") {
            a.output = requireValue(i, argc, argv, arg);
        } else if (arg == "--board-w") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseIntValue(value, a.board_w)) {
                fprintf(stderr, "[ERROR] Invalid --board-w value: %s\n", value);
                std::exit(1);
            }
        } else if (arg == "--board-h") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseIntValue(value, a.board_h)) {
                fprintf(stderr, "[ERROR] Invalid --board-h value: %s\n", value);
                std::exit(1);
            }
        } else if (arg == "--no-vis") {
            a.no_vis = true;
        } else if (arg == "--gpu-preprocess") {
            a.use_gpu_preprocess = true;
        } else if (arg == "--optimize-intrinsics") {
            a.fix_intrinsics = false;
        } else if (arg == "--sb") {
            a.use_sb = true;
        } else if (arg == "--exhaustive") {
            a.use_exhaustive = true;
        } else if (arg == "--jobs") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseIntValue(value, a.jobs)) {
                fprintf(stderr, "[ERROR] Invalid --jobs value: %s\n", value);
                std::exit(1);
            }
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s -s SQUARE_SIZE [options]\n"
                   "  -s, --square-size MM   Square size in mm (required)\n"
                   "  -d, --images-dir DIR   Image directory (default: calibration_images)\n"
                   "  -o, --output FILE      Output YAML (default: stereo_calib.yaml)\n"
                   "  --board-w N            Inner corners width (default: 5)\n"
                   "  --board-h N            Inner corners height (default: 8)\n"
                   "  --no-vis               Skip visualization\n"
                   "  --gpu-preprocess       Use CUDA for Bayer/BGR to gray preprocessing when available\n"
                   "  --optimize-intrinsics  Let stereoCalibrate refine intrinsics (default: fix monocular intrinsics)\n"
                   "  --sb                   Enable SB chessboard fallback after classic detectors\n"
                   "  --exhaustive           Enable slow SB exhaustive fallback for difficult boards\n"
                   "  --jobs N               Parallel image-pair detection jobs (default: CPU cores; 1 with --gpu-preprocess)\n"
                   "  -h, --help             Show help\n",
                   argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "[ERROR] Unknown or incomplete argument: %s\n",
                    arg.c_str());
            std::exit(1);
        }
    }
    if (a.square_size <= 0.0f) {
        fprintf(stderr, "[ERROR] Square size required: -s <mm>\n");
        std::exit(1);
    }
    if (a.board_w < 2 || a.board_h < 2) {
        fprintf(stderr, "[ERROR] Invalid chessboard inner corners: %dx%d\n",
                a.board_w, a.board_h);
        std::exit(1);
    }
    if (a.jobs < 0) {
        fprintf(stderr, "[ERROR] --jobs must be >= 0, got %d\n", a.jobs);
        std::exit(1);
    }
    if (a.use_exhaustive) {
        a.use_sb = true;
    }
    return a;
}
