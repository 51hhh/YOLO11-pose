/**
 * @file args.cpp
 * @brief Command-line options for the stereo_calibrate tool.
 */

#include "args.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

bool requireValue(int& i, int argc, char* argv[],
                  const std::string& arg, const char*& value) {
    if (i + 1 >= argc) {
        fprintf(stderr, "[ERROR] %s requires a value\n", arg.c_str());
        return false;
    }
    value = argv[++i];
    return true;
}

}  // namespace

void printStereoCalibrateUsage(const char* argv0, std::FILE* stream) {
    std::fprintf(
        stream,
        "Usage: %s -s SQUARE_SIZE [options]\n"
        "  -s, --square-size MM   Square size in mm (required)\n"
        "  -d, --images-dir DIR   Image directory [calibration_images]\n"
        "  -o, --output FILE      New output YAML [stereo_calib.yaml]\n"
        "  --board-w N            Inner corners width [5]\n"
        "  --board-h N            Inner corners height [8]\n"
        "  --no-vis               Skip visualization\n"
        "  --gpu-preprocess       Use CUDA for image-to-gray preprocessing when available\n"
        "  --optimize-intrinsics  Refine intrinsics during stereo calibration\n"
        "  --sb                   Enable SB chessboard fallback\n"
        "  --exhaustive           Enable slow exhaustive SB fallback\n"
        "  --jobs N               Detection workers [CPU cores; 1 with GPU preprocessing]\n"
        "  -h, --help             Show this help\n",
        argv0);
}

bool parseStereoCalibrateArgs(int argc,
                              char* argv[],
                              StereoCalibrateArgs& a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const char* value = nullptr;
        if (arg == "-s" || arg == "--square-size") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseFloatValue(value, a.square_size)) {
                fprintf(stderr, "[ERROR] Invalid square size: %s\n",
                        value ? value : "");
                return false;
            }
        } else if (arg == "-d" || arg == "--images-dir") {
            if (!requireValue(i, argc, argv, arg, value)) return false;
            a.images_dir = value;
        } else if (arg == "-o" || arg == "--output") {
            if (!requireValue(i, argc, argv, arg, value)) return false;
            a.output = value;
        } else if (arg == "--board-w") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.board_w)) {
                fprintf(stderr, "[ERROR] Invalid --board-w value: %s\n",
                        value ? value : "");
                return false;
            }
        } else if (arg == "--board-h") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.board_h)) {
                fprintf(stderr, "[ERROR] Invalid --board-h value: %s\n",
                        value ? value : "");
                return false;
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
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.jobs)) {
                fprintf(stderr, "[ERROR] Invalid --jobs value: %s\n",
                        value ? value : "");
                return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            printStereoCalibrateUsage(argv[0]);
            a.help_requested = true;
            return true;
        } else {
            fprintf(stderr, "[ERROR] Unknown or incomplete argument: %s\n",
                    arg.c_str());
            printStereoCalibrateUsage(argv[0], stderr);
            return false;
        }
    }
    if (a.use_exhaustive) a.use_sb = true;
    return true;
}

bool validateStereoCalibrateArgs(const StereoCalibrateArgs& a) {
    if (a.help_requested) return true;
    if (a.square_size <= 0.0f) {
        fprintf(stderr, "[ERROR] Square size required: -s <mm>\n");
        return false;
    }
    if (a.images_dir.empty()) {
        fprintf(stderr, "[ERROR] Image directory must not be empty\n");
        return false;
    }
    if (a.output.empty()) {
        fprintf(stderr, "[ERROR] Output path must not be empty\n");
        return false;
    }
    if (a.board_w < 2 || a.board_h < 2) {
        fprintf(stderr, "[ERROR] Invalid chessboard inner corners: %dx%d\n",
                a.board_w, a.board_h);
        return false;
    }
    if (a.jobs < 0) {
        fprintf(stderr, "[ERROR] --jobs must be >= 0, got %d\n", a.jobs);
        return false;
    }
    return true;
}

bool validateStereoCalibrateOutputPath(const std::string& output) {
    namespace fs = std::filesystem;
    const fs::path output_path(output);
    if (output_path.empty() || output_path.filename().empty()) {
        fprintf(stderr, "[ERROR] Output must name a calibration file\n");
        return false;
    }

    std::error_code error;
    if (fs::exists(output_path, error)) {
        fprintf(stderr,
                "[ERROR] Refusing to overwrite existing calibration: %s\n"
                "        Choose a new -o/--output path.\n",
                output.c_str());
        return false;
    }
    if (error) {
        fprintf(stderr, "[ERROR] Cannot inspect output path %s: %s\n",
                output.c_str(), error.message().c_str());
        return false;
    }

    const fs::path parent = output_path.parent_path();
    if (!parent.empty() && fs::exists(parent, error)) {
        if (error || !fs::is_directory(parent, error) || error) {
            fprintf(stderr, "[ERROR] Output parent is not a directory: %s\n",
                    parent.string().c_str());
            return false;
        }
    } else if (error) {
        fprintf(stderr, "[ERROR] Cannot inspect output parent %s: %s\n",
                parent.string().c_str(), error.message().c_str());
        return false;
    }
    return true;
}
