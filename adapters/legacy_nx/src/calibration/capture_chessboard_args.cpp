/**
 * @file capture_chessboard_args.cpp
 * @brief Command-line options for capture_chessboard.
 */

#include "capture_chessboard_args.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

bool parseCaptureChessboardArgs(int argc,
                                char* argv[],
                                CaptureChessboardArgs& a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const char* value = nullptr;
        if (arg == "--free-run") a.free_run = true;
        else if (arg == "--no-pwm") a.no_pwm = true;
        else if (arg == "--headless") a.headless = true;
        else if (arg == "-n") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.auto_count)) {
                fprintf(stderr, "[ERROR] Invalid -n value: %s\n",
                        value ? value : "");
                return false;
            }
            a.headless = true;
        }
        else if (arg == "-o") {
            if (!requireValue(i, argc, argv, arg, value)) return false;
            a.output_dir = value;
        }
        else if (arg == "-e") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.exposure_us)) {
                fprintf(stderr, "[ERROR] Invalid -e value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "-g") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseFloatValue(value, a.gain_db)) {
                fprintf(stderr, "[ERROR] Invalid -g value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "--width") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.cam_width)) {
                fprintf(stderr, "[ERROR] Invalid --width value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "--height") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.cam_height)) {
                fprintf(stderr, "[ERROR] Invalid --height value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "--left-index") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.left_index)) {
                fprintf(stderr, "[ERROR] Invalid --left-index value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "--right-index") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.right_index)) {
                fprintf(stderr, "[ERROR] Invalid --right-index value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "--serial-left") {
            if (!requireValue(i, argc, argv, arg, value)) return false;
            a.serial_left = value;
        }
        else if (arg == "--serial-right") {
            if (!requireValue(i, argc, argv, arg, value)) return false;
            a.serial_right = value;
        }
        else if (arg == "--image-node-num") {
            if (!requireValue(i, argc, argv, arg, value) ||
                !parseIntValue(value, a.image_node_num)) {
                fprintf(stderr, "[ERROR] Invalid --image-node-num value: %s\n",
                        value ? value : "");
                return false;
            }
        }
        else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s [options]\n"
                   "  -o DIR          Output directory [calibration_images]\n"
                   "  -e US           Exposure time in us [9867]\n"
                   "  -g DB           Gain in dB [11.9906]\n"
                   "  --free-run      Free-run mode (no HW trigger)\n"
                   "  --no-pwm        Disable PWM output\n"
                   "  --headless      No GUI (for SSH sessions)\n"
                   "  -n COUNT        Auto-capture COUNT pairs then exit (implies --headless)\n"
                   "  --width W       Camera width [1440]\n"
                   "  --height H      Camera height [1080]\n"
                   "  --left-index N  Left camera index [0]\n"
                   "  --right-index N Right camera index [1]\n"
                   "  --serial-left S Bind left camera by serial number\n"
                   "  --serial-right S Bind right camera by serial number\n"
                   "  --image-node-num N SDK FIFO depth [3]\n"
                   "  -h, --help      Show this help\n",
                   argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "[ERROR] Unknown or incomplete argument: %s\n",
                    arg.c_str());
            return false;
        }
    }
    return true;
}

bool validateCaptureChessboardArgs(const CaptureChessboardArgs& a) {
    if (a.exposure_us <= 0) {
        fprintf(stderr, "[ERROR] Exposure must be positive, got %d\n",
                a.exposure_us);
        return false;
    }
    if (a.cam_width <= 0 || a.cam_height <= 0) {
        fprintf(stderr, "[ERROR] Invalid image size: %dx%d\n",
                a.cam_width, a.cam_height);
        return false;
    }
    if (a.auto_count < 0) {
        fprintf(stderr, "[ERROR] Auto capture count must be >= 0, got %d\n",
                a.auto_count);
        return false;
    }
    if (!a.serial_left.empty() && !a.serial_right.empty() &&
        a.serial_left == a.serial_right) {
        fprintf(stderr, "[ERROR] Left/right serial numbers must be different: %s\n",
                a.serial_left.c_str());
        return false;
    }
    if ((a.serial_left.empty() || a.serial_right.empty()) &&
        a.left_index == a.right_index) {
        fprintf(stderr, "[ERROR] Left/right camera indices must be different when serial binding is incomplete: %d\n",
                a.left_index);
        return false;
    }
    return true;
}
