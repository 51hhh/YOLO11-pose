#include "args.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace hik_metadata_probe {
namespace {

bool parseInt(const char* text, int& output) {
    if (text == nullptr || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        return false;
    }
    output = static_cast<int>(value);
    return true;
}

bool parseUnsigned(const char* text, unsigned int& output) {
    if (text == nullptr || *text == '\0' || *text == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    output = static_cast<unsigned int>(value);
    return true;
}

bool parseDouble(const char* text, double& output) {
    if (text == nullptr || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !std::isfinite(value)) {
        return false;
    }
    output = value;
    return true;
}

bool requireValue(int& index, int argc, char** argv,
                  const char* option, const char*& value) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", option);
        return false;
    }
    value = argv[++index];
    return true;
}

template <typename Value, typename Parser>
bool parseOptionValue(int& index, int argc, char** argv,
                      const char* option, Value& output, Parser parser) {
    const char* value = nullptr;
    if (!requireValue(index, argc, argv, option, value)) return false;
    if (!parser(value, output)) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
        return false;
    }
    return true;
}

}  // namespace

void printUsage(const char* argv0, std::FILE* stream) {
    std::fprintf(
        stream,
        "Usage: %s [--left-sn SN] [--right-sn SN] [--left-index N] [--right-index N]\n"
        "          [--frames N] [--timeout-ms N] [--exposure-us US] [--gain-db DB]\n"
        "          [--no-pwm] [--trigger-chip CHIP] [--trigger-line N] [--pwm-hz HZ]\n",
        argv0);
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        const char* value = nullptr;
        if (a == "--left-sn") {
            if (!requireValue(i, argc, argv, "--left-sn", value)) return false;
            args.left_sn = value;
        } else if (a == "--right-sn") {
            if (!requireValue(i, argc, argv, "--right-sn", value)) return false;
            args.right_sn = value;
        } else if (a == "--left-index") {
            if (!parseOptionValue(i, argc, argv, "--left-index",
                                  args.left_index, parseInt)) return false;
        } else if (a == "--right-index") {
            if (!parseOptionValue(i, argc, argv, "--right-index",
                                  args.right_index, parseInt)) return false;
        } else if (a == "--frames") {
            if (!parseOptionValue(i, argc, argv, "--frames",
                                  args.frames, parseInt)) return false;
        } else if (a == "--timeout-ms") {
            if (!parseOptionValue(i, argc, argv, "--timeout-ms",
                                  args.timeout_ms, parseInt)) return false;
        } else if (a == "--exposure-us") {
            double parsed = 0.0;
            if (!parseOptionValue(i, argc, argv, "--exposure-us",
                                  parsed, parseDouble)) return false;
            args.exposure_us = static_cast<float>(parsed);
        } else if (a == "--gain-db") {
            double parsed = 0.0;
            if (!parseOptionValue(i, argc, argv, "--gain-db",
                                  parsed, parseDouble)) return false;
            args.gain_db = static_cast<float>(parsed);
        } else if (a == "--trigger-chip") {
            if (!requireValue(i, argc, argv, "--trigger-chip", value)) return false;
            args.trigger_chip = value;
        } else if (a == "--trigger-line") {
            if (!parseOptionValue(i, argc, argv, "--trigger-line",
                                  args.trigger_line, parseUnsigned)) return false;
        } else if (a == "--pwm-hz") {
            if (!parseOptionValue(i, argc, argv, "--pwm-hz",
                                  args.pwm_hz, parseDouble)) return false;
        }
        else if (a == "--no-pwm") args.start_pwm = false;
        else if (a == "-h" || a == "--help") {
            printUsage(argv[0]);
            args.help_requested = true;
            return true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            printUsage(argv[0], stderr);
            return false;
        }
    }
    return true;
}

bool validateArgs(const Args& args) {
    if (args.help_requested) return true;
    if (args.frames < 2) {
        std::fprintf(stderr, "--frames must be at least 2, got %d\n", args.frames);
        return false;
    }
    if (args.timeout_ms <= 0) {
        std::fprintf(stderr, "--timeout-ms must be positive, got %d\n",
                     args.timeout_ms);
        return false;
    }
    if (!std::isfinite(args.exposure_us) || args.exposure_us <= 0.0f) {
        std::fprintf(stderr, "--exposure-us must be positive and finite\n");
        return false;
    }
    if (!std::isfinite(args.gain_db) || args.gain_db < 0.0f) {
        std::fprintf(stderr, "--gain-db must be non-negative and finite\n");
        return false;
    }
    if (!std::isfinite(args.pwm_hz) || args.pwm_hz <= 0.0) {
        std::fprintf(stderr, "--pwm-hz must be positive and finite\n");
        return false;
    }
    if (args.start_pwm && args.trigger_chip.empty()) {
        std::fprintf(stderr, "--trigger-chip must not be empty when PWM is enabled\n");
        return false;
    }
    if (args.left_index < 0 || args.right_index < 0) {
        std::fprintf(stderr, "camera indices must be non-negative\n");
        return false;
    }
    if (!args.left_sn.empty() && !args.right_sn.empty() &&
        args.left_sn == args.right_sn) {
        std::fprintf(stderr, "left/right serial numbers must be different\n");
        return false;
    }
    if ((args.left_sn.empty() || args.right_sn.empty()) &&
        args.left_index == args.right_index) {
        std::fprintf(stderr,
                     "left/right camera indices must differ when serial binding is incomplete\n");
        return false;
    }
    return true;
}

}  // namespace hik_metadata_probe
