#include "hik_frame_metadata_probe_args.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace hik_metadata_probe {
namespace {

void usage(const char* argv0) {
    std::printf(
        "Usage: %s [--left-sn SN] [--right-sn SN] [--left-index N] [--right-index N]\n"
        "          [--frames N] [--timeout-ms N] [--exposure-us US] [--gain-db DB]\n"
        "          [--no-pwm] [--trigger-chip gpiochip2] [--trigger-line 7] [--pwm-hz 100]\n",
        argv0);
}

}  // namespace

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--left-sn") args.left_sn = need("--left-sn");
        else if (a == "--right-sn") args.right_sn = need("--right-sn");
        else if (a == "--left-index") args.left_index = std::atoi(need("--left-index"));
        else if (a == "--right-index") args.right_index = std::atoi(need("--right-index"));
        else if (a == "--frames") args.frames = std::atoi(need("--frames"));
        else if (a == "--timeout-ms") args.timeout_ms = std::atoi(need("--timeout-ms"));
        else if (a == "--exposure-us") args.exposure_us = std::atof(need("--exposure-us"));
        else if (a == "--gain-db") args.gain_db = std::atof(need("--gain-db"));
        else if (a == "--trigger-chip") args.trigger_chip = need("--trigger-chip");
        else if (a == "--trigger-line") args.trigger_line = static_cast<unsigned int>(std::atoi(need("--trigger-line")));
        else if (a == "--pwm-hz") args.pwm_hz = std::atof(need("--pwm-hz"));
        else if (a == "--no-pwm") args.start_pwm = false;
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

}  // namespace hik_metadata_probe
