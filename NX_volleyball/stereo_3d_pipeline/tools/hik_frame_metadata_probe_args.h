#pragma once

#include <string>

namespace hik_metadata_probe {

struct Args {
    std::string left_sn;
    std::string right_sn;
    int left_index = 0;
    int right_index = 1;
    int frames = 300;
    int timeout_ms = 1000;
    float exposure_us = 9867.0f;
    float gain_db = 11.99f;
    bool start_pwm = true;
    std::string trigger_chip = "gpiochip2";
    unsigned int trigger_line = 7;
    double pwm_hz = 100.0;
};

bool parseArgs(int argc, char** argv, Args& args);

}  // namespace hik_metadata_probe
