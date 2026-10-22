#include "pipeline/fusion_timing.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

}  // namespace

int main() {
    constexpr uint64_t frame = 11111111ULL;  // approximately 90 Hz
    const double default_dt = 1.0 / 90.0;

    const double first = stereo3d::chooseFusionDtSeconds(
        1000000000ULL, 0, 0.0, default_dt);
    check(std::abs(first - default_dt) < 1e-9,
          "first frame must use the configured default interval");

    const double next = stereo3d::chooseFusionDtSeconds(
        1000000000ULL + frame, 1000000000ULL, 0.050, default_dt);
    check(std::abs(next - static_cast<double>(frame) * 1e-9) < 1e-12,
          "capture timestamps must take precedence over wall time");

    // A predict-only frame advances the stored capture timestamp. The next
    // measurement therefore advances by one frame, not by the full gap again.
    const uint64_t predict_stamp = 1000000000ULL + frame;
    const uint64_t measurement_stamp = predict_stamp + frame;
    const double after_predict = stereo3d::chooseFusionDtSeconds(
        measurement_stamp, predict_stamp, 0.022, default_dt);
    check(std::abs(after_predict - static_cast<double>(frame) * 1e-9) < 1e-12,
          "predict-only frames must advance the capture-time baseline");

    // If an entire frame is intentionally dropped without prediction, the
    // capture delta correctly spans both frame periods.
    const double skipped = stereo3d::chooseFusionDtSeconds(
        1000000000ULL + 2 * frame, 1000000000ULL, 0.005, default_dt);
    check(std::abs(skipped - static_cast<double>(2 * frame) * 1e-9) < 1e-12,
          "an omitted frame must retain the full capture-time gap");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "fusion timing tests passed\n";
    return EXIT_SUCCESS;
}
