#include "pipeline/exposure_timestamp.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

stereo3d::ExposureTimestampConfig measuredConfig() {
    stereo3d::ExposureTimestampConfig config;
    config.contract_valid = true;
    config.delay_measured = true;
    config.delay_ns = 200;
    config.delay_uncertainty_ns = 100;
    config.midpoint_offset_ns = 4'933'500;
    return config;
}

}  // namespace

int main() {
    {
        const auto estimate = stereo3d::estimateExposureMidpointTimestamp(
            1'000'000'000, 50, measuredConfig());
        check(estimate.timestamp_ns == 1'004'933'700,
              "timestamp must equal GPIO edge plus delay plus half exposure");
        check(estimate.uncertainty_ns == 150,
              "measured delay and GPIO-write uncertainty must be combined");
        check(estimate.mapping_valid,
              "measured exposure delay must produce a calibrated mapping");
    }

    {
        auto config = measuredConfig();
        config.delay_measured = false;
        const auto estimate = stereo3d::estimateExposureMidpointTimestamp(
            1'000'000'000, 50, config);
        check(estimate.timestamp_ns == 1'004'933'700,
              "zero-delay assumption must still produce an explicit estimate");
        check(estimate.uncertainty_ns ==
                  std::numeric_limits<std::uint64_t>::max(),
              "unmeasured delay must never claim finite uncertainty");
        check(!estimate.mapping_valid,
              "unmeasured delay must remain visibly uncalibrated");
    }

    {
        auto config = measuredConfig();
        config.contract_valid = false;
        const auto unsupported =
            stereo3d::estimateExposureMidpointTimestamp(1, 0, config);
        check(unsupported.timestamp_ns == 0 && !unsupported.mapping_valid,
              "unsupported shutter/source semantics must not fabricate time");

        config.contract_valid = true;
        const auto missing_edge =
            stereo3d::estimateExposureMidpointTimestamp(0, 0, config);
        check(missing_edge.timestamp_ns == 0,
              "missing GPIO edge must not fall back to arrival time");
    }

    {
        auto config = measuredConfig();
        const auto overflow = stereo3d::estimateExposureMidpointTimestamp(
            std::numeric_limits<std::uint64_t>::max() - 100, 0, config);
        check(overflow.timestamp_ns == 0 && !overflow.mapping_valid,
              "timestamp overflow must be rejected");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "exposure timestamp tests passed\n";
    return EXIT_SUCCESS;
}
