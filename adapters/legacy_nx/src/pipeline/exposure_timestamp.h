#ifndef STEREO_3D_PIPELINE_EXPOSURE_TIMESTAMP_H_
#define STEREO_3D_PIPELINE_EXPOSURE_TIMESTAMP_H_

#include <cstdint>
#include <limits>

namespace stereo3d {

struct ExposureTimestampConfig {
    bool contract_valid = false;
    bool delay_measured = false;
    std::uint64_t delay_ns = 0;
    std::uint64_t delay_uncertainty_ns = 0;
    std::uint64_t midpoint_offset_ns = 0;
};

struct ExposureTimestampEstimate {
    std::uint64_t timestamp_ns = 0;
    std::uint64_t uncertainty_ns =
        std::numeric_limits<std::uint64_t>::max();
    bool mapping_valid = false;
};

inline bool checkedTimestampAdd(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t* output) {
    if (output == nullptr ||
        rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return false;
    }
    *output = lhs + rhs;
    return true;
}

inline ExposureTimestampEstimate estimateExposureMidpointTimestamp(
    std::uint64_t trigger_edge_realtime_ns,
    std::uint64_t trigger_write_uncertainty_ns,
    const ExposureTimestampConfig& config) {
    ExposureTimestampEstimate estimate;
    if (!config.contract_valid || trigger_edge_realtime_ns == 0 ||
        config.midpoint_offset_ns == 0) {
        return estimate;
    }

    std::uint64_t exposure_start_ns = 0;
    if (!checkedTimestampAdd(trigger_edge_realtime_ns, config.delay_ns,
                             &exposure_start_ns) ||
        !checkedTimestampAdd(exposure_start_ns, config.midpoint_offset_ns,
                             &estimate.timestamp_ns)) {
        estimate.timestamp_ns = 0;
        return estimate;
    }

    if (!config.delay_measured) {
        return estimate;
    }
    if (!checkedTimestampAdd(trigger_write_uncertainty_ns,
                             config.delay_uncertainty_ns,
                             &estimate.uncertainty_ns)) {
        estimate.uncertainty_ns =
            std::numeric_limits<std::uint64_t>::max();
        return estimate;
    }
    estimate.mapping_valid = true;
    return estimate;
}

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_EXPOSURE_TIMESTAMP_H_
