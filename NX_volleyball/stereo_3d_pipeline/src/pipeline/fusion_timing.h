#ifndef STEREO_3D_PIPELINE_FUSION_TIMING_H_
#define STEREO_3D_PIPELINE_FUSION_TIMING_H_

#include <algorithm>
#include <cstdint>

namespace stereo3d {

inline double chooseFusionDtSeconds(uint64_t capture_timestamp_ns,
                                    uint64_t last_capture_timestamp_ns,
                                    double wall_dt_seconds,
                                    double default_dt_seconds) {
    double dt = default_dt_seconds;
    if (capture_timestamp_ns > 0 && last_capture_timestamp_ns > 0 &&
        capture_timestamp_ns > last_capture_timestamp_ns) {
        dt = static_cast<double>(capture_timestamp_ns -
                                 last_capture_timestamp_ns) * 1e-9;
    } else if (wall_dt_seconds > 0.0) {
        dt = wall_dt_seconds;
    }
    return std::clamp(dt, 0.002, 0.1);
}

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_FUSION_TIMING_H_
