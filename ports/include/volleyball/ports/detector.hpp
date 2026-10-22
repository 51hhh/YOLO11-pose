#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/rectified_frame.hpp"

#include <string>
#include <vector>

namespace volleyball::ports {

struct DetectionBatch {
    volleyball::core::FrameReference frame;
    volleyball::core::PipelineId pipeline_id;
    std::vector<volleyball::core::Detection2D> candidates;
};

struct DetectionResult {
    bool success{false};
    DetectionBatch batch;
    std::string message;
};

class Detector {
public:
    virtual ~Detector() = default;
    [[nodiscard]] virtual DetectionResult detect(
        const RectifiedCameraFrame& frame) = 0;
};

}  // namespace volleyball::ports

