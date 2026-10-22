#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/direct_observation_estimator.hpp"

#include <string>

namespace volleyball::ports {

// 对外发布所需的完整直接观测证据。这里只组合已经验证过的原始事实，
// 不携带 TrackEstimate 或 LandingPrediction 等派生状态。
struct ObservationPublication {
    volleyball::core::ValidatedObservation observation;
    volleyball::core::CameraFrameRecord left_frame;
    volleyball::core::CameraFrameRecord right_frame;
    volleyball::core::Detection2D left_detection;
    volleyball::core::Detection2D right_detection;
    volleyball::core::StereoAssociationAttempt association;
    DirectObservationEvidence estimation;
};

struct PublishResult {
    bool success{false};
    std::string message;
};

class ObservationPublisher {
public:
    virtual ~ObservationPublisher() = default;

    // 发布给 RDK 的 measurement 仍是直接三角化观测，不得用 TrackEstimate 伪装。
    [[nodiscard]] virtual PublishResult publish(
        const ObservationPublication& publication) = 0;
};

}  // namespace volleyball::ports

