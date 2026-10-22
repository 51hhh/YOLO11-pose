#pragma once

#include "volleyball/core/observation.hpp"

#include <string>

namespace volleyball::ports {

struct PublishResult {
    bool success{false};
    std::string message;
};

class ObservationPublisher {
public:
    virtual ~ObservationPublisher() = default;

    // 发布给 RDK 的 measurement 仍是直接三角化观测，不得用 TrackEstimate 伪装。
    [[nodiscard]] virtual PublishResult publish(
        const volleyball::core::ValidatedObservation& observation) = 0;
};

}  // namespace volleyball::ports

