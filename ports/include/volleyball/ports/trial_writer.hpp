#pragma once

#include "volleyball/core/observation.hpp"

#include <string>

namespace volleyball::ports {

enum class WriteStatus { accepted, backpressure, failed };

struct WriteResult {
    WriteStatus status{WriteStatus::failed};
    std::string message;
};

// 写入器只接收原始契约对象。滤波与预测结果必须使用独立派生写入器。
class TrialWriter {
public:
    virtual ~TrialWriter() = default;
    [[nodiscard]] virtual WriteResult append(
        const volleyball::core::CameraFrameRecord& record) = 0;
    [[nodiscard]] virtual WriteResult append(
        const volleyball::core::Detection2D& record) = 0;
    [[nodiscard]] virtual WriteResult append(
        const volleyball::core::StereoAssociationAttempt& record) = 0;
    [[nodiscard]] virtual WriteResult append(
        const volleyball::core::TriangulatedObservation& record) = 0;
    [[nodiscard]] virtual WriteResult finalize() = 0;
    virtual void abort(std::string reason) noexcept = 0;
};

}  // namespace volleyball::ports

