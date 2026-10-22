#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/captured_frame.hpp"

#include <string>

namespace volleyball::ports {

// 采集模式的队列背压必须由写入器内部阻塞吸收；返回层只表达本条记录
// 已经被接管，或 trial 必须失败，避免调用方误以为存在未实现的重试协议。
enum class WriteStatus { accepted, failed };

struct WriteResult {
    WriteStatus status{WriteStatus::failed};
    std::string message;
};

// 写入器只接收原始契约对象。滤波与预测结果必须使用独立派生写入器。
class TrialWriter {
public:
    virtual ~TrialWriter() = default;
    // 左右帧必须作为一个不可分割的采集事实交给写入器。这样生产实现既能
    // 保存图像所有权，也不会在第二侧失败时留下半个双目帧。
    [[nodiscard]] virtual WriteResult append(const StereoFrame& frame) = 0;
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

