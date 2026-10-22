#pragma once

#include "volleyball/ports/captured_frame.hpp"

#include <optional>
#include <string>

namespace volleyball::ports {

enum class SourceStatus { frame, end_of_stream, timeout, error };

struct SourceResult {
    SourceStatus status{SourceStatus::error};
    std::optional<StereoFrame> stereo_frame;
    std::string message;
};

class CameraSource {
public:
    virtual ~CameraSource() = default;
    [[nodiscard]] virtual SourceResult next() = 0;
};

// 实机来源由应用层显式管理生命周期。start 成功后才允许启动外部触发，
// stop 必须幂等且不抛异常，以便所有失败路径都能安全排空硬件。
class ManagedCameraSource : public CameraSource {
public:
    ~ManagedCameraSource() override = default;
    [[nodiscard]] virtual bool start(std::string& error) = 0;
    virtual void stop() noexcept = 0;
    [[nodiscard]] virtual bool isStarted() const noexcept = 0;
};

}  // namespace volleyball::ports

