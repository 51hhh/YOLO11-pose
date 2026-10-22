#pragma once

#include <string>

namespace volleyball::ports {

// 相机外部触发的最小生命周期。应用层只依赖该契约，不接触 GPIO SDK。
class TriggerController {
public:
    virtual ~TriggerController() = default;
    [[nodiscard]] virtual bool start(std::string& error) = 0;
    virtual void stop() noexcept = 0;
    [[nodiscard]] virtual bool isRunning() const noexcept = 0;
};

}  // namespace volleyball::ports
