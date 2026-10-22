#pragma once

#include "runtime_config.hpp"

#include "volleyball/services/stereo_pipeline_runner.hpp"

#include <functional>

namespace volleyball::apps::nx_mainline {

// 组装并运行一次完整的 NX 主线生命周期。硬件、GPU、录制与 ROS2
// 对象都局限在应用层，服务层只看到端口。
[[nodiscard]] services::RunSummary runMainlineApplication(
    const MainlineRuntimeConfig& config,
    std::function<bool()> stop_requested);

}  // namespace volleyball::apps::nx_mainline
