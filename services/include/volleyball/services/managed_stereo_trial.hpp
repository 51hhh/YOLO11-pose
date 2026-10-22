#pragma once

#include "volleyball/ports/camera_source.hpp"
#include "volleyball/ports/trigger_controller.hpp"
#include "volleyball/services/stereo_observation_service.hpp"
#include "volleyball/services/stereo_pipeline_runner.hpp"

namespace volleyball::services {

// 执行一次实机 trial 的完整启停顺序：先启动相机抓取，再启动外部触发；
// 无论成功或失败，退出时都先停止触发，再停止相机。
[[nodiscard]] RunSummary runManagedStereoTrial(
    StereoPipelineRunnerConfig config,
    ports::ManagedCameraSource& source,
    ports::TriggerController& trigger,
    StereoObservationService& service);

}  // namespace volleyball::services
