#pragma once

#include "volleyball/ports/camera_source.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace volleyball::services {

enum class SourceQueueOverflowPolicy {
    block,
    drop_oldest,
};

struct RunSummary;

struct StereoPipelineRunnerConfig {
    // 允许连续出现的超时次数；0 表示第一次超时即终止 trial。
    std::size_t max_consecutive_timeouts{0};
    // 由应用层信号处理器设置的只读查询。回调必须无阻塞且不得抛异常。
    std::function<bool()> stop_requested;
    // 实时无录制模式可在停止时正常关闭；采集必须 abort，禁止发布部分 trial。
    bool finalize_on_stop{false};
    // 0 保留同步读取语义；正值启用独立采集线程和有界 FIFO。
    std::size_t source_queue_capacity{0};
    SourceQueueOverflowPolicy source_queue_overflow{
        SourceQueueOverflowPolicy::block};
    // 非零时，每处理相应数量的帧回调一次当前摘要。
    std::size_t progress_interval_frames{0};
    std::function<void(const RunSummary&)> progress_callback;
};

enum class RunStatus {
    completed,
    stopped,
    already_finished,
    source_timeout,
    source_error,
    source_protocol_error,
    processing_error,
    finalize_error,
    camera_start_error,
    trigger_start_error,
    lifecycle_error,
};

struct RunSummary {
    RunStatus status{RunStatus::source_error};
    std::size_t frames_received{0};
    std::size_t observations{0};
    std::size_t no_matches{0};
    std::size_t timeouts{0};
    std::string message;
    // 仅 drop_oldest 队列可以产生该计数；capture/block 必须始终为 0。
    std::size_t frames_dropped{0};
    std::size_t queue_high_watermark{0};

    [[nodiscard]] bool ok() const noexcept {
        return status == RunStatus::completed;
    }
};

// 平台无关的单次 trial 编排器。硬件线程、SDK 生命周期和 GPU 资源仍由
// CameraSource/Detector adapter 所有；本类只保证服务恰好进入 finalize 或 abort。
class StereoPipelineRunner {
public:
    StereoPipelineRunner(
        StereoPipelineRunnerConfig config,
        ports::CameraSource& source,
        StereoObservationService& service);

    [[nodiscard]] RunSummary run();

private:
    StereoPipelineRunnerConfig config_;
    ports::CameraSource& source_;
    StereoObservationService& service_;
    bool finished_{false};
};

}  // namespace volleyball::services
