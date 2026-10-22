#pragma once

#include "volleyball/adapters/hik_mvs/hikvision_camera.hpp"
#include "volleyball/core/ids.hpp"
#include "volleyball/ports/camera_source.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace volleyball::adapters::hik_mvs {

// GPIO 适配器记录的单次上升沿事实。这里使用回调边界，避免海康适配器
// 反向依赖某一种 GPIO 实现，同时保留完整的双时钟审计字段。
struct TriggerEdgeEvidence {
    std::uint64_t sequence{0};
    std::uint64_t scheduled_monotonic_ns{0};
    std::uint64_t edge_monotonic_ns{0};
    std::uint64_t edge_realtime_ns{0};
    std::uint64_t write_uncertainty_ns{0};
};

using TriggerEvidenceLookup =
    std::function<bool(std::uint64_t, TriggerEdgeEvidence&)>;

struct HikStereoCameraSourceConfig {
    CameraConfig camera;
    core::CameraId left_camera_id;
    core::CameraId right_camera_id;
    core::ClockDomainId left_device_clock;
    core::ClockDomainId right_device_clock;
    core::ClockDomainId capture_realtime_clock;
    core::ClockDomainId host_monotonic_clock;
    unsigned int timeout_ms{1000};
    std::size_t buffer_pool_pairs{3};
    // 空值表示持续运行；采集 trial 应提供有限帧数以产生 end_of_stream。
    std::optional<std::uint64_t> frame_limit;
};

// 把海康双目 SDK 帧适配为平台无关 CameraSource。相机必须先 start()，
// 调用方随后启动 GPIO，再把本对象交给 StereoPipelineRunner。
class HikStereoCameraSource final : public ports::CameraSource {
public:
    HikStereoCameraSource(
        HikStereoCameraSourceConfig config,
        TriggerEvidenceLookup trigger_lookup);
    ~HikStereoCameraSource() override;

    HikStereoCameraSource(const HikStereoCameraSource&) = delete;
    HikStereoCameraSource& operator=(const HikStereoCameraSource&) = delete;

    [[nodiscard]] bool start(std::string& error);
    void stop() noexcept;
    [[nodiscard]] bool isStarted() const noexcept;
    [[nodiscard]] ports::SourceResult next() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volleyball::adapters::hik_mvs
