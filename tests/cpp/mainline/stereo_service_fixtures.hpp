#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/detector.hpp"
#include "volleyball/ports/image_buffer.hpp"
#include "volleyball/ports/observation_publisher.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include "stereo_domain_fixtures.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace volleyball::test_support {

class TestImage final : public ports::ImageBuffer {
public:
    explicit TestImage(
        std::uint32_t width = 1440,
        std::uint32_t height = 1080,
        core::PixelFormat format = core::PixelFormat::gray8,
        std::size_t stride = 0,
        std::size_t size = 0)
        : width_(width),
          height_(height),
          format_(format),
          stride_(stride == 0 ? width * channels(format) : stride),
          size_(size == 0 ? stride_ * height : size) {}

    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::host;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override {
        return width_;
    }
    [[nodiscard]] std::uint32_t height() const noexcept override {
        return height_;
    }
    [[nodiscard]] std::size_t strideBytes() const noexcept override {
        return stride_;
    }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return format_;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return 1;
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return size_;
    }

private:
    static std::size_t channels(core::PixelFormat format) noexcept {
        return format == core::PixelFormat::bgr8 ||
                       format == core::PixelFormat::rgb8
            ? 3U
            : 1U;
    }

    std::uint32_t width_;
    std::uint32_t height_;
    core::PixelFormat format_;
    std::size_t stride_;
    std::size_t size_;
};

class StaticDetector final : public ports::Detector {
public:
    explicit StaticDetector(std::vector<core::Detection2D> candidates)
        : candidates_(std::move(candidates)) {}

    [[nodiscard]] ports::DetectionResult detect(
        const ports::CapturedFrame& frame) override {
        if (fail_) {
            return {false, {}, "injected detector failure"};
        }
        ports::DetectionBatch batch;
        batch.frame = {
            frame.metadata.camera_id,
            frame.metadata.role,
            frame.metadata.frame_index,
        };
        batch.pipeline_id = core::PipelineId{"fixture-detector"};
        batch.candidates = candidates_;
        return {true, std::move(batch), {}};
    }

    void setFail(bool value) noexcept { fail_ = value; }

private:
    std::vector<core::Detection2D> candidates_;
    bool fail_{false};
};

class CollectingPublisher final : public ports::ObservationPublisher {
public:
    [[nodiscard]] ports::PublishResult publish(
        const core::ValidatedObservation& observation) override {
        observations.push_back(observation);
        return {true, {}};
    }

    std::vector<core::ValidatedObservation> observations;
};

inline core::CameraFrameRecord frame(
    core::CameraRole role,
    const char* id,
    std::int64_t receive_ns) {
    core::CameraFrameRecord output;
    output.camera_id = core::CameraId{id};
    output.role = role;
    output.frame_index = 7;
    output.hardware_timestamp = {
        900,
        core::ClockDomainId{std::string{id} + "-device-clock"},
    };
    output.exposure_start = time(1000);
    output.capture_timestamp = time(1000);
    output.receive_timestamp = time(receive_ns);
    output.inference_done_timestamp = time(1400);
    output.roi = {0, 0, 1440, 1080};
    output.pixel_format = core::PixelFormat::gray8;
    output.acquisition.receive_monotonic_timestamp = {
        receive_ns - 700,
        core::ClockDomainId{"nx-monotonic"},
    };
    return output;
}

inline services::StereoObservationServiceConfig serviceConfig() {
    services::StereoObservationServiceConfig output;
    output.association = associationConfig();
    output.calibration = calibration();
    output.triangulation = triangulationConfig();
    output.gate.min_depth_m = 0.8;
    output.gate.max_depth_m = 20.0;
    output.gate.max_reprojection_error_px = 2.0;
    return output;
}

inline ports::StereoFrame stereoFrame() {
    // 左帧比右帧晚到，但共享同一 GPIO 曝光时刻。
    return {
        {frame(core::CameraRole::left, "left", 1300),
         std::make_shared<TestImage>()},
        {frame(core::CameraRole::right, "right", 1200),
         std::make_shared<TestImage>()},
    };
}

}  // namespace volleyball::test_support
