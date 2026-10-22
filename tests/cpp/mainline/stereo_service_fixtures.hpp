#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/detector.hpp"
#include "volleyball/ports/image_buffer.hpp"
#include "volleyball/ports/observation_publisher.hpp"
#include "volleyball/ports/stereo_preprocessor.hpp"
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
        const ports::RectifiedCameraFrame& frame) override {
        ++calls_;
        last_detector_format_ = frame.detector_input
            ? std::optional<core::PixelFormat>{
                  frame.detector_input->pixelFormat()}
            : std::nullopt;
        if (fail_) {
            return {false, {}, "injected detector failure"};
        }
        ports::DetectionBatch batch;
        batch.frame = frame.source;
        batch.pipeline_id = core::PipelineId{"fixture-detector"};
        batch.candidates = candidates_;
        return {true, std::move(batch), {}};
    }

    void setFail(bool value) noexcept { fail_ = value; }
    [[nodiscard]] std::size_t calls() const noexcept { return calls_; }
    [[nodiscard]] std::optional<core::PixelFormat> lastDetectorFormat()
        const noexcept {
        return last_detector_format_;
    }

private:
    std::vector<core::Detection2D> candidates_;
    bool fail_{false};
    std::size_t calls_{0};
    std::optional<core::PixelFormat> last_detector_format_;
};

class StaticStereoPreprocessor final : public ports::StereoPreprocessor {
public:
    [[nodiscard]] ports::PreprocessResult prepare(
        const ports::StereoFrame& frame) override {
        if (fail_) {
            return {false, std::nullopt, "injected preprocess failure"};
        }
        auto prepared = ports::RectifiedStereoFrame{
            prepareCamera(frame.left),
            prepareCamera(frame.right),
        };
        if (swap_sources_) {
            std::swap(prepared.left.source, prepared.right.source);
        }
        if (invalid_left_detector_format_) {
            prepared.left.detector_input = std::make_shared<TestImage>(
                frame.left.metadata.roi.width,
                frame.left.metadata.roi.height,
                core::PixelFormat::gray8);
        }
        return {true, std::move(prepared), {}};
    }

    void setFail(bool value) noexcept { fail_ = value; }
    void setSwapSources(bool value) noexcept { swap_sources_ = value; }
    void setInvalidLeftDetectorFormat(bool value) noexcept {
        invalid_left_detector_format_ = value;
    }

private:
    static ports::RectifiedCameraFrame prepareCamera(
        const ports::CapturedFrame& frame) {
        const auto width = frame.metadata.roi.width;
        const auto height = frame.metadata.roi.height;
        return {
            {
                frame.metadata.camera_id,
                frame.metadata.role,
                frame.metadata.frame_index,
            },
            std::make_shared<TestImage>(
                width, height, core::PixelFormat::bgr8),
            std::make_shared<TestImage>(
                width, height, core::PixelFormat::gray8),
        };
    }

    bool fail_{false};
    bool swap_sources_{false};
    bool invalid_left_detector_format_{false};
};

inline StaticStereoPreprocessor& passingPreprocessor() {
    static StaticStereoPreprocessor preprocessor;
    return preprocessor;
}

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
    output.pixel_format = core::PixelFormat::bayer8;
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
         std::make_shared<TestImage>(
             1440, 1080, core::PixelFormat::bayer8)},
        {frame(core::CameraRole::right, "right", 1200),
         std::make_shared<TestImage>(
             1440, 1080, core::PixelFormat::bayer8)},
    };
}

}  // namespace volleyball::test_support
