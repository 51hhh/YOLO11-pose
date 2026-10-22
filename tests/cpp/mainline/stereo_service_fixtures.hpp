#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/detector.hpp"
#include "volleyball/ports/image_buffer.hpp"
#include "volleyball/ports/observation_publisher.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
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

inline core::Timestamp time(std::int64_t value) {
    return {value, core::ClockDomainId{"gpio-clock"}};
}

inline core::CameraFrameRecord frame(
    core::CameraRole role,
    const char* id,
    std::int64_t receive_ns) {
    core::CameraFrameRecord output;
    output.camera_id = core::CameraId{id};
    output.role = role;
    output.frame_index = 7;
    output.hardware_timestamp = time(900);
    output.exposure_start = time(1000);
    output.capture_timestamp = time(1000);
    output.receive_timestamp = time(receive_ns);
    output.inference_done_timestamp = time(1400);
    output.roi = {0, 0, 1440, 1080};
    output.pixel_format = core::PixelFormat::gray8;
    return output;
}

inline core::Detection2D detection(
    core::CameraRole role,
    const char* camera,
    core::CandidateId id,
    double x,
    double y,
    double score) {
    core::Detection2D output;
    output.frame = {core::CameraId{camera}, role, 7};
    output.candidate_id = id;
    output.center_px = {x, y};
    output.bounding_box =
        core::BoundingBox2D{x - 20.0, y - 20.0, 40.0, 40.0};
    output.score = score;
    output.source_pipeline = core::PipelineId{"fixture-detector"};
    return output;
}

inline services::StereoObservationServiceConfig serviceConfig() {
    services::StereoObservationServiceConfig output;
    output.association.max_capture_delta_ns = 100;
    output.association.epipolar_tolerance_px = 2.0;
    output.association.max_size_ratio = 2.0;
    output.association.min_shifted_iou = 0.5;
    output.calibration.left = {1000.0, 1000.0, 720.0, 540.0};
    output.calibration.right = output.calibration.left;
    output.calibration.baseline_m = 0.8;
    output.calibration.rig_frame = core::CoordinateFrameId{"stereo-rig"};
    output.triangulation.center_sigma_px = 0.5;
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
