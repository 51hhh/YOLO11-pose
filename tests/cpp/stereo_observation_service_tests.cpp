#include "volleyball/adapters/fake/in_memory_trial_writer.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace core = volleyball::core;
namespace fake = volleyball::adapters::fake;
namespace ports = volleyball::ports;
namespace services = volleyball::services;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

class TestImage final : public ports::ImageBuffer {
public:
    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::host;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override { return 1440; }
    [[nodiscard]] std::uint32_t height() const noexcept override { return 1080; }
    [[nodiscard]] std::size_t strideBytes() const noexcept override { return 1440; }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return core::PixelFormat::gray8;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override { return 1; }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return 1440U * 1080U;
    }
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

core::Timestamp time(std::int64_t value) {
    return {value, core::ClockDomainId{"gpio-clock"}};
}

core::CameraFrameRecord frame(core::CameraRole role, const char* id,
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

core::Detection2D detection(core::CameraRole role, const char* camera,
                            core::CandidateId id, double x, double y,
                            double score) {
    core::Detection2D output;
    output.frame = {core::CameraId{camera}, role, 7};
    output.candidate_id = id;
    output.center_px = {x, y};
    output.bounding_box = core::BoundingBox2D{x - 20.0, y - 20.0, 40.0, 40.0};
    output.score = score;
    output.source_pipeline = core::PipelineId{"fixture-detector"};
    return output;
}

services::StereoObservationServiceConfig config() {
    services::StereoObservationServiceConfig output;
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

ports::StereoFrame stereoFrame() {
    // 左帧比右帧晚到，但共享同一 GPIO 曝光时刻。
    return {
        {frame(core::CameraRole::left, "left", 1300),
         std::make_shared<TestImage>()},
        {frame(core::CameraRole::right, "right", 1200),
         std::make_shared<TestImage>()},
    };
}

void testServicePreservesAllAttemptsAndPublishesDirectObservation() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
        detection(core::CameraRole::left, "left", 1, 680.0, 600.0, 0.40),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.5, 0.90),
        detection(core::CameraRole::right, "right", 1, 660.0, 300.0, 0.30),
    });
    fake::InMemoryTrialWriter writer;
    CollectingPublisher publisher;
    services::StereoObservationService service(
        config(), left, right, writer, &publisher);

    const auto result = service.process(stereoFrame());
    check(result.ok(), "valid stereo frame must be processed");
    check(result.status == services::ProcessStatus::observation,
          "valid pair must produce an observation");
    check(result.associations.size() == 4,
          "every left/right candidate combination must be retained");
    check(writer.frames().size() == 2, "both camera facts must be written");
    check(writer.detections().size() == 4, "all detector candidates must be written");
    check(writer.associations().size() == 4, "all pair attempts must be written");
    check(writer.observations().size() == 1, "one direct observation must be written");
    check(publisher.observations.size() == 1, "accepted direct observation must publish");
    check(result.observation &&
              std::abs(result.observation->position_m.z - 16.0) < 1.0e-9,
          "rectified triangulation must use z=fx*baseline/disparity");
    check(result.observation && result.observation->covariance_m2.has_value(),
          "triangulation must propagate center uncertainty");
    check(service.finalize().status == ports::WriteStatus::accepted,
          "successful service must finalize the raw writer");
}

void testDetectorFailureAbortsTrial() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    right.setFail(true);
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(config(), left, right, writer);
    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "detector error must be explicit");
    check(!writer.abortReason().empty(), "partial trial must be aborted");
    check(service.finalize().status == ports::WriteStatus::failed,
          "aborted service must not publish a trial");
}

void testArrivalOrderNeverDefinesStereoIdentity() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(config(), left, right, writer);
    auto input = stereoFrame();
    check(input.left.metadata.receive_timestamp->nanoseconds >
              input.right.metadata.receive_timestamp->nanoseconds,
          "fixture must contain reversed USB arrival order");
    check(service.process(input).status == services::ProcessStatus::observation,
          "USB arrival order must not alter watermark-aligned pairing");
}

}  // namespace

int main() {
    testServicePreservesAllAttemptsAndPublishesDirectObservation();
    testDetectorFailureAbortsTrial();
    testArrivalOrderNeverDefinesStereoIdentity();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stereo observation service tests passed\n";
    return EXIT_SUCCESS;
}
