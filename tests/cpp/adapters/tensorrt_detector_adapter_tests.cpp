#include "tensorrt_runtime_fake.hpp"

#include "volleyball/adapters/tensorrt/detector_adapter.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace trt = volleyball::adapters::tensorrt;
namespace core = volleyball::core;
namespace ports = volleyball::ports;
namespace runtime_fake = volleyball::tests::tensorrt_runtime_fake;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class TestImage final : public ports::ImageBuffer {
public:
    TestImage(
        ports::MemoryKind memory_kind = ports::MemoryKind::cuda_device,
        core::PixelFormat pixel_format = core::PixelFormat::bgr8)
        : memory_kind_(memory_kind), pixel_format_(pixel_format) {}

    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return memory_kind_;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override { return 100; }
    [[nodiscard]] std::uint32_t height() const noexcept override { return 80; }
    [[nodiscard]] std::size_t strideBytes() const noexcept override { return 320; }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return pixel_format_;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return 0x1234;
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return 320 * 80;
    }

private:
    ports::MemoryKind memory_kind_;
    core::PixelFormat pixel_format_;
};

ports::RectifiedCameraFrame frame(
    ports::MemoryKind memory_kind = ports::MemoryKind::cuda_device,
    core::PixelFormat pixel_format = core::PixelFormat::bgr8) {
    auto image = std::make_shared<TestImage>(memory_kind, pixel_format);
    return {
        {core::CameraId{"left"}, core::CameraRole::left, 9},
        image,
        image,
    };
}

trt::TensorRtDetectorConfig config() {
    return {
        "models/test.engine",
        core::PipelineId{"volleyball-detector"},
        640,
        0.25F,
        0.4F,
        10,
        0,
        false,
        0,
    };
}

void testInitializationAndZeroDetection() {
    runtime_fake::reset();
    trt::TensorRtDetector detector(config());
    std::string error;
    check(detector.initialize(error) && detector.isInitialized(),
          "valid detector configuration should initialize");
    check(!detector.initialize(error),
          "one detector instance must not initialize twice");

    const auto result = detector.detect(frame());
    check(result.success && result.batch.candidates.empty(),
          "zero detections must remain a successful mainline batch");
    check(runtime_fake::detection_calls == 1 &&
              runtime_fake::last_pointer == 0x1234 &&
              runtime_fake::last_pitch == 320 &&
              runtime_fake::last_width == 100 &&
              runtime_fake::last_height == 80,
          "adapter should forward CUDA pointer and exact image geometry");
}

void testRuntimeFailureAndCandidateMapping() {
    runtime_fake::reset();
    trt::TensorRtDetector detector(config());
    std::string error;
    check(detector.initialize(error), "detector should initialize for mapping test");

    runtime_fake::detection_result = {
        true,
        {{20.0F, 30.0F, 10.0F, 12.0F, 0.9F, 0}},
        {},
    };
    auto result = detector.detect(frame());
    check(result.success && result.batch.candidates.size() == 1 &&
              result.batch.candidates.front().frame.frame_index == 9,
          "adapter should map runtime detections onto the source frame");

    runtime_fake::detection_result = {false, {}, "injected runtime failure"};
    result = detector.detect(frame());
    check(!result.success && result.message == "injected runtime failure",
          "runtime failures must not be converted into zero detections");
}

void testRejectsWrongMemoryAndEngineShape() {
    runtime_fake::reset();
    trt::TensorRtDetector detector(config());
    std::string error;
    check(detector.initialize(error), "detector should initialize for input test");

    const auto host_result = detector.detect(frame(ports::MemoryKind::host));
    check(!host_result.success && runtime_fake::detection_calls == 0,
          "host images must be rejected before calling TensorRT");

    runtime_fake::reset();
    runtime_fake::input_size = 320;
    trt::TensorRtDetector wrong_shape(config());
    check(!wrong_shape.initialize(error) && !wrong_shape.isInitialized(),
          "engine input size must match the frozen detector profile");
}

}  // namespace

int main() {
    testInitializationAndZeroDetection();
    testRuntimeFailureAndCandidateMapping();
    testRejectsWrongMemoryAndEngineShape();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "TensorRT detector adapter tests passed\n";
    return EXIT_SUCCESS;
}
