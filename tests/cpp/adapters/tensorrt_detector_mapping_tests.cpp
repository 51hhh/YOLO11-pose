#include "volleyball/adapters/tensorrt/detector_mapping.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace trt = volleyball::adapters::tensorrt;
namespace core = volleyball::core;
namespace ports = volleyball::ports;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool near(double actual, double expected, double tolerance = 1.0e-6) {
    return std::fabs(actual - expected) <= tolerance;
}

class TestImage final : public ports::ImageBuffer {
public:
    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::cuda_device;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override { return 100; }
    [[nodiscard]] std::uint32_t height() const noexcept override { return 80; }
    [[nodiscard]] std::size_t strideBytes() const noexcept override { return 300; }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return core::PixelFormat::bgr8;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return 1;
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return 24000;
    }
};

ports::RectifiedCameraFrame frame() {
    return {
        {core::CameraId{"left"}, core::CameraRole::left, 7},
        std::make_shared<TestImage>(),
        std::make_shared<TestImage>(),
    };
}

trt::DetectorMappingConfig config() {
    return {core::PipelineId{"volleyball-detector"}, 2, 0};
}

void testMapsStableConfidenceOrderAndCap() {
    const std::vector<trt::Detection> detections = {
        {20.0F, 30.0F, 8.0F, 10.0F, 0.7F, 0},
        {40.0F, 50.0F, 12.0F, 14.0F, 0.9F, 0},
        {60.0F, 20.0F, 6.0F, 6.0F, 0.8F, 0},
    };
    const auto result = trt::mapDetections(frame(), detections, config());
    check(result.success && result.batch.candidates.size() == 2,
          "mapping should cap candidates after stable confidence ordering");
    check(result.batch.frame.frame_index == 7 &&
              result.batch.pipeline_id == core::PipelineId{"volleyball-detector"},
          "mapping should preserve frame and pipeline identity");
    if (result.batch.candidates.size() == 2) {
        const auto& first = result.batch.candidates[0];
        const auto& second = result.batch.candidates[1];
        check(first.candidate_id == 0 && second.candidate_id == 1 &&
                  first.score && second.score &&
                  near(*first.score, 0.9) && near(*second.score, 0.8),
              "mapped candidate IDs should follow deterministic confidence order");
        check(first.bounding_box && first.bounding_box->x == 34.0 &&
                  first.bounding_box->y == 43.0 &&
                  first.bounding_box->width == 12.0 &&
                  first.bounding_box->height == 14.0,
              "mapping should convert center-size boxes to top-left boxes");
    }
}

void testZeroDetectionsIsSuccessful() {
    const auto result = trt::mapDetections(frame(), {}, config());
    check(result.success && result.batch.candidates.empty(),
          "a valid frame with no detections must remain a successful batch");
}

void testRejectsUnrepresentableOrInvalidOutput() {
    auto unexpected_class = trt::mapDetections(
        frame(), {{20.0F, 20.0F, 5.0F, 5.0F, 0.8F, 1}}, config());
    check(!unexpected_class.success && unexpected_class.batch.candidates.empty(),
          "unexpected classes must not be silently discarded");

    auto invalid = trt::mapDetections(
        frame(),
        {{std::numeric_limits<float>::quiet_NaN(),
          20.0F, 5.0F, 5.0F, 0.8F, 0}},
        config());
    check(!invalid.success && invalid.batch.candidates.empty(),
          "non-finite detector output must fail the batch");

    auto outside = trt::mapDetections(
        frame(), {{100.0F, 20.0F, 5.0F, 5.0F, 0.8F, 0}}, config());
    check(!outside.success && outside.batch.candidates.empty(),
          "out-of-image centers must fail before entering the service");
}

}  // namespace

int main() {
    testMapsStableConfidenceOrderAndCap();
    testZeroDetectionsIsSuccessful();
    testRejectsUnrepresentableOrInvalidOutput();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "TensorRT detector mapping tests passed\n";
    return EXIT_SUCCESS;
}
