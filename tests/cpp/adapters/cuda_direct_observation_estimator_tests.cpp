#include "volleyball/adapters/cuda_depth/direct_observation_estimator.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace adapter = volleyball::adapters::cuda_depth;
namespace core = volleyball::core;
namespace ports = volleyball::ports;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

void printFailureDetail(const ports::DirectObservationResult& result) {
    if (result.status != ports::DirectObservationStatus::observation ||
        !result.observation) {
        std::cerr << "  status=" << static_cast<int>(result.status)
                  << " attempted=" << result.attempted_method.value()
                  << " message=" << result.message << '\n';
    }
}

class DeviceGrayImage final : public ports::ImageBuffer {
public:
    DeviceGrayImage(
        std::uint32_t width,
        std::uint32_t height,
        const std::vector<std::uint8_t>& pixels)
        : width_(width), height_(height) {
        if (pixels.size() !=
            static_cast<std::size_t>(width_) * height_) {
            throw std::invalid_argument("invalid test image extent");
        }
        const auto bytes = pixels.size();
        if (cudaMalloc(reinterpret_cast<void**>(&device_), bytes) != cudaSuccess ||
            cudaMemcpy(
                device_, pixels.data(), bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            if (device_ != nullptr) cudaFree(device_);
            throw std::runtime_error("failed to allocate test CUDA image");
        }
    }

    ~DeviceGrayImage() override {
        if (device_ != nullptr) cudaFree(device_);
    }

    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::cuda_device;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override {
        return width_;
    }
    [[nodiscard]] std::uint32_t height() const noexcept override {
        return height_;
    }
    [[nodiscard]] std::size_t strideBytes() const noexcept override {
        return width_;
    }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return core::PixelFormat::gray8;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return reinterpret_cast<std::uintptr_t>(device_);
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return static_cast<std::size_t>(width_) * height_;
    }

private:
    std::uint32_t width_{0};
    std::uint32_t height_{0};
    std::uint8_t* device_{nullptr};
};

struct Fixture {
    static constexpr std::uint32_t width = 160;
    static constexpr std::uint32_t height = 120;
    static constexpr int disparity = 10;

    core::Detection2D left;
    core::Detection2D right;
    core::StereoAssociationAttempt association;

    Fixture() {
        left.frame = {
            core::CameraId{"left"}, core::CameraRole::left, 11};
        left.candidate_id = 4;
        left.center_px = {90.0, 60.0};
        left.bounding_box = {70.0, 40.0, 40.0, 40.0};
        left.score = 0.95;
        left.selected = true;
        left.source_pipeline = core::PipelineId{"test-detector"};
        right.frame = {
            core::CameraId{"right"}, core::CameraRole::right, 12};
        right.candidate_id = 7;
        right.center_px = {80.0, 60.0};
        right.bounding_box = {60.0, 40.0, 40.0, 40.0};
        right.score = 0.94;
        right.selected = true;
        right.source_pipeline = core::PipelineId{"test-detector"};

        association.pair_id = 3;
        association.left_frame_index = 11;
        association.right_frame_index = 12;
        association.left_candidate_id = 4;
        association.right_candidate_id = 7;
        association.left_capture = {
            1000, core::ClockDomainId{"gpio-clock"}};
        association.right_capture = {
            1002, core::ClockDomainId{"gpio-clock"}};
        association.delta_t_lr_ns = 2;
        association.disparity_px = 10.0;
        association.epipolar_error_px = 0.0;
        association.decision = core::AssociationDecision::accepted;
        association.reject_reason = core::AssociationRejectReason::none;
    }
};

adapter::CudaDirectObservationEstimatorConfig config() {
    adapter::CudaDirectObservationEstimatorConfig output;
    output.calibration.left = {800.0, 800.0, 80.0, 60.0};
    output.calibration.right = output.calibration.left;
    output.calibration.baseline_m = 0.1;
    output.calibration.rig_frame = core::CoordinateFrameId{"stereo-rig"};
    output.min_depth_m = 0.2;
    output.max_depth_m = 20.0;
    output.method_order = {
        core::StereoDepthMethodId{"circle_center"},
        core::StereoDepthMethodId{"bbox_center"},
    };
    return output;
}

std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>
shiftedImages(bool textured) {
    std::vector<std::uint8_t> left(Fixture::width * Fixture::height, 90U);
    std::vector<std::uint8_t> right(Fixture::width * Fixture::height, 90U);
    if (!textured) return {left, right};
    for (std::uint32_t y = 0; y < Fixture::height; ++y) {
        for (std::uint32_t x = 0; x < Fixture::width; ++x) {
            left[static_cast<std::size_t>(y) * Fixture::width + x] =
                static_cast<std::uint8_t>(
                    (x * 37U + y * 53U + (x * y) % 97U) % 251U);
        }
    }
    for (std::uint32_t y = 0; y < Fixture::height; ++y) {
        for (std::uint32_t x = 0;
             x + static_cast<std::uint32_t>(Fixture::disparity) <
                 Fixture::width;
             ++x) {
            right[static_cast<std::size_t>(y) * Fixture::width + x] =
                left[static_cast<std::size_t>(y) * Fixture::width + x +
                     static_cast<std::uint32_t>(Fixture::disparity)];
        }
    }
    return {left, right};
}

ports::RectifiedStereoFrame frame(
    const Fixture& fixture,
    ports::ImageBufferPtr left,
    ports::ImageBufferPtr right) {
    ports::RectifiedStereoFrame output;
    output.left.source = fixture.left.frame;
    output.left.geometry_gray = std::move(left);
    output.right.source = fixture.right.frame;
    output.right.geometry_gray = std::move(right);
    return output;
}

void testPublishesConfiguredBallCenterMeasurement() {
    // 纹理输入用于验证候选语义；没有球体恢复的 ROI 不再作为正式球心发布。
    const Fixture fixture;
    auto pixels = shiftedImages(true);
    auto left = std::make_shared<DeviceGrayImage>(
        Fixture::width, Fixture::height, pixels.first);
    auto right = std::make_shared<DeviceGrayImage>(
        Fixture::width, Fixture::height, pixels.second);
    auto stereo_frame = frame(fixture, left, right);
    auto settings = config();
    settings.method_order = {core::StereoDepthMethodId{"bbox_center"}};
    adapter::CudaDirectObservationEstimator estimator(settings);
    const auto result = estimator.estimate({
        stereo_frame,
        fixture.left,
        fixture.right,
        fixture.association,
        19,
    });
    printFailureDetail(result);
    check(result.status == ports::DirectObservationStatus::observation &&
              result.observation.has_value(),
          "textured shifted images must produce a direct observation");
    check(result.observation &&
              result.observation->evidence.method ==
                  core::StereoDepthMethodId{"bbox_center"} &&
              std::abs(
                  result.observation->evidence.disparity_px -
                  Fixture::disparity) < 0.2 &&
              result.observation->evidence.support_points &&
              *result.observation->evidence.support_points >= 1U,
          "configured center evidence must remain identifiable");
    check(result.observation &&
              std::abs(result.observation->measurement.position_m.z - 8.0) <
                  0.2 &&
              result.observation->measurement.capture_reference.nanoseconds ==
                  1001 &&
              result.observation->measurement.observation_id == 19,
          "refined disparity must drive 3D while preserving time and IDs");
}

void testFallsBackToBboxWithoutRenamingIt() {
    const Fixture fixture;
    auto pixels = shiftedImages(false);
    auto left = std::make_shared<DeviceGrayImage>(
        Fixture::width, Fixture::height, pixels.first);
    auto right = std::make_shared<DeviceGrayImage>(
        Fixture::width, Fixture::height, pixels.second);
    auto stereo_frame = frame(fixture, left, right);
    adapter::CudaDirectObservationEstimator estimator(config());
    const auto result = estimator.estimate({
        stereo_frame,
        fixture.left,
        fixture.right,
        fixture.association,
        20,
    });
    printFailureDetail(result);
    check(result.status == ports::DirectObservationStatus::observation &&
              result.observation &&
              result.observation->evidence.method ==
                  core::StereoDepthMethodId{"bbox_center"},
          "texture failure must report the actual bbox fallback method");
}

void testRejectsInvalidConfigurationAndIdentity() {
    auto invalid = config();
    invalid.method_order = {core::StereoDepthMethodId{"roi_multi_point"}};
    check(!adapter::isValid(invalid), "raw surface disparity must not be used as ball center");
    invalid.method_order = {
        core::StereoDepthMethodId{"cuda_template_match"}};
    check(!adapter::isValid(invalid),
          "unimplemented final method must fail static validation");

    const Fixture fixture;
    auto pixels = shiftedImages(true);
    auto left = std::make_shared<DeviceGrayImage>(
        Fixture::width, Fixture::height, pixels.first);
    auto right = std::make_shared<DeviceGrayImage>(
        Fixture::width, Fixture::height, pixels.second);
    auto stereo_frame = frame(fixture, left, right);
    auto wrong_association = fixture.association;
    ++wrong_association.left_candidate_id;
    adapter::CudaDirectObservationEstimator estimator(config());
    const auto result = estimator.estimate({
        stereo_frame,
        fixture.left,
        fixture.right,
        wrong_association,
        21,
    });
    check(result.status == ports::DirectObservationStatus::failed &&
              !result.attempted_method.empty(),
          "identity mismatch must be an explicit integration failure");
}

}  // namespace

int main() {
    testPublishesConfiguredBallCenterMeasurement();
    testFallsBackToBboxWithoutRenamingIt();
    testRejectsInvalidConfigurationAndIdentity();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "CUDA direct observation estimator tests passed\n";
    return EXIT_SUCCESS;
}
