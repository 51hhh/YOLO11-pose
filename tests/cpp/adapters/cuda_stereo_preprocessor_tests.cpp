#include "volleyball/adapters/cuda_preprocess/stereo_preprocessor.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace adapter = volleyball::adapters::cuda_preprocess;
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

class HostImage final : public ports::ImageBuffer {
public:
    HostImage(
        std::uint32_t width,
        std::uint32_t height,
        std::uint8_t value,
        core::PixelFormat format = core::PixelFormat::bayer8)
        : width_(width),
          height_(height),
          format_(format),
          bytes_(static_cast<std::size_t>(width) * height, value) {}

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
        return width_;
    }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return format_;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return reinterpret_cast<std::uintptr_t>(bytes_.data());
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return bytes_.size();
    }

private:
    std::uint32_t width_;
    std::uint32_t height_;
    core::PixelFormat format_;
    std::vector<std::uint8_t> bytes_;
};

adapter::StereoRemapMaps identityMaps(
    std::uint32_t width,
    std::uint32_t height) {
    adapter::StereoRemapMaps maps;
    maps.width = width;
    maps.height = height;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    maps.left_x.resize(count);
    maps.left_y.resize(count);
    maps.right_x.resize(count);
    maps.right_y.resize(count);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * width + x;
            maps.left_x[index] = static_cast<float>(x);
            maps.left_y[index] = static_cast<float>(y);
            maps.right_x[index] = static_cast<float>(x);
            maps.right_y[index] = static_cast<float>(y);
        }
    }
    return maps;
}

core::CameraFrameRecord metadata(
    core::CameraRole role,
    const char* camera_id,
    std::uint32_t width,
    std::uint32_t height) {
    core::CameraFrameRecord value;
    value.camera_id = core::CameraId{camera_id};
    value.role = role;
    value.frame_index = 42;
    value.roi = {0, 0, width, height};
    value.pixel_format = core::PixelFormat::bayer8;
    return value;
}

ports::StereoFrame frame(
    std::uint32_t width,
    std::uint32_t height,
    core::PixelFormat left_format = core::PixelFormat::bayer8) {
    ports::StereoFrame value;
    value.left.metadata = metadata(
        core::CameraRole::left, "left-camera", width, height);
    value.right.metadata = metadata(
        core::CameraRole::right, "right-camera", width, height);
    value.left.image = std::make_shared<HostImage>(
        width, height, 77U, left_format);
    value.right.image = std::make_shared<HostImage>(
        width, height, 77U);
    return value;
}

std::vector<std::uint8_t> download(
    const ports::ImageBufferPtr& image,
    std::size_t row_bytes) {
    std::vector<std::uint8_t> output(row_bytes * image->height());
    const cudaError_t status = cudaMemcpy2D(
        output.data(), row_bytes,
        reinterpret_cast<const void*>(image->nativeHandle()),
        image->strideBytes(), row_bytes, image->height(),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return output;
}

void checkCameraOutput(
    const ports::RectifiedCameraFrame& output,
    core::CameraRole role,
    const char* camera_id) {
    check(output.source.camera_id == core::CameraId{camera_id},
          "preprocessor must preserve camera ID");
    check(output.source.role == role && output.source.frame_index == 42,
          "preprocessor must preserve role and frame index");
    check(output.detector_input && output.geometry_gray,
          "preprocessor must publish both image products");
    check(
        output.detector_input->memoryKind() ==
                ports::MemoryKind::cuda_device &&
            output.detector_input->pixelFormat() == core::PixelFormat::bgr8,
        "detector image must be CUDA BGR8");
    check(
        output.geometry_gray->memoryKind() ==
                ports::MemoryKind::cuda_device &&
            output.geometry_gray->pixelFormat() == core::PixelFormat::gray8,
        "geometry image must be CUDA Gray8");

    const auto bgr = download(output.detector_input, 4U * 3U);
    const auto gray = download(output.geometry_gray, 4U);
    check(std::all_of(bgr.begin(), bgr.end(), [](std::uint8_t value) {
              return value == 77U;
          }),
          "constant Bayer input must remain constant after identity remap");
    check(std::all_of(gray.begin(), gray.end(), [](std::uint8_t value) {
              return value == 77U;
          }),
          "constant BGR input must produce the same gray value");
}

void testPreparesAuditedCudaImagesAndBoundsPool() {
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 4;
    adapter::CudaStereoPreprocessor preprocessor({width, height, 1});
    std::string error;
    check(preprocessor.initialize(identityMaps(width, height), error),
          "valid identity maps must initialize the preprocessor");
    check(preprocessor.isInitialized(),
          "successful initialization must be observable");

    auto first = preprocessor.prepare(frame(width, height));
    check(first.success && first.frame.has_value(),
          "valid host Bayer pair must preprocess successfully");
    if (first.frame) {
        checkCameraOutput(
            first.frame->left, core::CameraRole::left, "left-camera");
        checkCameraOutput(
            first.frame->right, core::CameraRole::right, "right-camera");
    }

    const auto exhausted = preprocessor.prepare(frame(width, height));
    check(!exhausted.success && !exhausted.frame,
          "retained outputs must exhaust a one-slot pool explicitly");
    first.frame.reset();

    const auto recovered = preprocessor.prepare(frame(width, height));
    check(recovered.success && recovered.frame.has_value(),
          "releasing image views must make the slot reusable");
}

void testRejectsInvalidMapsAndInput() {
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 4;
    auto invalid_maps = identityMaps(width, height);
    invalid_maps.right_y.pop_back();
    adapter::CudaStereoPreprocessor invalid_preprocessor({width, height, 1});
    std::string error;
    check(!invalid_preprocessor.initialize(invalid_maps, error),
          "incomplete remap maps must be rejected");

    adapter::CudaStereoPreprocessor preprocessor({width, height, 1});
    check(preprocessor.initialize(identityMaps(width, height), error),
          "valid preprocessor setup must succeed");
    check(!preprocessor.initialize(identityMaps(width, height), error),
          "a live preprocessor must reject repeated initialization");

    const auto invalid_frame =
        preprocessor.prepare(frame(width, height, core::PixelFormat::gray8));
    check(!invalid_frame.success && !invalid_frame.frame,
          "non-Bayer image buffers must be rejected");
}

void testOutputOwnsDeviceMemoryPastPreprocessorLifetime() {
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 4;
    std::optional<ports::RectifiedStereoFrame> retained;
    {
        adapter::CudaStereoPreprocessor preprocessor({width, height, 1});
        std::string error;
        check(preprocessor.initialize(identityMaps(width, height), error),
              "lifetime fixture must initialize");
        auto result = preprocessor.prepare(frame(width, height));
        check(result.success && result.frame.has_value(),
              "lifetime fixture must produce a frame");
        retained = std::move(result.frame);
    }
    if (retained) {
        const auto gray = download(retained->left.geometry_gray, width);
        check(std::all_of(gray.begin(), gray.end(), [](std::uint8_t value) {
                  return value == 77U;
              }),
              "image views must keep CUDA slot memory alive");
    }
}

}  // namespace

int main() {
    try {
        testPreparesAuditedCudaImagesAndBoundsPool();
        testRejectsInvalidMapsAndInput();
        testOutputOwnsDeviceMemoryPastPreprocessorLifetime();
    } catch (const std::exception& exception) {
        std::cerr << "FAILED with exception: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "CUDA stereo preprocessor tests passed\n";
    return EXIT_SUCCESS;
}
