#include "volleyball/adapters/cuda_preprocess/kernels.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_preprocess = volleyball::adapters::cuda_preprocess;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            operation + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        requireCuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t sizeBytes() const noexcept { return count_ * sizeof(T); }

private:
    T* data_{nullptr};
    std::size_t count_{};
};

void testBayerConstantImage() {
    constexpr int width = 4;
    constexpr int height = 4;
    const std::vector<std::uint8_t> input(width * height, 77U);
    DeviceBuffer<std::uint8_t> device_input(input.size());
    DeviceBuffer<std::uint8_t> device_output(input.size() * 3U);
    requireCuda(
        cudaMemcpy(
            device_input.data(), input.data(), input.size(),
            cudaMemcpyHostToDevice),
        "copy Bayer input");

    requireCuda(
        cuda_preprocess::launchBayerRggbToBgr8(
            device_input.data(), width, device_output.data(), width * 3U,
            width, height, nullptr),
        "launch Bayer conversion");
    requireCuda(cudaDeviceSynchronize(), "synchronize Bayer conversion");

    std::vector<std::uint8_t> output(input.size() * 3U);
    requireCuda(
        cudaMemcpy(
            output.data(), device_output.data(), output.size(),
            cudaMemcpyDeviceToHost),
        "copy Bayer output");
    for (const auto value : output) {
        require(value == 77U, "constant Bayer image must remain constant");
    }
}

void testIdentityRemapAndGray() {
    constexpr int width = 3;
    constexpr int height = 2;
    const std::array<std::uint8_t, width * height * 3> source{
        10, 20, 30, 40, 50, 60, 70, 80, 90,
        15, 25, 35, 45, 55, 65, 75, 85, 95,
    };
    const std::array<float, width * height> map_x{0, 1, 2, 0, 1, 2};
    const std::array<float, width * height> map_y{0, 0, 0, 1, 1, 1};

    DeviceBuffer<std::uint8_t> device_source(source.size());
    DeviceBuffer<float> device_map_x(map_x.size());
    DeviceBuffer<float> device_map_y(map_y.size());
    DeviceBuffer<std::uint8_t> device_bgr(source.size());
    DeviceBuffer<std::uint8_t> device_gray(width * height);
    requireCuda(
        cudaMemcpy(
            device_source.data(), source.data(), device_source.sizeBytes(),
            cudaMemcpyHostToDevice),
        "copy source BGR");
    requireCuda(
        cudaMemcpy(
            device_map_x.data(), map_x.data(), device_map_x.sizeBytes(),
            cudaMemcpyHostToDevice),
        "copy map x");
    requireCuda(
        cudaMemcpy(
            device_map_y.data(), map_y.data(), device_map_y.sizeBytes(),
            cudaMemcpyHostToDevice),
        "copy map y");

    requireCuda(
        cuda_preprocess::launchRemapBgrToBgrGray8(
            device_source.data(), width * 3U, width, height,
            device_map_x.data(), device_map_y.data(), width * sizeof(float),
            device_bgr.data(), width * 3U, device_gray.data(), width,
            width, height, nullptr),
        "launch identity remap");
    requireCuda(cudaDeviceSynchronize(), "synchronize identity remap");

    std::array<std::uint8_t, source.size()> bgr{};
    std::array<std::uint8_t, width * height> gray{};
    requireCuda(
        cudaMemcpy(
            bgr.data(), device_bgr.data(), device_bgr.sizeBytes(),
            cudaMemcpyDeviceToHost),
        "copy remapped BGR");
    requireCuda(
        cudaMemcpy(
            gray.data(), device_gray.data(), device_gray.sizeBytes(),
            cudaMemcpyDeviceToHost),
        "copy gray");
    require(bgr == source, "identity map must preserve BGR pixels");
    for (std::size_t pixel = 0; pixel < gray.size(); ++pixel) {
        const auto offset = pixel * 3U;
        const auto expected = static_cast<std::uint8_t>(
            source[offset] * 0.114 + source[offset + 1U] * 0.587 +
            source[offset + 2U] * 0.299 + 0.5);
        require(gray[pixel] == expected, "gray conversion mismatch");
    }
}

void testInvalidAndNonFiniteInputs() {
    constexpr int width = 1;
    constexpr int height = 1;
    require(
        cuda_preprocess::launchBayerRggbToBgr8(
            nullptr, width, nullptr, width * 3U, width, height, nullptr) ==
            cudaErrorInvalidValue,
        "null Bayer buffers must be rejected");

    const std::array<std::uint8_t, 3> source{10, 20, 30};
    const std::array<float, 1> map_x{
        std::numeric_limits<float>::quiet_NaN()};
    const std::array<float, 1> map_y{0.0F};
    DeviceBuffer<std::uint8_t> device_source(source.size());
    DeviceBuffer<float> device_map_x(map_x.size());
    DeviceBuffer<float> device_map_y(map_y.size());
    DeviceBuffer<std::uint8_t> device_bgr(3);
    DeviceBuffer<std::uint8_t> device_gray(1);
    requireCuda(
        cudaMemcpy(
            device_source.data(), source.data(), device_source.sizeBytes(),
            cudaMemcpyHostToDevice),
        "copy finite source");
    requireCuda(
        cudaMemcpy(
            device_map_x.data(), map_x.data(), device_map_x.sizeBytes(),
            cudaMemcpyHostToDevice),
        "copy non-finite map x");
    requireCuda(
        cudaMemcpy(
            device_map_y.data(), map_y.data(), device_map_y.sizeBytes(),
            cudaMemcpyHostToDevice),
        "copy finite map y");

    requireCuda(
        cuda_preprocess::launchRemapBgrToBgrGray8(
            device_source.data(), 3, width, height,
            device_map_x.data(), device_map_y.data(), sizeof(float),
            device_bgr.data(), 3, device_gray.data(), 1,
            width, height, nullptr),
        "launch non-finite remap");
    requireCuda(cudaDeviceSynchronize(), "synchronize non-finite remap");

    std::array<std::uint8_t, 3> bgr{1, 1, 1};
    std::uint8_t gray = 1;
    requireCuda(
        cudaMemcpy(
            bgr.data(), device_bgr.data(), device_bgr.sizeBytes(),
            cudaMemcpyDeviceToHost),
        "copy non-finite BGR");
    requireCuda(
        cudaMemcpy(
            &gray, device_gray.data(), device_gray.sizeBytes(),
            cudaMemcpyDeviceToHost),
        "copy non-finite gray");
    require(
        bgr == std::array<std::uint8_t, 3>{0, 0, 0} && gray == 0,
        "non-finite map coordinates must produce a zero border");
}

}  // namespace

int main() {
    try {
        testBayerConstantImage();
        testIdentityRemapAndGray();
        testInvalidAndNonFiniteInputs();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
