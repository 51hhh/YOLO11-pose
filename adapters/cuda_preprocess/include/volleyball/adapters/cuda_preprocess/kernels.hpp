#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace volleyball::adapters::cuda_preprocess {

[[nodiscard]] cudaError_t launchBayerRggbToBgr8(
    const std::uint8_t* bayer,
    std::size_t bayer_pitch_bytes,
    std::uint8_t* bgr,
    std::size_t bgr_pitch_bytes,
    int width,
    int height,
    cudaStream_t stream) noexcept;

// map_x/map_y 是 OpenCV initUndistortRectifyMap 生成的目标到源坐标。
// 一次采样同时写出校正 BGR8 和同坐标系 Gray8，越界采用零边界。
[[nodiscard]] cudaError_t launchRemapBgrToBgrGray8(
    const std::uint8_t* source_bgr,
    std::size_t source_pitch_bytes,
    int source_width,
    int source_height,
    const float* map_x,
    const float* map_y,
    std::size_t map_pitch_bytes,
    std::uint8_t* rectified_bgr,
    std::size_t rectified_bgr_pitch_bytes,
    std::uint8_t* rectified_gray,
    std::size_t rectified_gray_pitch_bytes,
    int output_width,
    int output_height,
    cudaStream_t stream) noexcept;

}  // namespace volleyball::adapters::cuda_preprocess
