/**
 * @file cuda_kernels.hpp
 * @brief TensorRT detector 使用的 CUDA 图像预处理和 DFL 解码入口。
 *
 * 该头文件只允许硬件适配器包含；平台无关 core、ports 和 services 不得
 * 依赖 CUDA 类型。
 */

#pragma once

#include <cuda_runtime_api.h>

extern "C" void launchGrayToRGBKernel(
    const unsigned char* gray,
    float* destination,
    int source_width,
    int source_height,
    int source_pitch,
    int destination_width,
    int destination_height,
    cudaStream_t stream);

extern "C" void launchGrayToRGBLetterboxKernel(
    const unsigned char* gray,
    float* destination,
    int source_width,
    int source_height,
    int source_pitch,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int padding_x,
    int padding_y,
    cudaStream_t stream);

extern "C" void launchBayerToRGBKernel(
    const unsigned char* bayer,
    float* destination,
    int source_width,
    int source_height,
    int source_pitch,
    int destination_width,
    int destination_height,
    cudaStream_t stream);

extern "C" void launchBayerToRGBLetterboxKernel(
    const unsigned char* bayer,
    float* destination,
    int source_width,
    int source_height,
    int source_pitch,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int padding_x,
    int padding_y,
    cudaStream_t stream);

extern "C" void launchBGRToRGBLetterboxKernel(
    const unsigned char* bgr,
    float* destination,
    int source_width,
    int source_height,
    int source_step,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int padding_x,
    int padding_y,
    cudaStream_t stream);

extern "C" void launchDFLDecodeKernel(
    const float* classification,
    const float* boxes,
    float* output_boxes,
    int* output_count,
    int height,
    int width,
    int class_count,
    int regression_bins,
    int stride,
    float confidence_threshold,
    int maximum_detections,
    cudaStream_t stream);
