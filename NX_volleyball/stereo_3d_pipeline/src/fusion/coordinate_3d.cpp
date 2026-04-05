/**
 * @file coordinate_3d.cpp
 * @brief 3D 坐标融合实现
 */

#include "coordinate_3d.h"
#include "../utils/logger.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// CUDA kernel 声明 (定义在 depth_extract.cu)
extern "C" void launchDepthExtractKernel(
    const int16_t* disparity, int dispPitch, int imgWidth,
    const int* bboxes, int numBoxes,
    float* depths,
    cudaStream_t stream);

namespace stereo3d {

Coordinate3D::~Coordinate3D() {
    freeGPUBuffers();
}

void Coordinate3D::init(const cv::Mat& P1, float baseline,
                        float minDepth, float maxDepth)
{
    baseline_ = baseline;
    minDepth_ = minDepth;
    maxDepth_ = maxDepth;

    // 从 P1 提取焦距和主点
    if (!P1.empty() && P1.rows == 3 && P1.cols == 4) {
        focal_ = static_cast<float>(P1.at<double>(0, 0));
        cx_    = static_cast<float>(P1.at<double>(0, 2));
        cy_    = static_cast<float>(P1.at<double>(1, 2));
    }

    LOG_INFO("Coordinate3D: focal=%.1f, cx=%.1f, cy=%.1f, baseline=%.4fm",
             focal_, cx_, cy_, baseline_);

    allocateGPUBuffers();
}

bool Coordinate3D::allocateGPUBuffers() {
    cudaError_t err;

    // 深度结果 (每个框一个 float)
    err = cudaMalloc(&depthResults_device_, maxBoxes_ * sizeof(float));
    if (err != cudaSuccess) { freeGPUBuffers(); return false; }

    err = cudaHostAlloc(reinterpret_cast<void**>(&depthResults_host_),
                        maxBoxes_ * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) { freeGPUBuffers(); return false; }

    // BBox 数据 [x1, y1, x2, y2] * maxBoxes
    err = cudaMalloc(&bboxes_device_, maxBoxes_ * 4 * sizeof(int));
    if (err != cudaSuccess) { freeGPUBuffers(); return false; }

    return true;
}

void Coordinate3D::freeGPUBuffers() {
    if (depthResults_device_) { cudaFree(depthResults_device_); depthResults_device_ = nullptr; }
    if (depthResults_host_)   { cudaFreeHost(depthResults_host_); depthResults_host_ = nullptr; }
    if (bboxes_device_)       { cudaFree(bboxes_device_); bboxes_device_ = nullptr; }
}

Object3D Coordinate3D::compute(const Detection& det,
                               const int16_t* disparityGPU, int dispPitch,
                               int imgWidth, int imgHeight,
                               cudaStream_t stream,
                               float disparityScale)
{
    std::vector<Detection> dets = {det};
    auto results = computeBatch(dets, disparityGPU, dispPitch,
                                imgWidth, imgHeight, stream, disparityScale);
    return results.empty() ? Object3D() : results[0];
}

std::vector<Object3D> Coordinate3D::computeBatch(
    const std::vector<Detection>& dets,
    const int16_t* disparityGPU, int dispPitch,
    int imgWidth, int imgHeight,
    cudaStream_t stream,
    float disparityScale)
{
    int numBoxes = std::min(static_cast<int>(dets.size()), maxBoxes_);
    if (numBoxes == 0) return {};

    // 1. 准备 BBox 数据并上传到 GPU
    std::vector<int> bboxes_host(numBoxes * 4);
    for (int i = 0; i < numBoxes; ++i) {
        const auto& d = dets[i];
        int x1 = std::max(0, static_cast<int>(d.cx - d.width / 2));
        int y1 = std::max(0, static_cast<int>(d.cy - d.height / 2));
        int x2 = std::min(imgWidth - 1,  static_cast<int>(d.cx + d.width / 2));
        int y2 = std::min(imgHeight - 1, static_cast<int>(d.cy + d.height / 2));
        bboxes_host[i * 4 + 0] = x1;
        bboxes_host[i * 4 + 1] = y1;
        bboxes_host[i * 4 + 2] = x2;
        bboxes_host[i * 4 + 3] = y2;
    }

    cudaMemcpyAsync(bboxes_device_, bboxes_host.data(),
                    numBoxes * 4 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // 2. CUDA kernel: 每个 block 处理一个 BBox，用直方图法求峰值视差
    launchDepthExtractKernel(disparityGPU, dispPitch, imgWidth,
                             bboxes_device_, numBoxes,
                             depthResults_device_, stream);

    // 3. 拷回 CPU
    cudaMemcpyAsync(depthResults_host_, depthResults_device_,
                    numBoxes * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. 3D 投影
    std::vector<Object3D> results;
    results.reserve(numBoxes);

    for (int i = 0; i < numBoxes; ++i) {
        float d_peak = depthResults_host_[i];
        if (d_peak <= 0) continue;

        // 半分辨率视差需要 ×2 补偿
        d_peak *= disparityScale;

        // VPI S16 Q10.5 格式: 实际视差 = d_peak / 32.0
        // 但 depth_extract.cu 已经做了除法，这里 d_peak 就是像素视差
        float Z = (focal_ * baseline_) / d_peak;

        if (Z < minDepth_ || Z > maxDepth_) continue;

        Object3D obj;
        obj.x = (dets[i].cx - cx_) * Z / focal_;
        obj.y = (dets[i].cy - cy_) * Z / focal_;
        obj.z = Z;
        obj.confidence = dets[i].confidence;
        obj.class_id   = dets[i].class_id;
        results.push_back(obj);
    }

    return results;
}

}  // namespace stereo3d
