/**
 * @file roi_stereo_matcher.cpp
 * @brief ROI 多点立体匹配器实现
 */

#include "roi_stereo_matcher.h"
#include "../utils/logger.h"
#include <algorithm>
#include <cmath>

// CUDA kernel C 接口
extern "C" void launchROIMultiPointMatch(
    const uint8_t* leftImg,  int leftPitch,
    const uint8_t* rightImg, int rightPitch,
    int imgWidth, int imgHeight,
    const int* bboxes,
    const float* detCx, const float* detCy,
    int numBoxes,
    float* results,
    int maxDisparity, int patchRadius,
    float focal, float baseline,
    float cx0, float cy0,
    float minDepth, float maxDepth,
    cudaStream_t stream);

extern "C" void launchROICircleFitMatch(
    const uint8_t* leftImg,  int leftPitch,
    const uint8_t* rightImg, int rightPitch,
    int imgWidth, int imgHeight,
    const int* bboxes,
    const float* detCx, const float* detCy,
    int numBoxes,
    float* results,
    int maxDisparity,
    float focal, float baseline,
    float cx0, float cy0,
    float minDepth, float maxDepth,
    float objectDiameter,
    cudaStream_t stream);

namespace stereo3d {

ROIStereoMatcher::~ROIStereoMatcher() {
    freeBuffers();
}

void ROIStereoMatcher::init(float focal, float baseline, float cx, float cy,
                            const ROIMatchConfig& config)
{
    focal_    = focal;
    baseline_ = baseline;
    cx_       = cx;
    cy_       = cy;
    config_   = config;

    if (!allocateBuffers()) {
        LOG_ERROR("ROIStereoMatcher: CUDA buffer allocation failed");
        return;
    }

    LOG_INFO("ROIStereoMatcher: focal=%.1f, baseline=%.4fm, cx=%.1f, cy=%.1f",
             focal_, baseline_, cx_, cy_);
    LOG_INFO("  maxDisparity=%d, patchRadius=%d, depth=[%.1f, %.1f]m",
             config_.maxDisparity, config_.patchRadius,
             config_.minDepth, config_.maxDepth);
}

bool ROIStereoMatcher::allocateBuffers() {
    cudaError_t err;

    err = cudaMalloc(&bboxes_device_, kMaxBoxes * 4 * sizeof(int));
    if (err != cudaSuccess) { LOG_ERROR("cudaMalloc bboxes failed"); freeBuffers(); return false; }

    err = cudaMalloc(&detCx_device_, kMaxBoxes * sizeof(float));
    if (err != cudaSuccess) { freeBuffers(); return false; }

    err = cudaMalloc(&detCy_device_, kMaxBoxes * sizeof(float));
    if (err != cudaSuccess) { freeBuffers(); return false; }

    // results: [X, Y, Z, disp, conf] per detection = 5 floats
    err = cudaMalloc(&results_device_, kMaxBoxes * 5 * sizeof(float));
    if (err != cudaSuccess) { freeBuffers(); return false; }

    err = cudaHostAlloc(reinterpret_cast<void**>(&results_host_),
                        kMaxBoxes * 5 * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) { freeBuffers(); return false; }

    return true;
}

void ROIStereoMatcher::freeBuffers() {
    if (bboxes_device_)  { cudaFree(bboxes_device_);  bboxes_device_  = nullptr; }
    if (detCx_device_)   { cudaFree(detCx_device_);   detCx_device_   = nullptr; }
    if (detCy_device_)   { cudaFree(detCy_device_);   detCy_device_   = nullptr; }
    if (results_device_) { cudaFree(results_device_);  results_device_ = nullptr; }
    if (results_host_)   { cudaFreeHost(results_host_); results_host_  = nullptr; }
}

std::vector<Object3D> ROIStereoMatcher::match(
    const uint8_t* leftGPU, int leftPitch,
    const uint8_t* rightGPU, int rightPitch,
    int imgWidth, int imgHeight,
    const std::vector<Detection>& detections,
    cudaStream_t stream)
{
    int numBoxes = std::min(static_cast<int>(detections.size()), kMaxBoxes);
    if (numBoxes == 0) return {};

    // 1. 准备 BBox + 中心坐标并上传 GPU
    std::vector<int>   bboxes_h(numBoxes * 4);
    std::vector<float> cx_h(numBoxes);
    std::vector<float> cy_h(numBoxes);

    for (int i = 0; i < numBoxes; ++i) {
        const auto& d = detections[i];
        int x1 = std::max(0, static_cast<int>(d.cx - d.width / 2));
        int y1 = std::max(0, static_cast<int>(d.cy - d.height / 2));
        int x2 = std::min(imgWidth - 1,  static_cast<int>(d.cx + d.width / 2));
        int y2 = std::min(imgHeight - 1, static_cast<int>(d.cy + d.height / 2));
        bboxes_h[i * 4 + 0] = x1;
        bboxes_h[i * 4 + 1] = y1;
        bboxes_h[i * 4 + 2] = x2;
        bboxes_h[i * 4 + 3] = y2;
        cx_h[i] = d.cx;
        cy_h[i] = d.cy;
    }

    cudaMemcpyAsync(bboxes_device_, bboxes_h.data(),
                    numBoxes * 4 * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(detCx_device_, cx_h.data(),
                    numBoxes * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(detCy_device_, cy_h.data(),
                    numBoxes * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 2. 启动 CUDA Kernel
    if (config_.useCircleFit) {
        launchROICircleFitMatch(
            leftGPU, leftPitch, rightGPU, rightPitch,
            imgWidth, imgHeight,
            bboxes_device_, detCx_device_, detCy_device_, numBoxes,
            results_device_,
            config_.maxDisparity,
            focal_, baseline_, cx_, cy_,
            config_.minDepth, config_.maxDepth,
            config_.objectDiameter,
            stream);
    } else {
        launchROIMultiPointMatch(
            leftGPU, leftPitch, rightGPU, rightPitch,
            imgWidth, imgHeight,
            bboxes_device_, detCx_device_, detCy_device_, numBoxes,
            results_device_,
            config_.maxDisparity, config_.patchRadius,
            focal_, baseline_, cx_, cy_,
            config_.minDepth, config_.maxDepth,
            stream);
    }

    // 3. 拷回 CPU
    cudaMemcpyAsync(results_host_, results_device_,
                    numBoxes * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. 组装结果 — 始终为每个检测输出一个结果 (无效结果 confidence=0)
    //    保证 output[i] 与 detections[i] 一一对应, 避免索引错位
    std::vector<Object3D> output(numBoxes);

    for (int i = 0; i < numBoxes; ++i) {
        float X    = results_host_[i * 5 + 0];
        float Y    = results_host_[i * 5 + 1];
        float Z    = results_host_[i * 5 + 2];
        float disp = results_host_[i * 5 + 3];
        float conf = results_host_[i * 5 + 4];

        output[i].class_id = detections[i].class_id;

        if (Z > 0.0f && conf > 0.0f) {
            output[i].x = X;
            output[i].y = Y;
            output[i].z = Z;
            output[i].confidence = conf * detections[i].confidence;
        } else {
            output[i].x = 0;
            output[i].y = 0;
            output[i].z = -1.0f;
            output[i].confidence = 0;
        }
    }

    return output;
}

}  // namespace stereo3d
