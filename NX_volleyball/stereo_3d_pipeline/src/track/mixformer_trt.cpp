/**
 * @file mixformer_trt.cpp
 * @brief MixFormerV2-small TensorRT 推理实现
 */

#include "mixformer_trt.h"
#include "../utils/logger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace stereo3d {

namespace {
class TRTLoggerMix : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            LOG_WARN("[MixFormer-TRT] %s", msg);
    }
};
static TRTLoggerMix sTrtLogger;
}

MixFormerTRT::MixFormerTRT() = default;

MixFormerTRT::~MixFormerTRT() {
    auto freeDev = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto freeHost = [](float*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
    freeDev(d_template_patch_);
    freeDev(d_search_patch_);
    freeDev(d_output_);
    freeHost(h_output_);

    if (context_) { delete context_; context_ = nullptr; }
    if (engine_) { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
}

bool MixFormerTRT::init(const std::string& engine_path,
                         const std::string& /*head_engine_path*/,
                         cudaStream_t stream) {
    stream_ = stream;

    runtime_ = nvinfer1::createInferRuntime(sTrtLogger);
    if (!runtime_) {
        LOG_ERROR("[MixFormer] Failed to create TensorRT runtime");
        return false;
    }

    // Load engine
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR("[MixFormer] Cannot open engine: %s", engine_path.c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);

    engine_ = runtime_->deserializeCudaEngine(buf.data(), size);
    if (!engine_) {
        LOG_ERROR("[MixFormer] Failed to deserialize engine");
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        LOG_ERROR("[MixFormer] Failed to create execution context");
        return false;
    }

    // Query output size
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = engine_->getTensorShape(name);
            output_elements_ = 1;
            for (int d = 0; d < dims.nbDims; d++)
                output_elements_ *= dims.d[d];
            LOG_INFO("[MixFormer] Output: %d elements (%s)", output_elements_, name);
        }
    }

    // Allocate buffers
    cudaMalloc(&d_template_patch_, template_size_ * template_size_ * sizeof(float));
    cudaMalloc(&d_search_patch_, search_size_ * search_size_ * sizeof(float));
    if (output_elements_ > 0) {
        cudaMalloc(&d_output_, output_elements_ * sizeof(float));
        cudaMallocHost(&h_output_, output_elements_ * sizeof(float));
    }

    LOG_INFO("[MixFormer] Initialized: engine=%s, output=%d elements",
             engine_path.c_str(), output_elements_);
    return true;
}

void MixFormerTRT::setTarget(const void* gpu_image, int pitch,
                              int img_width, int img_height,
                              const Detection& det) {
    last_det_ = det;
    target_sz_[0] = det.width;
    target_sz_[1] = det.height;

    // Crop and cache template patch
    cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                  d_template_patch_, template_size_,
                  det.cx, det.cy, det.width, det.height,
                  template_context_, stream_);
    cudaStreamSynchronize(stream_);

    has_target_ = true;
}

SOTResult MixFormerTRT::track(const void* gpu_image, int pitch,
                               int img_width, int img_height) {
    SOTResult result;
    if (!has_target_) return result;

    // 1. Crop search patch centered on last_det_
    cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                  d_search_patch_, search_size_,
                  last_det_.cx, last_det_.cy, last_det_.width, last_det_.height,
                  search_context_, stream_);

    // 2. Bind I/O and run inference
    int inputIdx = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (inputIdx == 0)
                context_->setTensorAddress(name, d_template_patch_);
            else
                context_->setTensorAddress(name, d_search_patch_);
            inputIdx++;
        } else {
            context_->setTensorAddress(name, d_output_);
        }
    }
    context_->enqueueV3(stream_);

    // 3. D2H + decode
    cudaMemcpyAsync(h_output_, d_output_, output_elements_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Decode output: MixFormerV2 typically outputs [score, cx_norm, cy_norm, w_norm, h_norm]
    // where cx/cy/w/h are normalized to search region [0, 1]
    if (output_elements_ >= 5) {
        // score: ONNX wrapper 输出已是概率 [0,1], 无需 sigmoid
        float score = h_output_[0];
        float pred_cx = h_output_[1];
        float pred_cy = h_output_[2];
        float pred_w  = h_output_[3];
        float pred_h  = h_output_[4];

        // Convert from search region normalized coords to image coords
        // cropResizeGPU uses square ROI: s = sqrt((w*ctx)*(h*ctx))
        float s = std::sqrt((last_det_.width * search_context_) *
                            (last_det_.height * search_context_));
        float search_x0 = last_det_.cx - s * 0.5f;
        float search_y0 = last_det_.cy - s * 0.5f;

        result.cx = search_x0 + pred_cx * s;
        result.cy = search_y0 + pred_cy * s;
        result.width = pred_w * s;
        result.height = pred_h * s;

        // Smooth size
        result.width = target_sz_[0] * 0.6f + result.width * 0.4f;
        result.height = target_sz_[1] * 0.6f + result.height * 0.4f;
        target_sz_[0] = result.width;
        target_sz_[1] = result.height;

        // Clamp
        result.cx = std::clamp(result.cx, 0.0f, (float)img_width);
        result.cy = std::clamp(result.cy, 0.0f, (float)img_height);
        result.width = std::max(result.width, 5.0f);
        result.height = std::max(result.height, 5.0f);

        result.confidence = score;
        result.valid = (score > 0.1f);
    } else if (output_elements_ >= 1) {
        // Minimal output — score map with peak finding
        // Try to find peak in a 2D score map
        int map_size = (int)std::sqrt((float)output_elements_);
        if (map_size * map_size == output_elements_) {
            int best_idx = 0;
            float best_score = -1e9f;
            for (int i = 0; i < output_elements_; i++) {
                float s = 1.0f / (1.0f + std::exp(-h_output_[i]));
                if (s > best_score) { best_score = s; best_idx = i; }
            }
            int gy = best_idx / map_size, gx = best_idx % map_size;
            float cx_grid = (float)map_size / 2.0f;
            float cy_grid = (float)map_size / 2.0f;
            float srw = last_det_.width * search_context_;
            float srh = last_det_.height * search_context_;

            result.cx = last_det_.cx + ((float)gx - cx_grid) / map_size * srw;
            result.cy = last_det_.cy + ((float)gy - cy_grid) / map_size * srh;
            result.width = target_sz_[0];
            result.height = target_sz_[1];
            result.confidence = best_score;
            result.valid = (best_score > 0.1f);
        }
    }

    if (result.valid) {
        last_det_.cx = result.cx;
        last_det_.cy = result.cy;
        last_det_.width = result.width;
        last_det_.height = result.height;
    }

    return result;
}

void MixFormerTRT::reset() {
    has_target_ = false;
    last_det_ = Detection();
    target_sz_[0] = 0;
    target_sz_[1] = 0;
}

}  // namespace stereo3d
