/**
 * @file nanotrack_trt.cpp
 * @brief NanoTrack TensorRT 推理实现
 */

#include "nanotrack_trt.h"
#include "../utils/logger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace stereo3d {

// TRT Logger
namespace {
class TRTLoggerNano : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            LOG_WARN("[NanoTrack-TRT] %s", msg);
    }
};
static TRTLoggerNano sTrtLogger;
}

NanoTrackTRT::NanoTrackTRT() = default;

NanoTrackTRT::~NanoTrackTRT() {
    freeBuffers();
    if (backboneCtx_) { delete backboneCtx_; backboneCtx_ = nullptr; }
    if (backboneEngine_) { delete backboneEngine_; backboneEngine_ = nullptr; }
    if (headCtx_) { delete headCtx_; headCtx_ = nullptr; }
    if (headEngine_) { delete headEngine_; headEngine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
}

bool NanoTrackTRT::loadEngine(const std::string& path,
                               nvinfer1::ICudaEngine*& engine,
                               nvinfer1::IExecutionContext*& ctx) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR("[NanoTrack] Cannot open engine: %s", path.c_str());
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buf(size);
    file.read(buf.data(), size);

    engine = runtime_->deserializeCudaEngine(buf.data(), size);
    if (!engine) {
        LOG_ERROR("[NanoTrack] Failed to deserialize engine: %s", path.c_str());
        return false;
    }

    ctx = engine->createExecutionContext();
    if (!ctx) {
        LOG_ERROR("[NanoTrack] Failed to create execution context: %s", path.c_str());
        delete engine;
        engine = nullptr;
        return false;
    }

    return true;
}

bool NanoTrackTRT::init(const std::string& backbone_path,
                         const std::string& head_path,
                         cudaStream_t stream) {
    stream_ = stream;

    runtime_ = nvinfer1::createInferRuntime(sTrtLogger);
    if (!runtime_) {
        LOG_ERROR("[NanoTrack] Failed to create TensorRT runtime");
        return false;
    }

    if (!loadEngine(backbone_path, backboneEngine_, backboneCtx_)) return false;
    if (!loadEngine(head_path, headEngine_, headCtx_)) return false;

    // Query backbone output dimensions
    // Backbone: input [1,1,H,H] → output feature map
    int nbBindings = backboneEngine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; i++) {
        const char* name = backboneEngine_->getIOTensorName(i);
        if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = backboneEngine_->getTensorShape(name);
            template_feat_elements_ = 1;
            for (int d = 0; d < dims.nbDims; d++)
                template_feat_elements_ *= dims.d[d];
            search_feat_elements_ = template_feat_elements_;  // same backbone, different spatial
            LOG_INFO("[NanoTrack] Backbone output: %d elements (%s)", template_feat_elements_, name);
        }
    }

    // Query head output dimensions (cls + reg)
    nbBindings = headEngine_->getNbIOTensors();
    for (int i = 0; i < nbBindings; i++) {
        const char* name = headEngine_->getIOTensorName(i);
        if (headEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = headEngine_->getTensorShape(name);
            int elements = 1;
            for (int d = 0; d < dims.nbDims; d++)
                elements *= dims.d[d];

            std::string sname(name);
            if (sname.find("cls") != std::string::npos || sname.find("score") != std::string::npos) {
                cls_elements_ = elements;
                // Assume last two dims are spatial
                if (dims.nbDims >= 2) {
                    score_map_h_ = dims.d[dims.nbDims - 2];
                    score_map_w_ = dims.d[dims.nbDims - 1];
                }
                LOG_INFO("[NanoTrack] Head cls output: %d elements, map %dx%d", elements, score_map_h_, score_map_w_);
            } else {
                reg_elements_ = elements;
                LOG_INFO("[NanoTrack] Head reg output: %d elements", elements);
            }
        }
    }

    // Backbone will be run with different input sizes for template vs search
    // Need to figure out search output feature size
    // For search image (255x255), backbone output spatial dim differs from template (127x127)
    // Re-query with search input shape
    {
        const char* inputName = nullptr;
        for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
            const char* name = backboneEngine_->getIOTensorName(i);
            if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                inputName = name;
                break;
            }
        }
        if (inputName) {
            // Check if backbone supports dynamic shapes
            auto minDims = backboneEngine_->getProfileShape(inputName, 0, nvinfer1::OptProfileSelector::kMIN);
            auto maxDims = backboneEngine_->getProfileShape(inputName, 0, nvinfer1::OptProfileSelector::kMAX);
            if (minDims.nbDims > 0 && maxDims.d[maxDims.nbDims-1] >= search_size_) {
                // Dynamic shape backbone — set search shape to get output dims
                nvinfer1::Dims4 searchDims{1, 1, search_size_, search_size_};
                backboneCtx_->setInputShape(inputName, searchDims);
                for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
                    const char* name = backboneEngine_->getIOTensorName(i);
                    if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                        auto dims = backboneCtx_->getTensorShape(name);
                        search_feat_elements_ = 1;
                        for (int d = 0; d < dims.nbDims; d++)
                            search_feat_elements_ *= dims.d[d];
                    }
                }
            }
        }
    }

    allocateBuffers();

    LOG_INFO("[NanoTrack] Initialized: backbone=%s, head=%s",
             backbone_path.c_str(), head_path.c_str());
    return true;
}

void NanoTrackTRT::allocateBuffers() {
    cudaMalloc(&d_template_patch_, template_size_ * template_size_ * sizeof(float));
    cudaMalloc(&d_search_patch_, search_size_ * search_size_ * sizeof(float));
    cudaMalloc(&d_template_feat_, template_feat_elements_ * sizeof(float));
    cudaMalloc(&d_search_feat_, search_feat_elements_ * sizeof(float));

    if (cls_elements_ > 0) {
        cudaMalloc(&d_head_cls_, cls_elements_ * sizeof(float));
        cudaMallocHost(&h_head_cls_, cls_elements_ * sizeof(float));
    }
    if (reg_elements_ > 0) {
        cudaMalloc(&d_head_reg_, reg_elements_ * sizeof(float));
        cudaMallocHost(&h_head_reg_, reg_elements_ * sizeof(float));
    }
}

void NanoTrackTRT::freeBuffers() {
    auto freeDev = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto freeHost = [](float*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };

    freeDev(d_template_patch_);
    freeDev(d_search_patch_);
    freeDev(d_template_feat_);
    freeDev(d_search_feat_);
    freeDev(d_head_cls_);
    freeDev(d_head_reg_);
    freeHost(h_head_cls_);
    freeHost(h_head_reg_);
}

void NanoTrackTRT::setTarget(const void* gpu_image, int pitch,
                              int img_width, int img_height,
                              const Detection& det) {
    last_det_ = det;
    target_sz_[0] = det.width;
    target_sz_[1] = det.height;

    // Crop template patch (127x127, context_factor=2.0)
    cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                  d_template_patch_, template_size_,
                  det.cx, det.cy, det.width, det.height,
                  template_context_, stream_);

    // Run backbone on template to extract feature (cached)
    // Set input shape for template
    const char* inputName = nullptr;
    for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
        const char* name = backboneEngine_->getIOTensorName(i);
        if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
            break;
        }
    }
    if (inputName) {
        nvinfer1::Dims4 tmplDims{1, 1, template_size_, template_size_};
        backboneCtx_->setInputShape(inputName, tmplDims);
    }

    // Bind and execute backbone for template
    for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
        const char* name = backboneEngine_->getIOTensorName(i);
        if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            backboneCtx_->setTensorAddress(name, d_template_patch_);
        } else {
            backboneCtx_->setTensorAddress(name, d_template_feat_);
        }
    }
    backboneCtx_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    has_target_ = true;
}

SOTResult NanoTrackTRT::track(const void* gpu_image, int pitch,
                               int img_width, int img_height) {
    SOTResult result;
    if (!has_target_) return result;

    // 1. Crop search patch (255x255, context_factor=4.0 centered on last_det_)
    cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                  d_search_patch_, search_size_,
                  last_det_.cx, last_det_.cy, last_det_.width, last_det_.height,
                  search_context_, stream_);

    // 2. Backbone on search patch
    const char* inputName = nullptr;
    for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
        const char* name = backboneEngine_->getIOTensorName(i);
        if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
            break;
        }
    }
    if (inputName) {
        nvinfer1::Dims4 searchDims{1, 1, search_size_, search_size_};
        backboneCtx_->setInputShape(inputName, searchDims);
    }

    for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
        const char* name = backboneEngine_->getIOTensorName(i);
        if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            backboneCtx_->setTensorAddress(name, d_search_patch_);
        } else {
            backboneCtx_->setTensorAddress(name, d_search_feat_);
        }
    }
    backboneCtx_->enqueueV3(stream_);

    // 3. Head: cross-correlation
    // Head inputs: template_feat + search_feat, outputs: cls + reg
    int headTensorIdx = 0;
    for (int i = 0; i < headEngine_->getNbIOTensors(); i++) {
        const char* name = headEngine_->getIOTensorName(i);
        if (headEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            // First input = template_feat, second = search_feat
            if (headTensorIdx == 0)
                headCtx_->setTensorAddress(name, d_template_feat_);
            else
                headCtx_->setTensorAddress(name, d_search_feat_);
            headTensorIdx++;
        } else {
            // Output: cls or reg
            std::string sname(name);
            if (sname.find("cls") != std::string::npos || sname.find("score") != std::string::npos)
                headCtx_->setTensorAddress(name, d_head_cls_);
            else
                headCtx_->setTensorAddress(name, d_head_reg_);
        }
    }
    headCtx_->enqueueV3(stream_);

    // 4. D2H
    if (cls_elements_ > 0)
        cudaMemcpyAsync(h_head_cls_, d_head_cls_, cls_elements_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
    if (reg_elements_ > 0)
        cudaMemcpyAsync(h_head_reg_, d_head_reg_, reg_elements_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 5. Decode
    result = decodeScoreMap(img_width, img_height);
    if (result.valid) {
        last_det_.cx = result.cx;
        last_det_.cy = result.cy;
        last_det_.width = result.width;
        last_det_.height = result.height;
    }

    return result;
}

SOTResult NanoTrackTRT::decodeScoreMap(int img_width, int img_height) {
    SOTResult result;
    if (cls_elements_ <= 0 || score_map_h_ <= 0 || score_map_w_ <= 0)
        return result;

    // Find argmax of score map
    int best_idx = 0;
    float best_score = -1e9f;
    for (int i = 0; i < cls_elements_; i++) {
        // Apply sigmoid
        float score = 1.0f / (1.0f + std::exp(-h_head_cls_[i]));
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_score < 0.1f) return result;  // Too low confidence

    // Convert argmax to grid position
    int grid_y = best_idx / score_map_w_;
    int grid_x = best_idx % score_map_w_;

    // Score map center corresponds to last_det_ center in original image
    // Grid offset → pixel offset in search region
    float center_x = (float)score_map_w_ / 2.0f;
    float center_y = (float)score_map_h_ / 2.0f;

    // Search region size in original image (square, matching cropResizeGPU)
    float s = std::sqrt((last_det_.width * search_context_) *
                        (last_det_.height * search_context_));

    float offset_x = ((float)grid_x - center_x) / (float)score_map_w_ * s;
    float offset_y = ((float)grid_y - center_y) / (float)score_map_h_ * s;

    result.cx = last_det_.cx + offset_x;
    result.cy = last_det_.cy + offset_y;

    // Size from regression (if available)
    if (reg_elements_ >= 4) {
        // reg format: [N, 4, H, W] — 平面布局, 4个通道各自包含完整空间图
        int spatial = score_map_h_ * score_map_w_;
        if (spatial > 0 && 3 * spatial + best_idx < reg_elements_) {
            float dx1 = h_head_reg_[0 * spatial + best_idx];  // left
            float dy1 = h_head_reg_[1 * spatial + best_idx];  // top
            float dx2 = h_head_reg_[2 * spatial + best_idx];  // right
            float dy2 = h_head_reg_[3 * spatial + best_idx];  // bottom
            // Convert relative offsets to absolute size
            float pred_w = (dx1 + dx2) * s / (float)score_map_w_;
            float pred_h = (dy1 + dy2) * s / (float)score_map_h_;
            // Smooth size change
            result.width = target_sz_[0] * 0.6f + pred_w * 0.4f;
            result.height = target_sz_[1] * 0.6f + pred_h * 0.4f;
            target_sz_[0] = result.width;
            target_sz_[1] = result.height;
        } else {
            result.width = target_sz_[0];
            result.height = target_sz_[1];
        }
    } else {
        result.width = target_sz_[0];
        result.height = target_sz_[1];
    }

    // Clamp to image bounds
    result.cx = std::clamp(result.cx, 0.0f, (float)img_width);
    result.cy = std::clamp(result.cy, 0.0f, (float)img_height);
    result.width = std::max(result.width, 5.0f);
    result.height = std::max(result.height, 5.0f);

    result.confidence = best_score;
    result.valid = true;

    return result;
}

void NanoTrackTRT::reset() {
    has_target_ = false;
    last_det_ = Detection();
    target_sz_[0] = 0;
    target_sz_[1] = 0;
}

}  // namespace stereo3d
