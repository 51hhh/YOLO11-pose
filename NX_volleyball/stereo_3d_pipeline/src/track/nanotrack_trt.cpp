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
    if (templateCtx_) { delete templateCtx_; templateCtx_ = nullptr; }
    if (templateEngine_) { delete templateEngine_; templateEngine_ = nullptr; }
    if (searchCtx_) { delete searchCtx_; searchCtx_ = nullptr; }
    if (searchEngine_) { delete searchEngine_; searchEngine_ = nullptr; }
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

// Helper: query output dimensions from an engine
static int queryOutputElements(nvinfer1::ICudaEngine* engine, const char* tag) {
    int total = 0;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = engine->getTensorShape(name);
            int elements = 1;
            for (int d = 0; d < dims.nbDims; d++)
                elements *= dims.d[d];
            LOG_INFO("[NanoTrack] %s output: %s -> %d elements", tag, name, elements);
            total = std::max(total, elements);
        }
    }
    return total;
}

// Helper: query head outputs (cls + reg)
static void queryHeadOutputs(nvinfer1::ICudaEngine* engine,
                              int& cls_elements, int& reg_elements,
                              int& score_h, int& score_w) {
    // First pass: identify outputs
    int out_idx = 0;
    struct OutInfo { std::string name; int elements; int h; int w; };
    std::vector<OutInfo> outs;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = engine->getTensorShape(name);
            OutInfo info;
            info.name = name;
            info.elements = 1;
            for (int d = 0; d < dims.nbDims; d++)
                info.elements *= dims.d[d];
            info.h = (dims.nbDims >= 2) ? dims.d[dims.nbDims - 2] : 0;
            info.w = (dims.nbDims >= 2) ? dims.d[dims.nbDims - 1] : 0;
            outs.push_back(info);
        }
    }
    // Match by name or by size (smaller = cls, larger = reg)
    for (auto& o : outs) {
        bool is_cls = o.name.find("cls") != std::string::npos ||
                      o.name.find("score") != std::string::npos;
        // Fallback: if name has no keyword, first output (output1) = cls, second (output2) = reg
        if (!is_cls && outs.size() == 2 && &o == &outs[0]) is_cls = true;

        if (is_cls) {
            cls_elements = o.elements;
            score_h = o.h; score_w = o.w;
            LOG_INFO("[NanoTrack] Head cls: %s -> %d elements, map %dx%d", o.name.c_str(), o.elements, o.h, o.w);
        } else {
            reg_elements = o.elements;
            LOG_INFO("[NanoTrack] Head reg: %s -> %d elements", o.name.c_str(), o.elements);
        }
    }
}

bool NanoTrackTRT::init(const std::string& backbone_path,
                         const std::string& head_path,
                         cudaStream_t stream) {
    stream_ = stream;
    dual_backbone_ = false;
    input_channels_ = 1;

    runtime_ = nvinfer1::createInferRuntime(sTrtLogger);
    if (!runtime_) {
        LOG_ERROR("[NanoTrack] Failed to create TensorRT runtime");
        return false;
    }

    if (!loadEngine(backbone_path, backboneEngine_, backboneCtx_)) return false;
    if (!loadEngine(head_path, headEngine_, headCtx_)) return false;

    // Detect input channels from backbone
    for (int i = 0; i < backboneEngine_->getNbIOTensors(); i++) {
        const char* name = backboneEngine_->getIOTensorName(i);
        if (backboneEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            auto dims = backboneEngine_->getTensorShape(name);
            if (dims.nbDims >= 4) input_channels_ = dims.d[1];
            break;
        }
    }
    LOG_INFO("[NanoTrack] Single backbone mode, input_channels=%d", input_channels_);

    template_feat_elements_ = queryOutputElements(backboneEngine_, "Backbone");
    search_feat_elements_ = template_feat_elements_;

    queryHeadOutputs(headEngine_, cls_elements_, reg_elements_, score_map_h_, score_map_w_);
    // Re-query search feat size for dynamic-shape backbone
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
            auto minDims = backboneEngine_->getProfileShape(inputName, 0, nvinfer1::OptProfileSelector::kMIN);
            auto maxDims = backboneEngine_->getProfileShape(inputName, 0, nvinfer1::OptProfileSelector::kMAX);
            if (minDims.nbDims > 0 && maxDims.d[maxDims.nbDims-1] >= search_size_) {
                nvinfer1::Dims4 searchDims{1, input_channels_, search_size_, search_size_};
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
    LOG_INFO("[NanoTrack] Initialized (single backbone): %s, head=%s",
             backbone_path.c_str(), head_path.c_str());
    return true;
}

bool NanoTrackTRT::initDualBackbone(const std::string& template_path,
                                     const std::string& search_path,
                                     const std::string& head_path,
                                     cudaStream_t stream) {
    stream_ = stream;
    dual_backbone_ = true;
    input_channels_ = 3;  // dual-backbone models are 3ch

    runtime_ = nvinfer1::createInferRuntime(sTrtLogger);
    if (!runtime_) {
        LOG_ERROR("[NanoTrack] Failed to create TensorRT runtime");
        return false;
    }

    if (!loadEngine(template_path, templateEngine_, templateCtx_)) return false;
    if (!loadEngine(search_path, searchEngine_, searchCtx_)) return false;
    if (!loadEngine(head_path, headEngine_, headCtx_)) return false;

    // Detect actual input channels
    for (int i = 0; i < templateEngine_->getNbIOTensors(); i++) {
        const char* name = templateEngine_->getIOTensorName(i);
        if (templateEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            auto dims = templateEngine_->getTensorShape(name);
            if (dims.nbDims >= 4) input_channels_ = dims.d[1];
            break;
        }
    }
    LOG_INFO("[NanoTrack] Dual backbone mode, input_channels=%d", input_channels_);

    template_feat_elements_ = queryOutputElements(templateEngine_, "Backbone-Template");
    search_feat_elements_ = queryOutputElements(searchEngine_, "Backbone-Search");
    queryHeadOutputs(headEngine_, cls_elements_, reg_elements_, score_map_h_, score_map_w_);

    allocateBuffers();
    LOG_INFO("[NanoTrack] Initialized (dual backbone): tmpl=%s, search=%s, head=%s",
             template_path.c_str(), search_path.c_str(), head_path.c_str());
    return true;
}

void NanoTrackTRT::allocateBuffers() {
    int tmpl_pixels = input_channels_ * template_size_ * template_size_;
    int srch_pixels = input_channels_ * search_size_ * search_size_;
    auto chk = [](cudaError_t err, const char* tag) {
        if (err != cudaSuccess) { LOG_ERROR("cudaMalloc %s failed: %s", tag, cudaGetErrorString(err)); return false; }
        return true;
    };
    if (!chk(cudaMalloc(&d_template_patch_, tmpl_pixels * sizeof(float)), "tmpl_patch")) return;
    if (!chk(cudaMalloc(&d_search_patch_, srch_pixels * sizeof(float)), "srch_patch")) return;
    if (!chk(cudaMalloc(&d_template_feat_, template_feat_elements_ * sizeof(float)), "tmpl_feat")) return;
    if (!chk(cudaMalloc(&d_search_feat_, search_feat_elements_ * sizeof(float)), "srch_feat")) return;

    if (cls_elements_ > 0) {
        if (!chk(cudaMalloc(&d_head_cls_, cls_elements_ * sizeof(float)), "head_cls")) return;
        cudaMallocHost(&h_head_cls_, cls_elements_ * sizeof(float));
    }
    if (reg_elements_ > 0) {
        if (!chk(cudaMalloc(&d_head_reg_, reg_elements_ * sizeof(float)), "head_reg")) return;
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

    // Crop template patch
    if (input_channels_ == 3) {
        cropResizeGPU_3ch(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                          d_template_patch_, template_size_,
                          det.cx, det.cy, det.width, det.height,
                          template_context_, stream_);
    } else {
        cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                      d_template_patch_, template_size_,
                      det.cx, det.cy, det.width, det.height,
                      template_context_, stream_);
    }

    // Run backbone on template
    nvinfer1::ICudaEngine* engine = dual_backbone_ ? templateEngine_ : backboneEngine_;
    nvinfer1::IExecutionContext* ctx = dual_backbone_ ? templateCtx_ : backboneCtx_;

    if (!dual_backbone_) {
        // Set input shape for dynamic-shape backbone
        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                nvinfer1::Dims4 tmplDims{1, input_channels_, template_size_, template_size_};
                ctx->setInputShape(name, tmplDims);
                break;
            }
        }
    }

    int bbOutCount = 0;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            ctx->setTensorAddress(name, d_template_patch_);
        } else {
            ctx->setTensorAddress(name, d_template_feat_);
            bbOutCount++;
        }
    }
    if (bbOutCount != 1) LOG_WARN("[NanoTrack] Template backbone has %d outputs (expected 1)", bbOutCount);
    ctx->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    has_target_ = true;
}

SOTResult NanoTrackTRT::track(const void* gpu_image, int pitch,
                               int img_width, int img_height) {
    SOTResult result;
    if (!has_target_) return result;

    // 1. Crop search patch
    if (input_channels_ == 3) {
        cropResizeGPU_3ch(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                          d_search_patch_, search_size_,
                          last_det_.cx, last_det_.cy, last_det_.width, last_det_.height,
                          search_context_, stream_);
    } else {
        cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                      d_search_patch_, search_size_,
                      last_det_.cx, last_det_.cy, last_det_.width, last_det_.height,
                      search_context_, stream_);
    }

    // 2. Backbone on search patch
    nvinfer1::ICudaEngine* engine = dual_backbone_ ? searchEngine_ : backboneEngine_;
    nvinfer1::IExecutionContext* ctx = dual_backbone_ ? searchCtx_ : backboneCtx_;

    if (!dual_backbone_) {
        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                nvinfer1::Dims4 searchDims{1, input_channels_, search_size_, search_size_};
                ctx->setInputShape(name, searchDims);
                break;
            }
        }
    }

    int srchOutCount = 0;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            ctx->setTensorAddress(name, d_search_patch_);
        } else {
            ctx->setTensorAddress(name, d_search_feat_);
            srchOutCount++;
        }
    }
    if (srchOutCount != 1) LOG_WARN("[NanoTrack] Search backbone has %d outputs (expected 1)", srchOutCount);
    ctx->enqueueV3(stream_);

    // 3. Head: cross-correlation
    // Head inputs: template_feat + search_feat, outputs: cls + reg
    int headOutputIdx = 0;
    int headNumOutputs = 0;
    for (int i = 0; i < headEngine_->getNbIOTensors(); i++) {
        const char* name = headEngine_->getIOTensorName(i);
        if (headEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) headNumOutputs++;
    }
    int headInputIdx = 0;
    for (int i = 0; i < headEngine_->getNbIOTensors(); i++) {
        const char* name = headEngine_->getIOTensorName(i);
        if (headEngine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            // Match by name (template/search keyword), fallback by position
            std::string sname(name);
            bool is_template = sname.find("template") != std::string::npos ||
                               sname.find("input1") != std::string::npos;
            bool is_search   = sname.find("search") != std::string::npos ||
                               sname.find("input2") != std::string::npos;
            if (!is_template && !is_search) {
                // No keyword: first encountered input = template, second = search
                is_template = (headInputIdx == 0);
            }
            headCtx_->setTensorAddress(name, is_template ? d_template_feat_ : d_search_feat_);
            headInputIdx++;
        } else {
            // Output: match by name, fallback by position (first=cls, second=reg)
            std::string sname(name);
            bool is_cls = sname.find("cls") != std::string::npos ||
                          sname.find("score") != std::string::npos;
            if (!is_cls && headNumOutputs == 2 && headOutputIdx == 0) {
                is_cls = true;
                LOG_WARN("[NanoTrack] Head output '%s' has no cls/score keyword, assuming cls by position", name);
            }
            headCtx_->setTensorAddress(name, is_cls ? d_head_cls_ : d_head_reg_);
            headOutputIdx++;
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
