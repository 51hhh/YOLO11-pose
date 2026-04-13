/**
 * @file nanotrack_trt.h
 * @brief NanoTrack TensorRT 推理 (SOT 补帧)
 *
 * NanoTrack 架构:
 *   - Backbone: 共享特征提取 (template 127x127, search 255x255)
 *   - Head: Cross-correlation → score map + bbox offset
 *   - 两个独立 TRT 引擎
 *
 * 推理时序 (~0.5ms on Orin NX GPU):
 *   crop_resize → backbone(search) → head(template_feat, search_feat) → decode
 *   template backbone 在 setTarget() 中一次性计算并缓存
 */

#ifndef STEREO_3D_PIPELINE_NANOTRACK_TRT_H_
#define STEREO_3D_PIPELINE_NANOTRACK_TRT_H_

#include "sot_tracker.h"
#include "crop_resize.h"
#include <NvInfer.h>
#include <string>

namespace stereo3d {

class NanoTrackTRT : public SOTTracker {
public:
    NanoTrackTRT();
    ~NanoTrackTRT() override;

    bool init(const std::string& backbone_engine_path,
              const std::string& head_engine_path,
              cudaStream_t stream) override;

    void setTarget(const void* gpu_image, int pitch,
                   int img_width, int img_height,
                   const Detection& det) override;

    SOTResult track(const void* gpu_image, int pitch,
                    int img_width, int img_height) override;

    void reset() override;
    bool hasTarget() const override { return has_target_; }
    int getTemplateSize() const override { return template_size_; }
    int getSearchSize() const override { return search_size_; }

private:
    bool loadEngine(const std::string& path,
                    nvinfer1::ICudaEngine*& engine,
                    nvinfer1::IExecutionContext*& ctx);
    void allocateBuffers();
    void freeBuffers();
    SOTResult decodeScoreMap(int img_width, int img_height);

    // TRT runtime (shared)
    nvinfer1::IRuntime* runtime_ = nullptr;

    // Backbone engine + context
    nvinfer1::ICudaEngine* backboneEngine_ = nullptr;
    nvinfer1::IExecutionContext* backboneCtx_ = nullptr;

    // Head engine + context
    nvinfer1::ICudaEngine* headEngine_ = nullptr;
    nvinfer1::IExecutionContext* headCtx_ = nullptr;

    // GPU buffers
    float* d_template_patch_ = nullptr;   // [1, 1, 127, 127]
    float* d_search_patch_   = nullptr;   // [1, 1, 255, 255]
    float* d_template_feat_  = nullptr;   // backbone template output
    float* d_search_feat_    = nullptr;   // backbone search output
    float* d_head_cls_       = nullptr;   // head cls output (score map)
    float* d_head_reg_       = nullptr;   // head reg output (bbox offsets)
    float* h_head_cls_       = nullptr;   // pinned host cls
    float* h_head_reg_       = nullptr;   // pinned host reg

    // Sizes
    static constexpr int template_size_ = 127;
    static constexpr int search_size_ = 255;
    int template_feat_elements_ = 0;
    int search_feat_elements_ = 0;
    int score_map_h_ = 0, score_map_w_ = 0;
    int cls_elements_ = 0;
    int reg_elements_ = 0;

    // Context factors
    static constexpr float template_context_ = 2.0f;
    static constexpr float search_context_ = 4.0f;

    // State
    cudaStream_t stream_ = nullptr;
    bool has_target_ = false;
    Detection last_det_;
    float target_sz_[2] = {0, 0};  // [w, h] for scale estimation
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NANOTRACK_TRT_H_
