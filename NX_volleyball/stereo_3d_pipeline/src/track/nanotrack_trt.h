/**
 * @file nanotrack_trt.h
 * @brief NanoTrack TensorRT 推理 (SOT 补帧)
 *
 * NanoTrack 架构:
 *   - Backbone: 共享特征提取, 支持两种模式:
 *     a) 单 dynamic-shape backbone (1ch, 同时用于 template/search)
 *     b) 双 fixed-shape backbone (3ch, template + search 分别一个引擎)
 *   - Head: Cross-correlation → score map + bbox offset
 *
 * 推理时序 (~0.5–0.7ms on Orin NX GPU):
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

    /**
     * @brief 初始化 (单 backbone, 动态 shape, 1ch)
     * @param backbone_engine_path backbone 引擎 (支持 127x127 和 255x255)
     * @param head_engine_path head 引擎
     */
    bool init(const std::string& backbone_engine_path,
              const std::string& head_engine_path,
              cudaStream_t stream) override;

    /**
     * @brief 初始化 (双 backbone, 固定 shape, 3ch)
     * @param template_engine_path template backbone 引擎 (127x127)
     * @param search_engine_path search backbone 引擎 (255x255)
     * @param head_engine_path head 引擎
     */
    bool initDualBackbone(const std::string& template_engine_path,
                          const std::string& search_engine_path,
                          const std::string& head_engine_path,
                          cudaStream_t stream);

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

    // Mode: single dynamic backbone or dual fixed backbone
    bool dual_backbone_ = false;
    int input_channels_ = 1;  // 1 for single dynamic, 3 for dual fixed

    // Backbone engines + context (dual mode: template + search separate)
    nvinfer1::ICudaEngine* backboneEngine_ = nullptr;        // single-mode (dynamic)
    nvinfer1::IExecutionContext* backboneCtx_ = nullptr;
    nvinfer1::ICudaEngine* templateEngine_ = nullptr;        // dual-mode template
    nvinfer1::IExecutionContext* templateCtx_ = nullptr;
    nvinfer1::ICudaEngine* searchEngine_ = nullptr;          // dual-mode search
    nvinfer1::IExecutionContext* searchCtx_ = nullptr;

    // Head engine + context
    nvinfer1::ICudaEngine* headEngine_ = nullptr;
    nvinfer1::IExecutionContext* headCtx_ = nullptr;

    // GPU buffers
    float* d_template_patch_ = nullptr;   // [1, C, 127, 127]
    float* d_search_patch_   = nullptr;   // [1, C, 255, 255]
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
