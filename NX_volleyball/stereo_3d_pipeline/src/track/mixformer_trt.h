/**
 * @file mixformer_trt.h
 * @brief MixFormerV2-small TensorRT 推理 (SOT 补帧)
 *
 * MixFormerV2 架构:
 *   - 单一 TRT 引擎, 输入 template + search (分别绑定)
 *   - 输出: score + bbox [cx, cy, w, h] (normalized)
 *   - 推理 ~1ms on Orin NX GPU
 */

#ifndef STEREO_3D_PIPELINE_MIXFORMER_TRT_H_
#define STEREO_3D_PIPELINE_MIXFORMER_TRT_H_

#include "sot_tracker.h"
#include "crop_resize.h"
#include <NvInfer.h>
#include <string>

namespace stereo3d {

class MixFormerTRT : public SOTTracker {
public:
    MixFormerTRT();
    ~MixFormerTRT() override;

    bool init(const std::string& engine_path,
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
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    float* d_template_patch_ = nullptr;   // [1, 1, 128, 128]
    float* d_search_patch_   = nullptr;   // [1, 1, 256, 256]
    float* d_output_         = nullptr;   // [score, cx, cy, w, h]
    float* h_output_         = nullptr;   // pinned host

    static constexpr int template_size_ = 128;
    static constexpr int search_size_ = 256;
    static constexpr float template_context_ = 2.0f;
    static constexpr float search_context_ = 4.0f;

    int output_elements_ = 0;
    cudaStream_t stream_ = nullptr;
    bool has_target_ = false;
    Detection last_det_;
    float target_sz_[2] = {0, 0};
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_MIXFORMER_TRT_H_
