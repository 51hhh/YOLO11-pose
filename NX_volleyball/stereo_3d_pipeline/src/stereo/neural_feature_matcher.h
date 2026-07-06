/**
 * @file neural_feature_matcher.h
 * @brief TensorRT-backed learned ROI feature matching interface.
 *
 * Realtime policy:
 *   - Python is used only for offline correctness tests.
 *   - NX runtime uses pre-exported TensorRT engines with fixed ROI shape.
 *   - Matching output must still pass epipolar/disparity gates before depth use.
 */

#ifndef STEREO_3D_PIPELINE_NEURAL_FEATURE_MATCHER_H_
#define STEREO_3D_PIPELINE_NEURAL_FEATURE_MATCHER_H_

#include "neural_feature_config.h"
#include "neural_feature_direct_gpu_postprocess.h"
#include "neural_feature_xfeat_gpu_postprocess.h"
#include "roi_feature_result.h"
#include "pipeline/detection_types.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stereo3d {

struct NeuralFeaturePointMatch {
    float left_x = 0.0f;
    float left_y = 0.0f;
    float right_x = 0.0f;
    float right_y = 0.0f;
    float disparity = -1.0f;
    float score = 0.0f;
};

struct NeuralFeatureMatchResult {
    bool valid = false;
    std::string status;
    std::vector<NeuralFeaturePointMatch> matches;
    std::vector<SparseFeatureDebugPoint> debug_points;
    float disparity = -1.0f;
    float stddev_px = -1.0f;
    float depth_m = -1.0f;
    float confidence = 0.0f;
    float inference_ms = 0.0f;
};

class NeuralFeatureMatcher {
public:
    NeuralFeatureMatcher();
    ~NeuralFeatureMatcher();

    NeuralFeatureMatcher(const NeuralFeatureMatcher&) = delete;
    NeuralFeatureMatcher& operator=(const NeuralFeatureMatcher&) = delete;

    bool init(const NeuralFeatureConfig& config,
              float focal, float baseline, int max_disparity);

    bool isReady() const { return ready_; }
    const NeuralFeatureConfig& config() const { return config_; }
    bool requiresBgrInput() const;

    NeuralFeatureMatchResult matchGpuRoi(
        const uint8_t* left_gray_gpu, int left_gray_pitch,
        const uint8_t* right_gray_gpu, int right_gray_pitch,
        const uint8_t* left_bgr_gpu, int left_bgr_pitch,
        const uint8_t* right_bgr_gpu, int right_bgr_pitch,
        int img_width, int img_height,
        const Detection& left_det,
        const Detection& right_det,
        float initial_disparity,
        cudaStream_t stream);

private:
    struct TrtEngine {
        struct TensorBuffer {
            std::string name;
            bool is_input = false;
            nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
            nvinfer1::Dims dims{};
            void* device = nullptr;
            size_t bytes = 0;
            size_t elements = 0;
            std::vector<float> host_float;
            std::vector<int32_t> host_int32;
        };

        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        std::string path;
        std::vector<TensorBuffer> tensors;
        bool bindings_ready = false;
        int input_count = 0;
        int output_count = 0;
    };

    NeuralFeatureConfig config_;
    float focal_ = 0.0f;
    float baseline_ = 0.0f;
    int max_disparity_ = 0;
    bool ready_ = false;

    TrtEngine extractor_;
    TrtEngine matcher_;
    TrtEngine fused_;
    XFeatGpuWorkspace xfeat_gpu_workspace_;
    DirectFeatureGpuWorkspace direct_gpu_workspace_;
    void* plugin_library_handle_ = nullptr;

    bool loadPluginLibrary();
    void closePluginLibrary();
    bool loadEngine(const std::string& path, TrtEngine& out);
    void destroyEngine(TrtEngine& engine);
    bool validateConfig() const;
    bool prepareEngineBindings(TrtEngine& engine);

    NeuralFeatureMatchResult matchXFeatExtractorGpuRoi(
        const uint8_t* left_gray_gpu, int left_gray_pitch,
        const uint8_t* right_gray_gpu, int right_gray_pitch,
        const uint8_t* left_bgr_gpu, int left_bgr_pitch,
        const uint8_t* right_bgr_gpu, int right_bgr_pitch,
        int img_width, int img_height,
        const Detection& left_det,
        const Detection& right_det,
        float initial_disparity,
        cudaStream_t stream);

    NeuralFeatureMatchResult matchDirectExtractorGpuRoi(
        const uint8_t* left_gray_gpu, int left_gray_pitch,
        const uint8_t* right_gray_gpu, int right_gray_pitch,
        const uint8_t* left_bgr_gpu, int left_bgr_pitch,
        const uint8_t* right_bgr_gpu, int right_bgr_pitch,
        int img_width, int img_height,
        const Detection& left_det,
        const Detection& right_det,
        float initial_disparity,
        cudaStream_t stream);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NEURAL_FEATURE_MATCHER_H_
