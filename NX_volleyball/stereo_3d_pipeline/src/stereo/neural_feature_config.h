#ifndef STEREO_3D_PIPELINE_NEURAL_FEATURE_CONFIG_H_
#define STEREO_3D_PIPELINE_NEURAL_FEATURE_CONFIG_H_

#include <string>

namespace stereo3d {

enum class NeuralFeatureBackend {
    XFEAT,
    ALIKED,
    SUPERPOINT_LIGHTGLUE,
};

struct NeuralFeatureConfig {
    bool enabled = false;
    NeuralFeatureBackend backend = NeuralFeatureBackend::XFEAT;
    std::string backend_name = "xfeat";

    // XFeat / ALIKED: extractor engine output should be keypoints/descriptors/scores.
    // SuperPoint+LightGlue can use extractor+matcher engines or one fused engine.
    std::string extractor_engine_path;
    std::string matcher_engine_path;
    std::string fused_engine_path;

    int roi_size = 224;
    int top_k = 128;
    int descriptor_dim = 64;
    int realtime_stride = 1;
    int min_matches = 8;
    float max_y_error_px = 2.0f;
    float max_disp_delta_px = 32.0f;
    float final_disp_gate_px = 2.0f;
    float min_score = 0.0f;
    bool use_lightglue = false;
    bool gpu_postprocess = false;
    float match_margin = 0.015f;
    int min_spatial_quadrants = 2;
    float min_spatial_spread_ratio = 0.10f;
    bool final_geometry_gate_enabled = true;
};

NeuralFeatureBackend parseNeuralFeatureBackend(const std::string& name);
const char* neuralFeatureBackendName(NeuralFeatureBackend backend);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NEURAL_FEATURE_CONFIG_H_
