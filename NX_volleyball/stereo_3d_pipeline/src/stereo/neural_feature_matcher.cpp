/**
 * @file neural_feature_matcher.cpp
 * @brief TensorRT engine boundary for learned ROI feature matching.
 */

#include "neural_feature_matcher.h"

#include "utils/logger.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <vector>

namespace stereo3d {

namespace {

class NeuralFeatureTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) {
            LOG_WARN("[NeuralFeatureTRT] %s", msg);
        }
    }
};

NeuralFeatureTrtLogger gLogger;

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

}  // namespace

NeuralFeatureBackend parseNeuralFeatureBackend(const std::string& name) {
    const std::string v = lowerCopy(name);
    if (v == "aliked") return NeuralFeatureBackend::ALIKED;
    if (v == "superpoint_lightglue" || v == "superpoint+lightglue" ||
        v == "superpoint") {
        return NeuralFeatureBackend::SUPERPOINT_LIGHTGLUE;
    }
    return NeuralFeatureBackend::XFEAT;
}

const char* neuralFeatureBackendName(NeuralFeatureBackend backend) {
    switch (backend) {
    case NeuralFeatureBackend::XFEAT: return "xfeat";
    case NeuralFeatureBackend::ALIKED: return "aliked";
    case NeuralFeatureBackend::SUPERPOINT_LIGHTGLUE: return "superpoint_lightglue";
    }
    return "unknown";
}

NeuralFeatureMatcher::NeuralFeatureMatcher() = default;

NeuralFeatureMatcher::~NeuralFeatureMatcher() {
    destroyEngine(extractor_);
    destroyEngine(matcher_);
    destroyEngine(fused_);
}

bool NeuralFeatureMatcher::init(const NeuralFeatureConfig& config,
                                float focal, float baseline, int max_disparity) {
    config_ = config;
    config_.backend = parseNeuralFeatureBackend(config.backend_name);
    focal_ = focal;
    baseline_ = baseline;
    max_disparity_ = max_disparity;
    ready_ = false;

    if (!config_.enabled) {
        return true;
    }
    if (!validateConfig()) {
        return false;
    }

    if (!config_.fused_engine_path.empty()) {
        if (!loadEngine(config_.fused_engine_path, fused_)) {
            return false;
        }
    } else {
        if (!loadEngine(config_.extractor_engine_path, extractor_)) {
            return false;
        }
        if (!config_.matcher_engine_path.empty() &&
            !loadEngine(config_.matcher_engine_path, matcher_)) {
            return false;
        }
    }

    ready_ = true;
    LOG_INFO("Neural feature matcher ready: backend=%s roi=%d top_k=%d min_matches=%d",
             neuralFeatureBackendName(config_.backend),
             config_.roi_size,
             config_.top_k,
             config_.min_matches);
    return true;
}

bool NeuralFeatureMatcher::validateConfig() const {
    if (config_.roi_size < 64 || config_.roi_size > 512) {
        LOG_ERROR("neural_feature_matching.roi_size=%d out of range [64,512]",
                  config_.roi_size);
        return false;
    }
    if (config_.top_k < 8 || config_.top_k > 1024) {
        LOG_ERROR("neural_feature_matching.top_k=%d out of range [8,1024]",
                  config_.top_k);
        return false;
    }
    if (config_.min_matches < 1 || config_.min_matches > config_.top_k) {
        LOG_ERROR("neural_feature_matching.min_matches=%d out of range [1,top_k]",
                  config_.min_matches);
        return false;
    }
    if (config_.max_y_error_px <= 0.0f || config_.max_disp_delta_px <= 0.0f) {
        LOG_ERROR("neural_feature_matching geometry gates must be positive");
        return false;
    }
    if (config_.fused_engine_path.empty() && config_.extractor_engine_path.empty()) {
        LOG_ERROR("neural_feature_matching requires extractor_engine_path or fused_engine_path");
        return false;
    }
    if (config_.backend == NeuralFeatureBackend::SUPERPOINT_LIGHTGLUE &&
        config_.fused_engine_path.empty() &&
        config_.matcher_engine_path.empty()) {
        LOG_WARN("SuperPoint backend configured without matcher_engine_path; "
                 "runtime will use descriptor/geometry matching if implemented");
    }
    return true;
}

bool NeuralFeatureMatcher::loadEngine(const std::string& path, TrtEngine& out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open neural feature engine: %s", path.c_str());
        return false;
    }

    const std::streampos end_pos = file.tellg();
    if (end_pos <= 0) {
        LOG_ERROR("Invalid neural feature engine file: %s", path.c_str());
        return false;
    }
    const size_t file_size = static_cast<size_t>(end_pos);
    file.seekg(0);
    std::vector<char> data(file_size);
    file.read(data.data(), static_cast<std::streamsize>(file_size));
    if (file.gcount() != static_cast<std::streamsize>(file_size)) {
        LOG_ERROR("Failed to read neural feature engine: %s", path.c_str());
        return false;
    }

    out.runtime = nvinfer1::createInferRuntime(gLogger);
    if (!out.runtime) {
        LOG_ERROR("createInferRuntime failed for %s", path.c_str());
        return false;
    }
    out.engine = out.runtime->deserializeCudaEngine(data.data(), file_size);
    if (!out.engine) {
        LOG_ERROR("deserializeCudaEngine failed for %s", path.c_str());
        destroyEngine(out);
        return false;
    }
    out.context = out.engine->createExecutionContext();
    if (!out.context) {
        LOG_ERROR("createExecutionContext failed for %s", path.c_str());
        destroyEngine(out);
        return false;
    }
    out.path = path;

    const int nb = out.engine->getNbIOTensors();
    LOG_INFO("Neural feature engine loaded: %s (%d I/O tensors)", path.c_str(), nb);
    for (int i = 0; i < nb; ++i) {
        const char* name = out.engine->getIOTensorName(i);
        const auto mode = out.engine->getTensorIOMode(name);
        const auto dims = out.engine->getTensorShape(name);
        LOG_INFO("  %s tensor[%d]: %s dims=%d",
                 mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output",
                 i, name, dims.nbDims);
    }
    return true;
}

void NeuralFeatureMatcher::destroyEngine(TrtEngine& e) {
#if NV_TENSORRT_MAJOR >= 10
    if (e.context) { delete e.context; e.context = nullptr; }
    if (e.engine) { delete e.engine; e.engine = nullptr; }
    if (e.runtime) { delete e.runtime; e.runtime = nullptr; }
#else
    if (e.context) { e.context->destroy(); e.context = nullptr; }
    if (e.engine) { e.engine->destroy(); e.engine = nullptr; }
    if (e.runtime) { e.runtime->destroy(); e.runtime = nullptr; }
#endif
    e.path.clear();
}

NeuralFeatureMatchResult NeuralFeatureMatcher::matchGpuRoi(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_width, int img_height,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disparity,
    cudaStream_t stream) {
    (void)left_gpu;
    (void)left_pitch;
    (void)right_gpu;
    (void)right_pitch;
    (void)img_width;
    (void)img_height;
    (void)left_det;
    (void)right_det;
    (void)initial_disparity;
    (void)stream;

    NeuralFeatureMatchResult out;
    if (!ready_) {
        out.status = "not_ready";
        return out;
    }
    out.status = "tensor_binding_not_implemented";
    return out;
}

}  // namespace stereo3d
