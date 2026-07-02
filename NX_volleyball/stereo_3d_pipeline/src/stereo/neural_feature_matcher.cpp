/**
 * @file neural_feature_matcher.cpp
 * @brief TensorRT engine boundary for learned ROI feature matching.
 */

#include "neural_feature_matcher.h"

#include "track/crop_resize.h"
#include "utils/logger.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <utility>
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

size_t dataTypeBytes(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 0;
    }
}

bool hasDynamicDim(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return true;
    }
    return false;
}

size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) return 0;
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return 0;
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

int tensorChannels(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 4) return static_cast<int>(dims.d[1]);
    if (dims.nbDims == 3) return static_cast<int>(dims.d[0]);
    return 0;
}

int tensorHeight(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 4) return static_cast<int>(dims.d[2]);
    if (dims.nbDims == 3) return static_cast<int>(dims.d[1]);
    return 0;
}

int tensorWidth(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 4) return static_cast<int>(dims.d[3]);
    if (dims.nbDims == 3) return static_cast<int>(dims.d[2]);
    return 0;
}

float medianOf(std::vector<float>& values) {
    if (values.empty()) return -1.0f;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n & 1u) ? values[n / 2] : 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

float bilinearChannelSample(const std::vector<float>& chw,
                            int channels, int height, int width,
                            int channel, float x, float y) {
    if (channels <= 0 || height <= 0 || width <= 0 ||
        channel < 0 || channel >= channels || chw.empty()) {
        return 0.0f;
    }
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) {
        const size_t idx = (static_cast<size_t>(channel) * height +
                            static_cast<size_t>(yy)) * width +
                           static_cast<size_t>(xx);
        return idx < chw.size() ? chw[idx] : 0.0f;
    };
    const float v00 = at(x0, y0);
    const float v10 = at(x1, y0);
    const float v01 = at(x0, y1);
    const float v11 = at(x1, y1);
    return v00 * (1.0f - fx) * (1.0f - fy) +
           v10 * fx * (1.0f - fy) +
           v01 * (1.0f - fx) * fy +
           v11 * fx * fy;
}

struct XFeatRawOutput {
    std::vector<float> feats;
    std::vector<float> keypoints;
    std::vector<float> heatmap;
    int feat_h = 0;
    int feat_w = 0;
};

struct XFeatFeature {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
    std::vector<float> descriptor;
};

struct XFeatCandidate {
    float x = 0.0f;
    float y = 0.0f;
    float feat_x = 0.0f;
    float feat_y = 0.0f;
    float score = 0.0f;
};

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
    if (!prepareEngineBindings(out)) {
        LOG_WARN("Neural feature engine loaded but has unsupported/dynamic tensor "
                 "bindings for generic realtime parser: %s", path.c_str());
    }
    return true;
}

void NeuralFeatureMatcher::destroyEngine(TrtEngine& e) {
    for (auto& tensor : e.tensors) {
        if (tensor.device) {
            cudaFree(tensor.device);
            tensor.device = nullptr;
        }
    }
    e.tensors.clear();
    e.bindings_ready = false;
    e.input_count = 0;
    e.output_count = 0;
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

bool NeuralFeatureMatcher::prepareEngineBindings(TrtEngine& e) {
    if (!e.engine || !e.context) return false;

    e.tensors.clear();
    e.input_count = 0;
    e.output_count = 0;
    e.bindings_ready = false;

    const int nb = e.engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* cname = e.engine->getIOTensorName(i);
        if (!cname) return false;
        TrtEngine::TensorBuffer tensor;
        tensor.name = cname;
        tensor.is_input =
            e.engine->getTensorIOMode(cname) == nvinfer1::TensorIOMode::kINPUT;
        tensor.dtype = e.engine->getTensorDataType(cname);
        tensor.dims = e.engine->getTensorShape(cname);

        if (tensor.is_input && hasDynamicDim(tensor.dims) && tensor.dims.nbDims == 4) {
            int c = static_cast<int>(tensor.dims.d[1]);
            if (c <= 0) c = 1;
            nvinfer1::Dims4 shape{1, c, config_.roi_size, config_.roi_size};
            if (!e.context->setInputShape(cname, shape)) {
                LOG_WARN("Neural feature: setInputShape failed for %s", cname);
                return false;
            }
            tensor.dims = e.context->getTensorShape(cname);
        }

        if (!tensor.is_input && hasDynamicDim(tensor.dims)) {
            tensor.dims = e.context->getTensorShape(cname);
        }

        tensor.elements = volume(tensor.dims);
        const size_t dtype_bytes = dataTypeBytes(tensor.dtype);
        tensor.bytes = tensor.elements * dtype_bytes;
        if (tensor.elements == 0 || dtype_bytes == 0 || tensor.bytes == 0) {
            LOG_WARN("Neural feature: unsupported tensor shape/type for %s", cname);
            return false;
        }
        if (cudaMalloc(&tensor.device, tensor.bytes) != cudaSuccess) {
            LOG_WARN("Neural feature: cudaMalloc failed for tensor %s", cname);
            return false;
        }
        if (!e.context->setTensorAddress(tensor.name.c_str(), tensor.device)) {
            LOG_WARN("Neural feature: setTensorAddress failed for %s", cname);
            return false;
        }
        if (tensor.is_input) {
            ++e.input_count;
        } else {
            ++e.output_count;
            if (tensor.dtype == nvinfer1::DataType::kFLOAT) {
                tensor.host_float.resize(tensor.elements);
            }
        }
        e.tensors.push_back(std::move(tensor));
    }

    e.bindings_ready = e.input_count > 0 && e.output_count > 0;
    return e.bindings_ready;
}

NeuralFeatureMatchResult NeuralFeatureMatcher::matchXFeatExtractorGpuRoi(
    const uint8_t* left_gray_gpu, int left_gray_pitch,
    const uint8_t* right_gray_gpu, int right_gray_pitch,
    int img_width, int img_height,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disparity,
    cudaStream_t stream) {
    NeuralFeatureMatchResult out;
    if (!extractor_.engine || !extractor_.context || !extractor_.bindings_ready) {
        out.status = "extractor_not_ready";
        return out;
    }
    if (config_.backend != NeuralFeatureBackend::XFEAT) {
        out.status = "split_backend_not_implemented";
        return out;
    }

    TrtEngine::TensorBuffer* input = nullptr;
    std::vector<TrtEngine::TensorBuffer*> outputs;
    for (auto& tensor : extractor_.tensors) {
        extractor_.context->setTensorAddress(tensor.name.c_str(), tensor.device);
        if (tensor.is_input) input = &tensor;
        else outputs.push_back(&tensor);
    }
    if (!input || outputs.size() < 3 || input->dtype != nvinfer1::DataType::kFLOAT) {
        out.status = "unsupported_extractor_schema";
        return out;
    }

    auto copy_outputs = [&](XFeatRawOutput& raw) -> bool {
        TrtEngine::TensorBuffer* feats = nullptr;
        TrtEngine::TensorBuffer* keypoints = nullptr;
        TrtEngine::TensorBuffer* heatmap = nullptr;
        for (auto* tensor : outputs) {
            if (tensor->dtype != nvinfer1::DataType::kFLOAT ||
                tensor->host_float.empty()) {
                continue;
            }
            const int c = tensorChannels(tensor->dims);
            if (c == config_.descriptor_dim && !feats) feats = tensor;
            else if (c == 65 && !keypoints) keypoints = tensor;
            else if (c == 1 && !heatmap) heatmap = tensor;
        }
        if (!feats || !keypoints || !heatmap) return false;
        for (auto* tensor : {feats, keypoints, heatmap}) {
            const cudaError_t err = cudaMemcpyAsync(
                tensor->host_float.data(), tensor->device, tensor->bytes,
                cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
        raw.feats = feats->host_float;
        raw.keypoints = keypoints->host_float;
        raw.heatmap = heatmap->host_float;
        raw.feat_h = tensorHeight(feats->dims);
        raw.feat_w = tensorWidth(feats->dims);
        return raw.feat_h > 0 && raw.feat_w > 0 &&
               tensorHeight(keypoints->dims) == raw.feat_h &&
               tensorWidth(keypoints->dims) == raw.feat_w &&
               tensorHeight(heatmap->dims) == raw.feat_h &&
               tensorWidth(heatmap->dims) == raw.feat_w;
    };

    const float context = 1.20f;
    const int input_channels = tensorChannels(input->dims);
    auto run_one = [&](const Detection& det,
                       const uint8_t* gray, int pitch,
                       XFeatRawOutput& raw) -> bool {
        float* dst = static_cast<float*>(input->device);
        if (input_channels == 1) {
            cropResizeGPU(gray, pitch, img_width, img_height,
                          dst, config_.roi_size,
                          det.cx, det.cy, det.width, det.height,
                          context, stream);
        } else if (input_channels == 3) {
            cropResizeGPU_3ch(gray, pitch, img_width, img_height,
                              dst, config_.roi_size,
                              det.cx, det.cy, det.width, det.height,
                              context, stream);
        } else {
            return false;
        }
        if (!extractor_.context->enqueueV3(stream)) return false;
        return copy_outputs(raw);
    };

    const auto start = std::chrono::steady_clock::now();
    XFeatRawOutput left_raw;
    XFeatRawOutput right_raw;
    if (!run_one(left_det, left_gray_gpu, left_gray_pitch, left_raw) ||
        !run_one(right_det, right_gray_gpu, right_gray_pitch, right_raw)) {
        out.status = "extractor_enqueue_or_copy_failed";
        return out;
    }

    auto postprocess = [&](const XFeatRawOutput& raw) {
        std::vector<XFeatCandidate> candidates;
        candidates.reserve(static_cast<size_t>(raw.feat_h) *
                           static_cast<size_t>(raw.feat_w));
        for (int yy = 0; yy < raw.feat_h; ++yy) {
            for (int xx = 0; xx < raw.feat_w; ++xx) {
                float max_logit = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < 65; ++c) {
                    const size_t idx = (static_cast<size_t>(c) * raw.feat_h +
                                        static_cast<size_t>(yy)) * raw.feat_w +
                                       static_cast<size_t>(xx);
                    max_logit = std::max(max_logit, raw.keypoints[idx]);
                }
                float denom = 0.0f;
                float best_prob = 0.0f;
                int best_bin = -1;
                for (int c = 0; c < 65; ++c) {
                    const size_t idx = (static_cast<size_t>(c) * raw.feat_h +
                                        static_cast<size_t>(yy)) * raw.feat_w +
                                       static_cast<size_t>(xx);
                    const float e = std::exp(raw.keypoints[idx] - max_logit);
                    if (c < 64 && e > best_prob) {
                        best_prob = e;
                        best_bin = c;
                    }
                    denom += e;
                }
                if (denom <= 0.0f || best_bin < 0) continue;
                const float prob = best_prob / denom;
                if (prob <= 0.05f) continue;
                const int ox = best_bin & 7;
                const int oy = best_bin >> 3;
                const int x = xx * 8 + ox;
                const int y = yy * 8 + oy;
                if (x >= config_.roi_size || y >= config_.roi_size) continue;
                const float fx = static_cast<float>(raw.feat_w) *
                                 static_cast<float>(x) /
                                 static_cast<float>(std::max(1, config_.roi_size - 1)) -
                                 0.5f;
                const float fy = static_cast<float>(raw.feat_h) *
                                 static_cast<float>(y) /
                                 static_cast<float>(std::max(1, config_.roi_size - 1)) -
                                 0.5f;
                const float reliability =
                    bilinearChannelSample(raw.heatmap, 1, raw.feat_h, raw.feat_w,
                                          0, fx, fy);
                candidates.push_back(XFeatCandidate{
                    static_cast<float>(x),
                    static_cast<float>(y),
                    fx,
                    fy,
                    prob * reliability});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const XFeatCandidate& a, const XFeatCandidate& b) {
                      return a.score > b.score;
                  });
        if (static_cast<int>(candidates.size()) > config_.top_k) {
            candidates.resize(static_cast<size_t>(config_.top_k));
        }

        std::vector<XFeatFeature> features;
        features.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            XFeatFeature f;
            f.x = candidate.x;
            f.y = candidate.y;
            f.score = candidate.score;
            f.descriptor.resize(static_cast<size_t>(config_.descriptor_dim));
            float norm2 = 0.0f;
            for (int c = 0; c < config_.descriptor_dim; ++c) {
                const float d = bilinearChannelSample(raw.feats,
                                                      config_.descriptor_dim,
                                                      raw.feat_h, raw.feat_w,
                                                      c,
                                                      candidate.feat_x,
                                                      candidate.feat_y);
                f.descriptor[static_cast<size_t>(c)] = d;
                norm2 += d * d;
            }
            const float inv_norm =
                norm2 > 1e-12f ? 1.0f / std::sqrt(norm2) : 0.0f;
            for (float& d : f.descriptor) d *= inv_norm;
            features.push_back(std::move(f));
        }
        return features;
    };

    std::vector<XFeatFeature> left_features = postprocess(left_raw);
    std::vector<XFeatFeature> right_features = postprocess(right_raw);
    if (static_cast<int>(left_features.size()) < config_.min_matches ||
        static_cast<int>(right_features.size()) < config_.min_matches) {
        out.status = "not_enough_keypoints";
        return out;
    }

    auto dot = [&](const XFeatFeature& a, const XFeatFeature& b) {
        float s = 0.0f;
        const int n = std::min(static_cast<int>(a.descriptor.size()),
                               static_cast<int>(b.descriptor.size()));
        for (int i = 0; i < n; ++i) {
            s += a.descriptor[static_cast<size_t>(i)] *
                 b.descriptor[static_cast<size_t>(i)];
        }
        return s;
    };

    std::vector<int> left_best(left_features.size(), -1);
    std::vector<float> left_score(left_features.size(), -2.0f);
    std::vector<int> right_best(right_features.size(), -1);
    std::vector<float> right_score(right_features.size(), -2.0f);
    for (size_t i = 0; i < left_features.size(); ++i) {
        for (size_t j = 0; j < right_features.size(); ++j) {
            const float s = dot(left_features[i], right_features[j]);
            if (s > left_score[i]) {
                left_score[i] = s;
                left_best[i] = static_cast<int>(j);
            }
            if (s > right_score[j]) {
                right_score[j] = s;
                right_best[j] = static_cast<int>(i);
            }
        }
    }

    const auto map_to_frame = [&](const Detection& det, const XFeatFeature& f,
                                  float* x, float* y) {
        const float s = std::sqrt(std::max(1.0f, det.width * context *
                                                  det.height * context));
        const float roi_x = det.cx - 0.5f * s;
        const float roi_y = det.cy - 0.5f * s;
        *x = roi_x + (f.x + 0.5f) * s / static_cast<float>(config_.roi_size) - 0.5f;
        *y = roi_y + (f.y + 0.5f) * s / static_cast<float>(config_.roi_size) - 0.5f;
    };

    std::vector<NeuralFeaturePointMatch> candidates;
    std::vector<float> disparities;
    for (size_t i = 0; i < left_features.size(); ++i) {
        const int j = left_best[i];
        if (j < 0 || j >= static_cast<int>(right_features.size()) ||
            right_best[static_cast<size_t>(j)] != static_cast<int>(i)) {
            continue;
        }
        const float score = left_score[i];
        if (score < config_.min_score) continue;
        float lx, ly, rx, ry;
        map_to_frame(left_det, left_features[i], &lx, &ly);
        map_to_frame(right_det, right_features[static_cast<size_t>(j)], &rx, &ry);
        const float disp = lx - rx;
        if (disp <= 0.5f || disp > static_cast<float>(max_disparity_) ||
            std::fabs(ly - ry) > config_.max_y_error_px ||
            std::fabs(disp - initial_disparity) > config_.max_disp_delta_px) {
            continue;
        }
        NeuralFeaturePointMatch m;
        m.left_x = lx;
        m.left_y = ly;
        m.right_x = rx;
        m.right_y = ry;
        m.disparity = disp;
        m.score = score;
        candidates.push_back(m);
        disparities.push_back(disp);
    }

    if (static_cast<int>(disparities.size()) < config_.min_matches) {
        out.status = "not_enough_matches";
        return out;
    }
    const float median = medianOf(disparities);
    std::vector<float> abs_dev;
    abs_dev.reserve(disparities.size());
    for (float d : disparities) abs_dev.push_back(std::fabs(d - median));
    const float mad = medianOf(abs_dev);
    const float gate = std::max(config_.final_disp_gate_px, 1.4826f * mad * 2.5f);
    float sum = 0.0f;
    float sum2 = 0.0f;
    float score_sum = 0.0f;
    for (const auto& m : candidates) {
        if (std::fabs(m.disparity - median) > gate) continue;
        out.matches.push_back(m);
        sum += m.disparity;
        sum2 += m.disparity * m.disparity;
        score_sum += m.score;
    }
    if (static_cast<int>(out.matches.size()) < config_.min_matches) {
        out.status = "not_enough_inliers";
        out.matches.clear();
        return out;
    }
    const float kept = static_cast<float>(out.matches.size());
    out.disparity = sum / kept;
    const float var = std::max(0.0f, sum2 / kept - out.disparity * out.disparity);
    out.stddev_px = std::sqrt(var);
    out.depth_m = focal_ * baseline_ / std::max(0.5f, out.disparity);
    const float support_conf =
        std::min(1.0f, kept / static_cast<float>(std::max(1, config_.min_matches * 2)));
    const float score_conf = std::clamp((score_sum / kept + 1.0f) * 0.5f, 0.0f, 1.0f);
    const float consistency = std::clamp(1.0f / (1.0f + out.stddev_px), 0.0f, 1.0f);
    out.confidence = std::clamp(0.45f * support_conf +
                                0.35f * score_conf +
                                0.20f * consistency,
                                0.0f, 1.0f);
    out.inference_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    out.valid = true;
    out.status = "ok";
    return out;
}

NeuralFeatureMatchResult NeuralFeatureMatcher::matchGpuRoi(
    const uint8_t* left_gray_gpu, int left_gray_pitch,
    const uint8_t* right_gray_gpu, int right_gray_pitch,
    const uint8_t* left_bgr_gpu, int left_bgr_pitch,
    const uint8_t* right_bgr_gpu, int right_bgr_pitch,
    int img_width, int img_height,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disparity,
    cudaStream_t stream) {
    NeuralFeatureMatchResult out;
    if (!ready_) {
        out.status = "not_ready";
        return out;
    }
    if (!left_gray_gpu || !right_gray_gpu ||
        left_gray_pitch <= 0 || right_gray_pitch <= 0 ||
        img_width <= 0 || img_height <= 0 || stream == nullptr) {
        out.status = "invalid_input";
        return out;
    }
    if (!fused_.engine || !fused_.context) {
        if (extractor_.engine && extractor_.context &&
            config_.backend == NeuralFeatureBackend::XFEAT) {
            return matchXFeatExtractorGpuRoi(
                left_gray_gpu, left_gray_pitch,
                right_gray_gpu, right_gray_pitch,
                img_width, img_height,
                left_det, right_det,
                initial_disparity,
                stream);
        }
        out.status = "split_engine_runtime_not_implemented";
        return out;
    }
    if (!fused_.bindings_ready) {
        out.status = "unsupported_tensor_binding";
        return out;
    }

    std::vector<TrtEngine::TensorBuffer*> inputs;
    std::vector<TrtEngine::TensorBuffer*> outputs;
    for (auto& tensor : fused_.tensors) {
        if (tensor.is_input) inputs.push_back(&tensor);
        else outputs.push_back(&tensor);
        fused_.context->setTensorAddress(tensor.name.c_str(), tensor.device);
    }
    if (inputs.empty() || outputs.empty()) {
        out.status = "unsupported_tensor_binding";
        return out;
    }

    const auto start = std::chrono::steady_clock::now();
    const float context = 1.20f;
    auto crop_into = [&](const Detection& det,
                         const uint8_t* gray_gpu, int gray_pitch,
                         const uint8_t* bgr_gpu, int bgr_pitch,
                         TrtEngine::TensorBuffer* tensor,
                         float* dst) -> bool {
        const int c = tensorChannels(tensor->dims);
        const int size = config_.roi_size;
        if (c == 1) {
            cropResizeGPU(gray_gpu, gray_pitch, img_width, img_height,
                          dst, size,
                          det.cx, det.cy, det.width, det.height,
                          context, stream);
            return true;
        }
        if (c == 3) {
            if (!bgr_gpu || bgr_pitch <= 0) {
                return false;
            }
            cropResizeGPU_3ch(bgr_gpu, bgr_pitch, img_width, img_height,
                              dst, size,
                              det.cx, det.cy, det.width, det.height,
                              context, stream);
            return true;
        }
        return false;
    };

    if (inputs.size() == 2) {
        if (inputs[0]->dtype != nvinfer1::DataType::kFLOAT ||
            inputs[1]->dtype != nvinfer1::DataType::kFLOAT ||
            !crop_into(left_det,
                       left_gray_gpu, left_gray_pitch,
                       left_bgr_gpu, left_bgr_pitch,
                       inputs[0],
                       static_cast<float*>(inputs[0]->device)) ||
            !crop_into(right_det,
                       right_gray_gpu, right_gray_pitch,
                       right_bgr_gpu, right_bgr_pitch,
                       inputs[1],
                       static_cast<float*>(inputs[1]->device))) {
            out.status = "unsupported_input_schema";
            return out;
        }
    } else if (inputs.size() == 1) {
        TrtEngine::TensorBuffer* input = inputs[0];
        if (input->dtype != nvinfer1::DataType::kFLOAT ||
            input->dims.nbDims != 4) {
            out.status = "unsupported_input_schema";
            return out;
        }
        const int c = tensorChannels(input->dims);
        const int spatial = config_.roi_size * config_.roi_size;
        float* base = static_cast<float*>(input->device);
        if (c == 2) {
            cropResizeGPU(left_gray_gpu, left_gray_pitch, img_width, img_height,
                          base, config_.roi_size,
                          left_det.cx, left_det.cy, left_det.width, left_det.height,
                          context, stream);
            cropResizeGPU(right_gray_gpu, right_gray_pitch, img_width, img_height,
                          base + spatial, config_.roi_size,
                          right_det.cx, right_det.cy, right_det.width, right_det.height,
                          context, stream);
        } else if (c == 6) {
            if (!left_bgr_gpu || !right_bgr_gpu ||
                left_bgr_pitch <= 0 || right_bgr_pitch <= 0) {
                out.status = "unsupported_input_schema";
                return out;
            }
            cropResizeGPU_3ch(left_bgr_gpu, left_bgr_pitch, img_width, img_height,
                              base, config_.roi_size,
                              left_det.cx, left_det.cy, left_det.width, left_det.height,
                              context, stream);
            cropResizeGPU_3ch(right_bgr_gpu, right_bgr_pitch, img_width, img_height,
                              base + 3 * spatial, config_.roi_size,
                              right_det.cx, right_det.cy, right_det.width, right_det.height,
                              context, stream);
        } else {
            out.status = "unsupported_input_schema";
            return out;
        }
    } else {
        out.status = "unsupported_input_schema";
        return out;
    }

    if (!fused_.context->enqueueV3(stream)) {
        out.status = "enqueue_failed";
        return out;
    }

    TrtEngine::TensorBuffer* match_output = nullptr;
    for (auto* tensor : outputs) {
        if (tensor->dtype != nvinfer1::DataType::kFLOAT ||
            tensor->host_float.empty() || tensor->elements < 4) {
            continue;
        }
        const int last_dim =
            tensor->dims.nbDims > 0
                ? static_cast<int>(tensor->dims.d[tensor->dims.nbDims - 1])
                : 0;
        if (last_dim == 4 || last_dim == 5) {
            match_output = tensor;
            break;
        }
    }
    if (!match_output) {
        out.status = "unsupported_output_schema";
        return out;
    }

    cudaError_t err = cudaMemcpyAsync(match_output->host_float.data(),
                                      match_output->device,
                                      match_output->bytes,
                                      cudaMemcpyDeviceToHost,
                                      stream);
    if (err != cudaSuccess) {
        out.status = "copy_failed";
        return out;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        out.status = "sync_failed";
        return out;
    }
    out.inference_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());

    const int stride =
        static_cast<int>(match_output->dims.d[match_output->dims.nbDims - 1]);
    if (stride != 4 && stride != 5) {
        out.status = "unsupported_output_schema";
        return out;
    }
    const int rows = static_cast<int>(match_output->elements / stride);
    const float left_s = std::sqrt(std::max(1.0f, left_det.width * context *
                                                  left_det.height * context));
    const float right_s = std::sqrt(std::max(1.0f, right_det.width * context *
                                                   right_det.height * context));
    const float left_x0 = left_det.cx - 0.5f * left_s;
    const float left_y0 = left_det.cy - 0.5f * left_s;
    const float right_x0 = right_det.cx - 0.5f * right_s;
    const float right_y0 = right_det.cy - 0.5f * right_s;
    std::vector<NeuralFeaturePointMatch> candidates;
    std::vector<float> disparities;
    candidates.reserve(static_cast<size_t>(rows));
    disparities.reserve(static_cast<size_t>(rows));
    for (int i = 0; i < rows; ++i) {
        const float* row = match_output->host_float.data() + i * stride;
        const float score = stride == 5 ? row[4] : 1.0f;
        if (score < config_.min_score) continue;
        const bool normalized =
            std::fabs(row[0]) <= 2.0f && std::fabs(row[1]) <= 2.0f &&
            std::fabs(row[2]) <= 2.0f && std::fabs(row[3]) <= 2.0f;
        const float scale = normalized ? static_cast<float>(config_.roi_size) : 1.0f;
        const float lx = left_x0 + (row[0] * scale) * left_s /
                                  static_cast<float>(config_.roi_size);
        const float ly = left_y0 + (row[1] * scale) * left_s /
                                  static_cast<float>(config_.roi_size);
        const float rx = right_x0 + (row[2] * scale) * right_s /
                                   static_cast<float>(config_.roi_size);
        const float ry = right_y0 + (row[3] * scale) * right_s /
                                   static_cast<float>(config_.roi_size);
        const float disp = lx - rx;
        if (disp <= 0.5f || disp > static_cast<float>(max_disparity_) ||
            std::fabs(ly - ry) > config_.max_y_error_px ||
            std::fabs(disp - initial_disparity) > config_.max_disp_delta_px) {
            continue;
        }
        NeuralFeaturePointMatch m;
        m.left_x = lx;
        m.left_y = ly;
        m.right_x = rx;
        m.right_y = ry;
        m.disparity = disp;
        m.score = score;
        candidates.push_back(m);
        disparities.push_back(disp);
    }

    if (static_cast<int>(disparities.size()) < config_.min_matches) {
        out.status = "not_enough_matches";
        return out;
    }
    const float median = medianOf(disparities);
    std::vector<float> abs_dev;
    abs_dev.reserve(disparities.size());
    for (float d : disparities) abs_dev.push_back(std::fabs(d - median));
    const float mad = medianOf(abs_dev);
    const float gate = std::max(config_.final_disp_gate_px, 1.4826f * mad * 2.5f);
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (const auto& m : candidates) {
        if (std::fabs(m.disparity - median) > gate) continue;
        out.matches.push_back(m);
        sum += m.disparity;
        sum2 += m.disparity * m.disparity;
    }
    const int kept = static_cast<int>(out.matches.size());
    if (kept < config_.min_matches) {
        out.status = "not_enough_inliers";
        out.matches.clear();
        return out;
    }
    out.disparity = sum / static_cast<float>(kept);
    const float var = std::max(0.0f, sum2 / static_cast<float>(kept) -
                                      out.disparity * out.disparity);
    out.stddev_px = std::sqrt(var);
    out.depth_m = focal_ * baseline_ / std::max(0.5f, out.disparity);
    out.confidence = std::min(1.0f, static_cast<float>(kept) /
                                    std::max(1, config_.top_k));
    out.valid = true;
    out.status = "ok";
    return out;
}

}  // namespace stereo3d
