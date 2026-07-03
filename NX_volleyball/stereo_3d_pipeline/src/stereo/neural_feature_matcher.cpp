/**
 * @file neural_feature_matcher.cpp
 * @brief TensorRT engine boundary for learned ROI feature matching.
 */

#include "neural_feature_matcher.h"

#include "neural_feature_matcher_helpers.h"
#include "utils/logger.h"

#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>

namespace stereo3d {

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
        if (config_.use_lightglue &&
            !config_.matcher_engine_path.empty() &&
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
    if (config_.fused_engine_path.empty() &&
        !config_.matcher_engine_path.empty() &&
        !config_.use_lightglue) {
        LOG_WARN("neural_feature_matching.matcher_engine_path is configured but "
                 "ignored because use_lightglue=false");
    }
    if (config_.fused_engine_path.empty() &&
        config_.backend == NeuralFeatureBackend::SUPERPOINT_LIGHTGLUE) {
        LOG_WARN("SuperPoint+LightGlue split matcher runtime requires a supported "
                 "fixed TensorRT matcher schema; otherwise direct descriptor "
                 "matching is used");
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

    const bool is_extractor_engine = (&e == &extractor_);
    const bool is_fused_engine = (&e == &fused_);
    const bool is_matcher_engine = (&e == &matcher_);
    const bool is_roi_image_engine = is_extractor_engine || is_fused_engine;

    const int nb = e.engine->getNbIOTensors();
    auto set_matcher_input_shape =
        [&](const char* cname, const nvinfer1::Dims& dims) -> bool {
            if (!is_matcher_engine || !hasDynamicDim(dims)) return true;
            nvinfer1::Dims shape = dims;
            const std::string lname = lowerCopy(cname);
            const bool is_keypoints =
                lname.find("keypoint") != std::string::npos ||
                lname.find("kpt") != std::string::npos ||
                lname.find("point") != std::string::npos ||
                lname.find("coord") != std::string::npos;
            const bool is_descriptors =
                lname.find("descriptor") != std::string::npos ||
                lname.find("desc") != std::string::npos ||
                lname.find("feature") != std::string::npos;
            const bool is_scores =
                lname.find("score") != std::string::npos ||
                lname.find("conf") != std::string::npos ||
                lname.find("prob") != std::string::npos;
            const bool is_image_size = isImageSizeTensorName(lname);

            if (is_keypoints) {
                for (int d = 0; d < shape.nbDims; ++d) {
                    if (shape.d[d] > 0) continue;
                    if (d == 0 && shape.nbDims >= 3) shape.d[d] = 1;
                    else if (d == shape.nbDims - 1) shape.d[d] = 2;
                    else shape.d[d] = config_.top_k;
                }
            } else if (is_descriptors) {
                const bool channel_first =
                    shape.nbDims >= 2 &&
                    shape.d[shape.nbDims - 2] == config_.descriptor_dim;
                for (int d = 0; d < shape.nbDims; ++d) {
                    if (shape.d[d] > 0) continue;
                    if (d == 0 && shape.nbDims >= 3) shape.d[d] = 1;
                    else if (channel_first && d == shape.nbDims - 1) {
                        shape.d[d] = config_.top_k;
                    } else if (!channel_first && d == shape.nbDims - 1) {
                        shape.d[d] = config_.descriptor_dim;
                    } else {
                        shape.d[d] = channel_first
                            ? config_.descriptor_dim
                            : config_.top_k;
                    }
                }
            } else if (is_scores) {
                for (int d = 0; d < shape.nbDims; ++d) {
                    if (shape.d[d] > 0) continue;
                    shape.d[d] = (d == 0 && shape.nbDims >= 2)
                        ? 1
                        : config_.top_k;
                }
            } else if (is_image_size) {
                for (int d = 0; d < shape.nbDims; ++d) {
                    if (shape.d[d] > 0) continue;
                    shape.d[d] = (d == 0 && shape.nbDims >= 2) ? 1 : 2;
                }
            } else {
                LOG_WARN("Neural feature: unsupported dynamic matcher input %s",
                         cname);
                return false;
            }
            if (!e.context->setInputShape(cname, shape)) {
                LOG_WARN("Neural feature: set matcher input shape failed for %s",
                         cname);
                return false;
            }
            return true;
        };

    if (is_matcher_engine) {
        for (int i = 0; i < nb; ++i) {
            const char* cname = e.engine->getIOTensorName(i);
            if (!cname) return false;
            if (e.engine->getTensorIOMode(cname) !=
                nvinfer1::TensorIOMode::kINPUT) {
                continue;
            }
            if (!set_matcher_input_shape(cname,
                                         e.engine->getTensorShape(cname))) {
                return false;
            }
        }
    }

    for (int i = 0; i < nb; ++i) {
        const char* cname = e.engine->getIOTensorName(i);
        if (!cname) return false;
        TrtEngine::TensorBuffer tensor;
        tensor.name = cname;
        tensor.is_input =
            e.engine->getTensorIOMode(cname) == nvinfer1::TensorIOMode::kINPUT;
        tensor.dtype = e.engine->getTensorDataType(cname);
        tensor.dims = e.engine->getTensorShape(cname);

        if (tensor.is_input && is_roi_image_engine &&
            hasDynamicDim(tensor.dims) && tensor.dims.nbDims == 4) {
            int c = static_cast<int>(tensor.dims.d[1]);
            if (c <= 0) c = 1;
            nvinfer1::Dims4 shape{1, c, config_.roi_size, config_.roi_size};
            if (!e.context->setInputShape(cname, shape)) {
                LOG_WARN("Neural feature: setInputShape failed for %s", cname);
                return false;
            }
            tensor.dims = e.context->getTensorShape(cname);
        }
        if (tensor.is_input && is_matcher_engine && hasDynamicDim(tensor.dims)) {
            if (!set_matcher_input_shape(cname, tensor.dims)) {
                return false;
            }
            tensor.dims = e.context->getTensorShape(cname);
        }

        if (!tensor.is_input && hasDynamicDim(tensor.dims)) {
            tensor.dims = e.context->getTensorShape(cname);
        }

        if (tensor.is_input && is_roi_image_engine) {
            const int c = tensorChannels(tensor.dims);
            const int h = tensorHeight(tensor.dims);
            const int w = tensorWidth(tensor.dims);
            const bool channels_ok =
                is_extractor_engine ? (c == 1 || c == 3)
                                    : (c == 1 || c == 2 || c == 3 || c == 6);
            if ((tensor.dims.nbDims != 3 && tensor.dims.nbDims != 4) ||
                h != config_.roi_size || w != config_.roi_size ||
                !channels_ok) {
                LOG_WARN("Neural feature: unsupported input tensor %s shape "
                         "(channels=%d h=%d w=%d, expected roi=%d)",
                         cname, c, h, w, config_.roi_size);
                return false;
            }
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
            cudaFree(tensor.device);
            tensor.device = nullptr;
            return false;
        }
        if (tensor.is_input) {
            ++e.input_count;
        } else {
            ++e.output_count;
            if (tensor.dtype == nvinfer1::DataType::kFLOAT) {
                tensor.host_float.resize(tensor.elements);
            } else if (tensor.dtype == nvinfer1::DataType::kINT32) {
                tensor.host_int32.resize(tensor.elements);
            }
        }
        e.tensors.push_back(std::move(tensor));
    }

    e.bindings_ready = e.input_count > 0 && e.output_count > 0;
    if (e.bindings_ready && is_extractor_engine) {
        TrtEngine::TensorBuffer* input = nullptr;
        for (auto& tensor : e.tensors) {
            if (tensor.is_input) input = &tensor;
        }
        const int c = input ? tensorChannels(input->dims) : 0;
        if (e.input_count != 1 || (c != 1 && c != 3)) {
            LOG_WARN("Neural feature: extractor engine expects exactly one "
                     "1/3-channel ROI image input, got inputs=%d channels=%d",
                     e.input_count, c);
            e.bindings_ready = false;
        }
    }
    if (e.bindings_ready && is_fused_engine) {
        std::vector<int> input_channels;
        input_channels.reserve(static_cast<size_t>(e.input_count));
        for (const auto& tensor : e.tensors) {
            if (tensor.is_input) input_channels.push_back(tensorChannels(tensor.dims));
        }
        bool schema_ok = false;
        if (input_channels.size() == 1) {
            schema_ok = input_channels[0] == 2 || input_channels[0] == 6;
        } else if (input_channels.size() == 2) {
            schema_ok =
                (input_channels[0] == 1 || input_channels[0] == 3) &&
                (input_channels[1] == 1 || input_channels[1] == 3);
        }
        if (!schema_ok) {
            const int c0 = input_channels.empty() ? 0 : input_channels[0];
            const int c1 = input_channels.size() < 2 ? 0 : input_channels[1];
            LOG_WARN("Neural feature: fused engine expects one 2/6-channel "
                     "input or two 1/3-channel inputs, got inputs=%d "
                     "channels=(%d,%d)",
                     e.input_count, c0, c1);
            e.bindings_ready = false;
        }
    }
    if (e.bindings_ready && is_matcher_engine && e.input_count < 4) {
        LOG_WARN("Neural feature: split matcher expects keypoints/descriptors "
                 "for left and right, got inputs=%d", e.input_count);
        e.bindings_ready = false;
    }
    return e.bindings_ready;
}

bool NeuralFeatureMatcher::requiresBgrInput() const {
    auto engine_requires_bgr = [](const TrtEngine& e) {
        if (!e.bindings_ready) return false;
        for (const auto& tensor : e.tensors) {
            if (!tensor.is_input) continue;
            const int c = tensorChannels(tensor.dims);
            if (c == 3 || c == 6) return true;
        }
        return false;
    };
    return engine_requires_bgr(extractor_) || engine_requires_bgr(fused_);
}

}  // namespace stereo3d
