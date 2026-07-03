#include "neural_feature_matcher.h"

#include "neural_feature_matcher_helpers.h"
#include "track/crop_resize.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace stereo3d {

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
            NeuralFeatureMatchResult xfeat = matchXFeatExtractorGpuRoi(
                left_gray_gpu, left_gray_pitch,
                right_gray_gpu, right_gray_pitch,
                left_bgr_gpu, left_bgr_pitch,
                right_bgr_gpu, right_bgr_pitch,
                img_width, img_height,
                left_det, right_det,
                initial_disparity,
                stream);
            if (xfeat.status != "unsupported_extractor_schema") {
                return xfeat;
            }
        }
        if (extractor_.engine && extractor_.context) {
            return matchDirectExtractorGpuRoi(
                left_gray_gpu, left_gray_pitch,
                right_gray_gpu, right_gray_pitch,
                left_bgr_gpu, left_bgr_pitch,
                right_bgr_gpu, right_bgr_pitch,
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
