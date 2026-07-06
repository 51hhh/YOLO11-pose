#include "neural_feature_matcher.h"

#include "neural_feature_matcher_helpers.h"
#include "track/crop_resize.h"
#include "../utils/profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace stereo3d {

namespace {

struct DirectGpuStageEvents {
    cudaEvent_t crop_start = nullptr;
    cudaEvent_t crop_end = nullptr;
    cudaEvent_t trt_end = nullptr;
    cudaEvent_t post_end = nullptr;

    ~DirectGpuStageEvents() {
        if (crop_start) cudaEventDestroy(crop_start);
        if (crop_end) cudaEventDestroy(crop_end);
        if (trt_end) cudaEventDestroy(trt_end);
        if (post_end) cudaEventDestroy(post_end);
    }

    bool ensure() {
        auto make = [](cudaEvent_t& event) {
            return event ||
                   cudaEventCreateWithFlags(&event, cudaEventDefault) ==
                       cudaSuccess;
        };
        return make(crop_start) && make(crop_end) &&
               make(trt_end) && make(post_end);
    }
};

DirectGpuStageEvents& directGpuStageEvents() {
    thread_local DirectGpuStageEvents events;
    return events;
}

void recordGpuElapsed(const char* name, cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, stop) == cudaSuccess) {
        globalPerf().record(name, static_cast<double>(ms));
    }
}

}  // namespace

NeuralFeatureMatchResult NeuralFeatureMatcher::matchDirectExtractorGpuRoi(
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
    if (!extractor_.engine || !extractor_.context || !extractor_.bindings_ready) {
        out.status = "extractor_not_ready";
        return out;
    }

    TrtEngine::TensorBuffer* input = nullptr;
    std::vector<TrtEngine::TensorBuffer*> outputs;
    for (auto& tensor : extractor_.tensors) {
        extractor_.context->setTensorAddress(tensor.name.c_str(), tensor.device);
        if (tensor.is_input) input = &tensor;
        else outputs.push_back(&tensor);
    }
    if (!input || input->dtype != nvinfer1::DataType::kFLOAT || outputs.empty()) {
        out.status = "unsupported_direct_extractor_schema";
        return out;
    }

    const int input_channels = tensorChannels(input->dims);
    if (input_channels != 1 && input_channels != 3) {
        out.status = "unsupported_input_schema";
        return out;
    }
    if (input_channels == 3 &&
        (!left_bgr_gpu || !right_bgr_gpu ||
         left_bgr_pitch <= 0 || right_bgr_pitch <= 0)) {
        out.status = "unsupported_input_schema";
        return out;
    }

    auto tensor_name_has = [](const TrtEngine::TensorBuffer* tensor,
                              const char* needle) {
        return lowerCopy(tensor->name).find(needle) != std::string::npos;
    };

    auto find_keypoint_tensor = [&]() -> TrtEngine::TensorBuffer* {
        TrtEngine::TensorBuffer* fallback = nullptr;
        for (auto* tensor : outputs) {
            if (tensor->dtype != nvinfer1::DataType::kFLOAT ||
                tensor->host_float.empty()) {
                continue;
            }
            const int last = tensorLastDim(tensor->dims);
            const bool keypoint_shape =
                last == 2 || last == 3 ||
                (tensor->dims.nbDims >= 2 &&
                 tensor->dims.d[tensor->dims.nbDims - 2] == 2);
            const bool keypoint_name =
                tensor_name_has(tensor, "keypoint") ||
                tensor_name_has(tensor, "kpt") ||
                tensor_name_has(tensor, "coord") ||
                tensor_name_has(tensor, "point");
            if (keypoint_shape && keypoint_name) return tensor;
            if (keypoint_shape && !fallback) fallback = tensor;
        }
        return fallback;
    };

    auto find_descriptor_tensor = [&]() -> TrtEngine::TensorBuffer* {
        TrtEngine::TensorBuffer* fallback = nullptr;
        for (auto* tensor : outputs) {
            if (tensor->dtype != nvinfer1::DataType::kFLOAT ||
                tensor->host_float.empty()) {
                continue;
            }
            const int last = tensorLastDim(tensor->dims);
            const bool descriptor_shape =
                last == config_.descriptor_dim ||
                (tensor->dims.nbDims >= 2 &&
                 tensor->dims.d[tensor->dims.nbDims - 2] ==
                     config_.descriptor_dim);
            const bool descriptor_name =
                tensor_name_has(tensor, "descriptor") ||
                tensor_name_has(tensor, "desc") ||
                tensor_name_has(tensor, "feature");
            if (descriptor_shape && descriptor_name) return tensor;
            if (descriptor_shape && !fallback) fallback = tensor;
        }
        return fallback;
    };

    auto find_score_tensor = [&]() -> TrtEngine::TensorBuffer* {
        for (auto* tensor : outputs) {
            if (tensor->dtype != nvinfer1::DataType::kFLOAT ||
                tensor->host_float.empty()) {
                continue;
            }
            const bool score_name =
                tensor_name_has(tensor, "score") ||
                tensor_name_has(tensor, "conf") ||
                tensor_name_has(tensor, "prob");
            const int last = tensorLastDim(tensor->dims);
            const bool score_shape = last == 1 || tensor->dims.nbDims <= 2;
            if (score_shape && score_name) return tensor;
        }
        return nullptr;
    };

    auto tensor_batch = [](const nvinfer1::Dims& dims) {
        if (dims.nbDims >= 3 && dims.d[0] > 0 && dims.d[0] <= 8) {
            return static_cast<int>(dims.d[0]);
        }
        if (dims.nbDims == 2 && dims.d[0] > 0 && dims.d[0] <= 8) {
            return static_cast<int>(dims.d[0]);
        }
        return 1;
    };

    auto batch_stride = [&](const TrtEngine::TensorBuffer* tensor) -> size_t {
        if (!tensor) return 0;
        const int batch = std::max(1, tensor_batch(tensor->dims));
        return tensor->elements / static_cast<size_t>(batch);
    };

    auto keypoint_count = [&](const TrtEngine::TensorBuffer* tensor) -> int {
        if (!tensor) return 0;
        const int last = tensorLastDim(tensor->dims);
        const size_t per_batch = batch_stride(tensor);
        if (last == 2 || last == 3) {
            return static_cast<int>(per_batch /
                                    static_cast<size_t>(last));
        }
        if (tensor->dims.nbDims >= 2 &&
            tensor->dims.d[tensor->dims.nbDims - 2] == 2) {
            return last;
        }
        return 0;
    };

    auto descriptor_count = [&](const TrtEngine::TensorBuffer* tensor) -> int {
        if (!tensor || config_.descriptor_dim <= 0) return 0;
        const int last = tensorLastDim(tensor->dims);
        const size_t per_batch = batch_stride(tensor);
        if (last == config_.descriptor_dim) {
            return static_cast<int>(
                per_batch / static_cast<size_t>(config_.descriptor_dim));
        }
        if (tensor->dims.nbDims >= 2 &&
            tensor->dims.d[tensor->dims.nbDims - 2] ==
                config_.descriptor_dim) {
            return last;
        }
        return 0;
    };

    auto score_count = [&](const TrtEngine::TensorBuffer* tensor) -> int {
        if (!tensor) return 0;
        const size_t per_batch = batch_stride(tensor);
        const int last = tensorLastDim(tensor->dims);
        if (last == 1 && tensor->dims.nbDims >= 2) {
            return static_cast<int>(per_batch);
        }
        return static_cast<int>(per_batch);
    };

    auto keypoint_layout =
        [](const TrtEngine::TensorBuffer* tensor,
           DirectFeatureKeypointLayout* layout) -> bool {
            if (!tensor || !layout) return false;
            const int last = tensorLastDim(tensor->dims);
            if (last == 2) {
                *layout = DIRECT_KPTS_K2;
                return true;
            }
            if (tensor->dims.nbDims >= 2 &&
                tensor->dims.d[tensor->dims.nbDims - 2] == 2) {
                *layout = DIRECT_KPTS_2K;
                return true;
            }
            return false;
        };

    auto descriptor_layout =
        [this](const TrtEngine::TensorBuffer* tensor,
               DirectFeatureDescriptorLayout* layout) -> bool {
            if (!tensor || !layout) return false;
            const int last = tensorLastDim(tensor->dims);
            if (last == config_.descriptor_dim) {
                *layout = DIRECT_DESC_KD;
                return true;
            }
            if (tensor->dims.nbDims >= 2 &&
                tensor->dims.d[tensor->dims.nbDims - 2] ==
                    config_.descriptor_dim) {
                *layout = DIRECT_DESC_DK;
                return true;
            }
            return false;
        };

    auto read_keypoint = [this, &tensor_batch, &batch_stride, &keypoint_count](
                                const TrtEngine::TensorBuffer* tensor,
                                int index,
                                int batch_index,
                                float* x,
                                float* y,
                                float* score) -> bool {
        const int last = tensorLastDim(tensor->dims);
        const float* data = tensor->host_float.data();
        const int batch = std::max(1, tensor_batch(tensor->dims));
        if (batch_index < 0 || batch_index >= batch) return false;
        const size_t offset =
            static_cast<size_t>(batch_index) * batch_stride(tensor);
        if (last == 2 || last == 3) {
            const size_t base = offset + static_cast<size_t>(index) *
                                static_cast<size_t>(last);
            if (base + 1 >= tensor->host_float.size()) return false;
            *x = data[base];
            *y = data[base + 1];
            if (last == 3 && base + 2 < tensor->host_float.size()) {
                *score = data[base + 2];
            }
        } else if (tensor->dims.nbDims >= 2 &&
                   tensor->dims.d[tensor->dims.nbDims - 2] == 2) {
            const int count = keypoint_count(tensor);
            const size_t ix = offset + static_cast<size_t>(index);
            const size_t iy = offset + static_cast<size_t>(count) +
                              static_cast<size_t>(index);
            if (iy >= tensor->host_float.size()) return false;
            *x = data[ix];
            *y = data[iy];
        } else {
            return false;
        }

        auto to_roi = [this](float v) {
            if (std::fabs(v) <= 1.5f) {
                return v < -0.01f
                    ? (v + 1.0f) * 0.5f * static_cast<float>(config_.roi_size)
                    : v * static_cast<float>(config_.roi_size);
            }
            return v;
        };
        *x = to_roi(*x);
        *y = to_roi(*y);
        return std::isfinite(*x) && std::isfinite(*y) &&
               *x >= 0.0f && *y >= 0.0f &&
               *x < static_cast<float>(config_.roi_size) &&
               *y < static_cast<float>(config_.roi_size);
    };

    auto read_descriptor = [this, &tensor_batch, &batch_stride, &descriptor_count](
                                  const TrtEngine::TensorBuffer* tensor,
                                  int index,
                                  int batch_index,
                                  std::vector<float>* descriptor) -> bool {
        if (!tensor || config_.descriptor_dim <= 0) return false;
        descriptor->assign(static_cast<size_t>(config_.descriptor_dim), 0.0f);
        const int last = tensorLastDim(tensor->dims);
        const float* data = tensor->host_float.data();
        const int batch = std::max(1, tensor_batch(tensor->dims));
        if (batch_index < 0 || batch_index >= batch) return false;
        const size_t offset =
            static_cast<size_t>(batch_index) * batch_stride(tensor);
        if (last == config_.descriptor_dim) {
            const size_t base = offset + static_cast<size_t>(index) *
                                static_cast<size_t>(config_.descriptor_dim);
            if (base + static_cast<size_t>(config_.descriptor_dim) >
                tensor->host_float.size()) {
                return false;
            }
            std::copy(data + base,
                      data + base + config_.descriptor_dim,
                      descriptor->begin());
        } else if (tensor->dims.nbDims >= 2 &&
                   tensor->dims.d[tensor->dims.nbDims - 2] ==
                       config_.descriptor_dim) {
            const int count = descriptor_count(tensor);
            for (int c = 0; c < config_.descriptor_dim; ++c) {
                const size_t idx = offset + static_cast<size_t>(c) *
                                   static_cast<size_t>(count) +
                                   static_cast<size_t>(index);
                if (idx >= tensor->host_float.size()) return false;
                (*descriptor)[static_cast<size_t>(c)] = data[idx];
            }
        } else {
            return false;
        }

        float norm2 = 0.0f;
        for (float d : *descriptor) norm2 += d * d;
        const float inv_norm =
            norm2 > 1e-12f ? 1.0f / std::sqrt(norm2) : 0.0f;
        for (float& d : *descriptor) d *= inv_norm;
        return inv_norm > 0.0f;
    };

    auto score_at = [&](const TrtEngine::TensorBuffer* tensor,
                        int index,
                        int batch_index,
                        float fallback) {
        if (!tensor || tensor->host_float.empty()) return fallback;
        const int batch = std::max(1, tensor_batch(tensor->dims));
        if (batch_index < 0 || batch_index >= batch) return fallback;
        const size_t idx =
            static_cast<size_t>(batch_index) * batch_stride(tensor) +
            static_cast<size_t>(index);
        if (idx >= tensor->host_float.size()) return fallback;
        return tensor->host_float[idx];
    };

    TrtEngine::TensorBuffer* direct_keypoints = find_keypoint_tensor();
    TrtEngine::TensorBuffer* direct_descriptors = find_descriptor_tensor();
    TrtEngine::TensorBuffer* direct_scores = find_score_tensor();

    auto parse_features = [&](int batch_index,
                              std::vector<DirectFeature>* features) -> bool {
        const int count = std::min(keypoint_count(direct_keypoints),
                                   descriptor_count(direct_descriptors));
        if (!direct_keypoints || !direct_descriptors ||
            count < config_.min_matches) {
            return false;
        }
        features->clear();
        features->reserve(static_cast<size_t>(std::min(count, config_.top_k)));
        for (int i = 0; i < count; ++i) {
            DirectFeature f;
            f.score = score_at(direct_scores, i, batch_index, f.score);
            if (!read_keypoint(direct_keypoints, i, batch_index,
                               &f.x, &f.y, &f.score)) {
                continue;
            }
            if (!std::isfinite(f.score)) f.score = 1.0f;
            if (f.score <= 0.0f) continue;
            if (!read_descriptor(direct_descriptors, i, batch_index,
                                 &f.descriptor)) {
                continue;
            }
            features->push_back(std::move(f));
        }
        std::sort(features->begin(), features->end(),
                  [](const DirectFeature& a, const DirectFeature& b) {
                      return a.score > b.score;
                  });
        if (static_cast<int>(features->size()) > config_.top_k) {
            features->resize(static_cast<size_t>(config_.top_k));
        }
        return static_cast<int>(features->size()) >= config_.min_matches;
    };

    auto elapsed_ms_since = [](const std::chrono::steady_clock::time_point& t0) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    auto copy_outputs = [&]() -> bool {
        const auto copy_start = std::chrono::steady_clock::now();
        for (auto* tensor : outputs) {
            if (tensor->dtype != nvinfer1::DataType::kFLOAT ||
                tensor->host_float.empty()) {
                continue;
            }
            const cudaError_t err = cudaMemcpyAsync(
                tensor->host_float.data(), tensor->device, tensor->bytes,
                cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) return false;
        }
        const bool ok = cudaStreamSynchronize(stream) == cudaSuccess;
        globalPerf().record("Stage2_NeuralDirectD2HSync",
                            elapsed_ms_since(copy_start));
        return ok;
    };

    const float context = 1.20f;
    const auto roi_side = [](const Detection& det) {
        return std::max(1.0f, std::max(det.width, det.height));
    };
    const int input_batch = std::max(1, tensor_batch(input->dims));
    const size_t input_batch_stride = input->elements /
                                      static_cast<size_t>(input_batch);
    auto crop_one = [&](const Detection& det,
                        const uint8_t* gray, int pitch,
                        const uint8_t* bgr, int bgr_pitch,
                        int batch_index) -> bool {
        if (batch_index < 0 || batch_index >= input_batch) return false;
        const auto crop_start = std::chrono::steady_clock::now();
        float* dst = static_cast<float*>(input->device) +
                     static_cast<size_t>(batch_index) * input_batch_stride;
        const float side = roi_side(det);
        if (input_channels == 1) {
            cropResizeGPU(gray, pitch, img_width, img_height,
                          dst, config_.roi_size,
                          det.cx, det.cy, side, side,
                          context, stream);
        } else {
            cropResizeBgrGPU_3ch(bgr, bgr_pitch, img_width, img_height,
                                  dst, config_.roi_size,
                                  det.cx, det.cy, side, side,
                                  context, stream);
        }
        const bool ok = cudaPeekAtLastError() == cudaSuccess;
        globalPerf().record("Stage2_NeuralDirectCrop",
                            elapsed_ms_since(crop_start));
        return ok;
    };
    auto enqueue_extract = [&]() -> bool {
        const auto enqueue_start = std::chrono::steady_clock::now();
        const bool ok = extractor_.context->enqueueV3(stream);
        globalPerf().record("Stage2_NeuralDirectEnqueue",
                            elapsed_ms_since(enqueue_start));
        return ok;
    };
    auto run_one = [&](const Detection& det,
                       const uint8_t* gray, int pitch,
                       const uint8_t* bgr, int bgr_pitch,
                       int batch_index,
                       std::vector<DirectFeature>* features) -> bool {
        if (!crop_one(det, gray, pitch, bgr, bgr_pitch, batch_index) ||
            !enqueue_extract() || !copy_outputs()) {
            return false;
        }
        return parse_features(batch_index, features);
    };

    const auto start = std::chrono::steady_clock::now();
    std::vector<DirectFeature> left_features;
    std::vector<DirectFeature> right_features;

    struct IndexMatch {
        int query_idx = -1;
        int train_idx = -1;
        float score = 1.0f;
    };

    auto append_debug_point =
        [&](NeuralFeatureMatchResult* result,
            float lx, float ly, float rx, float ry,
            float disparity, float score, float second_score,
            SparseFeatureDebugStage stage,
            SparseFeatureRejectReason reason) {
            if (!result ||
                result->debug_points.size() >=
                    static_cast<size_t>(kMaxSparseFeatureDebugPoints)) {
                return;
            }
            SparseFeatureDebugPoint p;
            p.left_x = lx;
            p.left_y = ly;
            p.right_x = rx;
            p.right_y = ry;
            p.disparity = disparity;
            p.score = score;
            p.second_score = second_score;
            p.y_delta = ly - ry;
            p.y_residual = p.y_delta;
            p.disp_delta = disparity - initial_disparity;
            p.stage = static_cast<int>(stage);
            p.reject_reason = static_cast<int>(reason);
            result->debug_points.push_back(p);
        };

    auto append_debug_match =
        [&](NeuralFeatureMatchResult* result,
            const NeuralFeaturePointMatch& m,
            SparseFeatureDebugStage stage,
            SparseFeatureRejectReason reason,
            float second_score = std::numeric_limits<float>::quiet_NaN()) {
            append_debug_point(result,
                               m.left_x,
                               m.left_y,
                               m.right_x,
                               m.right_y,
                               m.disparity,
                               m.score,
                               second_score,
                               stage,
                               reason);
    };

    const auto map_to_frame = [&](const Detection& det, const DirectFeature& f,
                                  float* x, float* y) {
        const float s = roi_side(det) * context;
        const float roi_x = det.cx - 0.5f * s;
        const float roi_y = det.cy - 0.5f * s;
        *x = roi_x + (f.x + 0.5f) * s / static_cast<float>(config_.roi_size) - 0.5f;
        *y = roi_y + (f.y + 0.5f) * s / static_cast<float>(config_.roi_size) - 0.5f;
    };

    auto build_result_from_candidates =
        [&](const std::vector<NeuralFeaturePointMatch>& candidates,
            const char* ok_status) {
            NeuralFeatureMatchResult result;
            if (static_cast<int>(candidates.size()) < config_.min_matches) {
                result.status = "not_enough_matches";
                return result;
            }

            std::vector<float> disparities;
            disparities.reserve(candidates.size());
            for (const auto& m : candidates) disparities.push_back(m.disparity);
            const float median = medianOf(disparities);
            std::vector<float> abs_dev;
            abs_dev.reserve(disparities.size());
            for (float d : disparities) {
                abs_dev.push_back(std::fabs(d - median));
            }
            const float mad = medianOf(abs_dev);
            const float gate =
                std::max(config_.final_disp_gate_px, 1.4826f * mad * 2.5f);

            float sum = 0.0f;
            float sum2 = 0.0f;
            float score_sum = 0.0f;
            float min_x = std::numeric_limits<float>::infinity();
            float max_x = -std::numeric_limits<float>::infinity();
            float min_y = std::numeric_limits<float>::infinity();
            float max_y = -std::numeric_limits<float>::infinity();
            int quadrant_mask = 0;
            for (const auto& m : candidates) {
                if (std::fabs(m.disparity - median) > gate) {
                    append_debug_match(&result,
                                       m,
                                       SparseFeatureDebugStage::GEOMETRY,
                                       SparseFeatureRejectReason::MAD_OUTLIER);
                    continue;
                }
                result.matches.push_back(m);
                append_debug_match(&result,
                                   m,
                                   SparseFeatureDebugStage::INLIER,
                                   SparseFeatureRejectReason::NONE);
                sum += m.disparity;
                sum2 += m.disparity * m.disparity;
                score_sum += m.score;
                min_x = std::min(min_x, m.left_x);
                max_x = std::max(max_x, m.left_x);
                min_y = std::min(min_y, m.left_y);
                max_y = std::max(max_y, m.left_y);
                const int qx = m.left_x >= left_det.cx ? 1 : 0;
                const int qy = m.left_y >= left_det.cy ? 1 : 0;
                quadrant_mask |= 1 << (qy * 2 + qx);
            }
            if (static_cast<int>(result.matches.size()) < config_.min_matches) {
                result.status = "not_enough_inliers";
                result.matches.clear();
                return result;
            }

            const int quadrants =
                ((quadrant_mask & 1) ? 1 : 0) +
                ((quadrant_mask & 2) ? 1 : 0) +
                ((quadrant_mask & 4) ? 1 : 0) +
                ((quadrant_mask & 8) ? 1 : 0);
            const float spread =
                std::max(std::max(0.0f, max_x - min_x),
                         std::max(0.0f, max_y - min_y));
            const float min_spread =
                std::min(left_det.width, left_det.height) *
                std::max(0.0f, config_.min_spatial_spread_ratio);
            if (config_.min_spatial_quadrants > 0 &&
                quadrants < config_.min_spatial_quadrants) {
                result.status = "poor_spatial_quadrants";
                result.matches.clear();
                return result;
            }
            if (min_spread > 0.0f && spread < min_spread) {
                result.status = "poor_spatial_spread";
                result.matches.clear();
                return result;
            }

            const float kept = static_cast<float>(result.matches.size());
            result.disparity = sum / kept;
            const float var = std::max(
                0.0f, sum2 / kept - result.disparity * result.disparity);
            result.stddev_px = std::sqrt(var);
            result.depth_m =
                focal_ * baseline_ / std::max(0.5f, result.disparity);
            const float support_conf = std::min(
                1.0f,
                kept / static_cast<float>(
                    std::max(1, config_.min_matches * 2)));
            const float score_conf = std::clamp(
                (score_sum / kept + 1.0f) * 0.5f, 0.0f, 1.0f);
            const float consistency =
                std::clamp(1.0f / (1.0f + result.stddev_px), 0.0f, 1.0f);
            const float spatial_conf =
                std::clamp(0.5f * static_cast<float>(quadrants) / 4.0f +
                               0.5f * (min_spread > 0.0f
                                            ? std::min(1.0f, spread / min_spread)
                                            : 1.0f),
                           0.0f, 1.0f);
            result.confidence = std::clamp(0.40f * support_conf +
                                           0.30f * score_conf +
                                           0.20f * consistency +
                                           0.10f * spatial_conf,
                                           0.0f, 1.0f);
            result.inference_ms = static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            result.valid = true;
            result.status = ok_status;
            return result;
        };

    auto build_result_from_indices =
        [&](const std::vector<IndexMatch>& index_matches,
            const char* ok_status) {
            NeuralFeatureMatchResult result;
            std::vector<NeuralFeaturePointMatch> candidates;
            candidates.reserve(index_matches.size());
            for (const auto& im : index_matches) {
                if (im.query_idx < 0 || im.train_idx < 0 ||
                    im.query_idx >= static_cast<int>(left_features.size()) ||
                    im.train_idx >= static_cast<int>(right_features.size())) {
                    continue;
                }
                if (im.score < config_.min_score) continue;
                float lx, ly, rx, ry;
                map_to_frame(left_det,
                             left_features[static_cast<size_t>(im.query_idx)],
                             &lx, &ly);
                map_to_frame(right_det,
                             right_features[static_cast<size_t>(im.train_idx)],
                             &rx, &ry);
                const float disp = lx - rx;
                SparseFeatureRejectReason reject =
                    SparseFeatureRejectReason::NONE;
                if (disp <= 0.5f ||
                    disp > static_cast<float>(max_disparity_)) {
                    reject = SparseFeatureRejectReason::BAD_DISPARITY;
                } else if (std::fabs(ly - ry) > config_.max_y_error_px) {
                    reject = SparseFeatureRejectReason::Y_RESIDUAL;
                } else if (std::fabs(disp - initial_disparity) >
                           config_.max_disp_delta_px) {
                    reject = SparseFeatureRejectReason::DISP_DELTA;
                }
                if (reject != SparseFeatureRejectReason::NONE) {
                    append_debug_point(&result,
                                       lx,
                                       ly,
                                       rx,
                                       ry,
                                       disp,
                                       im.score,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       SparseFeatureDebugStage::GEOMETRY,
                                       reject);
                    continue;
                }
                NeuralFeaturePointMatch m;
                m.left_x = lx;
                m.left_y = ly;
                m.right_x = rx;
                m.right_y = ry;
                m.disparity = disp;
                m.score = im.score;
                append_debug_match(&result,
                                   m,
                                   SparseFeatureDebugStage::GEOMETRY,
                                   SparseFeatureRejectReason::NONE);
                candidates.push_back(m);
            }

            NeuralFeatureMatchResult finalized =
                build_result_from_candidates(candidates, ok_status);
            finalized.debug_points.insert(finalized.debug_points.begin(),
                                          result.debug_points.begin(),
                                          result.debug_points.end());
            if (finalized.debug_points.size() >
                static_cast<size_t>(kMaxSparseFeatureDebugPoints)) {
                finalized.debug_points.resize(
                    static_cast<size_t>(kMaxSparseFeatureDebugPoints));
            }
            return finalized;
        };

    const int direct_keypoint_count = keypoint_count(direct_keypoints);
    const int direct_descriptor_count = descriptor_count(direct_descriptors);
    const int direct_score_count =
        direct_scores ? score_count(direct_scores)
                      : std::min(direct_keypoint_count, direct_descriptor_count);
    const int output_batch = std::min({
        direct_keypoints ? tensor_batch(direct_keypoints->dims) : 0,
        direct_descriptors ? tensor_batch(direct_descriptors->dims) : 0,
        direct_scores ? tensor_batch(direct_scores->dims) : input_batch,
    });
    DirectFeatureKeypointLayout direct_kpt_layout = DIRECT_KPTS_K2;
    DirectFeatureDescriptorLayout direct_desc_layout = DIRECT_DESC_KD;

    if (config_.gpu_postprocess &&
        input_batch >= 2 &&
        output_batch >= 2 &&
        direct_keypoints &&
        direct_descriptors &&
        direct_keypoint_count >= config_.min_matches &&
        direct_descriptor_count >= config_.min_matches &&
        direct_score_count >= config_.min_matches &&
        keypoint_layout(direct_keypoints, &direct_kpt_layout) &&
        descriptor_layout(direct_descriptors, &direct_desc_layout) &&
        ensureDirectFeatureGpuWorkspace(direct_gpu_workspace_,
                                        std::max(1, config_.top_k),
                                        config_.descriptor_dim)) {
        DirectGpuStageEvents& gpu_events = directGpuStageEvents();
        const bool gpu_profile = stream && gpu_events.ensure();
        if (gpu_profile) {
            cudaEventRecord(gpu_events.crop_start, stream);
        }
        const bool crops_ok =
            crop_one(left_det,
                     left_gray_gpu, left_gray_pitch,
                     left_bgr_gpu, left_bgr_pitch,
                     0) &&
            crop_one(right_det,
                     right_gray_gpu, right_gray_pitch,
                     right_bgr_gpu, right_bgr_pitch,
                     1);
        if (gpu_profile && crops_ok) {
            cudaEventRecord(gpu_events.crop_end, stream);
        }
        const bool enqueue_ok = crops_ok && enqueue_extract();
        if (gpu_profile && enqueue_ok) {
            cudaEventRecord(gpu_events.trt_end, stream);
        }
        if (!enqueue_ok) {
            goto direct_gpu_b2_fallback;
        }
        const auto* kpt_base =
            static_cast<const float*>(direct_keypoints->device);
        const auto* desc_base =
            static_cast<const float*>(direct_descriptors->device);
        const auto* score_base = direct_scores
            ? static_cast<const float*>(direct_scores->device)
            : nullptr;
        const size_t kpt_stride = batch_stride(direct_keypoints);
        const size_t desc_stride = batch_stride(direct_descriptors);
        const size_t score_stride = direct_scores ? batch_stride(direct_scores) : 0;
        std::vector<DirectFeatureGpuMatch> gpu_matches;
        const auto post_start = std::chrono::steady_clock::now();
        const bool gpu_path_ok = runDirectFeatureGpuPostprocess(
            direct_gpu_workspace_,
            kpt_base,
            kpt_base + kpt_stride,
            desc_base,
            desc_base + desc_stride,
            score_base,
            score_base ? score_base + score_stride : nullptr,
            direct_keypoint_count,
            direct_descriptor_count,
            direct_score_count,
            config_.descriptor_dim,
            direct_kpt_layout,
            direct_desc_layout,
            config_.roi_size,
            config_.min_matches,
            config_.min_score,
            config_.match_margin,
            config_.max_y_error_px,
            config_.max_disp_delta_px,
            initial_disparity,
            max_disparity_,
            left_det.cx, left_det.cy, left_det.width, left_det.height,
            right_det.cx, right_det.cy, right_det.width, right_det.height,
            stream,
            &gpu_matches);
        if (gpu_profile) {
            cudaEventRecord(gpu_events.post_end, stream);
            if (cudaEventSynchronize(gpu_events.post_end) == cudaSuccess) {
                recordGpuElapsed("Stage2_NeuralDirectCropGpu",
                                 gpu_events.crop_start, gpu_events.crop_end);
                recordGpuElapsed("Stage2_NeuralDirectTrtGpu",
                                 gpu_events.crop_end, gpu_events.trt_end);
                recordGpuElapsed("Stage2_NeuralDirectPostprocessD2HGpu",
                                 gpu_events.trt_end, gpu_events.post_end);
                recordGpuElapsed("Stage2_NeuralDirectGpuTotal",
                                 gpu_events.crop_start, gpu_events.post_end);
            }
        }
        globalPerf().record("Stage2_NeuralDirectGpuPostprocessSync",
                            elapsed_ms_since(post_start));
        if (gpu_path_ok) {
            const auto finalize_start = std::chrono::steady_clock::now();
            std::vector<NeuralFeaturePointMatch> candidates;
            candidates.reserve(gpu_matches.size());
            for (const auto& src : gpu_matches) {
                NeuralFeaturePointMatch m;
                m.left_x = src.left_x;
                m.left_y = src.left_y;
                m.right_x = src.right_x;
                m.right_y = src.right_y;
                m.disparity = src.disparity;
                m.score = src.score;
                candidates.push_back(m);
            }
            NeuralFeatureMatchResult gpu_result =
                build_result_from_candidates(candidates, "ok_gpu_b2");
            globalPerf().record("Stage2_NeuralDirectFinalize",
                                elapsed_ms_since(finalize_start));
            return gpu_result;
        }
    }

direct_gpu_b2_fallback:
    bool features_ready = false;
    if (input_batch >= 2 && output_batch >= 2) {
        features_ready =
            crop_one(left_det,
                     left_gray_gpu, left_gray_pitch,
                     left_bgr_gpu, left_bgr_pitch,
                     0) &&
            crop_one(right_det,
                     right_gray_gpu, right_gray_pitch,
                     right_bgr_gpu, right_bgr_pitch,
                     1) &&
            enqueue_extract() &&
            copy_outputs() &&
            parse_features(0, &left_features) &&
            parse_features(1, &right_features);
    }
    if (!features_ready) {
        features_ready =
            run_one(left_det,
                    left_gray_gpu, left_gray_pitch,
                    left_bgr_gpu, left_bgr_pitch,
                    0,
                    &left_features) &&
            run_one(right_det,
                    right_gray_gpu, right_gray_pitch,
                    right_bgr_gpu, right_bgr_pitch,
                    0,
                    &right_features);
    }
    if (!features_ready) {
        out.status = "extractor_failed";
        out.inference_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        return out;
    }

    auto run_split_matcher = [&]() {
        NeuralFeatureMatchResult result;
        result.status = "unsupported_split_matcher_schema";
        if (!matcher_.engine || !matcher_.context || !matcher_.bindings_ready) {
            return result;
        }

        std::array<TrtEngine::TensorBuffer*, 2> keypoint_inputs{nullptr, nullptr};
        std::array<TrtEngine::TensorBuffer*, 2> descriptor_inputs{nullptr, nullptr};
        std::array<TrtEngine::TensorBuffer*, 2> score_inputs{nullptr, nullptr};
        std::array<TrtEngine::TensorBuffer*, 2> size_inputs{nullptr, nullptr};
        int next_keypoint = 0;
        int next_descriptor = 0;
        int next_score = 0;
        int next_size = 0;
        std::vector<TrtEngine::TensorBuffer*> matcher_outputs;

        auto assign_input =
            [](std::array<TrtEngine::TensorBuffer*, 2>& slots,
               int& next_slot,
               int side,
               TrtEngine::TensorBuffer* tensor) {
                if (side >= 0 && side < 2) {
                    slots[static_cast<size_t>(side)] = tensor;
                    return;
                }
                if (next_slot < 2) {
                    slots[static_cast<size_t>(next_slot++)] = tensor;
                }
            };

        for (auto& tensor : matcher_.tensors) {
            matcher_.context->setTensorAddress(tensor.name.c_str(),
                                               tensor.device);
            if (!tensor.is_input) {
                matcher_outputs.push_back(&tensor);
                continue;
            }
            if (tensor.dtype != nvinfer1::DataType::kFLOAT) {
                return result;
            }
            const std::string lname = lowerCopy(tensor.name);
            const int side = splitTensorSideFromName(lname);
            const bool is_keypoints =
                tensor_name_has(&tensor, "keypoint") ||
                tensor_name_has(&tensor, "kpt") ||
                tensor_name_has(&tensor, "point") ||
                tensor_name_has(&tensor, "coord");
            const bool is_descriptors =
                tensor_name_has(&tensor, "descriptor") ||
                tensor_name_has(&tensor, "desc") ||
                tensor_name_has(&tensor, "feature");
            const bool is_scores =
                tensor_name_has(&tensor, "score") ||
                tensor_name_has(&tensor, "conf") ||
                tensor_name_has(&tensor, "prob");
            const bool is_size =
                isImageSizeTensorName(lname);
            if (is_keypoints) {
                assign_input(keypoint_inputs, next_keypoint, side, &tensor);
            } else if (is_descriptors) {
                assign_input(descriptor_inputs, next_descriptor, side, &tensor);
            } else if (is_scores) {
                assign_input(score_inputs, next_score, side, &tensor);
            } else if (is_size) {
                assign_input(size_inputs, next_size, side, &tensor);
            } else {
                return result;
            }
        }
        if (!keypoint_inputs[0] || !keypoint_inputs[1] ||
            !descriptor_inputs[0] || !descriptor_inputs[1] ||
            matcher_outputs.empty()) {
            return result;
        }

        auto copy_vector_to_input =
            [&](TrtEngine::TensorBuffer* tensor,
                const std::vector<float>& values) {
                if (!tensor || values.size() > tensor->elements) return false;
                const cudaError_t err = cudaMemcpyAsync(
                    tensor->device,
                    values.data(),
                    values.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);
                if (err != cudaSuccess) return false;
                if (values.size() < tensor->elements) {
                    const size_t offset = values.size() * sizeof(float);
                    const size_t bytes = tensor->bytes - offset;
                    return cudaMemsetAsync(
                        static_cast<uint8_t*>(tensor->device) + offset,
                        0,
                        bytes,
                        stream) == cudaSuccess;
                }
                return true;
            };
        auto make_keypoint_input =
            [&](TrtEngine::TensorBuffer* tensor,
                const std::vector<DirectFeature>& features) {
                std::vector<float> values(tensor->elements, 0.0f);
                const int last = tensorLastDim(tensor->dims);
                const int count = keypoint_count(tensor);
                const int n = std::min(
                    std::min(count, config_.top_k),
                    static_cast<int>(features.size()));
                if (last == 2 || last == 3) {
                    for (int i = 0; i < n; ++i) {
                        const size_t base = static_cast<size_t>(i) *
                                            static_cast<size_t>(last);
                        values[base] = features[static_cast<size_t>(i)].x;
                        values[base + 1] = features[static_cast<size_t>(i)].y;
                        if (last == 3) {
                            values[base + 2] =
                                features[static_cast<size_t>(i)].score;
                        }
                    }
                } else if (tensor->dims.nbDims >= 2 &&
                           tensor->dims.d[tensor->dims.nbDims - 2] == 2) {
                    for (int i = 0; i < n; ++i) {
                        values[static_cast<size_t>(i)] =
                            features[static_cast<size_t>(i)].x;
                        values[static_cast<size_t>(count) +
                               static_cast<size_t>(i)] =
                            features[static_cast<size_t>(i)].y;
                    }
                } else {
                    return std::vector<float>{};
                }
                return values;
            };
        auto make_descriptor_input =
            [&](TrtEngine::TensorBuffer* tensor,
                const std::vector<DirectFeature>& features) {
                std::vector<float> values(tensor->elements, 0.0f);
                const int last = tensorLastDim(tensor->dims);
                const int count = descriptor_count(tensor);
                const int n = std::min(
                    std::min(count, config_.top_k),
                    static_cast<int>(features.size()));
                if (last == config_.descriptor_dim) {
                    for (int i = 0; i < n; ++i) {
                        const auto& desc =
                            features[static_cast<size_t>(i)].descriptor;
                        const size_t base = static_cast<size_t>(i) *
                                            static_cast<size_t>(config_.descriptor_dim);
                        for (int c = 0; c < config_.descriptor_dim; ++c) {
                            values[base + static_cast<size_t>(c)] =
                                c < static_cast<int>(desc.size())
                                    ? desc[static_cast<size_t>(c)]
                                    : 0.0f;
                        }
                    }
                } else if (tensor->dims.nbDims >= 2 &&
                           tensor->dims.d[tensor->dims.nbDims - 2] ==
                               config_.descriptor_dim) {
                    for (int i = 0; i < n; ++i) {
                        const auto& desc =
                            features[static_cast<size_t>(i)].descriptor;
                        for (int c = 0; c < config_.descriptor_dim; ++c) {
                            const size_t idx = static_cast<size_t>(c) *
                                               static_cast<size_t>(count) +
                                               static_cast<size_t>(i);
                            values[idx] = c < static_cast<int>(desc.size())
                                ? desc[static_cast<size_t>(c)]
                                : 0.0f;
                        }
                    }
                } else {
                    return std::vector<float>{};
                }
                return values;
            };
        auto make_score_input =
            [&](TrtEngine::TensorBuffer* tensor,
                const std::vector<DirectFeature>& features) {
                std::vector<float> values(tensor->elements, 0.0f);
                const int n = std::min(static_cast<int>(values.size()),
                                       static_cast<int>(features.size()));
                for (int i = 0; i < n; ++i) {
                    values[static_cast<size_t>(i)] =
                        features[static_cast<size_t>(i)].score;
                }
                return values;
            };
        auto make_size_input = [&](TrtEngine::TensorBuffer* tensor) {
            std::vector<float> values(tensor->elements, 0.0f);
            for (size_t i = 0; i + 1 < values.size(); i += 2) {
                values[i] = static_cast<float>(config_.roi_size);
                values[i + 1] = static_cast<float>(config_.roi_size);
            }
            return values;
        };

        const std::array<const std::vector<DirectFeature>*, 2> feature_sets{
            &left_features, &right_features};
        for (int side = 0; side < 2; ++side) {
            const auto& feats = *feature_sets[static_cast<size_t>(side)];
            std::vector<float> kpts =
                make_keypoint_input(keypoint_inputs[static_cast<size_t>(side)],
                                    feats);
            std::vector<float> desc =
                make_descriptor_input(descriptor_inputs[static_cast<size_t>(side)],
                                      feats);
            if (kpts.empty() || desc.empty() ||
                !copy_vector_to_input(keypoint_inputs[static_cast<size_t>(side)],
                                      kpts) ||
                !copy_vector_to_input(descriptor_inputs[static_cast<size_t>(side)],
                                      desc)) {
                result.status = "matcher_input_copy_failed";
                return result;
            }
            if (score_inputs[static_cast<size_t>(side)]) {
                std::vector<float> scores =
                    make_score_input(score_inputs[static_cast<size_t>(side)],
                                     feats);
                if (!copy_vector_to_input(score_inputs[static_cast<size_t>(side)],
                                          scores)) {
                    result.status = "matcher_input_copy_failed";
                    return result;
                }
            }
            if (size_inputs[static_cast<size_t>(side)]) {
                std::vector<float> sizes =
                    make_size_input(size_inputs[static_cast<size_t>(side)]);
                if (!copy_vector_to_input(size_inputs[static_cast<size_t>(side)],
                                          sizes)) {
                    result.status = "matcher_input_copy_failed";
                    return result;
                }
            }
        }

        if (!matcher_.context->enqueueV3(stream)) {
            result.status = "matcher_enqueue_failed";
            return result;
        }
        for (auto* tensor : matcher_outputs) {
            if (tensor->dtype == nvinfer1::DataType::kFLOAT &&
                !tensor->host_float.empty()) {
                if (cudaMemcpyAsync(tensor->host_float.data(),
                                    tensor->device,
                                    tensor->bytes,
                                    cudaMemcpyDeviceToHost,
                                    stream) != cudaSuccess) {
                    result.status = "matcher_copy_failed";
                    return result;
                }
            } else if (tensor->dtype == nvinfer1::DataType::kINT32 &&
                       !tensor->host_int32.empty()) {
                if (cudaMemcpyAsync(tensor->host_int32.data(),
                                    tensor->device,
                                    tensor->bytes,
                                    cudaMemcpyDeviceToHost,
                                    stream) != cudaSuccess) {
                    result.status = "matcher_copy_failed";
                    return result;
                }
            }
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            result.status = "matcher_sync_failed";
            return result;
        }

        TrtEngine::TensorBuffer* matches = nullptr;
        TrtEngine::TensorBuffer* matches0 = nullptr;
        TrtEngine::TensorBuffer* generic_scores = nullptr;
        TrtEngine::TensorBuffer* left_scores = nullptr;
        for (auto* tensor : matcher_outputs) {
            const std::string lname = lowerCopy(tensor->name);
            const bool is_match_name =
                lname.find("match") != std::string::npos ||
                lname.find("index") != std::string::npos ||
                lname.find("indices") != std::string::npos;
            const bool is_score_name =
                lname.find("score") != std::string::npos ||
                lname.find("conf") != std::string::npos ||
                lname.find("prob") != std::string::npos;
            const int last = tensorLastDim(tensor->dims);
            if (is_score_name && tensor->dtype == nvinfer1::DataType::kFLOAT) {
                if (isLeftScoreTensorName(lname)) {
                    left_scores = tensor;
                } else if (!isRightScoreTensorName(lname) && !generic_scores) {
                    generic_scores = tensor;
                }
            } else if (is_match_name && (last == 2 || last == 3)) {
                if (!matches) matches = tensor;
            } else if (is_match_name &&
                       isLeftMatchIndexTensorName(lname)) {
                matches0 = tensor;
            } else if (is_match_name &&
                       !isRightMatchIndexTensorName(lname) &&
                       !matches0) {
                matches0 = tensor;
            }
        }
        if (!matches && !matches0) {
            return result;
        }

        std::vector<IndexMatch> index_matches;
        TrtEngine::TensorBuffer* scores = left_scores ? left_scores
                                                      : generic_scores;
        auto score_at = [&](int idx, float fallback) {
            if (!scores || scores->host_float.empty() ||
                idx < 0 ||
                static_cast<size_t>(idx) >= scores->host_float.size()) {
                return fallback;
            }
            return scores->host_float[static_cast<size_t>(idx)];
        };
        auto push_pair = [&](int qi, int ti, float score) {
            if (qi < 0 || ti < 0 ||
                qi >= static_cast<int>(left_features.size()) ||
                ti >= static_cast<int>(right_features.size())) {
                return;
            }
            IndexMatch im;
            im.query_idx = qi;
            im.train_idx = ti;
            im.score = score;
            index_matches.push_back(im);
        };
        if (matches) {
            const int stride = tensorLastDim(matches->dims);
            const int rows = stride > 0
                ? static_cast<int>(matches->elements /
                                   static_cast<size_t>(stride))
                : 0;
            for (int i = 0; i < rows; ++i) {
                if (matches->dtype == nvinfer1::DataType::kINT32) {
                    const size_t base = static_cast<size_t>(i) *
                                        static_cast<size_t>(stride);
                    const int qi = matches->host_int32[base];
                    const int ti = matches->host_int32[base + 1];
                    const float score = stride >= 3
                        ? static_cast<float>(matches->host_int32[base + 2])
                        : score_at(i, 1.0f);
                    push_pair(qi, ti, score);
                } else if (matches->dtype == nvinfer1::DataType::kFLOAT) {
                    const size_t base = static_cast<size_t>(i) *
                                        static_cast<size_t>(stride);
                    const float fq = matches->host_float[base];
                    const float ft = matches->host_float[base + 1];
                    const int qi = static_cast<int>(std::round(fq));
                    const int ti = static_cast<int>(std::round(ft));
                    if (std::fabs(fq - static_cast<float>(qi)) > 1e-3f ||
                        std::fabs(ft - static_cast<float>(ti)) > 1e-3f) {
                        continue;
                    }
                    const float score = stride >= 3
                        ? matches->host_float[base + 2]
                        : score_at(i, 1.0f);
                    push_pair(qi, ti, score);
                }
            }
        } else if (matches0) {
            const int rows = static_cast<int>(matches0->elements);
            for (int qi = 0; qi < rows; ++qi) {
                int ti = -1;
                if (matches0->dtype == nvinfer1::DataType::kINT32) {
                    ti = matches0->host_int32[static_cast<size_t>(qi)];
                } else if (matches0->dtype == nvinfer1::DataType::kFLOAT) {
                    const float ft = matches0->host_float[static_cast<size_t>(qi)];
                    ti = static_cast<int>(std::round(ft));
                    if (std::fabs(ft - static_cast<float>(ti)) > 1e-3f) {
                        continue;
                    }
                }
                push_pair(qi, ti, score_at(qi, 1.0f));
            }
        }
        if (index_matches.empty()) {
            result.status = "not_enough_matches";
            return result;
        }
        return build_result_from_indices(index_matches, "ok_split_matcher");
    };

    if (config_.use_lightglue &&
        matcher_.engine && matcher_.context && matcher_.bindings_ready) {
        NeuralFeatureMatchResult split = run_split_matcher();
        if (split.status != "unsupported_split_matcher_schema") {
            return split;
        }
    }

    auto dot = [](const DirectFeature& a, const DirectFeature& b) {
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

    std::vector<IndexMatch> mutual_matches;
    mutual_matches.reserve(left_features.size());
    for (size_t i = 0; i < left_features.size(); ++i) {
        const int j = left_best[i];
        if (j < 0 || j >= static_cast<int>(right_features.size()) ||
            right_best[static_cast<size_t>(j)] != static_cast<int>(i)) {
            continue;
        }
        IndexMatch im;
        im.query_idx = static_cast<int>(i);
        im.train_idx = j;
        im.score = left_score[i];
        mutual_matches.push_back(im);
    }
    return build_result_from_indices(mutual_matches, "ok_cpu");
}


}  // namespace stereo3d
