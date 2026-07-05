#include "neural_feature_matcher.h"

#include "neural_feature_matcher_helpers.h"
#include "track/crop_resize.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace stereo3d {

NeuralFeatureMatchResult NeuralFeatureMatcher::matchXFeatExtractorGpuRoi(
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
    if (!feats || !keypoints || !heatmap) {
        out.status = "unsupported_extractor_schema";
        return out;
    }

    auto copy_outputs = [&](XFeatRawOutput& raw) -> bool {
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
    const auto tensor_batch = [](const nvinfer1::Dims& dims) {
        return dims.nbDims == 4 ? static_cast<int>(dims.d[0]) : 1;
    };
    const int input_batch = tensor_batch(input->dims);
    const int output_batch = std::min({tensor_batch(feats->dims),
                                       tensor_batch(keypoints->dims),
                                       tensor_batch(heatmap->dims)});
    const size_t input_batch_stride =
        static_cast<size_t>(std::max(1, input_channels)) *
        static_cast<size_t>(config_.roi_size) *
        static_cast<size_t>(config_.roi_size);
    auto crop_one = [&](const Detection& det,
                        const uint8_t* gray, int pitch,
                        const uint8_t* bgr, int bgr_pitch,
                        int batch_index = 0) -> bool {
        if (batch_index < 0 || batch_index >= std::max(1, input_batch)) {
            return false;
        }
        float* dst = static_cast<float*>(input->device) +
                     static_cast<size_t>(batch_index) * input_batch_stride;
        if (input_channels == 1) {
            cropResizeGPU(gray, pitch, img_width, img_height,
                          dst, config_.roi_size,
                          det.cx, det.cy, det.width, det.height,
                          context, stream);
        } else if (input_channels == 3) {
            if (!bgr || bgr_pitch <= 0) {
                return false;
            }
            cropResizeBgrGPU_3ch(bgr, bgr_pitch, img_width, img_height,
                                  dst, config_.roi_size,
                                  det.cx, det.cy, det.width, det.height,
                                  context, stream);
        } else {
            return false;
        }
        return cudaPeekAtLastError() == cudaSuccess;
    };
    auto enqueue_extract = [&]() -> bool {
        return extractor_.context->enqueueV3(stream);
    };
    auto run_one_extract = [&](const Detection& det,
                               const uint8_t* gray, int pitch,
                               const uint8_t* bgr, int bgr_pitch,
                               int batch_index = 0) -> bool {
        return crop_one(det, gray, pitch, bgr, bgr_pitch, batch_index) &&
               enqueue_extract();
    };
    auto run_one = [&](const Detection& det,
                       const uint8_t* gray, int pitch,
                       const uint8_t* bgr, int bgr_pitch,
                       XFeatRawOutput& raw) -> bool {
        return run_one_extract(det, gray, pitch, bgr, bgr_pitch) &&
               copy_outputs(raw);
    };

    const auto start = std::chrono::steady_clock::now();
    if (input_channels == 3 &&
        (!left_bgr_gpu || !right_bgr_gpu ||
         left_bgr_pitch <= 0 || right_bgr_pitch <= 0)) {
        out.status = "unsupported_input_schema";
        return out;
    }

    auto append_debug_point =
        [&](float lx, float ly, float rx, float ry,
            float disparity, float score, float second_score,
            SparseFeatureDebugStage stage,
            SparseFeatureRejectReason reason) {
            if (out.debug_points.size() >=
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
            out.debug_points.push_back(p);
        };

    auto append_debug_match =
        [&](const NeuralFeaturePointMatch& m,
            SparseFeatureDebugStage stage,
            SparseFeatureRejectReason reason,
            float second_score = std::numeric_limits<float>::quiet_NaN()) {
            append_debug_point(m.left_x, m.left_y,
                               m.right_x, m.right_y,
                               m.disparity, m.score, second_score,
                               stage, reason);
        };

    auto finalize_candidates =
        [&](const std::vector<NeuralFeaturePointMatch>& candidates,
            const char* ok_status) -> bool {
            if (static_cast<int>(candidates.size()) < config_.min_matches) {
                out.status = "not_enough_matches";
                return false;
            }
            std::vector<float> disparities;
            disparities.reserve(candidates.size());
            for (const auto& m : candidates) disparities.push_back(m.disparity);
            const float median = medianOf(disparities);
            std::vector<float> abs_dev;
            abs_dev.reserve(disparities.size());
            for (float d : disparities) abs_dev.push_back(std::fabs(d - median));
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
            out.matches.clear();
            for (const auto& m : candidates) {
                if (std::fabs(m.disparity - median) > gate) {
                    append_debug_match(m,
                                       SparseFeatureDebugStage::GEOMETRY,
                                       SparseFeatureRejectReason::MAD_OUTLIER);
                    continue;
                }
                out.matches.push_back(m);
                append_debug_match(m,
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
            if (static_cast<int>(out.matches.size()) < config_.min_matches) {
                out.status = "not_enough_inliers";
                out.matches.clear();
                return false;
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
                out.status = "poor_spatial_quadrants";
                out.matches.clear();
                return false;
            }
            if (min_spread > 0.0f && spread < min_spread) {
                out.status = "poor_spatial_spread";
                out.matches.clear();
                return false;
            }

            const float kept = static_cast<float>(out.matches.size());
            out.disparity = sum / kept;
            const float var =
                std::max(0.0f, sum2 / kept - out.disparity * out.disparity);
            out.stddev_px = std::sqrt(var);
            out.depth_m = focal_ * baseline_ / std::max(0.5f, out.disparity);
            const float support_conf =
                std::min(1.0f, kept /
                                   static_cast<float>(
                                       std::max(1, config_.min_matches * 2)));
            const float score_conf =
                std::clamp((score_sum / kept + 1.0f) * 0.5f, 0.0f, 1.0f);
            const float consistency =
                std::clamp(1.0f / (1.0f + out.stddev_px), 0.0f, 1.0f);
            const float spatial_conf =
                std::clamp(0.5f * static_cast<float>(quadrants) / 4.0f +
                               0.5f * (min_spread > 0.0f
                                            ? std::min(1.0f, spread / min_spread)
                                            : 1.0f),
                           0.0f, 1.0f);
            out.confidence = std::clamp(0.40f * support_conf +
                                            0.30f * score_conf +
                                            0.20f * consistency +
                                            0.10f * spatial_conf,
                                        0.0f, 1.0f);
            out.inference_ms = static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count());
            out.valid = true;
            out.status = ok_status;
            return true;
        };

    const int feat_h = tensorHeight(feats->dims);
    const int feat_w = tensorWidth(feats->dims);
    const bool xfeat_schema_ok =
        feat_h > 0 && feat_w > 0 &&
        tensorHeight(keypoints->dims) == feat_h &&
        tensorWidth(keypoints->dims) == feat_w &&
        tensorHeight(heatmap->dims) == feat_h &&
        tensorWidth(heatmap->dims) == feat_w;
    if (!xfeat_schema_ok) {
        out.status = "unsupported_extractor_schema";
        return out;
    }

    const size_t feat_batch_stride =
        static_cast<size_t>(config_.descriptor_dim) *
        static_cast<size_t>(feat_h) * static_cast<size_t>(feat_w);
    const size_t keypoint_batch_stride =
        static_cast<size_t>(65) *
        static_cast<size_t>(feat_h) * static_cast<size_t>(feat_w);
    const size_t heatmap_batch_stride =
        static_cast<size_t>(feat_h) * static_cast<size_t>(feat_w);
    const size_t feat_batch_bytes = feat_batch_stride * sizeof(float);
    const size_t keypoint_batch_bytes = keypoint_batch_stride * sizeof(float);
    const size_t heatmap_batch_bytes = heatmap_batch_stride * sizeof(float);

    if (config_.gpu_postprocess && input_batch >= 2 && output_batch >= 2 &&
        ensureXFeatGpuWorkspace(xfeat_gpu_workspace_,
                                feat_h,
                                feat_w,
                                config_.descriptor_dim,
                                config_.top_k) &&
        crop_one(left_det,
                 left_gray_gpu, left_gray_pitch,
                 left_bgr_gpu, left_bgr_pitch,
                 0) &&
        crop_one(right_det,
                 right_gray_gpu, right_gray_pitch,
                 right_bgr_gpu, right_bgr_pitch,
                 1) &&
        enqueue_extract()) {
        const float* feat_base = static_cast<const float*>(feats->device);
        const float* keypoint_base =
            static_cast<const float*>(keypoints->device);
        const float* heatmap_base = static_cast<const float*>(heatmap->device);
        std::vector<XFeatGpuMatch> gpu_matches;
        const bool gpu_path_ok = runXFeatGpuPostprocess(
            xfeat_gpu_workspace_,
            feat_base,
            keypoint_base,
            heatmap_base,
            feat_base + feat_batch_stride,
            keypoint_base + keypoint_batch_stride,
            heatmap_base + heatmap_batch_stride,
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
        if (gpu_path_ok) {
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
                append_debug_match(m,
                                   SparseFeatureDebugStage::GEOMETRY,
                                   SparseFeatureRejectReason::NONE);
                candidates.push_back(m);
            }
            finalize_candidates(candidates, "ok_gpu_b2");
            return out;
        }
    }

    if (config_.gpu_postprocess &&
        ensureXFeatGpuWorkspace(xfeat_gpu_workspace_,
                                feat_h,
                                feat_w,
                                config_.descriptor_dim,
                                config_.top_k)) {
        bool gpu_path_ok = false;
        if (run_one_extract(left_det,
                            left_gray_gpu, left_gray_pitch,
                            left_bgr_gpu, left_bgr_pitch)) {
            gpu_path_ok =
                cudaMemcpyAsync(xfeat_gpu_workspace_.left_feats,
                                feats->device,
                                feat_batch_bytes,
                                cudaMemcpyDeviceToDevice,
                                stream) == cudaSuccess &&
                cudaMemcpyAsync(xfeat_gpu_workspace_.left_keypoints,
                                keypoints->device,
                                keypoint_batch_bytes,
                                cudaMemcpyDeviceToDevice,
                                stream) == cudaSuccess &&
                cudaMemcpyAsync(xfeat_gpu_workspace_.left_heatmap,
                                heatmap->device,
                                heatmap_batch_bytes,
                                cudaMemcpyDeviceToDevice,
                                stream) == cudaSuccess &&
                run_one_extract(right_det,
                                right_gray_gpu, right_gray_pitch,
                                right_bgr_gpu, right_bgr_pitch);
        }
        if (gpu_path_ok) {
            std::vector<XFeatGpuMatch> gpu_matches;
            gpu_path_ok = runXFeatGpuPostprocess(
                xfeat_gpu_workspace_,
                xfeat_gpu_workspace_.left_feats,
                xfeat_gpu_workspace_.left_keypoints,
                xfeat_gpu_workspace_.left_heatmap,
                static_cast<const float*>(feats->device),
                static_cast<const float*>(keypoints->device),
                static_cast<const float*>(heatmap->device),
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
            if (gpu_path_ok) {
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
                    append_debug_match(m,
                                       SparseFeatureDebugStage::GEOMETRY,
                                       SparseFeatureRejectReason::NONE);
                    candidates.push_back(m);
                }
                finalize_candidates(candidates, "ok_gpu");
                return out;
            }
        }
    }

    XFeatRawOutput left_raw;
    XFeatRawOutput right_raw;
    if (!run_one(left_det,
                 left_gray_gpu, left_gray_pitch,
                 left_bgr_gpu, left_bgr_pitch,
                 left_raw) ||
        !run_one(right_det,
                 right_gray_gpu, right_gray_pitch,
                 right_bgr_gpu, right_bgr_pitch,
                 right_raw)) {
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
    for (size_t i = 0; i < left_features.size(); ++i) {
        const int j = left_best[i];
        if (j < 0 || j >= static_cast<int>(right_features.size()) ||
            right_best[static_cast<size_t>(j)] != static_cast<int>(i)) {
            float lx = std::numeric_limits<float>::quiet_NaN();
            float ly = std::numeric_limits<float>::quiet_NaN();
            map_to_frame(left_det, left_features[i], &lx, &ly);
            append_debug_point(lx, ly,
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN(),
                               left_score[i],
                               std::numeric_limits<float>::quiet_NaN(),
                               SparseFeatureDebugStage::RAW,
                               SparseFeatureRejectReason::NO_MUTUAL);
            continue;
        }
        const float score = left_score[i];
        float lx, ly, rx, ry;
        map_to_frame(left_det, left_features[i], &lx, &ly);
        map_to_frame(right_det, right_features[static_cast<size_t>(j)], &rx, &ry);
        const float disp = lx - rx;
        if (score < config_.min_score) {
            append_debug_point(lx, ly, rx, ry, disp, score,
                               std::numeric_limits<float>::quiet_NaN(),
                               SparseFeatureDebugStage::MATCH,
                               SparseFeatureRejectReason::LOW_SCORE);
            continue;
        }
        if (disp <= 0.5f || disp > static_cast<float>(max_disparity_)) {
            append_debug_point(lx, ly, rx, ry, disp, score,
                               std::numeric_limits<float>::quiet_NaN(),
                               SparseFeatureDebugStage::MATCH,
                               SparseFeatureRejectReason::BAD_DISPARITY);
            continue;
        }
        if (std::fabs(ly - ry) > config_.max_y_error_px) {
            append_debug_point(lx, ly, rx, ry, disp, score,
                               std::numeric_limits<float>::quiet_NaN(),
                               SparseFeatureDebugStage::MATCH,
                               SparseFeatureRejectReason::Y_RESIDUAL);
            continue;
        }
        if (std::fabs(disp - initial_disparity) > config_.max_disp_delta_px) {
            append_debug_point(lx, ly, rx, ry, disp, score,
                               std::numeric_limits<float>::quiet_NaN(),
                               SparseFeatureDebugStage::MATCH,
                               SparseFeatureRejectReason::DISP_DELTA);
            continue;
        }
        NeuralFeaturePointMatch m;
        m.left_x = lx;
        m.left_y = ly;
        m.right_x = rx;
        m.right_y = ry;
        m.disparity = disp;
        m.score = score;
        append_debug_match(m,
                           SparseFeatureDebugStage::GEOMETRY,
                           SparseFeatureRejectReason::NONE);
        candidates.push_back(m);
    }

    finalize_candidates(candidates, "ok_cpu");
    return out;
}


}  // namespace stereo3d
