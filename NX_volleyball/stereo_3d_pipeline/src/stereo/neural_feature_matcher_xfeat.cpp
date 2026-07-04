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
                       const uint8_t* bgr, int bgr_pitch,
                       XFeatRawOutput& raw) -> bool {
        float* dst = static_cast<float*>(input->device);
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
        if (!extractor_.context->enqueueV3(stream)) return false;
        return copy_outputs(raw);
    };

    const auto start = std::chrono::steady_clock::now();
    XFeatRawOutput left_raw;
    XFeatRawOutput right_raw;
    if (input_channels == 3 &&
        (!left_bgr_gpu || !right_bgr_gpu ||
         left_bgr_pitch <= 0 || right_bgr_pitch <= 0)) {
        out.status = "unsupported_input_schema";
        return out;
    }
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


}  // namespace stereo3d
