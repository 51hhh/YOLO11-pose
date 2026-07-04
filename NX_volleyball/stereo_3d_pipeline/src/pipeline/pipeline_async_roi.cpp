/**
 * @file pipeline_async_roi.cpp
 * @brief Async ROI Stage2 worker and snapshot management.
 */

#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "../stereo/roi_feature_match_gpu.h"
#include "../stereo/neural_feature_matcher.h"
#include "../utils/logger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cuda_runtime.h>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace stereo3d {

namespace {

using DiagnosticGpuMatchFn = SparseFeatureDisparityResult (*)(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream);

bool asyncRoiSubpixelDepthEnabled(const PipelineConfig::DualYoloConfig& cfg) {
    if (!cfg.depth_roi_subpixel || !cfg.subpixel_enabled) return false;
    std::string solver = cfg.depth_solver;
    std::transform(solver.begin(), solver.end(), solver.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return solver == "roi_subpixel_match" ||
           solver == "subpixel" ||
           solver == "multi_point";
}

bool asyncRoiNeedsHostImages(const PipelineConfig::DualYoloConfig& cfg) {
    const bool cpu_descriptor =
        cfg.depth_roi_brisk_points ||
        cfg.depth_roi_akaze_points ||
        cfg.depth_roi_sift_points;
    const bool cpu_fallback = cfg.fallback_epipolar_search &&
        (cfg.depth_epipolar_fallback ||
         cfg.depth_fallback_template ||
         cfg.depth_fallback_feature_points);
    if (cfg.gpu_candidate_refine) {
        return cpu_descriptor || cpu_fallback;
    }
    const bool circle_seed_refine = cfg.center_refine &&
        (cfg.depth_circle_center ||
         cfg.depth_circle_edges ||
         cfg.depth_roi_edge_centroid ||
         cfg.depth_roi_center_patch ||
         asyncRoiSubpixelDepthEnabled(cfg) ||
         cpu_fallback);
    return circle_seed_refine ||
           cfg.depth_roi_radial_center ||
           cfg.depth_roi_edge_pair_center ||
           cfg.depth_roi_corner_points ||
           cfg.depth_roi_texture_points ||
           cfg.depth_roi_binary_points ||
           cfg.depth_roi_orb_points ||
           cpu_descriptor ||
           cfg.depth_roi_iou_region_color_patch ||
           cfg.depth_roi_patch_iou_color_edge ||
           asyncRoiSubpixelDepthEnabled(cfg) ||
           cpu_fallback;
}

void restoreFrameMetadata(FrameSlot& slot, const FrameMetadata& meta) {
    slot.left_timestamp_us = meta.left_timestamp_us;
    slot.right_timestamp_us = meta.right_timestamp_us;
    slot.left_frame_number = meta.left_frame_number;
    slot.right_frame_number = meta.right_frame_number;
    slot.left_frame_counter = meta.left_frame_counter;
    slot.right_frame_counter = meta.right_frame_counter;
    slot.left_trigger_index = meta.left_trigger_index;
    slot.right_trigger_index = meta.right_trigger_index;
    slot.grab_failed = meta.grab_failed;
    slot.is_detect_frame = meta.is_detect_frame;
    slot.p2_depth_modes_enabled = meta.p2_depth_modes_enabled;
    slot.p2_depth_mode_mask = meta.p2_depth_mode_mask;
    slot.p2_feature_job_scaffold_enabled =
        meta.p2_feature_job_scaffold_enabled;
    slot.p2_realtime_requested = meta.p2_realtime_requested;
    slot.p2_diagnostic_requested = meta.p2_diagnostic_requested;
    slot.p2_realtime_triggers = meta.p2_realtime_triggers;
    slot.p2_diagnostic_triggers = meta.p2_diagnostic_triggers;
    slot.p2_realtime_skip_reasons = meta.p2_realtime_skip_reasons;
    slot.p2_diagnostic_skip_reasons = meta.p2_diagnostic_skip_reasons;
    slot.p2_feature_job_count = meta.p2_feature_job_count;
    slot.p2_left_count = meta.p2_left_count;
    slot.p2_right_count = meta.p2_right_count;
    slot.p2_valid_direct_pair_count = meta.p2_valid_direct_pair_count;
}

void applyP2FeatureJobDecisionToSlot(
    FrameSlot& slot,
    const P2FeatureJobDecision& decision,
    const std::vector<P2FeatureJobDescriptor>& jobs) {
    slot.p2_depth_modes_enabled = decision.p2_depth_modes_enabled;
    slot.p2_depth_mode_mask = decision.depth_mode_mask;
    slot.p2_feature_job_scaffold_enabled = decision.split_feature_jobs;
    slot.p2_realtime_requested = decision.realtime_requested;
    slot.p2_diagnostic_requested = decision.diagnostic_requested;
    slot.p2_realtime_triggers = decision.realtime_triggers;
    slot.p2_diagnostic_triggers = decision.diagnostic_triggers;
    slot.p2_realtime_skip_reasons = decision.realtime_skip_reasons;
    slot.p2_diagnostic_skip_reasons = decision.diagnostic_skip_reasons;
    slot.p2_feature_job_count = static_cast<int>(jobs.size());
    slot.p2_left_count = decision.left_count;
    slot.p2_right_count = decision.right_count;
    slot.p2_valid_direct_pair_count = decision.valid_direct_pair_count;
}

cudaError_t createLowPriorityNonBlockingStream(cudaStream_t* stream,
                                               const char* name) {
    int least_priority = 0;
    int greatest_priority = 0;
    const cudaError_t priority_err =
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    (void)greatest_priority;
    if (priority_err == cudaSuccess) {
        cudaError_t err = cudaStreamCreateWithPriority(
            stream, cudaStreamNonBlocking, least_priority);
        if (err == cudaSuccess) {
            return err;
        }
        LOG_WARN("Create low-priority %s stream failed (%s), falling back to "
                 "default priority",
                 name ? name : "cuda",
                 cudaGetErrorString(err));
    }
    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
}

bool chooseDiagnosticDirectPair(const std::vector<Detection>& left,
                                const std::vector<Detection>& right,
                                int max_disparity,
                                Detection* left_out,
                                Detection* right_out,
                                float* disparity_out) {
    if (!left_out || !right_out || !disparity_out) {
        return false;
    }
    float best_score = -1.0f;
    Detection best_left;
    Detection best_right;
    float best_disp = -1.0f;
    for (const auto& l : left) {
        for (const auto& r : right) {
            const float disp = l.cx - r.cx;
            if (!std::isfinite(disp) || disp <= 0.0f ||
                disp > static_cast<float>(std::max(1, max_disparity))) {
                continue;
            }
            const float y_delta = std::fabs(l.cy - r.cy);
            const float score =
                (l.confidence + r.confidence) * 0.5f -
                y_delta * 0.01f;
            if (score > best_score) {
                best_score = score;
                best_left = l;
                best_right = r;
                best_disp = disp;
            }
        }
    }
    if (best_score < 0.0f) {
        return false;
    }
    *left_out = best_left;
    *right_out = best_right;
    *disparity_out = best_disp;
    return true;
}

const char* p2DiagnosticModeName(uint32_t mode_bit) {
    switch (mode_bit) {
    case P2_DEPTH_MODE_ORB_POINTS:
        return "opencv_cuda_orb";
    case P2_DEPTH_MODE_CUDA_TEMPLATE:
        return "cuda_template";
    case P2_DEPTH_MODE_CUDA_STEREO_BM:
        return "cuda_stereo_bm";
    case P2_DEPTH_MODE_CUDA_STEREO_SGM:
        return "cuda_stereo_sgm";
    case P2_DEPTH_MODE_RING_EDGE_PROFILE:
        return "cuda_ring_edge_profile";
    case P2_DEPTH_MODE_VPI_TEMPLATE:
        return "vpi_template_match";
    case P2_DEPTH_MODE_VPI_STEREO:
        return "vpi_stereo_disparity";
    case P2_DEPTH_MODE_VPI_HARRIS_LK:
        return "vpi_harris_lk";
    case P2_DEPTH_MODE_VPI_ORB:
        return "vpi_orb";
    case P2_DEPTH_MODE_CUDA_GFTT_LK:
        return "opencv_cuda_gftt_lk";
    case P2_DEPTH_MODE_CUDA_SIFT:
        return "cuda_sift";
    case P2_DEPTH_MODE_LIBSGM:
        return "fixstars_libsgm";
    case P2_DEPTH_MODE_CUDA_HOUGH_CIRCLE:
        return "cuda_hough_circle";
    default:
        return "unknown";
    }
}

float p2DepthFromDisparity(float disparity, float focal, float baseline) {
    if (!std::isfinite(disparity) || disparity <= 0.0f ||
        !std::isfinite(focal) || focal <= 0.0f ||
        !std::isfinite(baseline) || baseline <= 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return focal * baseline / disparity;
}

void writeCsvFloat(std::ostream& out, float value) {
    if (std::isfinite(value)) {
        out << std::fixed << std::setprecision(6) << value;
    }
}

void writeCsvDouble(std::ostream& out, double value) {
    if (std::isfinite(value)) {
        out << std::fixed << std::setprecision(6) << value;
    }
}

void writeP2FeatureDiagnosticResultHeader(std::ostream& out) {
    out << "frame_id,left_timestamp_us,right_timestamp_us,"
        << "left_frame_number,right_frame_number,"
        << "left_frame_counter,right_frame_counter,"
        << "left_trigger_index,right_trigger_index,"
        << "lane,mode,status,valid,low_confidence,"
        << "disparity,z_m,confidence,stddev,support,attempted,"
        << "initial_disparity,"
        << "left_cx,left_cy,left_w,left_h,left_conf,"
        << "right_cx,right_cy,right_w,right_h,right_conf,"
        << "anchor_cx,anchor_cy,right_anchor_cx,right_anchor_cy,"
        << "debug_match_count,artifact_path,"
        << "algo_ms,queue_wait_ms,worker_elapsed_ms,deadline_ms,"
        << "over_deadline,depth_mode_mask,triggers\n";
}

bool finitePositive(float value) {
    return std::isfinite(value) && value > 0.0f;
}

bool finitePoint(float x, float y) {
    return std::isfinite(x) && std::isfinite(y);
}

std::string sanitizeArtifactToken(std::string token) {
    if (token.empty()) {
        return "unknown";
    }
    for (char& c : token) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') {
            c = '_';
        }
    }
    return token;
}

cv::Rect2f detectionRect(const Detection& det) {
    if (!finitePoint(det.cx, det.cy) ||
        !finitePositive(det.width) ||
        !finitePositive(det.height)) {
        return cv::Rect2f();
    }
    return cv::Rect2f(det.cx - det.width * 0.5f,
                      det.cy - det.height * 0.5f,
                      det.width,
                      det.height);
}

void includePointBounds(float x,
                        float y,
                        float& min_x,
                        float& min_y,
                        float& max_x,
                        float& max_y) {
    if (!finitePoint(x, y)) {
        return;
    }
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
}

void includeRectBounds(const cv::Rect2f& rect,
                       float& min_x,
                       float& min_y,
                       float& max_x,
                       float& max_y) {
    if (rect.width <= 0.0f || rect.height <= 0.0f) {
        return;
    }
    includePointBounds(rect.x, rect.y, min_x, min_y, max_x, max_y);
    includePointBounds(rect.x + rect.width,
                       rect.y + rect.height,
                       min_x, min_y, max_x, max_y);
}

cv::Rect cropAround(float cx,
                    float cy,
                    int crop_w,
                    int crop_h,
                    int img_w,
                    int img_h) {
    crop_w = std::clamp(crop_w, 1, std::max(1, img_w));
    crop_h = std::clamp(crop_h, 1, std::max(1, img_h));
    int x = static_cast<int>(std::lround(cx - static_cast<float>(crop_w) * 0.5f));
    int y = static_cast<int>(std::lround(cy - static_cast<float>(crop_h) * 0.5f));
    x = std::clamp(x, 0, std::max(0, img_w - crop_w));
    y = std::clamp(y, 0, std::max(0, img_h - crop_h));
    return cv::Rect(x, y, crop_w, crop_h);
}

cv::Point panelPoint(float x,
                     float y,
                     const cv::Rect& crop,
                     float scale,
                     int x_offset,
                     int y_offset) {
    return cv::Point(
        x_offset + static_cast<int>(std::lround((x - crop.x) * scale)),
        y_offset + static_cast<int>(std::lround((y - crop.y) * scale)));
}

void drawDetectionOnPanel(cv::Mat& canvas,
                          const Detection& det,
                          const cv::Rect& crop,
                          float scale,
                          int x_offset,
                          int y_offset,
                          const cv::Scalar& color) {
    const cv::Rect2f rect = detectionRect(det);
    if (rect.width <= 0.0f || rect.height <= 0.0f) {
        return;
    }
    const cv::Point p0 = panelPoint(rect.x, rect.y, crop, scale, x_offset, y_offset);
    const cv::Point p1 = panelPoint(rect.x + rect.width,
                                    rect.y + rect.height,
                                    crop, scale, x_offset, y_offset);
    cv::rectangle(canvas, p0, p1, color, 2, cv::LINE_AA);
}

void drawCross(cv::Mat& canvas,
               const cv::Point& p,
               const cv::Scalar& color,
               int radius = 7) {
    cv::line(canvas, cv::Point(p.x - radius, p.y),
             cv::Point(p.x + radius, p.y), color, 2, cv::LINE_AA);
    cv::line(canvas, cv::Point(p.x, p.y - radius),
             cv::Point(p.x, p.y + radius), color, 2, cv::LINE_AA);
}

cv::Mat renderDebugPatchPlane(const std::vector<float>& values,
                              int width,
                              int height,
                              float configured_min,
                              float configured_max,
                              int target_w,
                              int target_h) {
    if (width <= 0 || height <= 0 ||
        static_cast<int>(values.size()) < width * height ||
        target_w <= 0 || target_h <= 0) {
        return cv::Mat();
    }

    float min_value = configured_min;
    float max_value = configured_max;
    if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
        max_value <= min_value) {
        min_value = std::numeric_limits<float>::max();
        max_value = std::numeric_limits<float>::lowest();
        for (int i = 0; i < width * height; ++i) {
            const float v = values[static_cast<size_t>(i)];
            if (!std::isfinite(v)) {
                continue;
            }
            min_value = std::min(min_value, v);
            max_value = std::max(max_value, v);
        }
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
        max_value <= min_value) {
        return cv::Mat();
    }

    cv::Mat gray(height, width, CV_8UC1, cv::Scalar(0));
    const float inv_range = 1.0f / std::max(1e-6f, max_value - min_value);
    for (int y = 0; y < height; ++y) {
        auto* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            const float v = values[static_cast<size_t>(y * width + x)];
            if (!std::isfinite(v)) {
                row[x] = 0;
                continue;
            }
            const float t = std::clamp((v - min_value) * inv_range, 0.0f, 1.0f);
            row[x] = static_cast<uint8_t>(std::lround(24.0f + t * 231.0f));
        }
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(target_w, target_h),
               0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat color;
    cv::applyColorMap(resized, color, cv::COLORMAP_TURBO);
    return color;
}

void drawDebugPatchPanel(cv::Mat& canvas,
                         const SparseFeatureDebugPatch& patch,
                         int y_offset,
                         int width,
                         const cv::Scalar& white,
                         const cv::Scalar& muted) {
    if (!patch.valid || patch.width <= 0 || patch.height <= 0 ||
        patch.disparity.empty() || y_offset >= canvas.rows) {
        return;
    }
    constexpr int kInnerPad = 8;
    constexpr int kLabelH = 24;
    constexpr int kGap = 12;
    const int available_h = canvas.rows - y_offset - kInnerPad;
    if (available_h <= kLabelH + 8) {
        return;
    }
    const bool draw_conf = patch.has_confidence &&
        !patch.confidence.empty() &&
        static_cast<int>(patch.confidence.size()) >= patch.width * patch.height;
    const int panel_w = draw_conf
        ? std::max(1, (width - kGap) / 2)
        : std::max(1, width);
    const int heat_h = std::max(1, available_h - kLabelH);

    auto draw_one = [&](const char* title,
                        const std::vector<float>& values,
                        float min_value,
                        float max_value,
                        int x_offset) {
        cv::putText(canvas, title, cv::Point(x_offset, y_offset + 17),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, white, 1, cv::LINE_AA);
        cv::Mat heat = renderDebugPatchPlane(
            values, patch.width, patch.height, min_value, max_value,
            panel_w, heat_h);
        if (heat.empty()) {
            cv::putText(canvas, "no finite values",
                        cv::Point(x_offset, y_offset + kLabelH + 24),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, muted, 1, cv::LINE_AA);
            return;
        }
        heat.copyTo(canvas(cv::Rect(x_offset, y_offset + kLabelH,
                                    heat.cols, heat.rows)));
        cv::rectangle(canvas,
                      cv::Rect(x_offset, y_offset + kLabelH,
                               heat.cols, heat.rows),
                      muted, 1, cv::LINE_AA);
    };

    draw_one(patch.disparity_is_score ? "SCORE PATCH" : "DISPARITY PATCH",
             patch.disparity,
             patch.disparity_min, patch.disparity_max, 0);
    if (draw_conf) {
        draw_one("CONFIDENCE PATCH", patch.confidence,
                 patch.confidence_min, patch.confidence_max,
                 panel_w + kGap);
    }
}

bool p2DiagnosticOnlyFeatureJobsEnabled(const PipelineConfig& config) {
    return config.p2_feature_job_scaffold_enabled &&
           config.p2_diagnostic_lane_decision_enabled &&
           !config.p2_realtime_lane_decision_enabled;
}

}  // namespace

bool Pipeline::asyncRoiStage2Configured() const {
    return config_.async_roi_stage2 &&
           config_.disparity_strategy == DisparityStrategy::ROI_ONLY &&
           !config_.detection_only &&
           (!config_.tracker.enabled || config_.tracker.detect_interval <= 1);
}

bool Pipeline::initAsyncRoiStage2() {
    if (async_roi_ready_) {
        return true;
    }

    const int buffer_count = std::clamp(config_.async_roi_buffers, 2, 8);
    config_.async_roi_buffers = buffer_count;
    config_.async_roi_deadline_ms =
        std::max(1.0f, config_.async_roi_deadline_ms);

    cudaError_t err =
        createLowPriorityNonBlockingStream(&async_roi_stream_, "Async ROI worker");
    if (err != cudaSuccess) {
        LOG_ERROR("Async ROI: create worker stream failed: %s",
                  cudaGetErrorString(err));
        return false;
    }
    err = createLowPriorityNonBlockingStream(&async_roi_copy_stream_,
                                             "Async ROI copy");
    if (err != cudaSuccess) {
        LOG_ERROR("Async ROI: create copy stream failed: %s",
                  cudaGetErrorString(err));
        destroyAsyncRoiStage2();
        return false;
    }

    for (auto& evt : async_roi_slot_copy_done_) {
        err = cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            LOG_ERROR("Async ROI: create slot copy event failed: %s",
                      cudaGetErrorString(err));
            destroyAsyncRoiStage2();
            return false;
        }
    }
    async_roi_slot_copy_pending_.fill(false);

    async_roi_buffers_.resize(static_cast<size_t>(buffer_count));
    const size_t gray_width_bytes = static_cast<size_t>(config_.rect_width);
    const size_t bgr_width_bytes =
        static_cast<size_t>(config_.rect_width) * 3u;
    const size_t rows = static_cast<size_t>(config_.rect_height);
    const bool neural_needs_bgr =
        neural_feature_matcher_ && neural_feature_matcher_->requiresBgrInput();
    const bool allocate_host_gray =
        asyncRoiNeedsHostImages(config_.dual_yolo);
    const bool allocate_bgr =
        colorPipelineEnabled() &&
        (config_.dual_yolo.depth_roi_iou_region_color_patch ||
         config_.dual_yolo.depth_roi_patch_iou_color_edge ||
         neural_needs_bgr);

    for (int i = 0; i < buffer_count; ++i) {
        auto& b = async_roi_buffers_[static_cast<size_t>(i)];
        err = cudaEventCreateWithFlags(&b.copy_done, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            LOG_ERROR("Async ROI: create buffer copy event %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyAsyncRoiStage2();
            return false;
        }
        err = cudaEventCreateWithFlags(&b.p2_diag_copy_done,
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) {
            LOG_ERROR("Async ROI: create P2 diagnostic copy event %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyAsyncRoiStage2();
            return false;
        }
        err = cudaMallocPitch(reinterpret_cast<void**>(&b.left_gray_gpu),
                              &b.left_gray_pitch,
                              gray_width_bytes, rows);
        if (err != cudaSuccess) {
            LOG_ERROR("Async ROI: alloc left gray buffer %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyAsyncRoiStage2();
            return false;
        }
        err = cudaMallocPitch(reinterpret_cast<void**>(&b.right_gray_gpu),
                              &b.right_gray_pitch,
                              gray_width_bytes, rows);
        if (err != cudaSuccess) {
            LOG_ERROR("Async ROI: alloc right gray buffer %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyAsyncRoiStage2();
            return false;
        }
        if (allocate_host_gray) {
            b.left_gray_host_pitch = gray_width_bytes;
            b.right_gray_host_pitch = gray_width_bytes;
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.left_gray_host),
                                gray_width_bytes * rows,
                                cudaHostAllocDefault);
            if (err != cudaSuccess) {
                LOG_ERROR("Async ROI: alloc left gray host buffer %d failed: %s",
                          i, cudaGetErrorString(err));
                destroyAsyncRoiStage2();
                return false;
            }
            err = cudaHostAlloc(reinterpret_cast<void**>(&b.right_gray_host),
                                gray_width_bytes * rows,
                                cudaHostAllocDefault);
            if (err != cudaSuccess) {
                LOG_ERROR("Async ROI: alloc right gray host buffer %d failed: %s",
                          i, cudaGetErrorString(err));
                destroyAsyncRoiStage2();
                return false;
            }
        }

        if (allocate_bgr) {
            err = cudaMallocPitch(reinterpret_cast<void**>(&b.left_bgr_gpu),
                                  &b.left_bgr_pitch,
                                  bgr_width_bytes, rows);
            if (err != cudaSuccess) {
                LOG_ERROR("Async ROI: alloc left BGR buffer %d failed: %s",
                          i, cudaGetErrorString(err));
                destroyAsyncRoiStage2();
                return false;
            }
            err = cudaMallocPitch(reinterpret_cast<void**>(&b.right_bgr_gpu),
                                  &b.right_bgr_pitch,
                                  bgr_width_bytes, rows);
            if (err != cudaSuccess) {
                LOG_ERROR("Async ROI: alloc right BGR buffer %d failed: %s",
                          i, cudaGetErrorString(err));
                destroyAsyncRoiStage2();
                return false;
            }
        }
        async_roi_free_buffers_.push_back(i);
    }

    async_roi_thread_stop_ = false;
    async_roi_expire_before_frame_ = -1;
    async_roi_ready_ = true;
    LOG_INFO("Async ROI Stage2 buffers ready: count=%d gray=%dx%d host_gray=%d bgr=%d",
             buffer_count, config_.rect_width, config_.rect_height,
             allocate_host_gray ? 1 : 0,
             allocate_bgr ? 1 : 0);
    return true;
}

bool Pipeline::startAsyncRoiStage2() {
    if (!async_roi_ready_) {
        return true;
    }
    if (async_roi_thread_.joinable()) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        async_roi_thread_stop_ = false;
        async_roi_expire_before_frame_ = -1;
        async_roi_completed_.clear();
    }
    async_roi_thread_ = std::thread(&Pipeline::asyncRoiWorkerLoop, this);
    LOG_INFO("Async ROI Stage2 worker started");
    return true;
}

bool Pipeline::p2FeatureDiagnosticLaneConfigured() const {
    return config_.p2_feature_job_scaffold_enabled &&
           config_.p2_diagnostic_lane_decision_enabled &&
           pipelineP2DepthModesEnabled(config_);
}

bool Pipeline::startP2FeatureDiagnosticLane() {
    if (!p2FeatureDiagnosticLaneConfigured()) {
        return true;
    }
    if (p2_feature_diag_thread_.joinable()) {
        return true;
    }
    if (!initP2FeatureDiagnosticBuffers()) {
        return false;
    }
    p2_feature_diag_artifacts_saved_ = 0;
    openP2FeatureDiagnosticResults();
    {
        std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
        p2_feature_diag_thread_stop_ = false;
        p2_feature_diag_worker_busy_ = false;
        p2_feature_diag_pending_.clear();
    }
    p2_feature_diag_thread_ =
        std::thread(&Pipeline::p2FeatureDiagnosticWorkerLoop, this);
    LOG_INFO("P2 diagnostic FeatureJob worker started");
    return true;
}

void Pipeline::shutdownP2FeatureDiagnosticLane() {
    if (!p2_feature_diag_thread_.joinable()) {
        destroyP2FeatureDiagnosticBuffers();
        return;
    }
    {
        std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
        p2_feature_diag_thread_stop_ = true;
        if (!p2_feature_diag_pending_.empty()) {
            globalPerf().record("Stage2_P2FeatureJobDiagnosticDropShutdown",
                                static_cast<double>(
                                    p2_feature_diag_pending_.size()));
            while (!p2_feature_diag_pending_.empty()) {
                const int buffer_index =
                    p2_feature_diag_pending_.front().buffer_index;
                if (buffer_index >= 0) {
                    p2_feature_diag_free_buffers_.push_back(buffer_index);
                }
                p2_feature_diag_pending_.pop_front();
            }
        }
    }
    p2_feature_diag_cv_.notify_all();
    p2_feature_diag_thread_.join();
    closeP2FeatureDiagnosticResults();
    {
        std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
        p2_feature_diag_worker_busy_ = false;
        p2_feature_diag_thread_stop_ = false;
    }
    destroyP2FeatureDiagnosticBuffers();
}

bool Pipeline::initP2FeatureDiagnosticBuffers() {
    if (!p2_feature_diag_buffers_.empty()) {
        return true;
    }
    cudaError_t err = createLowPriorityNonBlockingStream(
        &p2_feature_diag_stream_, "P2 diagnostic worker");
    if (err != cudaSuccess) {
        LOG_ERROR("P2 diagnostic: create worker stream failed: %s",
                  cudaGetErrorString(err));
        destroyP2FeatureDiagnosticBuffers();
        return false;
    }
    err = createLowPriorityNonBlockingStream(
        &p2_feature_diag_copy_stream_, "P2 diagnostic copy");
    if (err != cudaSuccess) {
        LOG_ERROR("P2 diagnostic: create copy stream failed: %s",
                  cudaGetErrorString(err));
        destroyP2FeatureDiagnosticBuffers();
        return false;
    }

    const int buffer_count =
        std::clamp(config_.p2_diagnostic_max_in_flight, 1, 8);
    const size_t gray_width_bytes =
        static_cast<size_t>(config_.rect_width);
    const size_t rows = static_cast<size_t>(config_.rect_height);
    p2_feature_diag_buffers_.resize(static_cast<size_t>(buffer_count));
    for (int i = 0; i < buffer_count; ++i) {
        auto& b = p2_feature_diag_buffers_[static_cast<size_t>(i)];
        err = cudaMallocPitch(reinterpret_cast<void**>(&b.left_gray_gpu),
                              &b.left_gray_pitch,
                              gray_width_bytes, rows);
        if (err != cudaSuccess) {
            LOG_ERROR("P2 diagnostic: alloc left gray buffer %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyP2FeatureDiagnosticBuffers();
            return false;
        }
        err = cudaMallocPitch(reinterpret_cast<void**>(&b.right_gray_gpu),
                              &b.right_gray_pitch,
                              gray_width_bytes, rows);
        if (err != cudaSuccess) {
            LOG_ERROR("P2 diagnostic: alloc right gray buffer %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyP2FeatureDiagnosticBuffers();
            return false;
        }
        err = cudaEventCreateWithFlags(&b.copy_done, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            LOG_ERROR("P2 diagnostic: create copy event %d failed: %s",
                      i, cudaGetErrorString(err));
            destroyP2FeatureDiagnosticBuffers();
            return false;
        }
        p2_feature_diag_free_buffers_.push_back(i);
    }
    LOG_INFO("P2 diagnostic FeatureJob buffers ready: count=%d gray=%dx%d",
             buffer_count, config_.rect_width, config_.rect_height);
    return true;
}

void Pipeline::destroyP2FeatureDiagnosticBuffers() {
    for (auto& b : p2_feature_diag_buffers_) {
        if (b.copy_done) {
            if (b.copy_event_recorded) {
                cudaEventSynchronize(b.copy_done);
                b.copy_event_recorded = false;
            }
            cudaEventDestroy(b.copy_done);
            b.copy_done = nullptr;
        }
        if (b.left_gray_gpu) {
            cudaFree(b.left_gray_gpu);
            b.left_gray_gpu = nullptr;
        }
        if (b.right_gray_gpu) {
            cudaFree(b.right_gray_gpu);
            b.right_gray_gpu = nullptr;
        }
        b.left_gray_pitch = 0;
        b.right_gray_pitch = 0;
    }
    p2_feature_diag_buffers_.clear();
    p2_feature_diag_free_buffers_.clear();
    p2_feature_diag_pending_.clear();
    if (p2_feature_diag_stream_) {
        cudaStreamDestroy(p2_feature_diag_stream_);
        p2_feature_diag_stream_ = nullptr;
    }
    if (p2_feature_diag_copy_stream_) {
        cudaStreamDestroy(p2_feature_diag_copy_stream_);
        p2_feature_diag_copy_stream_ = nullptr;
    }
    closeP2FeatureDiagnosticResults();
}

bool Pipeline::openP2FeatureDiagnosticResults() {
    if (!config_.p2_diagnostic_results_enabled) {
        return true;
    }
    if (config_.p2_diagnostic_results_path.empty()) {
        LOG_WARN("P2 diagnostic results enabled but path is empty; results "
                 "CSV disabled");
        config_.p2_diagnostic_results_enabled = false;
        return true;
    }

    namespace fs = std::filesystem;
    std::lock_guard<std::mutex> lk(p2_feature_diag_results_mutex_);
    if (p2_feature_diag_results_file_.is_open()) {
        return true;
    }
    const fs::path path(config_.p2_diagnostic_results_path);
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            LOG_WARN("P2 diagnostic: create results directory failed: %s (%s); "
                     "results CSV disabled",
                      parent.string().c_str(),
                      ec.message().c_str());
            config_.p2_diagnostic_results_enabled = false;
            return true;
        }
    }
    p2_feature_diag_results_file_.open(path.string(),
                                       std::ios::out | std::ios::trunc);
    if (!p2_feature_diag_results_file_.is_open()) {
        LOG_WARN("P2 diagnostic: failed to open results CSV: %s; disabled",
                  path.string().c_str());
        config_.p2_diagnostic_results_enabled = false;
        return true;
    }
    writeP2FeatureDiagnosticResultHeader(p2_feature_diag_results_file_);
    p2_feature_diag_results_file_.flush();
    LOG_INFO("P2 diagnostic results CSV: %s", path.string().c_str());
    return true;
}

void Pipeline::closeP2FeatureDiagnosticResults() {
    std::lock_guard<std::mutex> lk(p2_feature_diag_results_mutex_);
    if (p2_feature_diag_results_file_.is_open()) {
        p2_feature_diag_results_file_.flush();
        p2_feature_diag_results_file_.close();
    }
}

void Pipeline::writeP2FeatureDiagnosticResults(
    const std::vector<P2FeatureDiagnosticResultRow>& rows) {
    if (rows.empty() || !config_.p2_diagnostic_results_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lk(p2_feature_diag_results_mutex_);
    if (!p2_feature_diag_results_file_.is_open()) {
        return;
    }
    for (const auto& row : rows) {
        p2_feature_diag_results_file_ << row.frame_id << ","
            << row.metadata.left_timestamp_us << ","
            << row.metadata.right_timestamp_us << ","
            << row.metadata.left_frame_number << ","
            << row.metadata.right_frame_number << ","
            << row.metadata.left_frame_counter << ","
            << row.metadata.right_frame_counter << ","
            << row.metadata.left_trigger_index << ","
            << row.metadata.right_trigger_index << ","
            << row.lane << "," << row.mode << "," << row.status << ","
            << (row.valid ? 1 : 0) << ","
            << (row.low_confidence ? 1 : 0) << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.disparity);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.z_m);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.confidence);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.stddev);
        p2_feature_diag_results_file_ << ","
            << row.support << "," << row.attempted << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.initial_disparity);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.left_det.cx);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.left_det.cy);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.left_det.width);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.left_det.height);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.left_det.confidence);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_det.cx);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_det.cy);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_det.width);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_det.height);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_det.confidence);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.anchor_cx);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.anchor_cy);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_anchor_cx);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.right_anchor_cy);
        p2_feature_diag_results_file_ << ","
            << row.debug_match_count << ","
            << row.artifact_path << ",";
        writeCsvDouble(p2_feature_diag_results_file_, row.algo_ms);
        p2_feature_diag_results_file_ << ",";
        writeCsvDouble(p2_feature_diag_results_file_, row.queue_wait_ms);
        p2_feature_diag_results_file_ << ",";
        writeCsvDouble(p2_feature_diag_results_file_, row.worker_elapsed_ms);
        p2_feature_diag_results_file_ << ",";
        writeCsvFloat(p2_feature_diag_results_file_, row.deadline_ms);
        p2_feature_diag_results_file_ << ","
            << (row.over_deadline ? 1 : 0) << ","
            << row.depth_mode_mask << "," << row.triggers << "\n";
    }
    p2_feature_diag_results_file_.flush();
}

void Pipeline::writeP2FeatureDiagnosticArtifacts(
    std::vector<P2FeatureDiagnosticResultRow>& rows,
    const P2FeatureDiagnosticBuffer& buffer,
    int width,
    int height) {
    if (rows.empty() || !config_.p2_diagnostic_artifacts_enabled ||
        config_.p2_diagnostic_artifacts_max <= 0 ||
        width <= 0 || height <= 0 ||
        !buffer.left_gray_gpu || !buffer.right_gray_gpu) {
        return;
    }

    namespace fs = std::filesystem;
    fs::path out_dir;
    if (!config_.p2_diagnostic_artifacts_dir.empty()) {
        out_dir = fs::path(config_.p2_diagnostic_artifacts_dir);
    } else if (!config_.p2_diagnostic_results_path.empty()) {
        const fs::path csv_path(config_.p2_diagnostic_results_path);
        const fs::path parent = csv_path.parent_path();
        out_dir = parent / (csv_path.stem().string() + ".artifacts");
    } else {
        out_dir = fs::path("p2_diagnostic_artifacts");
    }

    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        LOG_WARN("P2 diagnostic: create artifact directory failed: %s (%s)",
                 out_dir.string().c_str(),
                 ec.message().c_str());
        return;
    }

    if (p2_feature_diag_stream_) {
        const cudaError_t sync_err = cudaStreamSynchronize(p2_feature_diag_stream_);
        if (sync_err != cudaSuccess) {
            LOG_WARN("P2 diagnostic: synchronize before artifacts failed: %s",
                     cudaGetErrorString(sync_err));
            return;
        }
    }

    cv::Mat left_gray(height, width, CV_8UC1);
    cv::Mat right_gray(height, width, CV_8UC1);
    cudaError_t err = cudaMemcpy2D(left_gray.data,
                                   static_cast<size_t>(left_gray.step),
                                   buffer.left_gray_gpu,
                                   buffer.left_gray_pitch,
                                   static_cast<size_t>(width),
                                   static_cast<size_t>(height),
                                   cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_WARN("P2 diagnostic: download left artifact image failed: %s",
                 cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy2D(right_gray.data,
                       static_cast<size_t>(right_gray.step),
                       buffer.right_gray_gpu,
                       buffer.right_gray_pitch,
                       static_cast<size_t>(width),
                       static_cast<size_t>(height),
                       cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_WARN("P2 diagnostic: download right artifact image failed: %s",
                 cudaGetErrorString(err));
        return;
    }

    cv::Mat left_bgr;
    cv::Mat right_bgr;
    cv::cvtColor(left_gray, left_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(right_gray, right_bgr, cv::COLOR_GRAY2BGR);

    constexpr int kTopMargin = 78;
    constexpr int kGap = 24;
    constexpr int kMinCrop = 72;
    constexpr int kMaxPanel = 520;
    constexpr int kMaxDrawMatches = 24;
    const cv::Scalar yellow(0, 220, 255);
    const cv::Scalar cyan(255, 220, 0);
    const cv::Scalar white(245, 245, 245);
    const cv::Scalar muted(170, 170, 170);
    const std::array<cv::Scalar, 8> palette = {
        cv::Scalar(64, 220, 255),
        cv::Scalar(80, 255, 120),
        cv::Scalar(255, 170, 80),
        cv::Scalar(255, 100, 180),
        cv::Scalar(120, 180, 255),
        cv::Scalar(180, 255, 80),
        cv::Scalar(255, 220, 120),
        cv::Scalar(210, 130, 255),
    };

    std::vector<size_t> row_order;
    row_order.reserve(rows.size());
    std::vector<uint8_t> queued(rows.size(), 0);
    auto row_drawable = [](const P2FeatureDiagnosticResultRow& row) {
        return row.debug_match_count > 0 || row.debug_patch.valid;
    };
    auto enqueue_rows = [&](auto predicate) {
        for (size_t i = 0; i < rows.size(); ++i) {
            if (queued[i] || !row_drawable(rows[i]) || !predicate(rows[i])) {
                continue;
            }
            queued[i] = 1;
            row_order.push_back(i);
        }
    };
    enqueue_rows([](const P2FeatureDiagnosticResultRow& row) {
        return row.valid && row.debug_match_count > 0;
    });
    enqueue_rows([](const P2FeatureDiagnosticResultRow& row) {
        return row.valid;
    });
    enqueue_rows([](const P2FeatureDiagnosticResultRow& row) {
        return row.debug_match_count > 0;
    });
    enqueue_rows([](const P2FeatureDiagnosticResultRow&) {
        return true;
    });

    for (const size_t row_index : row_order) {
        auto& row = rows[row_index];
        const bool diagnostic_valid_priority =
            row.lane == "diagnostic" && row.valid && row.debug_match_count > 0;
        if (p2_feature_diag_artifacts_saved_ >=
                config_.p2_diagnostic_artifacts_max &&
            !diagnostic_valid_priority) {
            break;
        }
        if (row.debug_match_count <= 0 && !row.debug_patch.valid) {
            continue;
        }

        float left_min_x = std::numeric_limits<float>::max();
        float left_min_y = std::numeric_limits<float>::max();
        float left_max_x = std::numeric_limits<float>::lowest();
        float left_max_y = std::numeric_limits<float>::lowest();
        float right_min_x = std::numeric_limits<float>::max();
        float right_min_y = std::numeric_limits<float>::max();
        float right_max_x = std::numeric_limits<float>::lowest();
        float right_max_y = std::numeric_limits<float>::lowest();

        includeRectBounds(detectionRect(row.left_det),
                          left_min_x, left_min_y, left_max_x, left_max_y);
        includeRectBounds(detectionRect(row.right_det),
                          right_min_x, right_min_y, right_max_x, right_max_y);
        includePointBounds(row.anchor_cx, row.anchor_cy,
                           left_min_x, left_min_y, left_max_x, left_max_y);
        includePointBounds(row.right_anchor_cx, row.right_anchor_cy,
                           right_min_x, right_min_y, right_max_x, right_max_y);

        const int debug_count =
            std::clamp(row.debug_match_count, 0, kMaxSparseFeatureDebugMatches);
        for (int i = 0; i < debug_count; ++i) {
            const auto& m = row.debug_matches[static_cast<size_t>(i)];
            includePointBounds(m.left_x, m.left_y,
                               left_min_x, left_min_y, left_max_x, left_max_y);
            includePointBounds(m.right_x, m.right_y,
                               right_min_x, right_min_y, right_max_x, right_max_y);
        }

        const bool have_left_bounds = left_min_x <= left_max_x &&
                                      left_min_y <= left_max_y;
        const bool have_right_bounds = right_min_x <= right_max_x &&
                                       right_min_y <= right_max_y;
        float left_cx = finitePoint(row.left_det.cx, row.left_det.cy)
            ? row.left_det.cx : row.anchor_cx;
        float left_cy = finitePoint(row.left_det.cx, row.left_det.cy)
            ? row.left_det.cy : row.anchor_cy;
        float right_cx = finitePoint(row.right_det.cx, row.right_det.cy)
            ? row.right_det.cx : row.right_anchor_cx;
        float right_cy = finitePoint(row.right_det.cx, row.right_det.cy)
            ? row.right_det.cy : row.right_anchor_cy;
        if (!finitePoint(left_cx, left_cy)) {
            left_cx = static_cast<float>(width) * 0.5f;
            left_cy = static_cast<float>(height) * 0.5f;
        }
        if (!finitePoint(right_cx, right_cy)) {
            right_cx = static_cast<float>(width) * 0.5f;
            right_cy = static_cast<float>(height) * 0.5f;
        }

        const float left_extent_w = have_left_bounds
            ? (left_max_x - left_min_x) : static_cast<float>(kMinCrop);
        const float left_extent_h = have_left_bounds
            ? (left_max_y - left_min_y) : static_cast<float>(kMinCrop);
        const float right_extent_w = have_right_bounds
            ? (right_max_x - right_min_x) : static_cast<float>(kMinCrop);
        const float right_extent_h = have_right_bounds
            ? (right_max_y - right_min_y) : static_cast<float>(kMinCrop);
        const int margin = 24;
        const int crop_w = std::clamp(
            static_cast<int>(std::ceil(std::max(left_extent_w, right_extent_w))) +
                margin * 2,
            kMinCrop,
            std::max(1, width));
        const int crop_h = std::clamp(
            static_cast<int>(std::ceil(std::max(left_extent_h, right_extent_h))) +
                margin * 2,
            kMinCrop,
            std::max(1, height));
        const cv::Rect left_crop = cropAround(left_cx, left_cy,
                                             crop_w, crop_h, width, height);
        const cv::Rect right_crop = cropAround(right_cx, right_cy,
                                              crop_w, crop_h, width, height);
        const float scale = std::max(
            1.0f,
            std::min(4.0f, static_cast<float>(kMaxPanel) /
                               static_cast<float>(std::max(crop_w, crop_h))));
        const int panel_w = std::max(1, static_cast<int>(std::lround(crop_w * scale)));
        const int panel_h = std::max(1, static_cast<int>(std::lround(crop_h * scale)));

        cv::Mat left_panel;
        cv::Mat right_panel;
        cv::resize(left_bgr(left_crop), left_panel, cv::Size(panel_w, panel_h),
                   0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(right_bgr(right_crop), right_panel, cv::Size(panel_w, panel_h),
                   0.0, 0.0, cv::INTER_NEAREST);

        const int patch_area_h = row.debug_patch.valid ? 168 : 0;
        cv::Mat canvas(kTopMargin + panel_h + patch_area_h,
                       panel_w * 2 + kGap,
                       CV_8UC3,
                       cv::Scalar(20, 20, 20));
        left_panel.copyTo(canvas(cv::Rect(0, kTopMargin, panel_w, panel_h)));
        right_panel.copyTo(canvas(cv::Rect(panel_w + kGap,
                                           kTopMargin,
                                           panel_w,
                                           panel_h)));

        drawDetectionOnPanel(canvas, row.left_det, left_crop, scale,
                             0, kTopMargin, yellow);
        drawDetectionOnPanel(canvas, row.right_det, right_crop, scale,
                             panel_w + kGap, kTopMargin, yellow);
        if (finitePoint(row.anchor_cx, row.anchor_cy)) {
            const cv::Point p = panelPoint(row.anchor_cx, row.anchor_cy,
                                           left_crop, scale, 0, kTopMargin);
            drawCross(canvas, p, cyan);
        }
        if (finitePoint(row.right_anchor_cx, row.right_anchor_cy)) {
            const cv::Point p = panelPoint(row.right_anchor_cx, row.right_anchor_cy,
                                           right_crop, scale,
                                           panel_w + kGap, kTopMargin);
            drawCross(canvas, p, cyan);
        }

        const int draw_count = std::min(debug_count, kMaxDrawMatches);
        for (int i = 0; i < draw_count; ++i) {
            const auto& m = row.debug_matches[static_cast<size_t>(i)];
            if (!finitePoint(m.left_x, m.left_y) ||
                !finitePoint(m.right_x, m.right_y)) {
                continue;
            }
            const cv::Scalar color = palette[static_cast<size_t>(i) % palette.size()];
            const cv::Point lp = panelPoint(m.left_x, m.left_y,
                                            left_crop, scale, 0, kTopMargin);
            const cv::Point rp = panelPoint(m.right_x, m.right_y,
                                            right_crop, scale,
                                            panel_w + kGap, kTopMargin);
            cv::circle(canvas, lp, 5, color, 2, cv::LINE_AA);
            cv::circle(canvas, rp, 5, color, 2, cv::LINE_AA);
            cv::line(canvas, lp, rp, color, 1, cv::LINE_AA);
        }

        cv::putText(canvas, "LEFT", cv::Point(8, kTopMargin - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, white, 1, cv::LINE_AA);
        cv::putText(canvas, "RIGHT", cv::Point(panel_w + kGap + 8,
                                               kTopMargin - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, white, 1, cv::LINE_AA);

        std::ostringstream line1;
        line1 << "frame " << row.frame_id << "  " << row.mode
              << "  " << row.status
              << "  disp=";
        if (std::isfinite(row.disparity)) {
            line1 << std::fixed << std::setprecision(2) << row.disparity;
        } else {
            line1 << "nan";
        }
        line1 << " px";
        cv::putText(canvas, line1.str(), cv::Point(8, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv::LINE_AA);

        std::ostringstream line2;
        line2 << "z=";
        if (std::isfinite(row.z_m)) {
            line2 << std::fixed << std::setprecision(3) << row.z_m << "m";
        } else {
            line2 << "nan";
        }
        line2 << "  support=" << row.support << "/" << row.attempted
              << "  debug_matches=" << debug_count;
        if (row.debug_patch.valid) {
            line2 << "  patch=" << row.debug_patch.width
                  << "x" << row.debug_patch.height;
        }
        cv::putText(canvas, line2.str(), cv::Point(8, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.50,
                    white, 1, cv::LINE_AA);
        if (row.debug_patch.valid) {
            drawDebugPatchPanel(canvas, row.debug_patch,
                                kTopMargin + panel_h + 12,
                                canvas.cols, white, muted);
        }

        std::ostringstream filename;
        filename << "frame_" << std::setw(6) << std::setfill('0') << row.frame_id
                 << "_" << std::setw(2) << std::setfill('0') << row_index
                 << "_" << sanitizeArtifactToken(row.mode)
                 << "_" << sanitizeArtifactToken(row.status) << ".png";
        const fs::path path = out_dir / filename.str();
        if (cv::imwrite(path.string(), canvas)) {
            row.artifact_path = path.string();
            ++p2_feature_diag_artifacts_saved_;
        } else {
            LOG_WARN("P2 diagnostic: failed to write artifact: %s",
                     path.string().c_str());
        }
    }
}

void Pipeline::releaseP2FeatureDiagnosticBuffer(int buffer_index) {
    if (buffer_index < 0 ||
        buffer_index >= static_cast<int>(p2_feature_diag_buffers_.size())) {
        return;
    }
    std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
    p2_feature_diag_free_buffers_.push_back(buffer_index);
}

void Pipeline::enqueueP2FeatureDiagnosticJobs(
    const FrameMetadata& metadata,
    const std::vector<P2FeatureJobDescriptor>& jobs) {
    if (!p2FeatureDiagnosticLaneConfigured()) {
        return;
    }
    const int max_in_flight = std::max(1, config_.p2_diagnostic_max_in_flight);
    int enqueued = 0;
    int dropped = 0;
    {
        std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
        for (const auto& job : jobs) {
            if (job.lane != P2FeatureJobLane::DIAGNOSTIC) {
                continue;
            }
            const int in_flight =
                static_cast<int>(p2_feature_diag_pending_.size()) +
                (p2_feature_diag_worker_busy_ ? 1 : 0);
            if (in_flight >= max_in_flight) {
                ++dropped;
                continue;
            }
            P2FeatureDiagnosticTask task;
            task.job = job;
            task.metadata = metadata;
            task.enqueue_time = std::chrono::steady_clock::now();
            p2_feature_diag_pending_.push_back(std::move(task));
            ++enqueued;
        }
        globalPerf().record("Stage2_P2FeatureJobDiagnosticQueueDepth",
                            static_cast<double>(
                                p2_feature_diag_pending_.size()));
    }
    for (int i = 0; i < enqueued; ++i) {
        globalPerf().record("Stage2_P2FeatureJobDiagnosticEnqueued", 0.0);
    }
    for (int i = 0; i < dropped; ++i) {
        globalPerf().record("Stage2_P2FeatureJobDiagnosticDropFull", 0.0);
    }
    if (enqueued > 0) {
        p2_feature_diag_cv_.notify_one();
    }
}

void Pipeline::enqueueP2FeatureDiagnosticJobs(
    const FrameMetadata& metadata,
    const std::vector<P2FeatureJobDescriptor>& jobs,
    const RoiStage2Input& input,
    cudaEvent_t source_copy_done,
    bool source_copy_event_recorded,
    AsyncRoiBuffer& source_buffer) {
    if (!p2FeatureDiagnosticLaneConfigured()) {
        return;
    }
    if (!input.left_gray_gpu || !input.right_gray_gpu ||
        input.left_gray_pitch <= 0 || input.right_gray_pitch <= 0 ||
        input.width <= 0 || input.height <= 0 ||
        !p2_feature_diag_copy_stream_) {
        enqueueP2FeatureDiagnosticJobs(metadata, jobs);
        globalPerf().record("Stage2_P2FeatureJobDiagnosticNoImage", 0.0);
        return;
    }

    const int max_in_flight = std::max(1, config_.p2_diagnostic_max_in_flight);
    int enqueued = 0;
    int dropped = 0;
    const size_t gray_width_bytes = static_cast<size_t>(input.width);
    const size_t rows = static_cast<size_t>(input.height);
    for (const auto& job : jobs) {
        if (job.lane != P2FeatureJobLane::DIAGNOSTIC) {
            continue;
        }
        int buffer_index = -1;
        {
            std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
            const int in_flight =
                static_cast<int>(p2_feature_diag_pending_.size()) +
                (p2_feature_diag_worker_busy_ ? 1 : 0);
            if (in_flight >= max_in_flight ||
                p2_feature_diag_free_buffers_.empty()) {
                ++dropped;
                continue;
            }
            buffer_index = p2_feature_diag_free_buffers_.front();
            p2_feature_diag_free_buffers_.pop_front();
        }

        auto& dst =
            p2_feature_diag_buffers_[static_cast<size_t>(buffer_index)];
        dst.copy_event_recorded = false;
        cudaError_t err = cudaSuccess;
        if (source_copy_event_recorded && source_copy_done) {
            err = cudaStreamWaitEvent(p2_feature_diag_copy_stream_,
                                      source_copy_done, 0);
        }
        const auto copy_submit_start = std::chrono::high_resolution_clock::now();
        if (err == cudaSuccess) {
            err = cudaMemcpy2DAsync(
                dst.left_gray_gpu, dst.left_gray_pitch,
                input.left_gray_gpu,
                static_cast<size_t>(input.left_gray_pitch),
                gray_width_bytes, rows,
                cudaMemcpyDeviceToDevice,
                p2_feature_diag_copy_stream_);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpy2DAsync(
                dst.right_gray_gpu, dst.right_gray_pitch,
                input.right_gray_gpu,
                static_cast<size_t>(input.right_gray_pitch),
                gray_width_bytes, rows,
                cudaMemcpyDeviceToDevice,
                p2_feature_diag_copy_stream_);
        }
        if (err == cudaSuccess) {
            err = cudaEventRecord(dst.copy_done, p2_feature_diag_copy_stream_);
        }
        if (err == cudaSuccess && source_buffer.p2_diag_copy_done) {
            err = cudaEventRecord(source_buffer.p2_diag_copy_done,
                                  p2_feature_diag_copy_stream_);
        }
        if (err != cudaSuccess) {
            LOG_WARN("P2 diagnostic: enqueue image copy failed frame=%d err=%s",
                     job.frame_id, cudaGetErrorString(err));
            releaseP2FeatureDiagnosticBuffer(buffer_index);
            ++dropped;
            continue;
        }
        dst.copy_event_recorded = true;
        source_buffer.p2_diag_copy_event_recorded =
            source_buffer.p2_diag_copy_done != nullptr;
        const double copy_submit_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() -
                copy_submit_start).count();
        globalPerf().record("Stage2_P2FeatureJobDiagnosticD2DSubmit",
                            copy_submit_ms);

        P2FeatureDiagnosticTask task;
        task.job = job;
        task.metadata = metadata;
        task.buffer_index = buffer_index;
        task.left_detections = input.left_detections;
        task.right_detections = input.right_detections;
        task.width = input.width;
        task.height = input.height;
        task.copy_event_recorded = true;
        task.enqueue_time = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
            p2_feature_diag_pending_.push_back(std::move(task));
            globalPerf().record("Stage2_P2FeatureJobDiagnosticQueueDepth",
                                static_cast<double>(
                                    p2_feature_diag_pending_.size()));
        }
        ++enqueued;
    }

    for (int i = 0; i < enqueued; ++i) {
        globalPerf().record("Stage2_P2FeatureJobDiagnosticEnqueued", 0.0);
    }
    for (int i = 0; i < dropped; ++i) {
        globalPerf().record("Stage2_P2FeatureJobDiagnosticDropFull", 0.0);
    }
    if (enqueued > 0) {
        p2_feature_diag_cv_.notify_one();
    }
}

void Pipeline::p2FeatureDiagnosticWorkerLoop() {
    using Clock = std::chrono::steady_clock;
    while (true) {
        P2FeatureDiagnosticTask task;
        {
            std::unique_lock<std::mutex> lk(p2_feature_diag_mutex_);
            p2_feature_diag_cv_.wait(lk, [this] {
                return p2_feature_diag_thread_stop_ ||
                       !p2_feature_diag_pending_.empty();
            });
            if (p2_feature_diag_thread_stop_ &&
                p2_feature_diag_pending_.empty()) {
                break;
            }
            task = std::move(p2_feature_diag_pending_.front());
            p2_feature_diag_pending_.pop_front();
            p2_feature_diag_worker_busy_ = true;
        }

        const auto start = Clock::now();
        int release_buffer_index = task.buffer_index;
        double queue_wait_ms = 0.0;
        if (task.enqueue_time.time_since_epoch().count() > 0) {
            queue_wait_ms =
                std::chrono::duration<double, std::milli>(
                    start - task.enqueue_time).count();
            globalPerf().record("Stage2_P2FeatureJobDiagnosticQueueWait",
                                queue_wait_ms);
        }
        std::vector<P2FeatureDiagnosticResultRow> result_rows;
        auto make_status_row = [&](const std::string& mode,
                                   const std::string& status) {
            P2FeatureDiagnosticResultRow row;
            row.frame_id = task.job.frame_id;
            row.metadata = task.metadata;
            row.mode = mode;
            row.status = status;
            row.queue_wait_ms = queue_wait_ms;
            row.deadline_ms = task.job.deadline_ms;
            row.depth_mode_mask = task.job.depth_mode_mask;
            row.triggers = task.job.triggers;
            result_rows.push_back(std::move(row));
        };

        bool copy_ready = true;
        P2FeatureDiagnosticBuffer* buffer = nullptr;
        if (task.buffer_index >= 0 &&
            task.buffer_index <
                static_cast<int>(p2_feature_diag_buffers_.size())) {
            buffer =
                &p2_feature_diag_buffers_[static_cast<size_t>(
                    task.buffer_index)];
            if (task.copy_event_recorded && buffer->copy_done) {
                const auto copy_wait_start = Clock::now();
                const cudaError_t copy_err =
                    cudaEventSynchronize(buffer->copy_done);
                const double copy_wait_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - copy_wait_start).count();
                globalPerf().record("Stage2_P2FeatureJobDiagnosticCopyWait",
                                    copy_wait_ms);
                buffer->copy_event_recorded = false;
                if (copy_err != cudaSuccess) {
                    copy_ready = false;
                    LOG_WARN("P2 diagnostic: image copy failed frame=%d err=%s",
                             task.job.frame_id,
                             cudaGetErrorString(copy_err));
                }
            }
        }

        if (!copy_ready || !buffer) {
            globalPerf().record("Stage2_P2FeatureJobDiagnosticNoImage", 0.0);
            make_status_row("all", "no_image");
        } else {
            Detection left_det;
            Detection right_det;
            float initial_disp = -1.0f;
            if (!chooseDiagnosticDirectPair(
                    task.left_detections, task.right_detections,
                    config_.max_disparity,
                    &left_det, &right_det, &initial_disp)) {
                globalPerf().record(
                    "Stage2_P2FeatureJobDiagnosticNoDirectPair", 0.0);
                make_status_row("all", "no_direct_pair");
            } else if (!calibration_ || !p2_feature_diag_stream_) {
                globalPerf().record(
                    "Stage2_P2FeatureJobDiagnosticUnavailable", 0.0);
                make_status_row("all", "unavailable");
            } else {
                const auto& p_left = calibration_->getProjectionLeft();
                const float focal = static_cast<float>(p_left.at<double>(0, 0));
                const float baseline = calibration_->getBaseline();
                ROIFeatureMatchConfig feature_cfg =
                    makeROIFeatureMatchConfig(config_.dual_yolo,
                                              config_.depth);
                feature_cfg.debug_patch_enabled =
                    config_.p2_diagnostic_artifacts_enabled;

                auto run_diagnostic_match = [&](uint32_t mode_bit,
                                                const char* stage_name,
                                                const char* valid_name,
                                                const char* invalid_name,
                                                DiagnosticGpuMatchFn fn) {
                    if ((task.job.depth_mode_mask & mode_bit) == 0u || !fn) {
                        return false;
                    }
                    const auto algo_start = Clock::now();
                    const SparseFeatureDisparityResult result = fn(
                        buffer->left_gray_gpu,
                        static_cast<int>(buffer->left_gray_pitch),
                        buffer->right_gray_gpu,
                        static_cast<int>(buffer->right_gray_pitch),
                        task.width, task.height,
                        left_det, right_det,
                        initial_disp,
                        feature_cfg,
                        config_.max_disparity,
                        focal,
                        baseline,
                        p2_feature_diag_stream_);
                    const double algo_ms =
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - algo_start).count();
                    globalPerf().record(stage_name, algo_ms);
                    globalPerf().record(result.valid ? valid_name : invalid_name,
                                        0.0);
                    P2FeatureDiagnosticResultRow row;
                    row.frame_id = task.job.frame_id;
                    row.metadata = task.metadata;
                    row.mode = p2DiagnosticModeName(mode_bit);
                    row.status = result.valid
                        ? "valid"
                        : (result.unsupported ? "unsupported" : "invalid");
                    row.valid = result.valid;
                    row.low_confidence = result.low_confidence;
                    row.disparity = result.disparity;
                    row.z_m = result.valid
                        ? p2DepthFromDisparity(result.disparity,
                                               focal,
                                               baseline)
                        : std::numeric_limits<float>::quiet_NaN();
                    row.confidence = result.confidence;
                    row.stddev = result.stddev;
                    row.support = result.support;
                    row.attempted = result.attempted;
                    row.initial_disparity = initial_disp;
                    row.left_det = left_det;
                    row.right_det = right_det;
                    row.anchor_cx = result.anchor_cx;
                    row.anchor_cy = result.anchor_cy;
                    row.right_anchor_cx = result.right_anchor_cx;
                    row.right_anchor_cy = result.right_anchor_cy;
                    row.debug_match_count = std::clamp(
                        result.debug_match_count,
                        0,
                        kMaxSparseFeatureDebugMatches);
                    for (int i = 0; i < row.debug_match_count; ++i) {
                        row.debug_matches[static_cast<size_t>(i)] =
                            result.debug_matches[static_cast<size_t>(i)];
                    }
                    row.debug_patch = result.debug_patch;
                    row.algo_ms = algo_ms;
                    row.queue_wait_ms = queue_wait_ms;
                    row.deadline_ms = task.job.deadline_ms;
                    row.depth_mode_mask = task.job.depth_mode_mask;
                    row.triggers = task.job.triggers;
                    result_rows.push_back(std::move(row));
                    return true;
                };

                bool ran_any = false;
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_ORB_POINTS,
                    "Stage2_P2FeatureJobDiagnosticOpenCVCudaORB",
                    "Stage2_P2FeatureJobDiagnosticOpenCVCudaORBValid",
                    "Stage2_P2FeatureJobDiagnosticOpenCVCudaORBInvalid",
                    matchOpenCVORBDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_CUDA_TEMPLATE,
                    "Stage2_P2FeatureJobDiagnosticCudaTemplate",
                    "Stage2_P2FeatureJobDiagnosticCudaTemplateValid",
                    "Stage2_P2FeatureJobDiagnosticCudaTemplateInvalid",
                    matchCudaTemplateDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_CUDA_STEREO_BM,
                    "Stage2_P2FeatureJobDiagnosticCudaStereoBM",
                    "Stage2_P2FeatureJobDiagnosticCudaStereoBMValid",
                    "Stage2_P2FeatureJobDiagnosticCudaStereoBMInvalid",
                    matchCudaStereoBMDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_CUDA_STEREO_SGM,
                    "Stage2_P2FeatureJobDiagnosticCudaStereoSGM",
                    "Stage2_P2FeatureJobDiagnosticCudaStereoSGMValid",
                    "Stage2_P2FeatureJobDiagnosticCudaStereoSGMInvalid",
                    matchCudaStereoSGMDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_RING_EDGE_PROFILE,
                    "Stage2_P2FeatureJobDiagnosticCudaRingEdgeProfile",
                    "Stage2_P2FeatureJobDiagnosticCudaRingEdgeProfileValid",
                    "Stage2_P2FeatureJobDiagnosticCudaRingEdgeProfileInvalid",
                    matchCudaRingEdgeProfileDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_VPI_TEMPLATE,
                    "Stage2_P2FeatureJobDiagnosticVpiTemplate",
                    "Stage2_P2FeatureJobDiagnosticVpiTemplateValid",
                    "Stage2_P2FeatureJobDiagnosticVpiTemplateInvalid",
                    matchVpiTemplateDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_VPI_STEREO,
                    "Stage2_P2FeatureJobDiagnosticVpiStereo",
                    "Stage2_P2FeatureJobDiagnosticVpiStereoValid",
                    "Stage2_P2FeatureJobDiagnosticVpiStereoInvalid",
                    matchVpiStereoDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_VPI_HARRIS_LK,
                    "Stage2_P2FeatureJobDiagnosticVpiHarrisLk",
                    "Stage2_P2FeatureJobDiagnosticVpiHarrisLkValid",
                    "Stage2_P2FeatureJobDiagnosticVpiHarrisLkInvalid",
                    matchVpiHarrisLkDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_VPI_ORB,
                    "Stage2_P2FeatureJobDiagnosticVpiOrb",
                    "Stage2_P2FeatureJobDiagnosticVpiOrbValid",
                    "Stage2_P2FeatureJobDiagnosticVpiOrbInvalid",
                    matchVpiOrbDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_CUDA_GFTT_LK,
                    "Stage2_P2FeatureJobDiagnosticOpenCVCudaGfttLk",
                    "Stage2_P2FeatureJobDiagnosticOpenCVCudaGfttLkValid",
                    "Stage2_P2FeatureJobDiagnosticOpenCVCudaGfttLkInvalid",
                    matchOpenCVCudaGfttLkDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_CUDA_SIFT,
                    "Stage2_P2FeatureJobDiagnosticCudaSift",
                    "Stage2_P2FeatureJobDiagnosticCudaSiftValid",
                    "Stage2_P2FeatureJobDiagnosticCudaSiftInvalid",
                    matchCudaSiftDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_LIBSGM,
                    "Stage2_P2FeatureJobDiagnosticFixstarsLibSgm",
                    "Stage2_P2FeatureJobDiagnosticFixstarsLibSgmValid",
                    "Stage2_P2FeatureJobDiagnosticFixstarsLibSgmInvalid",
                    matchFixstarsLibSgmDisparityGPU);
                ran_any |= run_diagnostic_match(
                    P2_DEPTH_MODE_CUDA_HOUGH_CIRCLE,
                    "Stage2_P2FeatureJobDiagnosticCudaHoughCircle",
                    "Stage2_P2FeatureJobDiagnosticCudaHoughCircleValid",
                    "Stage2_P2FeatureJobDiagnosticCudaHoughCircleInvalid",
                    matchCudaCannyHoughCircleDisparityGPU);
                if (!ran_any) {
                    globalPerf().record("Stage2_P2FeatureJobDiagnosticNoop",
                                        0.0);
                    make_status_row("all", "noop");
                }
            }
        }

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
        globalPerf().record("Stage2_P2FeatureJobDiagnosticWorker", elapsed_ms);
        if (elapsed_ms > static_cast<double>(task.job.deadline_ms)) {
            globalPerf().record("Stage2_P2FeatureJobDiagnosticOverDeadline",
                                elapsed_ms);
        }
        for (auto& row : result_rows) {
            row.worker_elapsed_ms = elapsed_ms;
            row.over_deadline =
                elapsed_ms > static_cast<double>(task.job.deadline_ms);
        }
        const auto result_write_start = Clock::now();
        writeP2FeatureDiagnosticArtifacts(result_rows,
                                          *buffer,
                                          task.width,
                                          task.height);
        writeP2FeatureDiagnosticResults(result_rows);
        if (!result_rows.empty() && config_.p2_diagnostic_results_enabled) {
            const double result_write_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - result_write_start).count();
            globalPerf().record(
                "Stage2_P2FeatureJobDiagnosticResultsWrite",
                result_write_ms);
        }
        releaseP2FeatureDiagnosticBuffer(release_buffer_index);

        {
            std::lock_guard<std::mutex> lk(p2_feature_diag_mutex_);
            p2_feature_diag_worker_busy_ = false;
        }
    }
    LOG_INFO("P2 diagnostic FeatureJob worker exited");
}

void Pipeline::releaseAsyncRoiBuffer(int buffer_index, const char* reason) {
    if (buffer_index < 0 ||
        buffer_index >= static_cast<int>(async_roi_buffers_.size())) {
        return;
    }
    waitAsyncRoiBufferCopy(buffer_index, reason);
    std::lock_guard<std::mutex> lk(async_roi_mutex_);
    releaseAsyncRoiBufferLocked(buffer_index);
}

void Pipeline::releaseAsyncRoiBufferLocked(int buffer_index) {
    if (buffer_index >= 0 &&
        buffer_index < static_cast<int>(async_roi_buffers_.size())) {
        async_roi_free_buffers_.push_back(buffer_index);
    }
}

bool Pipeline::waitAsyncRoiBufferCopy(int buffer_index,
                                      const char* reason) {
    using Clock = std::chrono::high_resolution_clock;
    if (buffer_index < 0 ||
        buffer_index >= static_cast<int>(async_roi_buffers_.size())) {
        return false;
    }

    auto& buffer = async_roi_buffers_[static_cast<size_t>(buffer_index)];
    if (!buffer.copy_event_recorded || !buffer.copy_done) {
        if (!buffer.p2_diag_copy_event_recorded ||
            !buffer.p2_diag_copy_done) {
            return true;
        }
    } else {
        const auto t0 = Clock::now();
        const cudaError_t err = cudaEventSynchronize(buffer.copy_done);
        const double wait_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - t0).count();
        globalPerf().record("Stage2_AsyncRoiCopyWait", wait_ms);
        buffer.copy_event_recorded = false;
        if (err != cudaSuccess) {
            LOG_WARN("[AsyncROI] Buffer copy wait failed buffer=%d reason=%s err=%s",
                     buffer_index,
                     reason ? reason : "unknown",
                     cudaGetErrorString(err));
            return false;
        }
    }

    if (buffer.p2_diag_copy_event_recorded && buffer.p2_diag_copy_done) {
        const auto t0 = Clock::now();
        const cudaError_t err =
            cudaEventSynchronize(buffer.p2_diag_copy_done);
        const double wait_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - t0).count();
        globalPerf().record("Stage2_P2FeatureJobDiagnosticSourceCopyWait",
                            wait_ms);
        buffer.p2_diag_copy_event_recorded = false;
        if (err != cudaSuccess) {
            LOG_WARN("[AsyncROI] P2 diagnostic source copy wait failed "
                     "buffer=%d reason=%s err=%s",
                     buffer_index,
                     reason ? reason : "unknown",
                     cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

void Pipeline::markAsyncRoiSlotCopyPendingLocked(int slot_index) {
    if (slot_index >= 0 && slot_index < RING_BUFFER_SIZE) {
        async_roi_slot_copy_pending_[static_cast<size_t>(slot_index)] = true;
    }
}

void Pipeline::waitAsyncRoiSlotSnapshotDone(int slot_index,
                                            const char* reason) {
    using Clock = std::chrono::high_resolution_clock;
    if (!async_roi_ready_ ||
        slot_index < 0 ||
        slot_index >= RING_BUFFER_SIZE) {
        return;
    }

    cudaEvent_t evt = nullptr;
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        if (!async_roi_slot_copy_pending_[static_cast<size_t>(slot_index)]) {
            return;
        }
        evt = async_roi_slot_copy_done_[static_cast<size_t>(slot_index)];
    }
    if (!evt) {
        return;
    }

    const auto t0 = Clock::now();
    const cudaError_t err = cudaEventSynchronize(evt);
    const double wait_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    globalPerf().record("Stage2_AsyncRoiSlotCopyWait", wait_ms);

    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        async_roi_slot_copy_pending_[static_cast<size_t>(slot_index)] = false;
    }
    if (err != cudaSuccess) {
        LOG_WARN("[AsyncROI] Slot snapshot wait failed slot=%d reason=%s err=%s",
                 slot_index,
                 reason ? reason : "unknown",
                 cudaGetErrorString(err));
    }
}

void Pipeline::shutdownAsyncRoiStage2() {
    if (!async_roi_ready_ && !async_roi_thread_.joinable()) {
        return;
    }
    std::vector<int> pending_buffers;
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        async_roi_thread_stop_ = true;
        while (!async_roi_pending_.empty()) {
            pending_buffers.push_back(async_roi_pending_.front().buffer_index);
            async_roi_pending_.pop_front();
        }
    }
    async_roi_cv_.notify_all();
    for (int buffer_index : pending_buffers) {
        releaseAsyncRoiBuffer(buffer_index, "shutdown");
    }
    if (async_roi_thread_.joinable()) {
        async_roi_thread_.join();
    }
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        async_roi_completed_.clear();
        async_roi_slot_copy_pending_.fill(false);
        async_roi_worker_busy_ = false;
    }
}

void Pipeline::destroyAsyncRoiStage2() {
    shutdownAsyncRoiStage2();

    for (auto& b : async_roi_buffers_) {
        if (b.copy_done) {
            if (b.copy_event_recorded) {
                cudaEventSynchronize(b.copy_done);
                b.copy_event_recorded = false;
            }
            cudaEventDestroy(b.copy_done);
            b.copy_done = nullptr;
        }
        if (b.p2_diag_copy_done) {
            if (b.p2_diag_copy_event_recorded) {
                cudaEventSynchronize(b.p2_diag_copy_done);
                b.p2_diag_copy_event_recorded = false;
            }
            cudaEventDestroy(b.p2_diag_copy_done);
            b.p2_diag_copy_done = nullptr;
        }
        if (b.left_gray_gpu) {
            cudaFree(b.left_gray_gpu);
            b.left_gray_gpu = nullptr;
        }
        if (b.right_gray_gpu) {
            cudaFree(b.right_gray_gpu);
            b.right_gray_gpu = nullptr;
        }
        if (b.left_bgr_gpu) {
            cudaFree(b.left_bgr_gpu);
            b.left_bgr_gpu = nullptr;
        }
        if (b.right_bgr_gpu) {
            cudaFree(b.right_bgr_gpu);
            b.right_bgr_gpu = nullptr;
        }
        if (b.left_gray_host) {
            cudaFreeHost(b.left_gray_host);
            b.left_gray_host = nullptr;
        }
        if (b.right_gray_host) {
            cudaFreeHost(b.right_gray_host);
            b.right_gray_host = nullptr;
        }
        b.left_gray_pitch = 0;
        b.right_gray_pitch = 0;
        b.left_bgr_pitch = 0;
        b.right_bgr_pitch = 0;
        b.left_gray_host_pitch = 0;
        b.right_gray_host_pitch = 0;
    }
    async_roi_buffers_.clear();
    async_roi_free_buffers_.clear();
    async_roi_pending_.clear();
    async_roi_completed_.clear();

    for (size_t i = 0; i < async_roi_slot_copy_done_.size(); ++i) {
        auto& evt = async_roi_slot_copy_done_[i];
        if (evt) {
            if (async_roi_slot_copy_pending_[i]) {
                cudaEventSynchronize(evt);
            }
            cudaEventDestroy(evt);
            evt = nullptr;
        }
    }
    async_roi_slot_copy_pending_.fill(false);

    if (async_roi_stream_) {
        cudaStreamDestroy(async_roi_stream_);
        async_roi_stream_ = nullptr;
    }
    if (async_roi_copy_stream_) {
        cudaStreamDestroy(async_roi_copy_stream_);
        async_roi_copy_stream_ = nullptr;
    }
    async_roi_ready_ = false;
}

bool Pipeline::snapshotAsyncRoiImages(FrameSlot& slot,
                                      AsyncRoiBuffer& buffer,
                                      bool need_host_gray,
                                      bool need_bgr) {
    using SnapshotClock = std::chrono::high_resolution_clock;
    auto record_snapshot_elapsed = [](const char* name,
                                      const SnapshotClock::time_point& start) {
        const double ms = std::chrono::duration<double, std::milli>(
            SnapshotClock::now() - start).count();
        globalPerf().record(name, ms);
    };

    const uint8_t* left_src =
        static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    const uint8_t* right_src =
        static_cast<const uint8_t*>(slot.rectGray_R_gpu.data);
    const int left_src_pitch = slot.rectGray_L_gpu.pitchBytes;
    const int right_src_pitch = slot.rectGray_R_gpu.pitchBytes;
    const size_t gray_width_bytes = static_cast<size_t>(config_.rect_width);
    const size_t bgr_width_bytes =
        static_cast<size_t>(config_.rect_width) * 3u;
    const size_t rows = static_cast<size_t>(config_.rect_height);
    buffer.copy_event_recorded = false;
    buffer.p2_diag_copy_event_recorded = false;

    if (!left_src || !right_src ||
        left_src_pitch <= 0 || right_src_pitch <= 0 ||
        !buffer.left_gray_gpu || !buffer.right_gray_gpu) {
        LOG_WARN("Async ROI: invalid gray snapshot source/buffer frame=%d",
                 slot.frame_id);
        return false;
    }

    const uint8_t* left_bgr_src = nullptr;
    const uint8_t* right_bgr_src = nullptr;
    int left_bgr_src_pitch = 0;
    int right_bgr_src_pitch = 0;
    if (need_bgr) {
        left_bgr_src = static_cast<const uint8_t*>(slot.rectBGR_L_gpu.data);
        right_bgr_src = static_cast<const uint8_t*>(slot.rectBGR_R_gpu.data);
        left_bgr_src_pitch = slot.rectBGR_L_gpu.pitchBytes;
        right_bgr_src_pitch = slot.rectBGR_R_gpu.pitchBytes;
        if (!left_bgr_src || !right_bgr_src ||
            left_bgr_src_pitch <= 0 || right_bgr_src_pitch <= 0 ||
            !buffer.left_bgr_gpu || !buffer.right_bgr_gpu) {
            LOG_WARN("Async ROI: BGR snapshot requested but unavailable frame=%d",
                     slot.frame_id);
            return false;
        }
    }
    if (need_host_gray &&
        (!buffer.left_gray_host || !buffer.right_gray_host ||
         buffer.left_gray_host_pitch == 0 ||
         buffer.right_gray_host_pitch == 0)) {
        LOG_WARN("Async ROI: host gray snapshot requested but unavailable frame=%d",
                 slot.frame_id);
        return false;
    }

    const auto gray_submit_start = SnapshotClock::now();
    cudaError_t err = cudaMemcpy2DAsync(
        buffer.left_gray_gpu, buffer.left_gray_pitch,
        left_src, static_cast<size_t>(left_src_pitch),
        gray_width_bytes, rows,
        cudaMemcpyDeviceToDevice,
        async_roi_copy_stream_);
    if (err == cudaSuccess) {
        err = cudaMemcpy2DAsync(
            buffer.right_gray_gpu, buffer.right_gray_pitch,
            right_src, static_cast<size_t>(right_src_pitch),
            gray_width_bytes, rows,
            cudaMemcpyDeviceToDevice,
            async_roi_copy_stream_);
    }
    if (err == cudaSuccess) {
        record_snapshot_elapsed("Stage2_AsyncRoiGrayD2DSubmit",
                                gray_submit_start);
    }

    if (err == cudaSuccess && need_bgr) {
        const auto bgr_submit_start = SnapshotClock::now();
        err = cudaMemcpy2DAsync(
            buffer.left_bgr_gpu, buffer.left_bgr_pitch,
            left_bgr_src, static_cast<size_t>(left_bgr_src_pitch),
            bgr_width_bytes, rows,
            cudaMemcpyDeviceToDevice,
            async_roi_copy_stream_);
        if (err == cudaSuccess) {
            err = cudaMemcpy2DAsync(
                buffer.right_bgr_gpu, buffer.right_bgr_pitch,
                right_bgr_src, static_cast<size_t>(right_bgr_src_pitch),
                bgr_width_bytes, rows,
                cudaMemcpyDeviceToDevice,
                async_roi_copy_stream_);
        }
        if (err == cudaSuccess) {
            record_snapshot_elapsed("Stage2_AsyncRoiBgrD2DSubmit",
                                    bgr_submit_start);
        }
    }

    if (err == cudaSuccess && need_host_gray) {
        const auto host_submit_start = SnapshotClock::now();
        err = cudaMemcpy2DAsync(
            buffer.left_gray_host, buffer.left_gray_host_pitch,
            buffer.left_gray_gpu, buffer.left_gray_pitch,
            gray_width_bytes, rows,
            cudaMemcpyDeviceToHost,
            async_roi_copy_stream_);
        if (err == cudaSuccess) {
            err = cudaMemcpy2DAsync(
                buffer.right_gray_host, buffer.right_gray_host_pitch,
                buffer.right_gray_gpu, buffer.right_gray_pitch,
                gray_width_bytes, rows,
                cudaMemcpyDeviceToHost,
                async_roi_copy_stream_);
        }
        if (err == cudaSuccess) {
            record_snapshot_elapsed("Stage2_AsyncRoiHostGrayD2HSubmit",
                                    host_submit_start);
        }
    }

    if (err == cudaSuccess) {
        const auto event_submit_start = SnapshotClock::now();
        err = cudaEventRecord(buffer.copy_done, async_roi_copy_stream_);
        if (err == cudaSuccess) {
            buffer.copy_event_recorded = true;
            record_snapshot_elapsed("Stage2_AsyncRoiEventRecord",
                                    event_submit_start);
        }
    }
    if (err != cudaSuccess) {
        cudaStreamSynchronize(async_roi_copy_stream_);
        LOG_WARN("Async ROI: image snapshot failed frame=%d err=%s",
                 slot.frame_id, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool Pipeline::submitAsyncRoiStage2(FrameSlot& slot, int slot_index) {
    if (!async_roi_ready_) {
        return false;
    }

    collectRoiDetections(slot, slot_index);
    slot.results.clear();
    if (slot.detections.empty() && slot.detections_right.empty()) {
        globalPerf().record("Stage2_AsyncRoiNoDetections", 0.0);
        return false;
    }

    int buffer_index = -1;
    size_t queue_pending_depth = 0;
    size_t queue_free_buffers = 0;
    bool queue_worker_busy = false;
    bool no_buffer = false;
    size_t no_buffer_pending_depth = 0;
    bool no_buffer_worker_busy = false;
    int dropped_reuse_buffer = -1;
    int dropped_reuse_frame = -1;
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        queue_pending_depth = async_roi_pending_.size();
        queue_free_buffers = async_roi_free_buffers_.size();
        queue_worker_busy = async_roi_worker_busy_;
        if (async_roi_free_buffers_.empty() && !async_roi_pending_.empty()) {
            dropped_reuse_frame = async_roi_pending_.front().frame_id;
            dropped_reuse_buffer = async_roi_pending_.front().buffer_index;
            async_roi_pending_.pop_front();
            globalPerf().record("Stage2_AsyncRoiDropPending", 0.0);
        }
        if (!async_roi_free_buffers_.empty()) {
            buffer_index = async_roi_free_buffers_.front();
            async_roi_free_buffers_.pop_front();
        } else if (dropped_reuse_buffer < 0) {
            no_buffer = true;
            no_buffer_pending_depth = async_roi_pending_.size();
            no_buffer_worker_busy = async_roi_worker_busy_;
        }
    }
    if (dropped_reuse_buffer >= 0) {
        waitAsyncRoiBufferCopy(dropped_reuse_buffer, "replace_pending_reuse");
        if (buffer_index < 0) {
            buffer_index = dropped_reuse_buffer;
        } else {
            releaseAsyncRoiBuffer(dropped_reuse_buffer, "replace_pending_extra");
        }
        LOG_WARN("[AsyncROI] Replace pending ROI task frame=%d with frame=%d",
                 dropped_reuse_frame, slot.frame_id);
    }
    globalPerf().record("Stage2_AsyncRoiPendingDepth",
                        static_cast<double>(queue_pending_depth));
    globalPerf().record("Stage2_AsyncRoiFreeBuffers",
                        static_cast<double>(queue_free_buffers));
    globalPerf().record("Stage2_AsyncRoiWorkerBusy",
                        queue_worker_busy ? 1.0 : 0.0);
    if (no_buffer) {
        globalPerf().record("Stage2_AsyncRoiDropNoBuffer", 0.0);
        LOG_WARN("[AsyncROI] Drop frame=%d: no async ROI buffer free "
                 "(worker_busy=%d pending=%zu)",
                 slot.frame_id,
                 no_buffer_worker_busy ? 1 : 0,
                 no_buffer_pending_depth);
        return false;
    }

    AsyncRoiBuffer& buffer =
        async_roi_buffers_[static_cast<size_t>(buffer_index)];
    if (!waitAsyncRoiBufferCopy(buffer_index, "reuse_before_snapshot")) {
        releaseAsyncRoiBuffer(buffer_index, "reuse_wait_failed");
        return false;
    }
    const bool requested_host_gray =
        roiStage2NeedsHostImages(slot.detections, slot.detections_right);
    const bool neural_needs_bgr =
        neural_feature_matcher_ && neural_feature_matcher_->requiresBgrInput();
    const bool has_stereo_detections =
        !slot.detections.empty() && !slot.detections_right.empty();
    const bool requested_bgr =
        colorPipelineEnabled() &&
        has_stereo_detections &&
        (config_.dual_yolo.depth_roi_iou_region_color_patch ||
         config_.dual_yolo.depth_roi_patch_iou_color_edge ||
         neural_needs_bgr);
    const P2FeatureJobPolicy p2_policy = makeP2FeatureJobPolicy(config_);
    const P2FeatureJobDecision p2_decision = decideP2FeatureJobs(
        p2_policy,
        slot.frame_id,
        slot.detections,
        slot.detections_right,
        requested_host_gray,
        requested_bgr);
    const bool p2_inline_feature_jobs_enabled =
        (!p2_decision.p2_depth_modes_enabled ||
         !(p2_policy.split_feature_jobs && p2_policy.selective_trigger) ||
         p2_decision.realtime_requested) &&
        !p2DiagnosticOnlyFeatureJobsEnabled(config_);
    const bool fallback_needs_host_gray =
        roiStage2FallbackMayNeedHostImages(slot.detections,
                                           slot.detections_right);
    const bool need_host_gray =
        requested_host_gray &&
        (p2_inline_feature_jobs_enabled ||
         p2_decision.diagnostic_requested ||
         fallback_needs_host_gray);
    const bool need_bgr =
        requested_bgr &&
        (p2_inline_feature_jobs_enabled ||
         p2_decision.diagnostic_requested);
    std::vector<P2FeatureJobDescriptor> p2_feature_jobs =
        buildP2FeatureJobDescriptors(
            p2_policy,
            p2_decision,
            need_host_gray,
            need_bgr);
    if (need_host_gray) {
        globalPerf().record("Stage2_AsyncRoiNeedHostGray", 0.0);
    }
    if (requested_host_gray && !need_host_gray) {
        globalPerf().record("Stage2_AsyncRoiSkipHostGraySelective", 0.0);
    }
    if (need_bgr) {
        globalPerf().record("Stage2_AsyncRoiNeedBgr", 0.0);
    }
    if (requested_bgr && !need_bgr) {
        globalPerf().record("Stage2_AsyncRoiSkipBgrSelective", 0.0);
    }
    if (p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobConfigured", 0.0);
    }
    if (p2_decision.realtime_requested) {
        globalPerf().record("Stage2_P2FeatureJobRealtimeRequested", 0.0);
        if ((p2_decision.realtime_triggers & P2_TRIGGER_PAIR_LOW_IOU) != 0u) {
            globalPerf().record("Stage2_P2FeatureJobTriggerPairLowIou", 0.0);
        }
        if ((p2_decision.realtime_triggers &
             P2_TRIGGER_PAIR_EPIPOLAR_DY) != 0u) {
            globalPerf().record("Stage2_P2FeatureJobTriggerPairEpipolarDy",
                                0.0);
        }
        if ((p2_decision.realtime_triggers &
             P2_TRIGGER_PAIR_LOW_CONFIDENCE) != 0u) {
            globalPerf().record("Stage2_P2FeatureJobTriggerPairLowConf",
                                0.0);
        }
        if ((p2_decision.realtime_triggers &
             P2_TRIGGER_NO_VALID_DIRECT_PAIR) != 0u) {
            globalPerf().record("Stage2_P2FeatureJobTriggerNoValidPair",
                                0.0);
        }
    } else if (p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobRealtimeNotAttempted", 0.0);
        if ((p2_decision.realtime_skip_reasons &
             P2_SKIP_SELECTIVE_NOT_TRIGGERED) != 0u) {
            globalPerf().record("Stage2_P2FeatureJobSkipSelective", 0.0);
        }
    }
    if (p2_decision.diagnostic_requested) {
        globalPerf().record("Stage2_P2FeatureJobDiagnosticRequested", 0.0);
    }
    const bool p2_job_requested =
        p2_decision.realtime_requested || p2_decision.diagnostic_requested;
    if (p2_policy.split_feature_jobs && p2_job_requested &&
        p2_feature_jobs.empty() && p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobSplitNoJob", 0.0);
    }
    if (p2_policy.split_feature_jobs && !p2_job_requested &&
        p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobNotTriggered", 0.0);
    }
    applyP2FeatureJobDecisionToSlot(slot, p2_decision, p2_feature_jobs);

    ScopedTimer tsnap("Stage2_AsyncRoiSnapshot");
    if (!snapshotAsyncRoiImages(slot, buffer, need_host_gray, need_bgr)) {
        releaseAsyncRoiBuffer(buffer_index, "snapshot_failed");
        return false;
    }
    if (slot_index >= 0 && slot_index < RING_BUFFER_SIZE) {
        cudaEvent_t slot_copy_done =
            async_roi_slot_copy_done_[static_cast<size_t>(slot_index)];
        if (slot_copy_done) {
            const cudaError_t evt_err =
                cudaEventRecord(slot_copy_done, async_roi_copy_stream_);
            if (evt_err != cudaSuccess) {
                LOG_WARN("Async ROI: record slot copy event failed frame=%d err=%s",
                         slot.frame_id, cudaGetErrorString(evt_err));
                releaseAsyncRoiBuffer(buffer_index, "slot_event_failed");
                return false;
            }
            std::lock_guard<std::mutex> lk(async_roi_mutex_);
            markAsyncRoiSlotCopyPendingLocked(slot_index);
        }
    }
    globalPerf().record("Stage2_AsyncRoiSnapshot", tsnap.elapsedMs());

    AsyncRoiTask task;
    task.frame_id = slot.frame_id;
    task.slot_index = slot_index;
    task.buffer_index = buffer_index;
    task.host_gray_valid = need_host_gray;
    task.bgr_valid = need_bgr;
    task.copy_event_recorded = buffer.copy_event_recorded;
    task.metadata = makeFrameMetadata(slot);
    task.p2_feature_decision = p2_decision;
    task.p2_feature_jobs = std::move(p2_feature_jobs);
    task.input.frame_id = slot.frame_id;
    task.input.left_detections = slot.detections;
    task.input.right_detections = slot.detections_right;
    task.input.p2_inline_feature_jobs_enabled =
        p2_inline_feature_jobs_enabled;
    task.input.left_cpu = need_host_gray ? buffer.left_gray_host : nullptr;
    task.input.left_cpu_pitch =
        need_host_gray ? static_cast<int>(buffer.left_gray_host_pitch) : 0;
    task.input.right_cpu = need_host_gray ? buffer.right_gray_host : nullptr;
    task.input.right_cpu_pitch =
        need_host_gray ? static_cast<int>(buffer.right_gray_host_pitch) : 0;
    task.input.left_gray_gpu = buffer.left_gray_gpu;
    task.input.left_gray_pitch = static_cast<int>(buffer.left_gray_pitch);
    task.input.right_gray_gpu = buffer.right_gray_gpu;
    task.input.right_gray_pitch = static_cast<int>(buffer.right_gray_pitch);
    task.input.left_bgr_gpu = need_bgr ? buffer.left_bgr_gpu : nullptr;
    task.input.left_bgr_pitch =
        need_bgr ? static_cast<int>(buffer.left_bgr_pitch) : 0;
    task.input.right_bgr_gpu = need_bgr ? buffer.right_bgr_gpu : nullptr;
    task.input.right_bgr_pitch =
        need_bgr ? static_cast<int>(buffer.right_bgr_pitch) : 0;
    task.input.width = config_.rect_width;
    task.input.height = config_.rect_height;
    task.input.stream = async_roi_stream_;
    enqueueP2FeatureDiagnosticJobs(task.metadata,
                                   task.p2_feature_jobs,
                                   task.input,
                                   buffer.copy_done,
                                   buffer.copy_event_recorded,
                                   buffer);

    int dropped_pending_buffer = -1;
    int dropped_pending_frame = -1;
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        if (!async_roi_pending_.empty()) {
            dropped_pending_frame = async_roi_pending_.front().frame_id;
            dropped_pending_buffer = async_roi_pending_.front().buffer_index;
            async_roi_pending_.pop_front();
            globalPerf().record("Stage2_AsyncRoiDropPending", 0.0);
        }
        async_roi_pending_.push_back(std::move(task));
    }
    if (dropped_pending_buffer >= 0) {
        releaseAsyncRoiBuffer(dropped_pending_buffer, "replace_pending_after_submit");
        LOG_WARN("[AsyncROI] Replace pending ROI task frame=%d with frame=%d",
                 dropped_pending_frame, slot.frame_id);
    }
    async_roi_cv_.notify_one();
    globalPerf().record("Stage2_AsyncRoiSubmitted", 0.0);
    return true;
}

void Pipeline::asyncRoiWorkerLoop() {
    using Clock = std::chrono::high_resolution_clock;
    while (true) {
        AsyncRoiTask task;
        {
            std::unique_lock<std::mutex> lk(async_roi_mutex_);
            async_roi_cv_.wait(lk, [this] {
                return async_roi_thread_stop_ || !async_roi_pending_.empty();
            });
            if (async_roi_thread_stop_ && async_roi_pending_.empty()) {
                break;
            }
            task = std::move(async_roi_pending_.front());
            async_roi_pending_.pop_front();
            async_roi_worker_busy_ = true;
        }

        RoiStage2Output output;
        const auto t0 = Clock::now();
        bool copy_ready = true;
        AsyncRoiBuffer* artifact_buffer = nullptr;
        if (task.copy_event_recorded &&
            task.buffer_index >= 0 &&
            task.buffer_index < static_cast<int>(async_roi_buffers_.size())) {
            auto& buffer =
                async_roi_buffers_[static_cast<size_t>(task.buffer_index)];
            if (buffer.copy_done) {
                const auto copy_wait_start = Clock::now();
                const cudaError_t copy_err =
                    cudaEventSynchronize(buffer.copy_done);
                const double copy_wait_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - copy_wait_start).count();
                globalPerf().record("Stage2_AsyncRoiCopyWait", copy_wait_ms);
                buffer.copy_event_recorded = false;
                if (copy_err != cudaSuccess) {
                    copy_ready = false;
                    LOG_WARN("[AsyncROI] Copy event failed frame=%d err=%s",
                             task.frame_id, cudaGetErrorString(copy_err));
                }
                if (copy_ready) {
                    artifact_buffer = &buffer;
                }
            }
        }
        if (copy_ready) {
            if (task.host_gray_valid) {
                globalPerf().record("Stage2_AsyncRoiHostGrayTask", 0.0);
            }
            if (task.bgr_valid) {
                globalPerf().record("Stage2_AsyncRoiBgrTask", 0.0);
            }
            const bool p2_inline_enabled =
                task.input.p2_inline_feature_jobs_enabled;
            if (p2_inline_enabled &&
                task.p2_feature_decision.p2_depth_modes_enabled) {
                globalPerf().record("Stage2_P2FeatureJobInlineStage2", 0.0);
            }
            if (!p2_inline_enabled &&
                task.p2_feature_decision.p2_depth_modes_enabled) {
                if (p2DiagnosticOnlyFeatureJobsEnabled(config_)) {
                    globalPerf().record(
                        "Stage2_P2FeatureJobInlineSkippedDiagnosticOnly",
                        0.0);
                } else {
                    globalPerf().record(
                        "Stage2_P2FeatureJobInlineSkippedSelective",
                        0.0);
                }
            }
            if (p2_inline_enabled &&
                task.p2_feature_decision.split_feature_jobs &&
                !task.p2_feature_jobs.empty()) {
                globalPerf().record("Stage2_P2FeatureJobInlineFallback", 0.0);
            }
            std::lock_guard<std::mutex> post_lock(roi_postprocess_mutex_);
            output = runRoiStage2Core(task.input);
            if (artifact_buffer && !output.p2_artifact_rows.empty()) {
                P2FeatureDiagnosticBuffer artifact_view;
                artifact_view.left_gray_gpu = artifact_buffer->left_gray_gpu;
                artifact_view.left_gray_pitch = artifact_buffer->left_gray_pitch;
                artifact_view.right_gray_gpu = artifact_buffer->right_gray_gpu;
                artifact_view.right_gray_pitch = artifact_buffer->right_gray_pitch;
                writeP2FeatureDiagnosticArtifacts(output.p2_artifact_rows,
                                                  artifact_view,
                                                  task.input.width,
                                                  task.input.height);
            }
        } else {
            output.detections = task.input.left_detections;
            output.predict_only = true;
        }
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        globalPerf().record("Stage2_AsyncRoiWorker", elapsed_ms);
        if (elapsed_ms > static_cast<double>(config_.async_roi_deadline_ms)) {
            globalPerf().record("Stage2_AsyncRoiOverDeadline", elapsed_ms);
        }

        {
            std::lock_guard<std::mutex> lk(async_roi_mutex_);
            async_roi_worker_busy_ = false;
            const bool stale =
                async_roi_thread_stop_ ||
                (async_roi_expire_before_frame_ >= 0 &&
                 task.frame_id < async_roi_expire_before_frame_);
            releaseAsyncRoiBufferLocked(task.buffer_index);
            if (stale) {
                globalPerf().record("Stage2_AsyncRoiDropStaleResult",
                                    elapsed_ms);
            } else {
                AsyncRoiResult result;
                result.frame_id = task.frame_id;
                result.slot_index = task.slot_index;
                result.elapsed_ms = elapsed_ms;
                result.metadata = task.metadata;
                result.right_detections = task.input.right_detections;
                result.output = std::move(output);
                async_roi_completed_.push_back(std::move(result));
            }
        }
    }
    LOG_INFO("Async ROI Stage2 worker exited");
}

std::vector<int> Pipeline::drainCompletedAsyncRoiStage2() {
    std::deque<AsyncRoiResult> ready;
    int expire_before = -1;
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        ready.swap(async_roi_completed_);
        expire_before = async_roi_expire_before_frame_;
    }

    std::vector<int> accepted;
    while (!ready.empty()) {
        AsyncRoiResult result = std::move(ready.front());
        ready.pop_front();
        if (expire_before >= 0 && result.frame_id < expire_before) {
            globalPerf().record("Stage2_AsyncRoiDropStaleReady",
                                result.elapsed_ms);
            continue;
        }

        FrameSlot* live_slot = nullptr;
        if (result.slot_index >= 0 &&
            result.slot_index < RING_BUFFER_SIZE) {
            FrameSlot& candidate = slots_[result.slot_index];
            if (candidate.frame_id == result.frame_id) {
                live_slot = &candidate;
            }
        } else {
            globalPerf().record("Stage2_AsyncRoiBadSlotResult", 0.0);
        }

        if (live_slot) {
            applyRoiStage2Output(*live_slot, std::move(result.output));
            live_slot->bbox_source = BboxSource::YOLO;
            publishRoiFrameCallbacks(*live_slot);
            accepted.push_back(live_slot->frame_id);
        } else {
            FrameSlot shadow;
            shadow.frame_id = result.frame_id;
            restoreFrameMetadata(shadow, result.metadata);
            shadow.detections_right = std::move(result.right_detections);
            shadow.bbox_source = BboxSource::YOLO;
            applyRoiStage2Output(shadow, std::move(result.output));
            publishRoiResultCallback(shadow);
            accepted.push_back(shadow.frame_id);
            globalPerf().record("Stage2_AsyncRoiAcceptedReusedSlot",
                                result.elapsed_ms);
            globalPerf().record("Stage2_AsyncRoiFrameCallbackSkippedReusedSlot",
                                result.elapsed_ms);
        }
        globalPerf().record("Stage2_AsyncRoiAccepted", result.elapsed_ms);
    }
    return accepted;
}

void Pipeline::expireAsyncRoiBefore(int frame_id) {
    if (!async_roi_ready_) {
        return;
    }
    std::vector<int> expired_buffers;
    {
        std::lock_guard<std::mutex> lk(async_roi_mutex_);
        async_roi_expire_before_frame_ =
            std::max(async_roi_expire_before_frame_, frame_id);
        while (!async_roi_pending_.empty() &&
               async_roi_pending_.front().frame_id < async_roi_expire_before_frame_) {
            globalPerf().record("Stage2_AsyncRoiDropExpiredPending", 0.0);
            expired_buffers.push_back(async_roi_pending_.front().buffer_index);
            async_roi_pending_.pop_front();
        }
    }
    for (int buffer_index : expired_buffers) {
        releaseAsyncRoiBuffer(buffer_index, "expire_pending");
    }
}

}  // namespace stereo3d
