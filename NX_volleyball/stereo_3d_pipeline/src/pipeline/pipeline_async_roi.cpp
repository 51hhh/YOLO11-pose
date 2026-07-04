/**
 * @file pipeline_async_roi.cpp
 * @brief Async ROI Stage2 worker and snapshot management.
 */

#include "pipeline.h"
#include "../stereo/neural_feature_matcher.h"
#include "../utils/logger.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cuda_runtime.h>
#include <string>
#include <utility>

namespace stereo3d {

namespace {

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
    slot.p2_feature_job_count = meta.p2_feature_job_count;
    slot.p2_left_count = meta.p2_left_count;
    slot.p2_right_count = meta.p2_right_count;
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
    slot.p2_feature_job_count = static_cast<int>(jobs.size());
    slot.p2_left_count = decision.left_count;
    slot.p2_right_count = decision.right_count;
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

    int least_priority = 0;
    int greatest_priority = 0;
    const cudaError_t priority_err =
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    const bool use_low_priority = priority_err == cudaSuccess;
    (void)greatest_priority;
    auto create_async_stream = [&](cudaStream_t* stream,
                                   const char* name) -> cudaError_t {
        cudaError_t e = use_low_priority
            ? cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking,
                                           least_priority)
            : cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
        if (e != cudaSuccess && use_low_priority) {
            LOG_WARN("Async ROI: create low-priority %s stream failed (%s), "
                     "falling back to default priority",
                     name, cudaGetErrorString(e));
            e = cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
        }
        return e;
    };

    cudaError_t err = create_async_stream(&async_roi_stream_, "worker");
    if (err != cudaSuccess) {
        LOG_ERROR("Async ROI: create worker stream failed: %s",
                  cudaGetErrorString(err));
        return false;
    }
    err = create_async_stream(&async_roi_copy_stream_, "copy");
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
        return true;
    }

    const auto t0 = Clock::now();
    const cudaError_t err = cudaEventSynchronize(buffer.copy_done);
    const double wait_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    globalPerf().record("Stage2_AsyncRoiCopyWait", wait_ms);
    buffer.copy_event_recorded = false;
    if (err != cudaSuccess) {
        LOG_WARN("[AsyncROI] Buffer copy wait failed buffer=%d reason=%s err=%s",
                 buffer_index,
                 reason ? reason : "unknown",
                 cudaGetErrorString(err));
        return false;
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
    const bool need_host_gray =
        roiStage2NeedsHostImages(slot.detections, slot.detections_right);
    const bool neural_needs_bgr =
        neural_feature_matcher_ && neural_feature_matcher_->requiresBgrInput();
    const bool has_stereo_detections =
        !slot.detections.empty() && !slot.detections_right.empty();
    const bool need_bgr =
        colorPipelineEnabled() &&
        has_stereo_detections &&
        (config_.dual_yolo.depth_roi_iou_region_color_patch ||
         config_.dual_yolo.depth_roi_patch_iou_color_edge ||
         neural_needs_bgr);
    if (need_host_gray) {
        globalPerf().record("Stage2_AsyncRoiNeedHostGray", 0.0);
    }
    if (need_bgr) {
        globalPerf().record("Stage2_AsyncRoiNeedBgr", 0.0);
    }
    const P2FeatureJobPolicy p2_policy = makeP2FeatureJobPolicy(config_);
    const P2FeatureJobDecision p2_decision = decideP2FeatureJobs(
        p2_policy,
        slot.frame_id,
        slot.detections,
        slot.detections_right,
        need_host_gray,
        need_bgr);
    std::vector<P2FeatureJobDescriptor> p2_feature_jobs =
        buildP2FeatureJobDescriptors(
            p2_policy,
            p2_decision,
            need_host_gray,
            need_bgr);
    if (p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobConfigured", 0.0);
    }
    if (p2_decision.realtime_requested) {
        globalPerf().record("Stage2_P2FeatureJobRealtimeRequested", 0.0);
    } else if (p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobRealtimeNotAttempted", 0.0);
    }
    if (p2_decision.diagnostic_requested) {
        globalPerf().record("Stage2_P2FeatureJobDiagnosticRequested", 0.0);
    }
    if (p2_policy.split_feature_jobs && p2_feature_jobs.empty() &&
        p2_decision.p2_depth_modes_enabled) {
        globalPerf().record("Stage2_P2FeatureJobSplitNoJob", 0.0);
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
            }
        }
        if (copy_ready) {
            if (task.host_gray_valid) {
                globalPerf().record("Stage2_AsyncRoiHostGrayTask", 0.0);
            }
            if (task.bgr_valid) {
                globalPerf().record("Stage2_AsyncRoiBgrTask", 0.0);
            }
            if (task.p2_feature_decision.p2_depth_modes_enabled) {
                globalPerf().record("Stage2_P2FeatureJobInlineStage2", 0.0);
            }
            if (task.p2_feature_decision.split_feature_jobs &&
                !task.p2_feature_jobs.empty()) {
                globalPerf().record("Stage2_P2FeatureJobInlineFallback", 0.0);
            }
            std::lock_guard<std::mutex> post_lock(roi_postprocess_mutex_);
            output = runRoiStage2Core(task.input);
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
