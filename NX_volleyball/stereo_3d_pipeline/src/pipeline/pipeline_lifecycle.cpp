#include "pipeline.h"
#include "../stereo/neural_feature_matcher.h"
#include "../utils/logger.h"

#include <cstdint>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>

namespace stereo3d {

#ifdef HIK_CAMERA_ENABLED
void Pipeline::attachPwmTriggerMetadata(
    FrameSlot& slot,
    std::uint32_t frame_counter) const {
    if (!pwm_trigger_ || frame_counter == 0) return;
    PWMTriggerEdgeEvent event;
    if (!pwm_trigger_->eventForSequence(frame_counter, event)) return;
    slot.trigger_sequence = event.sequence;
    slot.trigger_scheduled_monotonic_ns = event.scheduled_monotonic_ns;
    slot.trigger_edge_monotonic_ns = event.edge_monotonic_ns;
    slot.trigger_edge_realtime_ns = event.edge_realtime_ns;
    slot.trigger_write_uncertainty_ns = event.write_uncertainty_ns;
}
#endif

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
    shutdownP2FeatureDiagnosticLane();
    destroyAsyncRoiStage2();
    shutdownP2InlineFeatureWorkers();
    for (auto& slot : slots_) {
        slot.destroy();
    }
    if (tnrPayloadL_) vpiPayloadDestroy(tnrPayloadL_);
    if (tnrPayloadR_) vpiPayloadDestroy(tnrPayloadR_);
    if (tnrNV12L_) vpiImageDestroy(tnrNV12L_);
    if (tnrNV12R_) vpiImageDestroy(tnrNV12R_);
    if (tnrOutNV12L_) vpiImageDestroy(tnrOutNV12L_);
    if (tnrOutNV12R_) vpiImageDestroy(tnrOutNV12R_);
    streams_.destroy();
}

bool Pipeline::start() {
    if (running_.exchange(true)) return true;

    last_fuse_time_ = {};
    last_fuse_capture_timestamp_ns_ = 0;
    if (hybrid_depth_) {
        hybrid_depth_->reset();
    }

    if (!startAsyncRoiStage2()) {
        running_ = false;
        return false;
    }
    if (!startP2FeatureDiagnosticLane()) {
        running_ = false;
        shutdownP2InlineFeatureWorkers();
        shutdownAsyncRoiStage2();
        return false;
    }
    if (!startP2InlineFeatureWorkers()) {
        running_ = false;
        shutdownP2FeatureDiagnosticLane();
        shutdownAsyncRoiStage2();
        return false;
    }

#ifdef HIK_CAMERA_ENABLED
    if (camera_ && !camera_->startGrabbing()) {
        LOG_ERROR("Failed to start camera grabbing");
        running_ = false;
        shutdownP2InlineFeatureWorkers();
        shutdownP2FeatureDiagnosticLane();
        shutdownAsyncRoiStage2();
        return false;
    }
    if (pwm_trigger_ && !pwm_trigger_->start()) {
        LOG_ERROR("Failed to start PWM trigger; refusing to run a triggerless pipeline");
        if (camera_) camera_->stopGrabbing();
        running_ = false;
        shutdownP2InlineFeatureWorkers();
        shutdownP2FeatureDiagnosticLane();
        shutdownAsyncRoiStage2();
        return false;
    }

    if (camera_) {
        grab_state_.start();
        grab_thread_ = std::thread(&Pipeline::grabLoop, this);
        LOG_INFO("Async grab thread started");
    }
#endif

    if (config_.disparity_strategy == DisparityStrategy::ROI_ONLY) {
        pipeline_thread_ = std::thread(&Pipeline::pipelineLoopROI, this);
        LOG_INFO("Pipeline thread started (ROI mode)");
    } else {
        pipeline_thread_ = std::thread(&Pipeline::pipelineLoop, this);
        LOG_INFO("Pipeline thread started (Full-frame mode)");
    }
    return true;
}

void Pipeline::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;

#ifdef HIK_CAMERA_ENABLED
    grab_state_.stop();
#endif

    if (pipeline_thread_.joinable()) {
        pipeline_thread_.join();
    }

    shutdownAsyncRoiStage2();
    shutdownP2FeatureDiagnosticLane();
    shutdownP2InlineFeatureWorkers();

#ifdef HIK_CAMERA_ENABLED
    if (grab_thread_.joinable()) {
        grab_thread_.join();
    }
#endif

    streams_.syncAll();

#ifdef HIK_CAMERA_ENABLED
    if (pwm_trigger_) pwm_trigger_->stop();
    if (camera_) camera_->stopGrabbing();
#endif

    globalPerf().printReport();
}

void Pipeline::p2InlineFeatureWorkerLoop(P2InlineFeatureWorker* worker,
                                         const char* label) {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(worker->mutex);
            worker->cv.wait(lk, [worker] {
                return worker->stop || worker->has_task;
            });
            if (worker->stop && !worker->has_task) {
                break;
            }
            task = std::move(worker->task);
            worker->has_task = false;
        }

        try {
            if (task) {
                task();
            }
        } catch (const std::exception& e) {
            LOG_ERROR("P2 inline %s worker task failed: %s",
                      label, e.what());
        } catch (...) {
            LOG_ERROR("P2 inline %s worker task failed with unknown error",
                      label);
        }

        {
            std::lock_guard<std::mutex> lk(worker->mutex);
            worker->done = true;
        }
        worker->cv.notify_all();
    }
}

bool Pipeline::startP2InlineFeatureWorkers() {
    auto start_one = [&](P2InlineFeatureWorker& worker,
                         const char* label) -> bool {
        if (worker.thread.joinable()) {
            return true;
        }
        {
            std::lock_guard<std::mutex> lk(worker.mutex);
            worker.stop = false;
            worker.has_task = false;
            worker.done = true;
            worker.task = nullptr;
        }
        try {
            worker.thread = std::thread(
                &Pipeline::p2InlineFeatureWorkerLoop, this, &worker, label);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to start P2 inline %s worker: %s",
                      label, e.what());
            return false;
        }
        return true;
    };

    if (!start_one(p2_inline_ncc_worker_, "NCC") ||
        !start_one(p2_inline_xfeat_worker_, "XFeat") ||
        !start_one(p2_inline_superpoint_worker_, "SuperPoint") ||
        !start_one(p2_inline_aliked_worker_, "ALIKED")) {
        shutdownP2InlineFeatureWorkers();
        return false;
    }
    p2_inline_feature_workers_ready_.store(true, std::memory_order_release);
    return true;
}

void Pipeline::shutdownP2InlineFeatureWorkers() {
    p2_inline_feature_workers_ready_.store(false, std::memory_order_release);
    auto stop_one = [](P2InlineFeatureWorker& worker) {
        {
            std::lock_guard<std::mutex> lk(worker.mutex);
            worker.stop = true;
        }
        worker.cv.notify_all();
        if (worker.thread.joinable()) {
            worker.thread.join();
        }
        {
            std::lock_guard<std::mutex> lk(worker.mutex);
            worker.stop = false;
            worker.has_task = false;
            worker.done = true;
            worker.task = nullptr;
        }
    };
    stop_one(p2_inline_ncc_worker_);
    stop_one(p2_inline_xfeat_worker_);
    stop_one(p2_inline_superpoint_worker_);
    stop_one(p2_inline_aliked_worker_);
}

bool Pipeline::submitP2InlineFeatureTask(P2InlineFeatureWorker& worker,
                                         std::function<void()> task) {
    if (!p2_inline_feature_workers_ready_.load(std::memory_order_acquire) ||
        !task) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(worker.mutex);
        if (worker.stop || worker.has_task || !worker.done) {
            return false;
        }
        worker.task = std::move(task);
        worker.done = false;
        worker.has_task = true;
    }
    worker.cv.notify_one();
    return true;
}

void Pipeline::waitP2InlineFeatureTask(P2InlineFeatureWorker& worker) {
    std::unique_lock<std::mutex> lk(worker.mutex);
    worker.cv.wait(lk, [&worker] {
        return worker.done && !worker.has_task;
    });
}

#ifdef HIK_CAMERA_ENABLED
void Pipeline::grabLoop() {
    int slot_idx = -1;
    while (grab_state_.waitForRequest(slot_idx)) {

        auto& slot = slots_[slot_idx];

        VPIImageData imgDataL, imgDataR;
        VPIStatus stL = vpiImageLockData(
            slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataL);
        VPIStatus stR = vpiImageLockData(
            slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataR);

        GrabResult resL, resR;
        bool ok = false;
        if (stL == VPI_SUCCESS && stR == VPI_SUCCESS) {
            ok = camera_->grabFramePair(
                static_cast<uint8_t*>(imgDataL.buffer.pitch.planes[0].data),
                static_cast<uint8_t*>(imgDataR.buffer.pitch.planes[0].data),
                imgDataL.buffer.pitch.planes[0].pitchBytes,
                imgDataR.buffer.pitch.planes[0].pitchBytes,
                1000, resL, resR);
            const uint64_t fallback_capture_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            slot.host_capture_timestamp_ns = chooseCaptureTimestampNs(
                resL.host_timestamp, resR.host_timestamp, fallback_capture_ns);
            slot.left_receive_monotonic_ns = resL.receive_monotonic_ns;
            slot.right_receive_monotonic_ns = resR.receive_monotonic_ns;
            slot.left_receive_realtime_ns = resL.receive_realtime_ns;
            slot.right_receive_realtime_ns = resR.receive_realtime_ns;
            slot.left_sdk_host_timestamp_raw = resL.host_timestamp;
            slot.right_sdk_host_timestamp_raw = resR.host_timestamp;
            slot.left_timestamp_us = resL.timestamp_us;
            slot.right_timestamp_us = resR.timestamp_us;
            slot.left_frame_number = resL.frame_number;
            slot.right_frame_number = resR.frame_number;
            slot.left_frame_counter = resL.frame_counter;
            slot.right_frame_counter = resR.frame_counter;
            slot.left_trigger_index = resL.trigger_index;
            slot.right_trigger_index = resR.trigger_index;
            attachPwmTriggerMetadata(slot, resL.frame_counter);
            slot.stereo_timestamp_residual_ns =
                resL.stereo_timestamp_residual_ns;
        } else {
            LOG_WARN("[Pipeline] grabLoop VPI raw lock failed: L=%d R=%d",
                     (int)stL, (int)stR);
        }

        if (stL == VPI_SUCCESS) {
            const VPIStatus unlock = vpiImageUnlock(slot.rawL);
            if (unlock != VPI_SUCCESS) {
                LOG_ERROR("[Pipeline] grabLoop left raw unlock failed: %d",
                          (int)unlock);
                ok = false;
            }
        }
        if (stR == VPI_SUCCESS) {
            const VPIStatus unlock = vpiImageUnlock(slot.rawR);
            if (unlock != VPI_SUCCESS) {
                LOG_ERROR("[Pipeline] grabLoop right raw unlock failed: %d",
                          (int)unlock);
                ok = false;
            }
        }

        if (!grab_state_.complete(ok)) {
            LOG_ERROR("[Pipeline] async grab completion state mismatch: slot=%d",
                      slot_idx);
        }
    }
}

bool Pipeline::requestGrab(int slot_idx) {
    if (!running_ || !grab_state_.request(slot_idx, RING_BUFFER_SIZE)) {
        LOG_ERROR("[Pipeline] reject async grab request: slot=%d running=%d",
                  slot_idx, running_.load() ? 1 : 0);
        return false;
    }
    return true;
}

bool Pipeline::waitGrab() {
    return grab_state_.waitForCompletion();
}
#endif

}  // namespace stereo3d
