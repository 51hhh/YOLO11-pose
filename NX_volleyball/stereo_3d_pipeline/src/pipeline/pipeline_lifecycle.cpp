#include "pipeline.h"
#include "../stereo/neural_feature_matcher.h"
#include "../utils/logger.h"

#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>

namespace stereo3d {

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

void Pipeline::start() {
    if (running_.exchange(true)) return;

    if (!startAsyncRoiStage2()) {
        running_ = false;
        return;
    }
    if (!startP2FeatureDiagnosticLane()) {
        running_ = false;
        shutdownP2InlineFeatureWorkers();
        shutdownAsyncRoiStage2();
        return;
    }
    if (!startP2InlineFeatureWorkers()) {
        running_ = false;
        shutdownP2FeatureDiagnosticLane();
        shutdownAsyncRoiStage2();
        return;
    }

#ifdef HIK_CAMERA_ENABLED
    if (camera_ && !camera_->startGrabbing()) {
        LOG_ERROR("Failed to start camera grabbing");
        running_ = false;
        shutdownP2InlineFeatureWorkers();
        shutdownP2FeatureDiagnosticLane();
        shutdownAsyncRoiStage2();
        return;
    }
    if (pwm_trigger_ && !pwm_trigger_->start()) {
        LOG_ERROR("Failed to start PWM trigger - camera may not receive triggers");
    }

    if (camera_) {
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
}

void Pipeline::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;

#ifdef HIK_CAMERA_ENABLED
    grab_request_cv_.notify_all();
    grab_done_cv_.notify_all();
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
    while (running_) {
        int slot_idx;
        {
            std::unique_lock<std::mutex> lk(grab_mutex_);
            grab_request_cv_.wait(lk, [this]{
                return grab_request_slot_ >= 0 || !running_;
            });
            if (!running_) break;
            slot_idx = grab_request_slot_;
            grab_request_slot_ = -1;
        }

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
            slot.left_timestamp_us = resL.timestamp_us;
            slot.right_timestamp_us = resR.timestamp_us;
            slot.left_frame_number = resL.frame_number;
            slot.right_frame_number = resR.frame_number;
            slot.left_frame_counter = resL.frame_counter;
            slot.right_frame_counter = resR.frame_counter;
            slot.left_trigger_index = resL.trigger_index;
            slot.right_trigger_index = resR.trigger_index;
        } else {
            LOG_WARN("[Pipeline] grabLoop VPI raw lock failed: L=%d R=%d",
                     (int)stL, (int)stR);
        }

        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rawL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rawR);

        {
            std::lock_guard<std::mutex> lk(grab_mutex_);
            grab_done_ = true;
            grab_done_ok_ = ok;
        }
        grab_done_cv_.notify_one();
    }
}

void Pipeline::requestGrab(int slot_idx) {
    {
        std::lock_guard<std::mutex> lk(grab_mutex_);
        grab_request_slot_ = slot_idx;
        grab_done_ = false;
    }
    grab_request_cv_.notify_one();
}

bool Pipeline::waitGrab() {
    std::unique_lock<std::mutex> lk(grab_mutex_);
    grab_done_cv_.wait(lk, [this]{ return grab_done_ || !running_; });
    return grab_done_ok_;
}
#endif

}  // namespace stereo3d
