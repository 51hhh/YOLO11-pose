#include "pipeline.h"
#include "exposure_timestamp.h"
#include "../stereo/neural_feature_matcher.h"
#include "volleyball/logging/log.hpp"
#include "../utils/vpi_plane_access.h"

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
    volleyball::adapters::gpio::PWMTriggerEdgeEvent event;
    if (!pwm_trigger_->eventForSequence(frame_counter, event)) return;
    slot.trigger_sequence = event.sequence;
    slot.trigger_scheduled_monotonic_ns = event.scheduled_monotonic_ns;
    slot.trigger_edge_monotonic_ns = event.edge_monotonic_ns;
    slot.trigger_edge_realtime_ns = event.edge_realtime_ns;
    slot.trigger_write_uncertainty_ns = event.write_uncertainty_ns;

    ExposureTimestampConfig timestamp_config;
    timestamp_config.contract_valid =
        config_.camera.shutter_type == "global" &&
        config_.camera.exposure_timestamp_source == "nx_gpio_rising_edge";
    timestamp_config.delay_measured =
        config_.camera.exposure_delay_status == "measured" &&
        config_.camera.exposure_delay_uncertainty_known;
    timestamp_config.delay_ns = config_.camera.exposure_delay_ns;
    timestamp_config.delay_uncertainty_ns =
        config_.camera.exposure_delay_uncertainty_ns;
    timestamp_config.midpoint_offset_ns =
        config_.camera.exposure_midpoint_offset_ns;
    const auto timestamp = estimateExposureMidpointTimestamp(
        event.edge_realtime_ns,
        event.write_uncertainty_ns,
        timestamp_config);
    if (timestamp.timestamp_ns > 0) {
        slot.host_capture_timestamp_ns = timestamp.timestamp_ns;
        slot.capture_timestamp_from_trigger = true;
        slot.capture_timestamp_mapping_valid = timestamp.mapping_valid;
        slot.capture_timestamp_uncertainty_ns = timestamp.uncertainty_ns;
    }
}
#endif

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
    (void)shutdownP2FeatureDiagnosticLane();
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
    if (!run_state_.begin()) {
        return run_state_.isRunning() && !run_state_.hasFailed();
    }

    auto fail_start = [this](const char* reason) {
        LOG_ERROR("Pipeline start rollback: %s", reason);
        run_state_.markFailed();
#ifdef HIK_CAMERA_ENABLED
        grab_state_.stop();
#endif
        stop();
        return false;
    };

    try {
        last_fuse_time_ = {};
        last_fuse_capture_timestamp_ns_ = 0;
        if (hybrid_depth_) {
            hybrid_depth_->reset();
        }

        if (!startAsyncRoiStage2()) {
            return fail_start("async ROI worker did not start");
        }
        if (!startP2FeatureDiagnosticLane()) {
            return fail_start("P2 diagnostic worker did not start");
        }
        if (!startP2InlineFeatureWorkers()) {
            return fail_start("P2 inline workers did not start");
        }

#ifdef HIK_CAMERA_ENABLED
        if (camera_ && !camera_->startGrabbing()) {
            return fail_start("camera grabbing did not start");
        }
        if (pwm_trigger_ && !pwm_trigger_->start()) {
            return fail_start(
                "PWM trigger did not start; triggerless run refused");
        }

        if (camera_) {
            grab_state_.start();
            grab_thread_ = std::thread(&Pipeline::grabThreadMain, this);
            LOG_INFO("Async grab thread started");
        }
#endif

        const bool roi_mode =
            config_.disparity_strategy == DisparityStrategy::ROI_ONLY;
        pipeline_thread_ = std::thread(
            &Pipeline::pipelineThreadMain, this, roi_mode);
        LOG_INFO("Pipeline thread started (%s mode)",
                 roi_mode ? "ROI" : "Full-frame");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Pipeline thread/resource start threw: %s", e.what());
        return fail_start("thread or resource construction threw");
    } catch (...) {
        LOG_ERROR("Pipeline thread/resource start threw an unknown exception");
        return fail_start("thread or resource construction threw");
    }
}

void Pipeline::stop() {
    // 回调运行在 pipeline thread。若用户从回调内直接 stop，不能 join 自己；
    // 只发停止信号，保留 cleanup token 给主线程或析构函数。
    if (pipeline_thread_.joinable() &&
        pipeline_thread_.get_id() == std::this_thread::get_id()) {
        LOG_ERROR("Pipeline::stop() called from pipeline callback; "
                  "deferring join/cleanup to owner thread");
        run_state_.requestStop();
#ifdef HIK_CAMERA_ENABLED
        grab_state_.stop();
#endif
        return;
    }

    run_state_.requestStop();
    if (!run_state_.claimCleanup()) return;

#ifdef HIK_CAMERA_ENABLED
    grab_state_.stop();
#endif

    if (pipeline_thread_.joinable()) {
        pipeline_thread_.join();
    }

    const bool async_roi_drained = shutdownAsyncRoiStage2();
    const bool p2_diagnostic_drained = shutdownP2FeatureDiagnosticLane();
    shutdownP2InlineFeatureWorkers();

#ifdef HIK_CAMERA_ENABLED
    if (grab_thread_.joinable()) {
        grab_thread_.join();
    }
#endif

    const bool primary_streams_drained = streams_.syncAll();
    const bool streams_drained = primary_streams_drained &&
                                 async_roi_drained &&
                                 p2_diagnostic_drained;
    if (streams_drained) {
        for (auto& slot : slots_) {
            unlockStage0CudaViews(slot);
            unlockRectifiedCudaViews(slot);
        }
    } else {
        LOG_ERROR("Pipeline stream drain failed; retaining exported image locks "
                  "until resource destruction");
        run_state_.markFailed();
    }

#ifdef HIK_CAMERA_ENABLED
    if (pwm_trigger_) pwm_trigger_->stop();
    if (camera_) camera_->stopGrabbing();
#endif

    globalPerf().printReport();
    if (!run_state_.finishCleanup()) {
        LOG_ERROR("Pipeline cleanup state transition failed");
        run_state_.markFailed();
    }
}

void Pipeline::handleWorkerFailure(const char* worker,
                                   const char* detail) noexcept {
    LOG_ERROR("[Pipeline] Unhandled %s failure: %s",
              worker ? worker : "worker",
              detail ? detail : "unknown exception");
    globalPerf().record("Pipeline_WorkerFailure", 0.0);
    run_state_.markFailed();
#ifdef HIK_CAMERA_ENABLED
    // 停止接收新请求，但让已接受的相机写入完成 single-flight 握手。
    grab_state_.stop();
#endif
}

void Pipeline::pipelineThreadMain(bool roi_mode) noexcept {
    try {
        if (roi_mode) {
            pipelineLoopROI();
        } else {
            pipelineLoop();
        }
        // 主循环只能因 stop/failure 退出。若内部错误路径直接 return/break
        // 却未改变运行态，主线程会永远等待；在统一入口把它升级为可见失败。
        if (run_state_.isRunning()) {
            handleWorkerFailure(
                "pipeline thread",
                "processing loop exited while the run was still active");
        }
    } catch (const std::exception& e) {
        handleWorkerFailure("pipeline thread", e.what());
    } catch (...) {
        handleWorkerFailure("pipeline thread", "unknown exception");
    }
}

void Pipeline::asyncRoiWorkerThreadMain() noexcept {
    try {
        asyncRoiWorkerLoop();
    } catch (const std::exception& e) {
        handleWorkerFailure("async ROI worker", e.what());
    } catch (...) {
        handleWorkerFailure("async ROI worker", "unknown exception");
    }
}

void Pipeline::p2FeatureDiagnosticThreadMain() noexcept {
    try {
        p2FeatureDiagnosticWorkerLoop();
    } catch (const std::exception& e) {
        handleWorkerFailure("P2 diagnostic worker", e.what());
    } catch (...) {
        handleWorkerFailure("P2 diagnostic worker", "unknown exception");
    }
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
    bool started_any = false;
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
        started_any = true;
        return true;
    };

    const bool started =
        (!streams_.cudaStreamP2Ncc ||
         start_one(p2_inline_ncc_worker_, "NCC")) &&
        (!streams_.cudaStreamP2XFeat ||
         start_one(p2_inline_xfeat_worker_, "XFeat")) &&
        (!streams_.cudaStreamP2SuperPoint ||
         start_one(p2_inline_superpoint_worker_, "SuperPoint")) &&
        (!streams_.cudaStreamP2Aliked ||
         start_one(p2_inline_aliked_worker_, "ALIKED"));
    if (!started) {
        shutdownP2InlineFeatureWorkers();
        return false;
    }
    p2_inline_feature_workers_ready_.store(
        started_any, std::memory_order_release);
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
        !worker.thread.joinable() || !task) {
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
void Pipeline::grabThreadMain() noexcept {
    try {
        grabLoop();
    } catch (const std::exception& e) {
        // pipeline 可能正阻塞在 waitGrab()；先发布失败 completion 再停接新请求。
        grab_state_.complete(false);
        handleWorkerFailure("grab worker", e.what());
    } catch (...) {
        grab_state_.complete(false);
        handleWorkerFailure("grab worker", "unknown exception");
    }
}

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
        std::exception_ptr request_error;
        if (stL == VPI_SUCCESS && stR == VPI_SUCCESS) {
            try {
                const auto& plane_l = imgDataL.buffer.pitch.planes[0];
                const auto& plane_r = imgDataR.buffer.pitch.planes[0];
                const bool views_valid =
                    vpiPlaneCovers(plane_l, config_.camera.width,
                                   config_.camera.height,
                                   config_.camera.width) &&
                    vpiPlaneCovers(plane_r, config_.camera.width,
                                   config_.camera.height,
                                   config_.camera.width);
                if (!views_valid) {
                    LOG_ERROR("[Pipeline] invalid async raw host view: "
                              "L=%dx%d/%d R=%dx%d/%d expected=%dx%d",
                              plane_l.width, plane_l.height,
                              plane_l.pitchBytes,
                              plane_r.width, plane_r.height,
                              plane_r.pitchBytes,
                              config_.camera.width, config_.camera.height);
                } else {
                    ok = camera_->grabFramePair(
                        vpiPlaneData(imgDataL.buffer.pitch.planes[0]),
                        vpiPlaneData(imgDataR.buffer.pitch.planes[0]),
                        plane_l.pitchBytes, plane_r.pitchBytes,
                        1000, resL, resR);
                }
                const uint64_t fallback_capture_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
                slot.host_capture_timestamp_ns = chooseCaptureTimestampNs(
                    resL.host_timestamp, resR.host_timestamp,
                    fallback_capture_ns);
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
            } catch (...) {
                ok = false;
                request_error = std::current_exception();
            }
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
        if (request_error) {
            std::rethrow_exception(request_error);
        }
    }
}

bool Pipeline::requestGrab(int slot_idx) {
    if (!run_state_.isRunning() ||
        !grab_state_.request(slot_idx, RING_BUFFER_SIZE)) {
        LOG_ERROR("[Pipeline] reject async grab request: slot=%d running=%d",
                  slot_idx, run_state_.isRunning() ? 1 : 0);
        return false;
    }
    return true;
}

bool Pipeline::waitGrab() {
    return grab_state_.waitForCompletion();
}
#endif

}  // namespace stereo3d
