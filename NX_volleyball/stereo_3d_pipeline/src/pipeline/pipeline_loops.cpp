/**
 * @file pipeline_loops.cpp
 * @brief Main pipeline loop scheduling implementations.
 */

#include "pipeline.h"
#include "../utils/logger.h"

#include <chrono>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

namespace stereo3d {

void Pipeline::pipelineLoop() {
    using Clock = std::chrono::high_resolution_clock;
    auto fps_start = Clock::now();
    int fps_count = 0;

    int next_grab_frame = 0;      // 下一次 Stage0 要抓取的 frame id
    int next_detect_frame = 0;    // 下一次 Stage1/2 要提交的 frame id
    int next_fuse_frame = 0;      // 下一次 Stage3 要输出的 frame id

    // ===== 填充: 先抓首帧 =====
    {
        int slot_idx = next_grab_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        slot.reset();
        slot.frame_id = next_grab_frame;

        ScopedTimer t0("Stage0_GrabRect");
        stage0_grab_and_rectify(slot);
        globalPerf().record("Stage0_GrabRect", t0.elapsedMs());

        next_grab_frame++;
    }

    while (running_) {
        // --- Stage 3: 融合上一帧 (若已提交 detect/stereo) ---
        if (next_fuse_frame < next_detect_frame) {
            int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
            auto& slot = slots_[slot_idx];

            // 等 Detect 完成
            {
                ScopedTimer tw("Stage3_WaitDetect");
                waitDetectDone(streams_.cudaStreamFuse, slot);
                cudaStreamSynchronize(streams_.cudaStreamFuse);
                globalPerf().record("Stage3_WaitDetect", tw.elapsedMs());
            }

            // 等 Stereo 完成（VPI stream 级同步，放在 Stage3 前，避免阻塞 Stage1/2 提交）
            {
                ScopedTimer tws("Stage3_WaitStereo");
                vpiStreamSync(streams_.vpiStreamGPU);
                globalPerf().record("Stage3_WaitStereo", tws.elapsedMs());
            }

            {
                ScopedTimer t3("Stage3_Fuse");
                stage3_fuse(slot, slot_idx);
                globalPerf().record("Stage3_Fuse", t3.elapsedMs());
            }

            if (result_callback_) {
                result_callback_(slot.frame_id, slot.results);
            }

            if (frame_callback_) {
                FrameCallbackData frame{
                    slot.frame_id,
                    slot.rectGray_vpiL,
                    slot.rectGray_vpiR,
                    slot.rectBGR_vpiL,
                    slot.rectBGR_vpiR,
                    slot.rawL,
                    slot.rawR,
                    slot.detections,
                    slot.detections_right,
                    slot.results,
                    makeFrameMetadata(slot),
                    current_fps_.load()
                };
                frame_callback_(frame);
            }

            next_fuse_frame++;

            // ---- FPS 统计基于输出帧 ----
            fps_count++;
            auto now = Clock::now();
            double elapsed_s = std::chrono::duration<double>(now - fps_start).count();
            if (elapsed_s >= 1.0) {
                current_fps_ = static_cast<float>(fps_count / elapsed_s);
                if (config_.stats_interval > 0 &&
                    next_fuse_frame % config_.stats_interval == 0) {
                    LOG_INFO("FPS: %.1f  (Output frame %d)", current_fps_.load(), next_fuse_frame);
                }
                fps_count = 0;
                fps_start = now;
            }
        }

        // --- Stage 1 + Stage 2: 异步提交当前帧 ---
        if (next_detect_frame < next_grab_frame) {
            int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
            auto& slot = slots_[slot_idx];

            // 帧同步跳变: 跳过此帧
            if (slot.grab_failed) {
                vpiStreamSync(streams_.vpiStreamPVA);
                next_detect_frame++;
                next_fuse_frame = next_detect_frame;
            } else {
                // 等待 VPI remap 完成 (stage0 异步提交)
                vpiStreamSync(streams_.vpiStreamPVA);
                cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);

                // DLA/GPU 都等 rect 完成后开始
                cudaStreamWaitEvent(getDLAStream(slot.frame_id),
                                    slot.evtRectDone, 0);
                cudaStreamWaitEvent(streams_.cudaStreamGPU,
                                    slot.evtRectDone, 0);

                {
                    ScopedTimer t1("Stage1_DetectSubmit");
                    stage1_detect(slot, slot_idx);
                    globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
                }
                recordDetectDoneEvents(slot);

                {
                    ScopedTimer t2("Stage2_StereoSubmit");
                    stage2_stereo(slot);
                    globalPerf().record("Stage2_StereoSubmit", t2.elapsedMs());
                }

                next_detect_frame++;
            } // end else (grab_failed)
        }

        // --- Stage 0: 抓取下一帧 ---
        int slot_idx = next_grab_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        slot.reset();
        slot.frame_id = next_grab_frame;

        {
            ScopedTimer t0("Stage0_GrabRect");
            stage0_grab_and_rectify(slot);
            globalPerf().record("Stage0_GrabRect", t0.elapsedMs());
        }
        next_grab_frame++;
    }

    // ===== 排空阶段 =====
    // 同步最后的 VPI remap
    vpiStreamSync(streams_.vpiStreamPVA);

    // 1) 提交所有已抓取但尚未提交 detect/stereo 的帧
    while (next_detect_frame < next_grab_frame) {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];

        cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);
        cudaStreamWaitEvent(getDLAStream(slot.frame_id), slot.evtRectDone, 0);
        cudaStreamWaitEvent(streams_.cudaStreamGPU, slot.evtRectDone, 0);

        stage1_detect(slot, slot_idx);
        recordDetectDoneEvents(slot);
        stage2_stereo(slot);

        next_detect_frame++;
    }

    // 2) 融合所有已提交 detect/stereo 但尚未输出的帧
    while (next_fuse_frame < next_detect_frame) {
        int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];

        waitDetectDone(streams_.cudaStreamFuse, slot);
        cudaStreamSynchronize(streams_.cudaStreamFuse);
        vpiStreamSync(streams_.vpiStreamGPU);

        stage3_fuse(slot, slot_idx);
        if (result_callback_) result_callback_(slot.frame_id, slot.results);

        next_fuse_frame++;
    }

    LOG_INFO("Pipeline loop exited");
}

// ===================================================================
// Stage 实现
// ===================================================================

void Pipeline::pipelineLoopROI() {
    using Clock = std::chrono::high_resolution_clock;
    auto fps_start = Clock::now();
    int fps_count = 0;
    int stale_drop_count = 0;
    constexpr double kStage1SubmitOutlierMs = 15.0;
    constexpr double kStage2WaitYoloOutlierMs = 8.0;
    constexpr double kStage0WaitGrabOutlierMs = 8.0;

    int next_grab_frame   = 0;
    int next_detect_frame = 0;
    int next_fuse_frame   = 0;

    auto sync_rect_for_detect = [&](FrameSlot& slot, int slot_idx) -> bool {
        ScopedTimer tw("Stage1_WaitRect");
        VPIStatus st = vpiStreamSync(streams_.vpiStreamPVA);
        const double wait_rect_ms = tw.elapsedMs();
        globalPerf().record("Stage1_WaitRect", wait_rect_ms);
        if (st != VPI_SUCCESS) {
            LOG_ERROR("[Pipeline] VPI rectification sync failed before detect: "
                      "frame=%d slot=%d err=%d",
                      slot.frame_id, slot_idx, (int)st);
            slot.grab_failed = true;
            return false;
        }
        cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);
        return true;
    };

    auto newer_yolo_ready = [&](int frame_id) -> bool {
        if (!config_.drop_stale_roi_frames || frame_id >= next_detect_frame) {
            return false;
        }
        const FrameSlot& newer = slots_[frame_id % RING_BUFFER_SIZE];
        return newer.frame_id == frame_id &&
               !newer.grab_failed &&
               newer.is_detect_frame &&
               detectEventsReady(newer);
    };

    auto account_output_frame = [&](int output_frame_id) {
        fps_count++;
        auto now = Clock::now();
        double elapsed_s = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed_s >= 1.0) {
            current_fps_ = static_cast<float>(fps_count / elapsed_s);
            fps_count = 0;
            fps_start = now;
            if (config_.stats_interval > 0) {
                LOG_INFO("[ROI] FPS: %.1f  (Output frame %d, stale_drop=%d)",
                         current_fps_.load(), output_frame_id, stale_drop_count);
            }
        }
    };

    auto drain_async_outputs = [&]() {
        if (!async_roi_ready_) {
            return;
        }
        const std::vector<int> accepted = drainCompletedAsyncRoiStage2();
        for (int frame_id : accepted) {
            account_output_frame(frame_id);
        }
    };

    // ===== 填充: 同步抓取 + 处理首帧 =====
    {
        int slot_idx = next_grab_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        waitAsyncRoiSlotSnapshotDone(slot_idx, "roi_initial_grab_reuse");
        slot.reset();
        slot.frame_id = next_grab_frame;

        bool grab_preloaded = false;
#ifdef HIK_CAMERA_ENABLED
        if (camera_) {
            requestGrab(slot_idx);
            if (!waitGrab()) {
                slot.grab_failed = true;
            }
            grab_preloaded = true;
        }
#endif
        ScopedTimer t0("Stage0_Process");
        stage0_grab_and_rectify(slot, grab_preloaded);
        globalPerf().record("Stage0_Process", t0.elapsedMs());
        next_grab_frame++;
    }

    // ===== 填充: 提交首帧检测 =====
    {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        slot.is_detect_frame = true;  // 首帧必为检测帧

        if (slot.grab_failed) {
            next_detect_frame++;
        } else if (sync_rect_for_detect(slot, slot_idx)) {
            auto dlaStream = getDLAStream(slot.frame_id);
            cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);

            ScopedTimer t1("Stage1_DetectSubmit");
            stage1_detect(slot, slot_idx);
            const double stage1_ms = t1.elapsedMs();
            globalPerf().record("Stage1_DetectSubmit", stage1_ms);
            if (stage1_ms > kStage1SubmitOutlierMs) {
                LOG_WARN("[PerfOutlier] Stage1_DetectSubmit frame=%d slot=%d ms=%.2f right_submitted=%d",
                         slot.frame_id, slot_idx, stage1_ms,
                         slot.right_detection_submitted ? 1 : 0);
            }
            recordDetectDoneEvents(slot);
        }
        next_detect_frame++;
    }

    while (running_) {
        drain_async_outputs();

        // ====================================================================
        // Phase A: 发起异步采集 (极速, ~0.01ms)
        //   grab 线程锁定 VPI Image → 阻塞等待相机 USB 传输
        //   pipeline 线程继续执行 Stage1/Stage2, 与 grab 并行
        // ====================================================================
        int grab_slot_idx = next_grab_frame % RING_BUFFER_SIZE;
        {
            auto& slot = slots_[grab_slot_idx];
            waitAsyncRoiSlotSnapshotDone(grab_slot_idx, "roi_grab_reuse");
            slot.reset();
            slot.frame_id = next_grab_frame;
#ifdef HIK_CAMERA_ENABLED
            if (camera_) {
                requestGrab(grab_slot_idx);
            }
#endif
        }

        // ====================================================================
        // Phase B: Stage1 — 提交检测 (与 grab 并行, ~3ms)
        //   此帧已在上一轮 Phase D 中完成 rectify
        //   SOT 模式: 检测帧→YOLO enqueue, 填充帧→tracker infill
        // ====================================================================
        if (next_detect_frame < next_grab_frame) {
            int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
            auto& slot = slots_[slot_idx];

            if (slot.grab_failed) {
                next_detect_frame++;
            } else {
                // 判断是否为检测帧
                // 固定节拍检测：按 detect_interval 控制 YOLO 频率，
                // 不再因 tracker 是否处于 TRACKING 状态而抢跑检测。
                bool is_detect = !tracker_ ||
                                 (slot.frame_id % effective_detect_interval_ == 0);
                slot.is_detect_frame = is_detect;

                if (is_detect) {
                    // ---- YOLO 检测帧 ----
                    if (!sync_rect_for_detect(slot, slot_idx)) {
                        next_detect_frame++;
                        continue;
                    }

                    auto dlaStream = getDLAStream(slot.frame_id);
                    cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);

                    {
                        ScopedTimer t1("Stage1_DetectSubmit");
                        stage1_detect(slot, slot_idx);
                        const double stage1_ms = t1.elapsedMs();
                        globalPerf().record("Stage1_DetectSubmit", stage1_ms);
                        if (stage1_ms > kStage1SubmitOutlierMs) {
                            LOG_WARN("[PerfOutlier] Stage1_DetectSubmit frame=%d slot=%d ms=%.2f right_submitted=%d",
                                     slot.frame_id, slot_idx, stage1_ms,
                                     slot.right_detection_submitted ? 1 : 0);
                        }
                    }
                    recordDetectDoneEvents(slot);
                } else {
                    // ---- Tracker 填充帧 ----
                    // 等 rectify 完成 (tracker 需要 rectified BGR)
                    vpiStreamSync(streams_.vpiStreamPVA);
                    cudaStreamSynchronize(streams_.cudaStreamGPU);

                    // *** GPU Power Control: 强制等待所有之前的YOLO任务完成 ***
                    // 防止YOLO+tracker并行运行导致功率爆口
                    // 在detect_interval=3的策略下，最近的YOLO应该已在Phase C collect了
                    // 等待检测完成
                    {
                        ScopedTimer tw("Stage1_WaitYOLOComplete");
                        cudaStreamSynchronize(streams_.cudaStreamDLA);
                        if (dualYoloEnabled()) {
                            cudaStreamSynchronize(streams_.cudaStreamDLA_R);
                        }
                        globalPerf().record("Stage1_WaitYOLOComplete", tw.elapsedMs());
                    }

                    {
                        ScopedTimer tt("Stage1_TrackerInfill");
                        tracker_infill(slot);
                        globalPerf().record("Stage1_TrackerInfill", tt.elapsedMs());
                    }
                    // 标记 "detect done" 让 Phase C 的等待逻辑兼容
                    cudaEventRecord(slot.evtDetectDone, streams_.cudaStreamGPU);
                }
                next_detect_frame++;
            }
        }

        drain_async_outputs();

        // ====================================================================
        // Phase C: Stage2 — ROI 匹配 + 融合帧 N-1 (与 grab 并行, ~0.1ms)
        //   检测帧: collect YOLO → ROI match → depth fuse + 刷新 tracker template
        //   填充帧: tracker bbox → ROI match → depth fuse
        // ====================================================================
        if (next_fuse_frame < next_detect_frame - 1) {
            int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
            auto& slot = slots_[slot_idx];

            if (slot.grab_failed) {
                next_fuse_frame++;
                continue;
            }

            if (config_.drop_stale_roi_frames &&
                !config_.detection_only &&
                slot.is_detect_frame &&
                next_fuse_frame + 1 < next_detect_frame &&
                newer_yolo_ready(next_fuse_frame + 1)) {
                ++stale_drop_count;
                globalPerf().record("Stage2_DropStaleROI", 0.0);
                if (config_.stats_interval > 0 &&
                    stale_drop_count % config_.stats_interval == 0) {
                    LOG_WARN("[ROI] Dropped %d stale frame(s); latest ready frame=%d",
                             stale_drop_count, next_fuse_frame + 1);
                }
                expireAsyncRoiBefore(next_fuse_frame + 1);
                next_fuse_frame++;
                continue;
            }

            {
                ScopedTimer tw("Stage2_WaitDetect");
                waitDetectDone(streams_.cudaStreamFuse, slot);
                cudaStreamSynchronize(streams_.cudaStreamFuse);
                globalPerf().record("Stage2_WaitDetect", tw.elapsedMs());
            }

            if (slot.is_detect_frame) {
                // ---- YOLO 检测帧: 完全等待DLA + collect + ROI + depth ----
                // *** GPU Power Control: 显式同步确保YOLO完全finish ***
                {
                    ScopedTimer tw("Stage2_WaitYOLOComplete");
                    auto dlaStream = getDLAStream(slot.frame_id);
                    cudaStreamSynchronize(dlaStream);
                    if (dualYoloEnabled() && slot.right_detection_submitted) {
                        cudaStreamSynchronize(getRightDLAStream(slot.frame_id));
                    }
                    const double wait_yolo_ms = tw.elapsedMs();
                    globalPerf().record("Stage2_WaitYOLOComplete", wait_yolo_ms);
                    if (wait_yolo_ms > kStage2WaitYoloOutlierMs) {
                        LOG_WARN("[PerfOutlier] Stage2_WaitYOLOComplete frame=%d slot=%d ms=%.2f right_submitted=%d",
                                 slot.frame_id, slot_idx, wait_yolo_ms,
                                 slot.right_detection_submitted ? 1 : 0);
                    }
                }

                bool publish_now = true;
                if (async_roi_ready_) {
                    // 当前帧 YOLO 已 ready；上一检测帧的异步 ROI 截止线到达。
                    drain_async_outputs();
                    expireAsyncRoiBefore(slot.frame_id);

                    ScopedTimer t2("Stage2_ROIMatchFuseSubmit");
                    const bool submitted = submitAsyncRoiStage2(slot, slot_idx);
                    globalPerf().record("Stage2_ROIMatchFuseSubmit", t2.elapsedMs());
                    slot.bbox_source = BboxSource::YOLO;
                    tracker_handle_detect_result(slot);
                    publish_now = false;
                    if (!submitted) {
                        if (!slot.detections.empty() ||
                            !slot.detections_right.empty()) {
                            ++stale_drop_count;
                            globalPerf().record("Stage2_AsyncRoiSubmitDrop", 0.0);
                        }
                        RoiStage2Output dropped;
                        dropped.detections = slot.detections;
                        dropped.predict_only = true;
                        applyRoiStage2Output(slot, std::move(dropped));
                        publish_now = true;
                    }
                } else {
                    {
                        ScopedTimer t2("Stage2_ROIMatchFuse");
                        stage2_roi_match_fuse(slot, slot_idx);
                        globalPerf().record("Stage2_ROIMatchFuse", t2.elapsedMs());
                    }
                    slot.bbox_source = BboxSource::YOLO;
                    // 用 YOLO 结果刷新 tracker template
                    tracker_handle_detect_result(slot);
                }

                if (publish_now) {
                    publishRoiFrameCallbacks(slot);
                    account_output_frame(slot.frame_id);
                }
            } else {
                // ---- Tracker 填充帧: tracker bbox → ROI + depth ----
                {
                    ScopedTimer t2("Stage2_ROIFuseTracker");
                    stage2_roi_fuse_tracker(slot, slot_idx);
                    globalPerf().record("Stage2_ROIFuseTracker", t2.elapsedMs());
                }
                publishRoiFrameCallbacks(slot);
                account_output_frame(slot.frame_id);
            }

            next_fuse_frame++;
        }

        // ====================================================================
        // Phase D: 等待 grab 完成 + 执行 rectify (bayer + remap)
        //   grab 线程已在 Phase A-C 期间运行 ~3ms
        //   剩余等待时间: max(0, grab_time - 3ms) ≈ 2ms
        //   然后处理 bayer→BGR + remap submit (~2ms)
        // ====================================================================
        {
            auto& slot = slots_[grab_slot_idx];
            bool grab_preloaded = false;
#ifdef HIK_CAMERA_ENABLED
            if (camera_) {
                ScopedTimer tw("Stage0_WaitGrab");
                bool ok = waitGrab();
                const double wait_grab_ms = tw.elapsedMs();
                globalPerf().record("Stage0_WaitGrab", wait_grab_ms);
                if (wait_grab_ms > kStage0WaitGrabOutlierMs) {
                    LOG_WARN("[PerfOutlier] Stage0_WaitGrab frame=%d slot=%d ms=%.2f ok=%d",
                             slot.frame_id, grab_slot_idx, wait_grab_ms, ok ? 1 : 0);
                }
                if (!ok) slot.grab_failed = true;
                grab_preloaded = true;
            }
#endif
            {
                ScopedTimer tp("Stage0_Process");
                stage0_grab_and_rectify(slot, grab_preloaded);
                globalPerf().record("Stage0_Process", tp.elapsedMs());
            }
            next_grab_frame++;
        }

    }

    // ===== 排空 =====
    vpiStreamSync(streams_.vpiStreamPVA);
    drain_async_outputs();

    while (next_detect_frame < next_grab_frame) {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        if (slot.grab_failed) {
            next_detect_frame++;
            continue;
        }
        if (!sync_rect_for_detect(slot, slot_idx)) {
            next_detect_frame++;
            continue;
        }
        auto dlaStream = getDLAStream(slot.frame_id);
        cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);
        stage1_detect(slot, slot_idx);
        recordDetectDoneEvents(slot);
        next_detect_frame++;
    }

    while (next_fuse_frame < next_detect_frame) {
        int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        if (slot.grab_failed) {
            next_fuse_frame++;
            continue;
        }
        waitDetectDone(streams_.cudaStreamFuse, slot);
        cudaStreamSynchronize(streams_.cudaStreamFuse);
        stage2_roi_match_fuse(slot, slot_idx);
        publishRoiFrameCallbacks(slot);
        account_output_frame(slot.frame_id);
        next_fuse_frame++;
    }

    LOG_INFO("ROI Pipeline loop exited");
}

void Pipeline::printPerfReport() const {
    globalPerf().printReport();
}

}  // namespace stereo3d
