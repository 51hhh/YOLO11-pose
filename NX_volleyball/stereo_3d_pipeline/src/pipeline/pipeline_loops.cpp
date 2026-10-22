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
                synchronizeDetectDone(slot, "synchronize Stage3 detect dependencies");
                globalPerf().record("Stage3_WaitDetect", tw.elapsedMs());
            }
            // BGR 检测可与灰度全帧视差并行；到 Stage3 时检测已完成，
            // 先释放外部 CUDA view，回调才能重新 lock 同一 VPI 图像。
            unlockRectifiedCudaViews(slot);

            // 等 Stereo 完成（VPI stream 级同步，放在 Stage3 前，避免阻塞 Stage1/2 提交）
            if (!slot.grab_failed) {
                ScopedTimer tws("Stage3_WaitStereo");
                const VPIStatus status = vpiStreamSync(streams_.vpiStreamGPU);
                if (status != VPI_SUCCESS) {
                    LOG_ERROR("[Pipeline] Stage3 stereo sync failed: frame=%d status=%d",
                              slot.frame_id, (int)status);
                    slot.grab_failed = true;
                }
                globalPerf().record("Stage3_WaitStereo", tws.elapsedMs());
            }

            if (!slot.grab_failed) {
                ScopedTimer t3("Stage3_Fuse");
                stage3_fuse(slot, slot_idx);
                globalPerf().record("Stage3_Fuse", t3.elapsedMs());
            }

            if (result_callback_) {
                result_callback_(slot.frame_id,
                                 slot.results,
                                 makeFrameMetadata(slot));
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
                    slot.detector_candidates_left,
                    slot.detector_candidates_right,
                    slot.stereo_pair_attempts,
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
                // 失败帧也必须进入帧回调，使 fail_trial 录制器能立即终止，
                // 而不是等待一帧永远不会到达的后继帧来发现序号缺口。
                publishRoiFrameCallbacks(slot);
                next_detect_frame++;
                next_fuse_frame = next_detect_frame;
            } else {
                // 等待 VPI remap 完成 (stage0 异步提交)
                const VPIStatus rect_status = vpiStreamSync(streams_.vpiStreamPVA);
                if (rect_status != VPI_SUCCESS) {
                    LOG_ERROR("[Pipeline] rectification sync failed before Stage1: "
                              "frame=%d status=%d",
                              slot.frame_id, (int)rect_status);
                    slot.grab_failed = true;
                }
                if (!slot.grab_failed) {
                    lockRectifiedCudaViews(slot);
                }
                if (!slot.grab_failed) {
                    checkCudaFrame(
                        cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU),
                        slot, "record rectification completion event");
                }
                if (!slot.grab_failed) {
                    checkCudaFrame(
                        cudaStreamWaitEvent(streams_.cudaStreamGPU,
                                            slot.evtRectDone, 0),
                        slot, "GPU stage wait for rectification");
                }

                if (!slot.grab_failed) {
                    ScopedTimer t1("Stage1_DetectSubmit");
                    stage1_detect(slot, slot_idx);
                    globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
                }
                recordDetectDoneEvents(slot);

                // 灰度检测与 VPI stereo 读取的是同一 VPIImage。VPI 文档要求
                // image 被外部 lock 时不得由 VPI stream 访问，因此这里先等
                // detector 用完并解锁；BGR 检测仍可和灰度 stereo 并行。
                if (rectifiedGrayCudaViewsLocked(slot)) {
                    synchronizeDetectDone(
                        slot, "synchronize gray detector before VPI stereo");
                    unlockRectifiedCudaViews(slot);
                }

                if (!slot.grab_failed) {
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
    const VPIStatus drain_rect_status = vpiStreamSync(streams_.vpiStreamPVA);
    if (drain_rect_status != VPI_SUCCESS) {
        LOG_ERROR("[Pipeline] final rectification drain failed: status=%d",
                  (int)drain_rect_status);
    }

    // 1) 提交所有已抓取但尚未提交 detect/stereo 的帧
    while (next_detect_frame < next_grab_frame) {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];

        if (slot.grab_failed) {
            publishRoiFrameCallbacks(slot);
            next_detect_frame++;
            next_fuse_frame = next_detect_frame;
            continue;
        }

        checkCudaFrame(cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU),
                       slot, "record final rectification completion event");
        if (!slot.grab_failed) lockRectifiedCudaViews(slot);
        if (!slot.grab_failed) {
            checkCudaFrame(cudaStreamWaitEvent(streams_.cudaStreamGPU,
                                               slot.evtRectDone, 0),
                           slot, "final GPU stage wait for rectification");
        }
        if (!slot.grab_failed) stage1_detect(slot, slot_idx);
        recordDetectDoneEvents(slot);
        if (rectifiedGrayCudaViewsLocked(slot)) {
            synchronizeDetectDone(
                slot, "synchronize final gray detector before VPI stereo");
            unlockRectifiedCudaViews(slot);
        }
        if (!slot.grab_failed) stage2_stereo(slot);

        next_detect_frame++;
    }

    // 2) 融合所有已提交 detect/stereo 但尚未输出的帧
    while (next_fuse_frame < next_detect_frame) {
        int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];

        synchronizeDetectDone(slot, "synchronize final detect dependencies");
        unlockRectifiedCudaViews(slot);
        if (!slot.grab_failed) {
            const VPIStatus status = vpiStreamSync(streams_.vpiStreamGPU);
            if (status != VPI_SUCCESS) {
                LOG_ERROR("[Pipeline] final stereo sync failed: frame=%d status=%d",
                          slot.frame_id, (int)status);
                slot.grab_failed = true;
            }
        }

        if (!slot.grab_failed) stage3_fuse(slot, slot_idx);
        if (result_callback_) {
            result_callback_(slot.frame_id,
                             slot.results,
                             makeFrameMetadata(slot));
        }

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
        if (!lockRectifiedCudaViews(slot)) {
            return false;
        }
        return checkCudaFrame(
            cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU),
            slot, "record ROI rectification completion event");
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
        if (!waitAsyncRoiSlotSnapshotDone(
                slot_idx, "roi_initial_grab_reuse") ||
            !unlockRectifiedCudaViews(slot)) {
            LOG_ERROR("[ROI] Failed to prepare initial slot %d for reuse",
                      slot_idx);
            return;
        }
        slot.reset();
        slot.frame_id = next_grab_frame;

        bool grab_preloaded = false;
#ifdef HIK_CAMERA_ENABLED
        if (camera_) {
            const bool requested = requestGrab(slot_idx);
            if (!requested) {
                return;
            }
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
            publishRoiFrameCallbacks(slot);
        } else if (sync_rect_for_detect(slot, slot_idx)) {
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
#ifdef HIK_CAMERA_ENABLED
        bool grab_requested = false;
#endif
        {
            auto& slot = slots_[grab_slot_idx];
            if (!waitAsyncRoiSlotSnapshotDone(
                    grab_slot_idx, "roi_grab_reuse") ||
                !unlockRectifiedCudaViews(slot)) {
                LOG_ERROR("[ROI] Failed to prepare slot %d for reuse",
                          grab_slot_idx);
                break;
            }
            slot.reset();
            slot.frame_id = next_grab_frame;
#ifdef HIK_CAMERA_ENABLED
            if (camera_) {
                grab_requested = requestGrab(grab_slot_idx);
                if (!grab_requested) break;
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
                publishRoiFrameCallbacks(slot);
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
                    if (sync_rect_for_detect(slot, slot_idx)) {
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
                } else {
                    // ---- Tracker 填充帧 ----
                    // 等 rectify 完成 (tracker 需要 rectified BGR)
                    const VPIStatus tracker_rect_status =
                        vpiStreamSync(streams_.vpiStreamPVA);
                    if (tracker_rect_status != VPI_SUCCESS) {
                        LOG_ERROR("[Pipeline] tracker rectification sync failed: "
                                  "frame=%d status=%d",
                                  slot.frame_id, (int)tracker_rect_status);
                        slot.grab_failed = true;
                    }
                    if (!slot.grab_failed) {
                        lockRectifiedCudaViews(slot);
                    }
                    if (!slot.grab_failed) {
                        checkCudaFrame(
                            cudaStreamSynchronize(streams_.cudaStreamGPU), slot,
                            "synchronize tracker input stream");
                    }

                    // *** GPU Power Control: 强制等待所有之前的YOLO任务完成 ***
                    // 防止YOLO+tracker并行运行导致功率爆口
                    // 在detect_interval=3的策略下，最近的YOLO应该已在Phase C collect了
                    // 等待检测完成
                    {
                        ScopedTimer tw("Stage1_WaitYOLOComplete");
                        if (!slot.grab_failed) {
                            checkCudaFrame(
                                cudaStreamSynchronize(streams_.cudaStreamDLA), slot,
                                "synchronize previous left YOLO before tracker");
                        }
                        if (!slot.grab_failed && dualYoloEnabled()) {
                            checkCudaFrame(
                                cudaStreamSynchronize(streams_.cudaStreamDLA_R), slot,
                                "synchronize previous right YOLO before tracker");
                        }
                        globalPerf().record("Stage1_WaitYOLOComplete", tw.elapsedMs());
                    }

                    if (!slot.grab_failed) {
                        ScopedTimer tt("Stage1_TrackerInfill");
                        tracker_infill(slot);
                        globalPerf().record("Stage1_TrackerInfill", tt.elapsedMs());
                    }
                    // 标记 "detect done" 让 Phase C 的等待逻辑兼容
                    if (!slot.grab_failed && checkCudaFrame(
                            cudaEventRecord(slot.evtDetectDone,
                                            streams_.cudaStreamGPU),
                            slot, "record tracker completion event")) {
                        slot.detect_done_event_recorded = true;
                    }
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

            // Phase A 已发起本轮异步抓帧，因此这里不能 continue 外层
            // while；局部 break 只结束本帧融合，Phase D 始终负责收割抓帧。
            do {
            if (slot.grab_failed) {
                if (slot.detection_submitted ||
                    slot.right_detection_submitted ||
                    slot.detect_done_event_recorded ||
                    slot.detect_right_done_event_recorded) {
                    synchronizeDetectDone(
                        slot, "synchronize failed ROI detect dependencies");
                }
                unlockRectifiedCudaViews(slot);
                publishRoiFrameCallbacks(slot);
                next_fuse_frame++;
                break;
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
                synchronizeDetectDone(
                    slot, "synchronize stale ROI detect dependencies");
                unlockRectifiedCudaViews(slot);
                expireAsyncRoiBefore(next_fuse_frame + 1);
                next_fuse_frame++;
                break;
            }

            {
                ScopedTimer tw("Stage2_WaitDetect");
                synchronizeDetectDone(
                    slot, "synchronize ROI Stage2 detect dependencies");
                globalPerf().record("Stage2_WaitDetect", tw.elapsedMs());
            }

            if (slot.grab_failed) {
                unlockRectifiedCudaViews(slot);
                publishRoiFrameCallbacks(slot);
                next_fuse_frame++;
                break;
            }

            if (slot.is_detect_frame) {
                // ---- YOLO 检测帧: 完全等待DLA + collect + ROI + depth ----
                // *** GPU Power Control: 显式同步确保YOLO完全finish ***
                {
                    ScopedTimer tw("Stage2_WaitYOLOComplete");
                    auto dlaStream = getDLAStream(slot.frame_id);
                    checkCudaFrame(cudaStreamSynchronize(dlaStream), slot,
                                   "synchronize left YOLO before collect");
                    if (dualYoloEnabled() && slot.right_detection_submitted) {
                        checkCudaFrame(
                            cudaStreamSynchronize(
                                getRightDLAStream(slot.frame_id)),
                            slot, "synchronize right YOLO before collect");
                    }
                    const double wait_yolo_ms = tw.elapsedMs();
                    globalPerf().record("Stage2_WaitYOLOComplete", wait_yolo_ms);
                    if (wait_yolo_ms > kStage2WaitYoloOutlierMs) {
                        LOG_WARN("[PerfOutlier] Stage2_WaitYOLOComplete frame=%d slot=%d ms=%.2f right_submitted=%d",
                                 slot.frame_id, slot_idx, wait_yolo_ms,
                                 slot.right_detection_submitted ? 1 : 0);
                    }
                }

                if (slot.grab_failed) {
                    unlockRectifiedCudaViews(slot);
                    publishRoiFrameCallbacks(slot);
                    next_fuse_frame++;
                    break;
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
                    const bool snapshot_ready =
                        waitAsyncRoiSlotSnapshotDone(
                            slot_idx, "roi_async_snapshot_release");
                    if (!snapshot_ready) {
                        slot.grab_failed = true;
                        expireAsyncRoiBefore(slot.frame_id + 1);
                    }
                    unlockRectifiedCudaViews(slot);
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
                    checkCudaFrame(
                        cudaStreamSynchronize(streams_.cudaStreamFuse), slot,
                        "synchronize ROI Stage2 before image unlock");
                    unlockRectifiedCudaViews(slot);
                }

                if (slot.grab_failed) publish_now = true;
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
                checkCudaFrame(
                    cudaStreamSynchronize(streams_.cudaStreamFuse), slot,
                    "synchronize tracker ROI Stage2 before image unlock");
                unlockRectifiedCudaViews(slot);
                publishRoiFrameCallbacks(slot);
                account_output_frame(slot.frame_id);
            }

            next_fuse_frame++;
            } while (false);
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
                const bool ok = grab_requested && waitGrab();
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
    const VPIStatus roi_drain_status = vpiStreamSync(streams_.vpiStreamPVA);
    if (roi_drain_status != VPI_SUCCESS) {
        LOG_ERROR("[Pipeline] ROI final rectification drain failed: status=%d",
                  (int)roi_drain_status);
    }
    drain_async_outputs();

    while (next_detect_frame < next_grab_frame) {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        if (slot.grab_failed) {
            publishRoiFrameCallbacks(slot);
            next_detect_frame++;
            continue;
        }
        if (!sync_rect_for_detect(slot, slot_idx)) {
            next_detect_frame++;
            continue;
        }
        stage1_detect(slot, slot_idx);
        recordDetectDoneEvents(slot);
        if (slot.grab_failed) {
            synchronizeDetectDone(
                slot, "synchronize failed final ROI detect dependencies");
            unlockRectifiedCudaViews(slot);
            publishRoiFrameCallbacks(slot);
        }
        next_detect_frame++;
    }

    while (next_fuse_frame < next_detect_frame) {
        int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        if (slot.grab_failed) {
            if (slot.detection_submitted ||
                slot.right_detection_submitted ||
                slot.detect_done_event_recorded ||
                slot.detect_right_done_event_recorded) {
                synchronizeDetectDone(
                    slot, "synchronize failed ROI drain dependencies");
            }
            unlockRectifiedCudaViews(slot);
            publishRoiFrameCallbacks(slot);
            next_fuse_frame++;
            continue;
        }
        synchronizeDetectDone(slot, "synchronize ROI drain dependencies");
        if (slot.grab_failed) {
            unlockRectifiedCudaViews(slot);
            publishRoiFrameCallbacks(slot);
            next_fuse_frame++;
            continue;
        }
        stage2_roi_match_fuse(slot, slot_idx);
        checkCudaFrame(
            cudaStreamSynchronize(streams_.cudaStreamFuse), slot,
            "synchronize final ROI Stage2 before image unlock");
        unlockRectifiedCudaViews(slot);
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
