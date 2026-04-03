/**
 * @file pipeline.cpp
 * @brief 四级流水线实现
 *
 * 核心调度逻辑 (帧间流水线重叠):
 *   每次迭代中，3 个不同的帧同时处于不同 Stage:
 *     Stage 3 处理帧 N    [GPU + CPU]      (等待上一帧 detect/stereo)
 *     Stage 1 处理帧 N+1  [DLA/GPU]        ┐ 异步提交
 *     Stage 2 处理帧 N+1  [GPU CUDA/VPI]   ┘
 *     Stage 0 处理帧 N+2  [CPU + VPI Remap]
 *
 *   吞吐量 = 1 / max(Stage_i latency) → 60-100 FPS
 *
 * CUDA/VPI 同步:
 *   evtRectDone   → Stage 0 完成后记录, Stage 1/2 作为 gate
 *   evtDetectDone → Stage 1 D2H 完成后记录, Stage 3 等待
 *   Stage 2       → VPI stream 异步提交, Stage 3 使用 vpiStreamSync 等待
 */

#include "pipeline.h"
#include "../utils/logger.h"
#include <chrono>

namespace stereo3d {

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
    for (auto& slot : slots_) {
        slot.destroy();
    }
    streams_.destroy();
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

    LOG_INFO("========================================");
    LOG_INFO("Stereo 3D Pipeline (4-Stage)");
    LOG_INFO("========================================");

    // 1. 初始化 CUDA/VPI Streams
    LOG_INFO("Initializing streams...");
    if (!streams_.init()) {
        LOG_ERROR("Failed to initialize streams");
        return false;
    }

    // 2. 初始化三缓冲 FrameSlots
    LOG_INFO("Creating FrameSlots (triple buffer)...");
    for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
        if (!slots_[i].createEvents()) {
            LOG_ERROR("Failed to create events for slot %d", i);
            return false;
        }
    }

    // 3. 初始化标定
    LOG_INFO("Loading stereo calibration: %s", config_.calibration_file.c_str());
    calibration_ = std::make_unique<StereoCalibration>();
    if (!calibration_->load(config_.calibration_file)) {
        LOG_ERROR("Failed to load calibration");
        return false;
    }

    // 4. 初始化 VPI Rectifier (PVA backend)
    //    Rectifier 按校正后分辨率初始化
    LOG_INFO("Initializing VPI Rectifier (PVA) at %dx%d...",
             config_.rect_width, config_.rect_height);
    rectifier_ = std::make_unique<VPIRectifier>();
    if (!rectifier_->init(*calibration_, config_.rect_width, config_.rect_height)) {
        LOG_ERROR("Failed to initialize VPI Rectifier");
        return false;
    }

    // 5. 分配 VPI Images (使用 VPI zero-copy buffers)
    LOG_INFO("Allocating VPI images for %d slots...", RING_BUFFER_SIZE);
    for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
        VPIStatus err;
        // CPU flag needed for host-side camera memcpy into rawL/rawR
        uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_PVA | VPI_BACKEND_CPU;

        // 原始图像 → 使用 raw_width x raw_height (相机原始分辨率)
        err = vpiImageCreate(config_.raw_width, config_.raw_height,
                             VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rawL);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rawL create failed"); return false; }

        err = vpiImageCreate(config_.raw_width, config_.raw_height,
                             VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rawR);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rawR create failed"); return false; }

        // 校正后图像 → 使用 rect_width x rect_height
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rectL);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rectL create failed"); return false; }

        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rectR);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rectR create failed"); return false; }

        // 视差图 (S16 格式, Q10.5 定点数) → 校正后分辨率
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_S16, VPI_BACKEND_CUDA, &slots_[i].disparityMap);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI disparity create failed"); return false; }

        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_U16, VPI_BACKEND_CUDA, &slots_[i].confidenceMap);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI confidence create failed"); return false; }
    }

    // 6. 初始化海康双目相机 (单实例管理左右)
#ifdef HIK_CAMERA_ENABLED
    LOG_INFO("Opening dual cameras...");
    camera_ = std::make_unique<HikvisionCamera>();
    CameraConfig cam_cfg;
    cam_cfg.camera_index_left  = config_.cam_left_index;
    cam_cfg.camera_index_right = config_.cam_right_index;
    cam_cfg.serial_left  = config_.cam_left_serial;
    cam_cfg.serial_right = config_.cam_right_serial;
    cam_cfg.exposure_us  = config_.exposure_us;
    cam_cfg.gain_db      = config_.gain_db;
    cam_cfg.use_trigger  = config_.use_trigger;
    cam_cfg.trigger_source     = config_.trigger_source;
    cam_cfg.trigger_activation = config_.trigger_activation;
    cam_cfg.width  = config_.raw_width;
    cam_cfg.height = config_.raw_height;
    if (!camera_->open(cam_cfg)) {
        LOG_WARN("Failed to open cameras - running in dry-run mode (synthetic frames)");
        camera_.reset();  // 释放相机, 标记为 dry-run
    }

    // 初始化 PWM 触发器 (硬件触发模式时)
    if (camera_ && config_.use_trigger) {
        pwm_trigger_ = std::make_unique<PWMTrigger>(
            config_.trigger_chip, config_.trigger_line, config_.trigger_freq_hz);
        LOG_INFO("PWM trigger configured: chip=%s line=%d freq=%dHz",
                 config_.trigger_chip.c_str(), config_.trigger_line, config_.trigger_freq_hz);
    }
#else
    LOG_WARN("Camera support disabled (HIK SDK not found) - pipeline runs without camera");
#endif

    // 7. 初始化 TensorRT 检测器 (DLA/GPU)
    LOG_INFO("Initializing TRT Detector (DLA=%d, core=%d)...",
             config_.use_dla, config_.dla_core);
    detector_ = std::make_unique<TRTDetector>();
    if (!detector_->init(config_.engine_file, config_.use_dla, config_.dla_core,
                         config_.conf_threshold, config_.nms_threshold)) {
        LOG_ERROR("Failed to initialize TRT Detector");
        return false;
    }

    // 8. 初始化 VPI 视差计算器 (校正后分辨率)
    LOG_INFO("Initializing VPI Stereo (maxDisp=%d, winSize=%d, %dx%d)...",
             config_.max_disparity, config_.window_size,
             config_.rect_width, config_.rect_height);
    stereo_ = std::make_unique<VPIStereo>();
    if (!stereo_->init(config_.max_disparity, config_.window_size,
                       config_.rect_width, config_.rect_height)) {
        LOG_ERROR("Failed to initialize VPI Stereo");
        return false;
    }

    // 9. 初始化 3D 融合
    fusion_ = std::make_unique<Coordinate3D>();
    fusion_->init(calibration_->getProjectionLeft(),
                  calibration_->getBaseline(),
                  config_.min_depth, config_.max_depth);

    LOG_INFO("========================================");
    LOG_INFO("Pipeline initialized successfully");
    LOG_INFO("  Raw resolution:  %dx%d", config_.raw_width, config_.raw_height);
    LOG_INFO("  Rect resolution: %dx%d", config_.rect_width, config_.rect_height);
    LOG_INFO("  Trigger: %d Hz", config_.trigger_freq_hz);
    LOG_INFO("  Detect: %s (DLA=%d)", config_.engine_file.c_str(), config_.use_dla);
    LOG_INFO("  Disparity: %s", config_.disparity_strategy == DisparityStrategy::FULL_FRAME
             ? "Full Frame" : (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION
             ? "Half Resolution" : "ROI Only"));
    LOG_INFO("========================================");

    return true;
}

void Pipeline::start() {
    if (running_.exchange(true)) return;

#ifdef HIK_CAMERA_ENABLED
    // 先启动相机采集, 再启动 PWM 触发
    if (camera_ && !camera_->startGrabbing()) {
        LOG_ERROR("Failed to start camera grabbing");
        running_ = false;
        return;
    }
    if (pwm_trigger_ && !pwm_trigger_->start()) {
        LOG_ERROR("Failed to start PWM trigger - camera may not receive triggers");
        // 非致命: 外部 PWM 可能已运行
    }
#endif

    // 在独立线程中运行 Pipeline 循环
    pipeline_thread_ = std::thread(&Pipeline::pipelineLoop, this);
    LOG_INFO("Pipeline thread started");
}

void Pipeline::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;

    // 等待工作线程退出
    if (pipeline_thread_.joinable()) {
        pipeline_thread_.join();
    }

    streams_.syncAll();

#ifdef HIK_CAMERA_ENABLED
    if (pwm_trigger_) pwm_trigger_->stop();
    if (camera_) camera_->stopGrabbing();
#endif

    globalPerf().printReport();
}

// ===================================================================
// Pipeline 主循环 (帧间流水线三级重叠)
//
// 调度策略:
//   1) Stage 3 处理上一帧 (等待上一帧 detect/stereo 完成)
//   2) Stage 1+2 异步提交当前帧 (不阻塞)
//   3) Stage 0 抓取下一帧 (与 1/2 重叠)
//
// 通过将 Stage 3 前置，避免 vpiStreamSync 等待“当前帧”Stereo，
// 实现真实帧间重叠。
// ===================================================================

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
                cudaStreamWaitEvent(streams_.cudaStreamFuse, slot.evtDetectDone, 0);
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

            // DLA/GPU 都等 rect 完成后开始
            cudaStreamWaitEvent(streams_.cudaStreamDLA, slot.evtRectDone, 0);
            cudaStreamWaitEvent(streams_.cudaStreamGPU, slot.evtRectDone, 0);

            {
                ScopedTimer t1("Stage1_DetectSubmit");
                stage1_detect(slot, slot_idx);
                globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
            }
            cudaEventRecord(slot.evtDetectDone, streams_.cudaStreamDLA);

            {
                ScopedTimer t2("Stage2_StereoSubmit");
                stage2_stereo(slot);
                globalPerf().record("Stage2_StereoSubmit", t2.elapsedMs());
            }

            next_detect_frame++;
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
    // 1) 提交所有已抓取但尚未提交 detect/stereo 的帧
    while (next_detect_frame < next_grab_frame) {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];

        cudaStreamWaitEvent(streams_.cudaStreamDLA, slot.evtRectDone, 0);
        cudaStreamWaitEvent(streams_.cudaStreamGPU, slot.evtRectDone, 0);

        stage1_detect(slot, slot_idx);
        cudaEventRecord(slot.evtDetectDone, streams_.cudaStreamDLA);
        stage2_stereo(slot);

        next_detect_frame++;
    }

    // 2) 融合所有已提交 detect/stereo 但尚未输出的帧
    while (next_fuse_frame < next_detect_frame) {
        int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];

        cudaStreamWaitEvent(streams_.cudaStreamFuse, slot.evtDetectDone, 0);
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

void Pipeline::stage0_grab_and_rectify(FrameSlot& slot) {
    NVTX_RANGE("Stage0_GrabRect");

#ifdef HIK_CAMERA_ENABLED
    if (camera_) {
        // 1. 抓取左右帧到 VPI Image
        //    Lock with HOST buffer → memcpy safe, VPI handles host→device transfer on unlock
        VPIImageData imgDataL, imgDataR;
        vpiImageLockData(slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataL);
        vpiImageLockData(slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataR);

        GrabResult resL, resR;
        camera_->grabFramePair(
            static_cast<uint8_t*>(imgDataL.buffer.pitch.planes[0].data),
            static_cast<uint8_t*>(imgDataR.buffer.pitch.planes[0].data),
            imgDataL.buffer.pitch.planes[0].pitchBytes,
            imgDataR.buffer.pitch.planes[0].pitchBytes,
            1000,  // timeout_ms
            resL, resR);

        vpiImageUnlock(slot.rawL);
        vpiImageUnlock(slot.rawR);
    }
    // else: dry-run mode — rawL/rawR contain zeroes (VPI images are zero-initialized)
#endif

    // 2. VPI Remap 校正 (异步提交)
    rectifier_->submit(streams_.vpiStreamPVA, slot.rawL, slot.rawR,
                       slot.rectL, slot.rectR);

    // 同步 rect stream 确保校正完成，然后记录 rect-done 事件
    vpiStreamSync(streams_.vpiStreamPVA);
    // 注意: evtRectDone 录在 cudaStreamGPU 上 (由 GPU 负责后续依赖);
    // VPI PVA 作业已经 sync 完成，此 event 仅作为下游 Stage 1/2 的 Gate
    cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);

    NVTX_RANGE_POP();
}

void Pipeline::stage1_detect(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage1_Detect");

    // 从 VPI Image 获取 GPU 指针，传给 TensorRT
    VPIImageData imgData;
    vpiImageLockData(slot.rectL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);

    void* gpu_ptr = imgData.buffer.pitch.planes[0].data;
    int pitch = imgData.buffer.pitch.planes[0].pitchBytes;

    // 异步推理提交: 仅 enqueue，不在此处同步
    detector_->enqueue(slot_index, gpu_ptr, pitch,
                       config_.rect_width, config_.rect_height,
                       streams_.cudaStreamDLA);

    vpiImageUnlock(slot.rectL);
    NVTX_RANGE_POP();
}

void Pipeline::stage2_stereo(FrameSlot& slot) {
    NVTX_RANGE("Stage2_Stereo");

    switch (config_.disparity_strategy) {
        case DisparityStrategy::FULL_FRAME:
            stereo_->compute(streams_.vpiStreamGPU,
                             slot.rectL, slot.rectR,
                             slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::HALF_RESOLUTION:
            stereo_->computeHalfRes(streams_.vpiStreamGPU,
                                    slot.rectL, slot.rectR,
                                    slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::ROI_ONLY:
            // ROI 模式需要先有检测结果 → 依赖 Stage 1
            // 此处使用全帧模式代替; ROI 路径在 Stage 3 中按需提取
            stereo_->compute(streams_.vpiStreamGPU,
                             slot.rectL, slot.rectR,
                             slot.disparityMap, slot.confidenceMap);
            break;
    }

    // 不在此处做 vpiStreamSync。
    // 下游 Stage3 统一在融合前对 vpiStreamGPU 做同步，避免串行化 Stage1/2 提交路径。

    NVTX_RANGE_POP();
}

void Pipeline::stage3_fuse(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage3_Fuse");

    slot.results.clear();

    // Detect 结果在 Stage1 中异步 D2H，现已通过 evtDetectDone 保证完成
    slot.detections = detector_->collect(slot_index,
                                         config_.rect_width, config_.rect_height);

    // 获取视差图 GPU 指针
    VPIImageData dispData;
    vpiImageLockData(slot.disparityMap, VPI_LOCK_READ,
                     VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dispData);

    const int16_t* disp_ptr = static_cast<const int16_t*>(dispData.buffer.pitch.planes[0].data);
    int disp_pitch = dispData.buffer.pitch.planes[0].pitchBytes;

    // 批量计算 3D 坐标
    // 半分辨率视差需要 ×2 补偿
    float dispScale = (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION)
                      ? 2.0f : 1.0f;
    slot.results = fusion_->computeBatch(slot.detections, disp_ptr, disp_pitch,
                                         config_.rect_width, config_.rect_height,
                                         streams_.cudaStreamFuse, dispScale);

    vpiImageUnlock(slot.disparityMap);
    NVTX_RANGE_POP();
}

void Pipeline::printPerfReport() const {
    globalPerf().printReport();
}

}  // namespace stereo3d
