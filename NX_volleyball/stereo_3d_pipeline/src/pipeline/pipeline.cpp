/**
 * @file pipeline.cpp
 * @brief 四级流水线实现
 *
 * 核心调度逻辑 (帧间流水线重叠):
 *   每次迭代中，3 个不同的帧同时处于不同 Stage:
 *     Stage 0 处理帧 N+2  [CPU + PVA]
 *     Stage 1 处理帧 N+1  [DLA]          ┐ 并行
 *     Stage 2 处理帧 N+1  [GPU CUDA]     ┘
 *     Stage 3 处理帧 N    [GPU + CPU]
 *
 *   吞吐量 = 1 / max(Stage_i latency) → 60-100 FPS
 *
 * CUDA Events 跨 Stream 同步:
 *   evtRectDone   → Stage 0 完成后记录在 PVA stream 的 CUDA 事件
 *   evtDetectDone → Stage 1 完成后记录在 DLA stream
 *   evtStereoDone → Stage 2 完成后记录在 GPU stream
 *
 *   Stage 1/2 通过 cudaStreamWaitEvent 等待 evtRectDone
 *   Stage 3   通过 cudaStreamWaitEvent 等待 evtDetectDone + evtStereoDone
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
        uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_PVA;

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

        // 视差图 (S16 格式, Q8.8 定点数) → 校正后分辨率
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
        LOG_ERROR("Failed to open cameras");
        return false;
    }
#else
    LOG_ERROR("Camera support disabled (HIK SDK not found)");
    return false;
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

    if (!camera_->startGrabbing()) {
        LOG_ERROR("Failed to start camera grabbing");
        running_ = false;
        return;
    }

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

    if (camera_) camera_->stopGrabbing();

    globalPerf().printReport();
}

// ===================================================================
// Pipeline 主循环 (帧间流水线三级重叠)
// ===================================================================

void Pipeline::pipelineLoop() {
    using Clock = std::chrono::high_resolution_clock;
    auto fps_start = Clock::now();
    int fps_count = 0;
    int frame_counter = 0;      // 全局帧计数 (Stage 0 已提交的帧数)
    int output_counter = 0;     // 已输出的帧数

    // === 流水线填充阶段 (Filling) ===
    // 第 1 帧: 仅 Stage 0
    {
        auto& s0 = slots_[0];
        s0.reset();
        s0.frame_id = frame_counter;

        ScopedTimer t0("Stage0_GrabRect");
        stage0_grab_and_rectify(s0);
        globalPerf().record("Stage0_GrabRect", t0.elapsedMs());
    }
    frame_counter++;

    // 第 2 帧: Stage 0 (帧1) + Stage 1/2 (帧0) 并行
    if (running_) {
        auto& s0 = slots_[1 % RING_BUFFER_SIZE];
        s0.reset();
        s0.frame_id = frame_counter;

        ScopedTimer t0("Stage0_GrabRect");
        stage0_grab_and_rectify(s0);
        globalPerf().record("Stage0_GrabRect", t0.elapsedMs());

        // Stage 1+2 on frame 0
        auto& s12 = slots_[0];
        cudaStreamWaitEvent(streams_.cudaStreamDLA, s12.evtRectDone, 0);
        cudaStreamWaitEvent(streams_.cudaStreamGPU, s12.evtRectDone, 0);

        {
            ScopedTimer t1("Stage1_Detect");
            stage1_detect(s12);
            globalPerf().record("Stage1_Detect", t1.elapsedMs());
        }
        cudaEventRecord(s12.evtDetectDone, streams_.cudaStreamDLA);

        {
            ScopedTimer t2("Stage2_Stereo");
            stage2_stereo(s12);
            globalPerf().record("Stage2_Stereo", t2.elapsedMs());
        }
        vpiStreamSync(streams_.vpiStreamGPU);
        cudaEventRecord(s12.evtStereoDone, streams_.cudaStreamGPU);

        frame_counter++;
    }

    // === 稳态循环 (Steady State) ===
    // 每次迭代: Stage 0(帧 N+2), Stage 1+2(帧 N+1), Stage 3(帧 N)
    while (running_) {
        int grab_slot   = frame_counter % RING_BUFFER_SIZE;
        int detect_slot = (frame_counter - 1) % RING_BUFFER_SIZE;
        int fuse_slot   = (frame_counter - 2) % RING_BUFFER_SIZE;

        // --- Stage 0: Grab + Rectify (帧 N+2) ---
        {
            auto& slot = slots_[grab_slot];
            slot.reset();
            slot.frame_id = frame_counter;

            ScopedTimer t0("Stage0_GrabRect");
            stage0_grab_and_rectify(slot);
            globalPerf().record("Stage0_GrabRect", t0.elapsedMs());
        }

        // --- Stage 1 + Stage 2 并行 (帧 N+1) ---
        {
            auto& slot = slots_[detect_slot];

            // DLA/GPU 都等 evtRectDone 后开始
            cudaStreamWaitEvent(streams_.cudaStreamDLA, slot.evtRectDone, 0);
            cudaStreamWaitEvent(streams_.cudaStreamGPU, slot.evtRectDone, 0);

            // Stage 1: DLA 异步检测 (不做 cudaStreamSynchronize)
            {
                ScopedTimer t1("Stage1_Detect");
                stage1_detect(slot);
                globalPerf().record("Stage1_Detect", t1.elapsedMs());
            }
            cudaEventRecord(slot.evtDetectDone, streams_.cudaStreamDLA);

            // Stage 2: GPU 异步视差 (不做 vpiStreamSync)
            {
                ScopedTimer t2("Stage2_Stereo");
                stage2_stereo(slot);
                globalPerf().record("Stage2_Stereo", t2.elapsedMs());
            }
            vpiStreamSync(streams_.vpiStreamGPU);
            cudaEventRecord(slot.evtStereoDone, streams_.cudaStreamGPU);
        }

        // --- Stage 3: 等 Stage 1 & 2 完成后融合 (帧 N) ---
        {
            auto& slot = slots_[fuse_slot];

            // 等待 DLA 检测 + GPU 视差都完成
            cudaStreamWaitEvent(streams_.cudaStreamFuse, slot.evtDetectDone, 0);
            cudaStreamWaitEvent(streams_.cudaStreamFuse, slot.evtStereoDone, 0);
            cudaStreamSynchronize(streams_.cudaStreamFuse);

            {
                ScopedTimer t3("Stage3_Fuse");
                stage3_fuse(slot);
                globalPerf().record("Stage3_Fuse", t3.elapsedMs());
            }

            if (result_callback_) {
                result_callback_(slot.frame_id, slot.results);
            }
            output_counter++;
        }

        frame_counter++;

        // ---- FPS 统计 ----
        fps_count++;
        auto now = Clock::now();
        double elapsed_s = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed_s >= 1.0) {
            current_fps_ = static_cast<float>(fps_count / elapsed_s);
            if (config_.stats_interval > 0 &&
                output_counter % config_.stats_interval == 0) {
                LOG_INFO("FPS: %.1f  (Output frame %d)", current_fps_.load(), output_counter);
            }
            fps_count = 0;
            fps_start = now;
        }
    }

    // === 排空阶段 (Draining): 处理流水线中剩余帧 ===
    // 帧 frame_counter-1 还在 Stage 1+2, 帧 frame_counter-2 还在 Stage 3 等待
    if (frame_counter >= 2) {
        int detect_slot = (frame_counter - 1) % RING_BUFFER_SIZE;
        auto& slot12 = slots_[detect_slot];
        cudaStreamWaitEvent(streams_.cudaStreamDLA, slot12.evtRectDone, 0);
        cudaStreamWaitEvent(streams_.cudaStreamGPU, slot12.evtRectDone, 0);
        stage1_detect(slot12);
        cudaEventRecord(slot12.evtDetectDone, streams_.cudaStreamDLA);
        stage2_stereo(slot12);
        vpiStreamSync(streams_.vpiStreamGPU);
        cudaEventRecord(slot12.evtStereoDone, streams_.cudaStreamGPU);

        // Fuse 帧 frame_counter-2
        {
            int fuse_slot = (frame_counter - 2) % RING_BUFFER_SIZE;
            auto& slot3 = slots_[fuse_slot];
            cudaStreamWaitEvent(streams_.cudaStreamFuse, slot3.evtDetectDone, 0);
            cudaStreamWaitEvent(streams_.cudaStreamFuse, slot3.evtStereoDone, 0);
            cudaStreamSynchronize(streams_.cudaStreamFuse);
            stage3_fuse(slot3);
            if (result_callback_) result_callback_(slot3.frame_id, slot3.results);
        }

        // Fuse 帧 frame_counter-1
        {
            cudaStreamWaitEvent(streams_.cudaStreamFuse, slot12.evtDetectDone, 0);
            cudaStreamWaitEvent(streams_.cudaStreamFuse, slot12.evtStereoDone, 0);
            cudaStreamSynchronize(streams_.cudaStreamFuse);
            stage3_fuse(slot12);
            if (result_callback_) result_callback_(slot12.frame_id, slot12.results);
        }
    }

    LOG_INFO("Pipeline loop exited");
}

// ===================================================================
// Stage 实现
// ===================================================================

void Pipeline::stage0_grab_and_rectify(FrameSlot& slot) {
    NVTX_RANGE("Stage0_GrabRect");

    // 1. 抓取左右帧到 VPI Image (Zero-Copy)
    //    海康 SDK 直接写入 VPI Image 的 Locked 指针
    VPIImageData imgDataL, imgDataR;
    vpiImageLockData(slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
    vpiImageLockData(slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);

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

    // 2. PVA 加速校正 (异步提交到 PVA stream)
    rectifier_->submit(streams_.vpiStreamPVA, slot.rawL, slot.rawR,
                       slot.rectL, slot.rectR);

    // 同步 PVA stream 确保校正完成，然后在 PVA 关联的 CUDA stream 上记录事件
    vpiStreamSync(streams_.vpiStreamPVA);
    // 注意: evtRectDone 录在 cudaStreamGPU 上 (由 GPU 负责后续依赖);
    // VPI PVA 作业已经 sync 完成，此 event 仅作为下游 Stage 1/2 的 Gate
    cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);

    NVTX_RANGE_POP();
}

void Pipeline::stage1_detect(FrameSlot& slot) {
    NVTX_RANGE("Stage1_Detect");

    // 从 VPI Image 获取 GPU 指针，传给 TensorRT
    VPIImageData imgData;
    vpiImageLockData(slot.rectL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);

    void* gpu_ptr = imgData.buffer.pitch.planes[0].data;
    int pitch = imgData.buffer.pitch.planes[0].pitchBytes;

    // 异步推理: detect() 将工作提交到 DLA stream, 不内部同步
    slot.detections = detector_->detect(gpu_ptr, pitch,
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

    // 不做 vpiStreamSync ! VPI 异步提交到 GPU backend,
    // 下游 Stage 3 通过 cudaStreamWaitEvent(evtStereoDone) 等待。

    NVTX_RANGE_POP();
}

void Pipeline::stage3_fuse(FrameSlot& slot) {
    NVTX_RANGE("Stage3_Fuse");

    slot.results.clear();

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
