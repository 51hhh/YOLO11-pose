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
#include "../track/nanotrack_trt.h"
#include "../track/mixformer_trt.h"
#include "../utils/logger.h"
#include <vpi/algo/ConvertImageFormat.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>

// 自定义 CUDA Bayer→BGR8 kernel (bilinear 插值, 在 detect_preprocess.cu 中)
extern "C" void launchBayerToBGR8(const unsigned char* bayer, unsigned char* bgr,
                                   int width, int height,
                                   int bayer_pitch, int bgr_pitch,
                                   cudaStream_t stream);

namespace stereo3d {

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
    for (auto& slot : slots_) {
        slot.destroy();
    }
    // TNR 资源清理
    if (tnrPayloadL_) vpiPayloadDestroy(tnrPayloadL_);
    if (tnrPayloadR_) vpiPayloadDestroy(tnrPayloadR_);
    if (tnrNV12L_) vpiImageDestroy(tnrNV12L_);
    if (tnrNV12R_) vpiImageDestroy(tnrNV12R_);
    if (tnrOutNV12L_) vpiImageDestroy(tnrOutNV12L_);
    if (tnrOutNV12R_) vpiImageDestroy(tnrOutNV12R_);
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

    // 4. 初始化 VPI Rectifier
    //    Rectifier 按校正后分辨率初始化
    uint64_t rectBackend = VPI_BACKEND_VIC;  // 默认 VIC (不占用 GPU)
    std::string rectBackendCfg = config_.rect_backend;
    std::transform(rectBackendCfg.begin(), rectBackendCfg.end(), rectBackendCfg.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (rectBackendCfg == "CUDA") rectBackend = VPI_BACKEND_CUDA;
    config_.rect_backend = rectBackendCfg;
    LOG_INFO("Initializing VPI Rectifier (%s) at %dx%d...",
             config_.rect_backend.c_str(), config_.rect_width, config_.rect_height);
    rectifier_ = std::make_unique<VPIRectifier>();
    if (!rectifier_->init(*calibration_, config_.rect_width, config_.rect_height, rectBackend)) {
        LOG_ERROR("Failed to initialize VPI Rectifier");
        return false;
    }

    // 5. 分配 VPI Images (使用 VPI zero-copy buffers)
    LOG_INFO("Allocating VPI images for %d slots...", RING_BUFFER_SIZE);
    for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
        VPIStatus err;
        // CPU flag needed for host-side camera memcpy into rawL/rawR
        // VIC flag needed for VIC-backend remap
        uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_PVA | VPI_BACKEND_VIC | VPI_BACKEND_CPU;

        // 原始图像 → 使用 camera.width x camera.height (相机原始分辨率)
        // BGR 模式: BayerRG8 格式 (用于 ConvertImageFormat debayer)
        // Gray 模式: U8 格式 (直接 remap, 与旧行为一致)
        const VPIImageFormat rawFmt = (config_.detector_input_format == "bgr")
            ? VPI_MAKE_RAW_IMAGE_FORMAT_ABBREV(BAYER_RGGB, PL, UNSIGNED, X000, 1, X8)
            : VPI_IMAGE_FORMAT_U8;
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             rawFmt, flags, &slots_[i].rawL);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rawL create failed (err=%d)", (int)err); return false; }

        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             rawFmt, flags, &slots_[i].rawR);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rawR create failed (err=%d)", (int)err); return false; }

        // --- Color pipeline images ---
        // Debayer 输出: raw res BGR
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].tempBGR_L);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI tempBGR_L create failed"); return false; }
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].tempBGR_R);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI tempBGR_R create failed"); return false; }
        // 校正后 BGR (检测用)
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].rectBGR_vpiL);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rectBGR_vpiL create failed"); return false; }
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].rectBGR_vpiR);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rectBGR_vpiR create failed"); return false; }
        // 校正后灰度 (立体匹配用)
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rectGray_vpiL);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rectGray_vpiL create failed"); return false; }
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rectGray_vpiR);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI rectGray_vpiR create failed"); return false; }

        // 视差图 (S16 格式, Q10.5 定点数) → 校正后分辨率
        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_S16, VPI_BACKEND_CUDA, &slots_[i].disparityMap);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI disparity create failed"); return false; }

        err = vpiImageCreate(config_.rect_width, config_.rect_height,
                             VPI_IMAGE_FORMAT_U16, VPI_BACKEND_CUDA, &slots_[i].confidenceMap);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI confidence create failed"); return false; }
    }

    // 5b. 缓存 Bayer→BGR 所需的 CUDA 指针 (Tegra 统一内存: 指针固定)
    //     避免每帧 8 次 VPI lock/unlock, 节省 ~2.4ms/frame
    if (config_.detector_input_format == "bgr") {
        LOG_INFO("Caching CUDA pointers for Bayer pipeline...");
        for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
            VPIImageData tmp;
            auto cachePtr = [](VPIImage img, FrameSlot::CachedGPU& out) {
                VPIImageData d;
                vpiImageLockData(img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &d);
                out.data = d.buffer.pitch.planes[0].data;
                out.pitchBytes = d.buffer.pitch.planes[0].pitchBytes;
                vpiImageUnlock(img);
            };
            cachePtr(slots_[i].rawL, slots_[i].rawL_gpu);
            cachePtr(slots_[i].rawR, slots_[i].rawR_gpu);
            cachePtr(slots_[i].tempBGR_L, slots_[i].tempBGR_L_gpu);
            cachePtr(slots_[i].tempBGR_R, slots_[i].tempBGR_R_gpu);
        }
    }

    // 6. 初始化 VPI TNR (时域降噪, 在校正前降噪)
    if (config_.tnr_enabled) {
        LOG_INFO("Initializing VPI TNR (preset=%d, strength=%.2f)...",
                 config_.tnr_preset, config_.tnr_strength);
        VPIStatus err;
        // 创建 NV12 缓冲 (用于 U8 → NV12 转换 + TNR 处理)
        // 使用 raw 分辨率, TNR 在校正前执行
        uint64_t nv12_flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             VPI_IMAGE_FORMAT_NV12_ER, nv12_flags, &tnrNV12L_);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI TNR NV12 L create failed"); return false; }
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             VPI_IMAGE_FORMAT_NV12_ER, nv12_flags, &tnrNV12R_);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI TNR NV12 R create failed"); return false; }
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             VPI_IMAGE_FORMAT_NV12_ER, nv12_flags, &tnrOutNV12L_);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI TNR output NV12 L create failed"); return false; }
        err = vpiImageCreate(config_.camera.width, config_.camera.height,
                             VPI_IMAGE_FORMAT_NV12_ER, nv12_flags, &tnrOutNV12R_);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI TNR output NV12 R create failed"); return false; }

        // 创建 TNR payload
        err = vpiCreateTemporalNoiseReduction(VPI_BACKEND_CUDA,
                  config_.camera.width, config_.camera.height,
                  VPI_IMAGE_FORMAT_NV12_ER, config_.tnr_version, &tnrPayloadL_);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI TNR payload L create failed"); return false; }
        err = vpiCreateTemporalNoiseReduction(VPI_BACKEND_CUDA,
                  config_.camera.width, config_.camera.height,
                  VPI_IMAGE_FORMAT_NV12_ER, config_.tnr_version, &tnrPayloadR_);
        if (err != VPI_SUCCESS) { LOG_ERROR("VPI TNR payload R create failed"); return false; }

        tnrFirstFrame_ = true;
        LOG_INFO("VPI TNR initialized (%dx%d, NV12_ER)", config_.camera.width, config_.camera.height);
    }

    // 7. 初始化海康双目相机
#ifdef HIK_CAMERA_ENABLED
    LOG_INFO("Opening dual cameras...");
    camera_ = std::make_unique<HikvisionCamera>();
    if (!camera_->open(config_.camera)) {
        LOG_WARN("Failed to open cameras - running in dry-run mode (synthetic frames)");
        camera_.reset();  // 释放相机, 标记为 dry-run
    }

    // 初始化 PWM 触发器 (硬件触发模式时)
    if (camera_ && config_.camera.use_trigger) {
        pwm_trigger_ = std::make_unique<PWMTrigger>(
            config_.trigger_chip, config_.trigger_line, config_.trigger_freq_hz);
        LOG_INFO("PWM trigger configured: chip=%s line=%d freq=%dHz",
                 config_.trigger_chip.c_str(), config_.trigger_line, config_.trigger_freq_hz);
    }
#else
    LOG_WARN("Camera support disabled (HIK SDK not found) - pipeline runs without camera");
#endif

    // 8. 初始化 TensorRT 检测器 (DLA/GPU)
    LOG_INFO("Initializing TRT Detector (DLA=%d, core=%d, dual=%d)...",
             config_.use_dla, config_.dla_core, config_.dual_dla);
    detector_ = std::make_unique<TRTDetector>();
    if (!detector_->init(config_.engine_file, config_.use_dla, config_.dla_core,
                         config_.conf_threshold, config_.nms_threshold,
                         config_.detector_input_format)) {
        LOG_ERROR("Failed to initialize TRT Detector");
        return false;
    }

    // 双 DLA 模式: 初始化第二个检测器 (DLA1)
    if ((config_.dual_dla || config_.triple_backend) && !config_.engine_file_dla1.empty()) {
        LOG_INFO("Initializing DLA1 Detector: %s", config_.engine_file_dla1.c_str());
        detector1_ = std::make_unique<TRTDetector>();
        if (!detector1_->init(config_.engine_file_dla1, true, 1,
                              config_.conf_threshold, config_.nms_threshold,
                              config_.detector_input_format)) {
            LOG_WARN("Failed to initialize DLA1 Detector - falling back to single DLA");
            detector1_.reset();
            config_.dual_dla = false;
            config_.triple_backend = false;
        }
    }

    // 三路轮转模式: 初始化 GPU 检测器
    if (config_.triple_backend && !config_.engine_file_gpu.empty()) {
        LOG_INFO("Initializing GPU Detector: %s", config_.engine_file_gpu.c_str());
        detector2_ = std::make_unique<TRTDetector>();
        if (!detector2_->init(config_.engine_file_gpu, false, 0,
                              config_.conf_threshold, config_.nms_threshold,
                              config_.detector_input_format)) {
            LOG_WARN("Failed to initialize GPU Detector - falling back to dual DLA");
            detector2_.reset();
            config_.triple_backend = false;
        }
    }

    // 8b. 初始化 SOT Tracker (YOLO 帧间填充)
    if (config_.tracker.enabled) {
        const auto& tcfg = config_.tracker;
        if (tcfg.type == "nanotrack") {
            tracker_ = std::make_unique<NanoTrackTRT>();
        } else if (tcfg.type == "mixformer") {
            tracker_ = std::make_unique<MixFormerTRT>();
        } else {
            LOG_ERROR("Unknown tracker type: %s (supported: nanotrack, mixformer)", tcfg.type.c_str());
            return false;
        }
        bool init_ok = false;
        if (tcfg.type == "nanotrack" && !tcfg.search_engine_path.empty()) {
            if (tcfg.head_engine_path.empty()) {
                LOG_ERROR("NanoTrack dual-backbone requires head_engine_path");
                return false;
            }
            auto* nt = dynamic_cast<NanoTrackTRT*>(tracker_.get());
            if (nt) init_ok = nt->initDualBackbone(tcfg.engine_path, tcfg.search_engine_path,
                                                    tcfg.head_engine_path, streams_.cudaStreamGPU);
        } else {
            init_ok = tracker_->init(tcfg.engine_path, tcfg.head_engine_path, streams_.cudaStreamGPU);
        }
        if (!init_ok) {
            LOG_ERROR("Failed to initialize SOT tracker (%s)", tcfg.type.c_str());
            return false;
        }
        tracker_state_ = TrackerState::IDLE;
        tracker_lost_count_ = 0;
        effective_detect_interval_ = tcfg.detect_interval;
        LOG_INFO("SOT Tracker initialized: %s (interval=%d, lost_thr=%d)",
                 tcfg.type.c_str(), tcfg.detect_interval, tcfg.lost_threshold);

        // 缓存 rectGray_vpiL 的 CUDA 指针 (Tegra 统一内存: 指针固定)
        // 避免 tracker 路径每帧 VPI lock/unlock (~0.6ms)
        for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
            if (slots_[i].rectGray_vpiL) {
                VPIImageData d;
                vpiImageLockData(slots_[i].rectGray_vpiL, VPI_LOCK_READ,
                                 VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &d);
                slots_[i].rectGray_L_gpu.data = d.buffer.pitch.planes[0].data;
                slots_[i].rectGray_L_gpu.pitchBytes = d.buffer.pitch.planes[0].pitchBytes;
                vpiImageUnlock(slots_[i].rectGray_vpiL);
            }
        }
    }

    // 9. 初始化立体匹配 (根据策略选择)
    if (config_.disparity_strategy == DisparityStrategy::ROI_ONLY) {
        // ROI 模式: 多点块匹配, 不需要全帧视差
        LOG_INFO("Initializing ROI Stereo Matcher (maxDisp=%d, patchR=%d)...",
                 config_.max_disparity, 5);
        roi_matcher_ = std::make_unique<ROIStereoMatcher>();
        const auto& P1 = calibration_->getProjectionLeft();
        float focal = static_cast<float>(P1.at<double>(0, 0));
        float cx    = static_cast<float>(P1.at<double>(0, 2));
        float cy    = static_cast<float>(P1.at<double>(1, 2));
        ROIMatchConfig roi_cfg;
        roi_cfg.maxDisparity    = config_.max_disparity;
        roi_cfg.patchRadius     = 5;
        roi_cfg.minDepth        = config_.depth.min_depth;
        roi_cfg.maxDepth        = config_.depth.max_depth;
        roi_cfg.objectDiameter  = config_.depth.object_diameter;
        roi_cfg.useCircleFit    = true;
        roi_matcher_->init(focal, calibration_->getBaseline(), cx, cy, roi_cfg);

        // 初始化混合深度估计 (单目+双目+Kalman)
        hybrid_depth_ = std::make_unique<HybridDepthEstimator>();
        auto hd_cfg = config_.depth;
        hd_cfg.dt = 1.0f / config_.trigger_freq_hz;
        hybrid_depth_->init(focal, calibration_->getBaseline(), cx, cy, hd_cfg);
        LOG_INFO("  Hybrid Depth: mono(<%.0fm) + stereo(>%.0fm) + Kalman @ %.0fHz",
                 hd_cfg.mono_max_z, hd_cfg.stereo_min_z, 1.0f / hd_cfg.dt);
    } else {
        // 全帧/半分辨率模式: VPI SGM
        LOG_INFO("Initializing VPI Stereo (maxDisp=%d, winSize=%d, %dx%d)...",
                 config_.max_disparity, config_.window_size,
                 config_.rect_width, config_.rect_height);
        stereo_ = std::make_unique<VPIStereo>();
        if (!stereo_->init(config_.max_disparity, config_.window_size,
                           config_.rect_width, config_.rect_height)) {
            LOG_ERROR("Failed to initialize VPI Stereo");
            return false;
        }

        // 全帧模式需要 Coordinate3D 融合器
        fusion_ = std::make_unique<Coordinate3D>();
        fusion_->init(calibration_->getProjectionLeft(),
                      calibration_->getBaseline(),
                      config_.depth.min_depth, config_.depth.max_depth);
    }

    const char* strategyStr = (config_.disparity_strategy == DisparityStrategy::ROI_ONLY)
        ? "ROI Multi-Point" : (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION
        ? "Half Resolution" : "Full Frame");

    LOG_INFO("========================================");
    LOG_INFO("Pipeline initialized successfully");
    LOG_INFO("  Raw resolution:  %dx%d", config_.camera.width, config_.camera.height);
    LOG_INFO("  Rect resolution: %dx%d", config_.rect_width, config_.rect_height);
    LOG_INFO("  Trigger: %d Hz", config_.trigger_freq_hz);
    LOG_INFO("  Detect: %s (DLA=%d)", config_.engine_file.c_str(), config_.use_dla);
    LOG_INFO("  Disparity: %s", strategyStr);
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

    // 启动异步采集线程 (解耦 USB 传输 ~5ms 阻塞)
    if (camera_) {
        grab_thread_ = std::thread(&Pipeline::grabLoop, this);
        LOG_INFO("Async grab thread started");
    }
#endif

    // 在独立线程中运行 Pipeline 循环
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
    // 唤醒可能在等待的 pipeline 线程和采集线程
    grab_request_cv_.notify_all();
    grab_done_cv_.notify_all();
#endif

    // 等待工作线程退出
    if (pipeline_thread_.joinable()) {
        pipeline_thread_.join();
    }

#ifdef HIK_CAMERA_ENABLED
    // 等待采集线程退出 (最多等待一个 camera grab timeout)
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

// ===================================================================
// 异步相机采集线程 (零拷贝, 按需模式)
//
// 工作流:
//   1. pipeline 调用 requestGrab(slot) → 采集线程唤醒
//   2. 采集线程: lock VPI Image → grabFramePair → unlock
//   3. 采集线程: signal grab_done → pipeline 端 waitGrab() 返回
//
// 关键优化:
//   - 直接写入 VPI Image (零拷贝, 无 staging buffer)
//   - grab 期间 pipeline 并行执行 stage1+stage2 (重叠 ~3ms)
//   - 总迭代时间: grab_wait(~2ms) + process(~4ms) ≈ 6-7ms
// ===================================================================

#ifdef HIK_CAMERA_ENABLED
void Pipeline::grabLoop() {
    while (running_) {
        // 等待 pipeline 的采集请求
        int slot_idx;
        {
            std::unique_lock<std::mutex> lk(grab_mutex_);
            grab_request_cv_.wait(lk, [this]{ return grab_request_slot_ >= 0 || !running_; });
            if (!running_) break;
            slot_idx = grab_request_slot_;
            grab_request_slot_ = -1;
        }

        auto& slot = slots_[slot_idx];

        // 直接锁定 VPI Image 并写入 (零拷贝: camera DMA → 统一内存)
        VPIImageData imgDataL, imgDataR;
        vpiImageLockData(slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataL);
        vpiImageLockData(slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataR);

        GrabResult resL, resR;
        bool ok = camera_->grabFramePair(
            static_cast<uint8_t*>(imgDataL.buffer.pitch.planes[0].data),
            static_cast<uint8_t*>(imgDataR.buffer.pitch.planes[0].data),
            imgDataL.buffer.pitch.planes[0].pitchBytes,
            imgDataR.buffer.pitch.planes[0].pitchBytes,
            1000, resL, resR);

        vpiImageUnlock(slot.rawL);
        vpiImageUnlock(slot.rawR);

        // 通知 pipeline 采集完成
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

            if (frame_callback_) {
                VPIImage vizImg = (config_.detector_input_format == "bgr")
                                  ? slot.rectBGR_vpiL : slot.rectGray_vpiL;
                frame_callback_(slot.frame_id, vizImg, slot.rawL,
                                slot.detections, slot.results,
                                current_fps_.load());
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
            cudaStreamWaitEvent(getDLAStream(slot.frame_id), slot.evtRectDone, 0);
            cudaStreamWaitEvent(streams_.cudaStreamGPU, slot.evtRectDone, 0);

            {
                ScopedTimer t1("Stage1_DetectSubmit");
                stage1_detect(slot, slot_idx);
                globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
            }
            cudaEventRecord(slot.evtDetectDone, getDLAStream(slot.frame_id));

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
        cudaEventRecord(slot.evtDetectDone, getDLAStream(slot.frame_id));
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

void Pipeline::stage0_grab_and_rectify(FrameSlot& slot, bool grab_preloaded) {
    NVTX_RANGE("Stage0_GrabRect");

#ifdef HIK_CAMERA_ENABLED
    if (camera_ && !grab_preloaded) {
        // 同步采集: 直接在 pipeline 线程阻塞抓取 (pipelineLoop 全帧模式使用)
        VPIImageData imgDataL, imgDataR;
        vpiImageLockData(slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataL);
        vpiImageLockData(slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataR);

        GrabResult resL, resR;
        bool grab_ok = camera_->grabFramePair(
            static_cast<uint8_t*>(imgDataL.buffer.pitch.planes[0].data),
            static_cast<uint8_t*>(imgDataR.buffer.pitch.planes[0].data),
            imgDataL.buffer.pitch.planes[0].pitchBytes,
            imgDataR.buffer.pitch.planes[0].pitchBytes,
            1000, resL, resR);

        vpiImageUnlock(slot.rawL);
        vpiImageUnlock(slot.rawR);

        if (!grab_ok) {
            slot.grab_failed = true;
            LOG_WARN("[Pipeline] Frame %d grab failed, skipping", slot.frame_id);
        }
    }
    // grab_preloaded == true: 异步采集线程已将数据写入 rawL/rawR (零拷贝)
#endif

    // === VPI 流同步: 确保上一帧的 remap/convert 完成后再提交新任务 ===
    // 放在 grab 之后: 相机 grab 阻塞 ~5ms 期间, VPI remap (~1ms) 已异步完成
    // 因此此处同步几乎不阻塞 (0~0.1ms)
    vpiStreamSync(streams_.vpiStreamPVA);

    // 2. VPI TNR 降噪 (可选, 在校正前对原始图降噪)
    if (config_.tnr_enabled) {
        ScopedTimer tt("TNR");

        // U8 → NV12_ER: Y = U8, UV = 0x80 (中性灰度)
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rawL, tnrNV12L_, nullptr);
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rawR, tnrNV12R_, nullptr);

        // TNR 处理
        VPITNRParams tnrParams;
        vpiInitTemporalNoiseReductionParams(&tnrParams);
        tnrParams.preset   = config_.tnr_preset;
        tnrParams.strength = config_.tnr_strength;

        VPIImage prevL = tnrFirstFrame_ ? nullptr : tnrOutNV12L_;
        VPIImage prevR = tnrFirstFrame_ ? nullptr : tnrOutNV12R_;

        vpiSubmitTemporalNoiseReduction(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                         tnrPayloadL_, prevL,
                                         tnrNV12L_, tnrOutNV12L_, &tnrParams);
        vpiSubmitTemporalNoiseReduction(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                         tnrPayloadR_, prevR,
                                         tnrNV12R_, tnrOutNV12R_, &tnrParams);

        // NV12_ER → U8: 提取 Y 通道回写 rawL/rawR (用于后续 Remap)
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    tnrOutNV12L_, slot.rawL, nullptr);
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    tnrOutNV12R_, slot.rawR, nullptr);

        tnrFirstFrame_ = false;
        globalPerf().record("TNR", tt.elapsedMs());
    }

    // 3. 校正路径 — 根据 input_format 选择 color 或 gray
    if (config_.detector_input_format == "bgr") {
        // Color Pipeline: Debayer → BGR Remap → Gray Convert
        //    a) Debayer: Bayer RG8 → BGR8 (自定义 CUDA bilinear kernel)
        //       使用 init() 时缓存的 CUDA 指针, 跳过 VPI lock/unlock (~2.4ms 优化)
        //       Tegra 统一内存: HOST 写入 (grab) 直接对 CUDA 可见, 无需显式同步
        {
            int rw = config_.camera.width;
            int rh = config_.camera.height;

            launchBayerToBGR8(
                static_cast<const unsigned char*>(slot.rawL_gpu.data),
                static_cast<unsigned char*>(slot.tempBGR_L_gpu.data),
                rw, rh,
                slot.rawL_gpu.pitchBytes,
                slot.tempBGR_L_gpu.pitchBytes,
                streams_.cudaStreamBGR);

            launchBayerToBGR8(
                static_cast<const unsigned char*>(slot.rawR_gpu.data),
                static_cast<unsigned char*>(slot.tempBGR_R_gpu.data),
                rw, rh,
                slot.rawR_gpu.pitchBytes,
                slot.tempBGR_R_gpu.pitchBytes,
                streams_.cudaStreamBGR);

            cudaStreamSynchronize(streams_.cudaStreamBGR);
        }

        //    b) BGR Remap: 校正 (与 init() 中 payload 相同 backend)
        rectifier_->submitBGR(streams_.vpiStreamPVA,
                              slot.tempBGR_L, slot.tempBGR_R,
                              slot.rectBGR_vpiL, slot.rectBGR_vpiR);

        //    c) BGR → Gray: 为立体匹配提供灰度图
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rectBGR_vpiL, slot.rectGray_vpiL, nullptr);
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rectBGR_vpiR, slot.rectGray_vpiR, nullptr);
    } else {
        // Gray Pipeline (Legacy): Bayer 原始图直接按 U8 单通道 remap
        rectifier_->submit(streams_.vpiStreamPVA,
                           slot.rawL, slot.rawR,
                           slot.rectGray_vpiL, slot.rectGray_vpiR);
    }

    // 不在此处同步! VPI remap (~1ms) 在下次 stage1 调用前由 caller 同步.
    // 三缓冲确保 slot 不被过早重用.

    NVTX_RANGE_POP();
}

// ===================================================================
// Dual DLA helpers: frame-based detector/stream selection
// ===================================================================

TRTDetector* Pipeline::getDetector(int frame_id) const {
    if (config_.triple_backend && detector1_ && detector2_) {
        switch (frame_id % 3) {
            case 0: return detector_.get();   // DLA0
            case 1: return detector1_.get();  // DLA1
            default: return detector2_.get(); // GPU
        }
    }
    if (config_.dual_dla && detector1_ && (frame_id & 1))
        return detector1_.get();
    return detector_.get();
}

cudaStream_t Pipeline::getDLAStream(int frame_id) const {
    if (config_.triple_backend && detector1_ && detector2_) {
        switch (frame_id % 3) {
            case 0: return streams_.cudaStreamDLA;
            case 1: return streams_.cudaStreamDLA1;
            default: return streams_.cudaStreamDetGPU;
        }
    }
    if (config_.dual_dla && detector1_ && (frame_id & 1))
        return streams_.cudaStreamDLA1;
    return streams_.cudaStreamDLA;
}

void Pipeline::stage1_detect(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage1_Detect");

    auto* det = getDetector(slot.frame_id);
    auto stream = getDLAStream(slot.frame_id);

    // 从 VPI Image 获取 GPU 指针，传给 TensorRT
    // BGR 模式: 使用校正后 BGR 图像; Gray 模式: 使用校正后灰度图
    VPIImage detectImg = (config_.detector_input_format == "bgr")
                         ? slot.rectBGR_vpiL : slot.rectGray_vpiL;
    VPIImageData imgData;
    VPIStatus st = vpiImageLockData(detectImg, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);
    if (st != VPI_SUCCESS) {
        LOG_WARN("stage1_detect: vpiImageLockData failed (%d)", (int)st);
        NVTX_RANGE_POP();
        return;
    }

    void* gpu_ptr = imgData.buffer.pitch.planes[0].data;
    int pitch = imgData.buffer.pitch.planes[0].pitchBytes;

    // 异步推理提交: 仅 enqueue，不在此处同步
    det->enqueue(slot_index, gpu_ptr, pitch,
                 config_.rect_width, config_.rect_height,
                 stream);

    vpiImageUnlock(detectImg);
    NVTX_RANGE_POP();
}

void Pipeline::stage2_stereo(FrameSlot& slot) {
    NVTX_RANGE("Stage2_Stereo");

    switch (config_.disparity_strategy) {
        case DisparityStrategy::FULL_FRAME:
            stereo_->compute(streams_.vpiStreamGPU,
                             slot.rectGray_vpiL, slot.rectGray_vpiR,
                             slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::HALF_RESOLUTION:
            stereo_->computeHalfRes(streams_.vpiStreamGPU,
                                    slot.rectGray_vpiL, slot.rectGray_vpiR,
                                    slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::ROI_ONLY:
            // ROI mode uses separate ROI matcher in pipelineLoopROI stage2.
            // stage2_stereo() should never be called in ROI mode.
            LOG_ERROR("stage2_stereo called in ROI_ONLY mode — this is a bug");
            return;
    }

    // 不在此处做 vpiStreamSync。
    // 下游 Stage3 统一在融合前对 vpiStreamGPU 做同步，避免串行化 Stage1/2 提交路径。

    NVTX_RANGE_POP();
}

void Pipeline::stage3_fuse(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage3_Fuse");

    slot.results.clear();

    // Detect 结果在 Stage1 中异步 D2H，现已通过 evtDetectDone 保证完成
    slot.detections = getDetector(slot.frame_id)->collect(slot_index,
                                         config_.rect_width, config_.rect_height);

    // 获取视差图 GPU 指针
    VPIImageData dispData;
    VPIStatus st = vpiImageLockData(slot.disparityMap, VPI_LOCK_READ,
                     VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dispData);
    if (st != VPI_SUCCESS) {
        LOG_WARN("stage3_fuse: vpiImageLockData failed (%d)", (int)st);
        NVTX_RANGE_POP();
        return;
    }

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

// ===================================================================
// ROI 模式: Stage 2 — 检测后 ROI 多点匹配 + 三角测距 (一步到位)
// ===================================================================

void Pipeline::stage2_roi_match_fuse(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage2_ROIMatchFuse");

    slot.results.clear();

    // 1. 收集检测结果 (Stage 1 完成后) — 使用与 stage1 相同的 detector
    auto* det = getDetector(slot.frame_id);
    slot.detections = det->collect(slot_index,
                                   config_.rect_width, config_.rect_height);

    if (slot.detections.empty()) {
        // 无检测: 仅 Kalman 预测
        if (hybrid_depth_) {
            slot.results = hybrid_depth_->predictOnly();
        }
        NVTX_RANGE_POP();
        return;
    }

    // 2. 获取校正后灰度左右图 GPU 指针 (color pipeline → rectGray)
    VPIImageData imgDataL, imgDataR;
    VPIStatus stL, stR;
    {
        ScopedTimer tvl("Stage2_VPILock");
        stL = vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
        stR = vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);
        globalPerf().record("Stage2_VPILock", tvl.elapsedMs());
    }
    if (stL != VPI_SUCCESS || stR != VPI_SUCCESS) {
        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiR);
        if (hybrid_depth_) slot.results = hybrid_depth_->predictOnly();
        NVTX_RANGE_POP();
        return;
    }

    const uint8_t* leftPtr  = static_cast<const uint8_t*>(imgDataL.buffer.pitch.planes[0].data);
    int leftPitch  = imgDataL.buffer.pitch.planes[0].pitchBytes;
    const uint8_t* rightPtr = static_cast<const uint8_t*>(imgDataR.buffer.pitch.planes[0].data);
    int rightPitch = imgDataR.buffer.pitch.planes[0].pitchBytes;

    // 3. ROI 多点立体匹配 (双目结果)
    std::vector<stereo3d::Object3D> roi_results;
    {
        ScopedTimer troi("Stage2_ROIMatch");
        roi_results = roi_matcher_->match(
            leftPtr, leftPitch, rightPtr, rightPitch,
            config_.rect_width, config_.rect_height,
            slot.detections, streams_.cudaStreamFuse);
        globalPerf().record("Stage2_ROIMatch", troi.elapsedMs());
    }

    vpiImageUnlock(slot.rectGray_vpiL);
    vpiImageUnlock(slot.rectGray_vpiR);

    // 4. 混合深度估计 (单目+双目融合+Kalman滤波)
    if (hybrid_depth_) {
        auto now = std::chrono::steady_clock::now();
        double dt = 0.01;
        if (last_fuse_time_.time_since_epoch().count() > 0) {
            dt = std::chrono::duration<double>(now - last_fuse_time_).count();
            dt = std::clamp(dt, 0.002, 0.1);
        }
        last_fuse_time_ = now;
        ScopedTimer thd("Stage2_HybridDepth");
        slot.results = hybrid_depth_->estimate(slot.detections, roi_results, dt);
        globalPerf().record("Stage2_HybridDepth", thd.elapsedMs());
    } else {
        slot.results = std::move(roi_results);
    }

    NVTX_RANGE_POP();
}

// ===================================================================
// SOT Tracker 辅助: 检测帧后刷新 tracker template
// ===================================================================
void Pipeline::tracker_handle_detect_result(FrameSlot& slot) {
    if (!tracker_) return;

    // 从 YOLO detections 中选最高置信度目标
    if (slot.detections.empty()) {
        tracker_lost_count_++;
        if (tracker_lost_count_ >= config_.tracker.lost_threshold) {
            if (tracker_state_ != TrackerState::IDLE) {
                tracker_state_ = TrackerState::LOST;
                tracker_->reset();
                tracker_state_ = TrackerState::IDLE;
                LOG_INFO("[Tracker] LOST → IDLE (no YOLO det for %d frames)", tracker_lost_count_);
            }
        }
        return;
    }

    // 选最高 confidence 的 detection
    const auto& best = *std::max_element(
        slot.detections.begin(), slot.detections.end(),
        [](const Detection& a, const Detection& b) { return a.confidence < b.confidence; });

    // 使用缓存的 rectGray_vpiL CUDA 指针 (避免 lock/unlock ~0.3ms)
    const uint8_t* imgPtr = static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    int imgPitch = slot.rectGray_L_gpu.pitchBytes;
    if (!imgPtr) return;

    if (tracker_state_ == TrackerState::IDLE || tracker_state_ == TrackerState::LOST) {
        // 首次目标 or 重新捕获 → setTarget
        tracker_->setTarget(imgPtr, imgPitch, config_.rect_width, config_.rect_height, best);
        tracker_state_ = TrackerState::TRACKING;
        tracker_lost_count_ = 0;
        LOG_INFO("[Tracker] setTarget: (%.0f,%.0f) %dx%d conf=%.2f",
                 best.cx, best.cy, (int)best.width, (int)best.height, best.confidence);
    } else {
        // TRACKING → 用 YOLO 结果刷新 template (纠正漂移)
        tracker_->setTarget(imgPtr, imgPitch, config_.rect_width, config_.rect_height, best);
        tracker_lost_count_ = 0;
    }
}

// ===================================================================
// SOT Tracker 辅助: 非检测帧运行 tracker 推理
// ===================================================================
void Pipeline::tracker_infill(FrameSlot& slot) {
    if (!tracker_ || tracker_state_ != TrackerState::TRACKING) {
        slot.sot_bbox_result = SOTResult{};
        slot.bbox_source = BboxSource::NONE;
        return;
    }

    // 使用缓存的 rectGray_vpiL CUDA 指针 (避免 lock/unlock ~0.3ms)
    const uint8_t* imgPtr = static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    int imgPitch = slot.rectGray_L_gpu.pitchBytes;
    if (!imgPtr) {
        slot.sot_bbox_result = SOTResult{};
        slot.bbox_source = BboxSource::NONE;
        return;
    }

    SOTResult result = tracker_->track(imgPtr, imgPitch,
                                       config_.rect_width, config_.rect_height);

    if (result.valid && result.confidence >= config_.tracker.min_confidence) {
        slot.sot_bbox_result = result;
        slot.bbox_source = BboxSource::TRACKER;
        tracker_lost_count_ = 0;
    } else {
        slot.sot_bbox_result = SOTResult{};
        slot.bbox_source = BboxSource::NONE;
        tracker_lost_count_++;
        if (tracker_lost_count_ >= config_.tracker.lost_threshold) {
            tracker_state_ = TrackerState::LOST;
            tracker_->reset();
            tracker_state_ = TrackerState::IDLE;
        }
    }
}

// ===================================================================
// ROI 模式: Stage 2 (tracker) — tracker bbox → ROI 匹配 + 深度融合
// ===================================================================
void Pipeline::stage2_roi_fuse_tracker(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage2_ROIFuseTracker");
    slot.results.clear();

    if (slot.bbox_source != BboxSource::TRACKER || !slot.sot_bbox_result.valid) {
        // 无有效 tracker 结果: 仅 Kalman 预测
        if (hybrid_depth_) {
            slot.results = hybrid_depth_->predictOnly();
        }
        NVTX_RANGE_POP();
        return;
    }

    // 将 tracker bbox 转换为 Detection 格式 (复用 ROI match 路径)
    const auto& sot = slot.sot_bbox_result;
    Detection pseudo_det;
    pseudo_det.cx = sot.cx;
    pseudo_det.cy = sot.cy;
    pseudo_det.width = sot.width;
    pseudo_det.height = sot.height;
    pseudo_det.confidence = sot.confidence;
    pseudo_det.class_id = 0;  // volleyball
    slot.detections = { pseudo_det };

    // 2. 获取校正后灰度左右图 GPU 指针
    VPIImageData imgDataL, imgDataR;
    VPIStatus stL = vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ,
                                      VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
    VPIStatus stR = vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ,
                                      VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);
    if (stL != VPI_SUCCESS || stR != VPI_SUCCESS) {
        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiR);
        if (hybrid_depth_) slot.results = hybrid_depth_->predictOnly();
        NVTX_RANGE_POP();
        return;
    }

    const uint8_t* leftPtr  = static_cast<const uint8_t*>(imgDataL.buffer.pitch.planes[0].data);
    int leftPitch  = imgDataL.buffer.pitch.planes[0].pitchBytes;
    const uint8_t* rightPtr = static_cast<const uint8_t*>(imgDataR.buffer.pitch.planes[0].data);
    int rightPitch = imgDataR.buffer.pitch.planes[0].pitchBytes;

    // 3. ROI 多点立体匹配
    std::vector<stereo3d::Object3D> roi_results;
    {
        ScopedTimer troi("Stage2_ROIMatchTracker");
        roi_results = roi_matcher_->match(
            leftPtr, leftPitch, rightPtr, rightPitch,
            config_.rect_width, config_.rect_height,
            slot.detections, streams_.cudaStreamFuse);
        globalPerf().record("Stage2_ROIMatchTracker", troi.elapsedMs());
    }

    vpiImageUnlock(slot.rectGray_vpiL);
    vpiImageUnlock(slot.rectGray_vpiR);

    // 4. 混合深度估计
    if (hybrid_depth_) {
        auto now = std::chrono::steady_clock::now();
        double dt = 0.01;
        if (last_fuse_time_.time_since_epoch().count() > 0) {
            dt = std::chrono::duration<double>(now - last_fuse_time_).count();
            dt = std::clamp(dt, 0.002, 0.1);
        }
        last_fuse_time_ = now;
        slot.results = hybrid_depth_->estimate(slot.detections, roi_results, dt);
    } else {
        slot.results = std::move(roi_results);
    }

    NVTX_RANGE_POP();
}

// ===================================================================
// ROI 模式 Pipeline 主循环
//
// 异步采集 + 三级流水线:
//   requestGrab(N+1) → Stage1 Detect(N) → Stage2 Fuse(N-1) → waitGrab(N+1) → Stage0 Process(N+1)
//
//   grab 线程: |-----grabFramePair ~5ms------|
//   pipeline:  |--detect 3ms--|--fuse 0.1ms--|--waitGrab ~2ms--|--process 2ms--|
//                    ↑ 与 grab 并行执行          ↑ grab 剩余时间     ↑ bayer+remap
//
//   每帧迭代: ~7ms (远低于 10ms/frame @ 100Hz)
//   吞吐量 = 100 FPS (camera rate limited)
// ===================================================================

void Pipeline::pipelineLoopROI() {
    using Clock = std::chrono::high_resolution_clock;
    auto fps_start = Clock::now();
    int fps_count = 0;

    int next_grab_frame   = 0;
    int next_detect_frame = 0;
    int next_fuse_frame   = 0;

    // ===== 填充: 同步抓取 + 处理首帧 =====
    {
        int slot_idx = next_grab_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        slot.reset();
        slot.frame_id = next_grab_frame;

        bool grab_preloaded = false;
#ifdef HIK_CAMERA_ENABLED
        if (camera_) {
            requestGrab(slot_idx);
            waitGrab();
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

        vpiStreamSync(streams_.vpiStreamPVA);
        cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);

        auto dlaStream = getDLAStream(slot.frame_id);
        cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);

        ScopedTimer t1("Stage1_DetectSubmit");
        stage1_detect(slot, slot_idx);
        globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
        cudaEventRecord(slot.evtDetectDone, dlaStream);
        next_detect_frame++;
    }

    while (running_) {
        // ====================================================================
        // Phase A: 发起异步采集 (极速, ~0.01ms)
        //   grab 线程锁定 VPI Image → 阻塞等待相机 USB 传输
        //   pipeline 线程继续执行 Stage1/Stage2, 与 grab 并行
        // ====================================================================
        int grab_slot_idx = next_grab_frame % RING_BUFFER_SIZE;
        {
            auto& slot = slots_[grab_slot_idx];
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
                vpiStreamSync(streams_.vpiStreamPVA);
                next_detect_frame++;
                next_fuse_frame = next_detect_frame - 1;
            } else {

            // 判断是否为检测帧
            // 固定节拍检测：按 detect_interval 控制 YOLO 频率，
            // 不再因 tracker 是否处于 TRACKING 状态而抢跑检测。
            bool is_detect = !tracker_ ||
                             (slot.frame_id % effective_detect_interval_ == 0);
            slot.is_detect_frame = is_detect;

            if (is_detect) {
                // ---- YOLO 检测帧 ----
                {
                    ScopedTimer tw("Stage1_WaitRect");
                    cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);
                    globalPerf().record("Stage1_WaitRect", tw.elapsedMs());
                }

                auto dlaStream = getDLAStream(slot.frame_id);
                cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);

                {
                    ScopedTimer t1("Stage1_DetectSubmit");
                    stage1_detect(slot, slot_idx);
                    globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
                }
                cudaEventRecord(slot.evtDetectDone, dlaStream);
            } else {
                // ---- Tracker 填充帧 ----
                // 等 rectify 完成 (tracker 需要 rectified BGR)
                vpiStreamSync(streams_.vpiStreamPVA);
                cudaStreamSynchronize(streams_.cudaStreamGPU);

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

        // ====================================================================
        // Phase C: Stage2 — ROI 匹配 + 融合帧 N-1 (与 grab 并行, ~0.1ms)
        //   检测帧: collect YOLO → ROI match → depth fuse + 刷新 tracker template
        //   填充帧: tracker bbox → ROI match → depth fuse
        // ====================================================================
        if (next_fuse_frame < next_detect_frame - 1) {
            int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
            auto& slot = slots_[slot_idx];

            {
                ScopedTimer tw("Stage2_WaitDetect");
                cudaStreamWaitEvent(streams_.cudaStreamFuse, slot.evtDetectDone, 0);
                cudaStreamSynchronize(streams_.cudaStreamFuse);
                globalPerf().record("Stage2_WaitDetect", tw.elapsedMs());
            }

            if (slot.is_detect_frame) {
                // ---- YOLO 检测帧: collect + ROI + depth ----
                {
                    ScopedTimer t2("Stage2_ROIMatchFuse");
                    stage2_roi_match_fuse(slot, slot_idx);
                    globalPerf().record("Stage2_ROIMatchFuse", t2.elapsedMs());
                }
                slot.bbox_source = BboxSource::YOLO;
                // 用 YOLO 结果刷新 tracker template
                tracker_handle_detect_result(slot);
            } else {
                // ---- Tracker 填充帧: tracker bbox → ROI + depth ----
                {
                    ScopedTimer t2("Stage2_ROIFuseTracker");
                    stage2_roi_fuse_tracker(slot, slot_idx);
                    globalPerf().record("Stage2_ROIFuseTracker", t2.elapsedMs());
                }
            }

            if (result_callback_) {
                ScopedTimer trc("Stage2_ResultCB");
                result_callback_(slot.frame_id, slot.results);
                globalPerf().record("Stage2_ResultCB", trc.elapsedMs());
            }

            if (frame_callback_) {
                ScopedTimer tfc("Stage2_FrameCB");
                VPIImage vizImg = (config_.detector_input_format == "bgr")
                                  ? slot.rectBGR_vpiL : slot.rectGray_vpiL;
                frame_callback_(slot.frame_id, vizImg, slot.rawL,
                                slot.detections, slot.results,
                                current_fps_.load());
                globalPerf().record("Stage2_FrameCB", tfc.elapsedMs());
            }

            next_fuse_frame++;

            fps_count++;
            auto now = Clock::now();
            double elapsed_s = std::chrono::duration<double>(now - fps_start).count();
            if (elapsed_s >= 1.0) {
                current_fps_ = static_cast<float>(fps_count / elapsed_s);
                fps_count = 0;
                fps_start = now;
                if (config_.stats_interval > 0) {
                    LOG_INFO("[ROI] FPS: %.1f  (Output frame %d)", current_fps_.load(), next_fuse_frame);
                }
            }
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
                globalPerf().record("Stage0_WaitGrab", tw.elapsedMs());
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

    while (next_detect_frame < next_grab_frame) {
        int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        auto dlaStream = getDLAStream(slot.frame_id);
        cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);
        cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);
        stage1_detect(slot, slot_idx);
        cudaEventRecord(slot.evtDetectDone, dlaStream);
        next_detect_frame++;
    }

    while (next_fuse_frame < next_detect_frame) {
        int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
        auto& slot = slots_[slot_idx];
        cudaStreamWaitEvent(streams_.cudaStreamFuse, slot.evtDetectDone, 0);
        cudaStreamSynchronize(streams_.cudaStreamFuse);
        stage2_roi_match_fuse(slot, slot_idx);
        if (result_callback_) result_callback_(slot.frame_id, slot.results);
        next_fuse_frame++;
    }

    LOG_INFO("ROI Pipeline loop exited");
}

void Pipeline::printPerfReport() const {
    globalPerf().printReport();
}

}  // namespace stereo3d
