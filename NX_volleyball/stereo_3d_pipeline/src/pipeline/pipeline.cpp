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
#include <cmath>
#include <limits>

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
        const VPIImageFormat rawFmt = colorPipelineEnabled()
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

        if (config_.disparity_strategy != DisparityStrategy::ROI_ONLY) {
            // 视差图 (S16 格式, Q10.5 定点数) → 校正后分辨率
            err = vpiImageCreate(config_.rect_width, config_.rect_height,
                                 VPI_IMAGE_FORMAT_S16, VPI_BACKEND_CUDA, &slots_[i].disparityMap);
            if (err != VPI_SUCCESS) { LOG_ERROR("VPI disparity create failed"); return false; }

            err = vpiImageCreate(config_.rect_width, config_.rect_height,
                                 VPI_IMAGE_FORMAT_U16, VPI_BACKEND_CUDA, &slots_[i].confidenceMap);
            if (err != VPI_SUCCESS) { LOG_ERROR("VPI confidence create failed"); return false; }
        }
    }

    // 5b. 缓存 Bayer→BGR 所需的 CUDA 指针 (Tegra 统一内存: 指针固定)
    //     避免每帧 8 次 VPI lock/unlock, 节省 ~2.4ms/frame
    if (colorPipelineEnabled()) {
        LOG_INFO("Caching CUDA pointers for Bayer pipeline...");
        for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
            auto cachePtr = [i](const char* name, VPIImage img,
                                FrameSlot::CachedGPU& out) -> bool {
                VPIImageData d;
                VPIStatus st = vpiImageLockData(
                    img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &d);
                if (st != VPI_SUCCESS) {
                    LOG_ERROR("Failed to cache %s CUDA pointer for slot %d (err=%d)",
                              name, i, (int)st);
                    return false;
                }
                out.data = d.buffer.pitch.planes[0].data;
                out.pitchBytes = d.buffer.pitch.planes[0].pitchBytes;
                vpiImageUnlock(img);
                return out.data != nullptr && out.pitchBytes > 0;
            };
            if (!cachePtr("rawL", slots_[i].rawL, slots_[i].rawL_gpu) ||
                !cachePtr("rawR", slots_[i].rawR, slots_[i].rawR_gpu) ||
                !cachePtr("tempBGR_L", slots_[i].tempBGR_L, slots_[i].tempBGR_L_gpu) ||
                !cachePtr("tempBGR_R", slots_[i].tempBGR_R, slots_[i].tempBGR_R_gpu)) {
                return false;
            }
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
        LOG_ERROR("Failed to open stereo cameras. Both Hikvision cameras are required; "
                  "check USB connection, serial/index config, and whether MVS holds a device.");
        camera_.reset();
        return false;
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

    // 8. 初始化 TensorRT 检测器 (GPU)
    LOG_INFO("Initializing TRT Detector (DLA=%d, core=%d)...",
             config_.use_dla, config_.dla_core);
    detector_ = std::make_unique<TRTDetector>();
    if (!detector_->init(config_.engine_file, config_.use_dla, config_.dla_core,
                         config_.conf_threshold, config_.nms_threshold,
                         config_.detector_input_format)) {
        LOG_ERROR("Failed to initialize TRT Detector");
        return false;
    }

    if (config_.dual_yolo.enabled) {
        const std::string right_engine = config_.dual_yolo.right_engine_file.empty()
            ? config_.engine_file : config_.dual_yolo.right_engine_file;
        const std::string right_format = config_.dual_yolo.right_input_format.empty()
            ? config_.detector_input_format : config_.dual_yolo.right_input_format;
        LOG_INFO("Initializing right TRT Detector for dual YOLO (DLA=%d, core=%d)...",
                 config_.dual_yolo.right_use_dla, config_.dual_yolo.right_dla_core);
        detector_right_ = std::make_unique<TRTDetector>();
        if (!detector_right_->init(right_engine,
                                   config_.dual_yolo.right_use_dla,
                                   config_.dual_yolo.right_dla_core,
                                   config_.conf_threshold,
                                   config_.nms_threshold,
                                   right_format)) {
            LOG_ERROR("Failed to initialize right TRT Detector");
            return false;
        }
        LOG_INFO("  Dual YOLO: right_engine=%s, use_for_depth=%d, fallback_roi=%d, "
                 "epipolar_fallback=%d, center_refine=%d, roi_denoise=%d, "
                 "depth_solver=%s, subpx=%d patch=%d search=%d pts=%d",
                 right_engine.c_str(),
                 config_.dual_yolo.use_for_depth,
                 config_.dual_yolo.fallback_to_roi_match,
                 config_.dual_yolo.fallback_epipolar_search,
                 config_.dual_yolo.center_refine,
                 config_.dual_yolo.roi_denoise,
                 config_.dual_yolo.depth_solver.c_str(),
                 config_.dual_yolo.subpixel_enabled,
                 config_.dual_yolo.subpixel_patch_radius,
                 config_.dual_yolo.subpixel_search_radius_px,
                 config_.dual_yolo.subpixel_max_points);
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
                VPIStatus st = vpiImageLockData(
                    slots_[i].rectGray_vpiL, VPI_LOCK_READ,
                    VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &d);
                if (st != VPI_SUCCESS) {
                    LOG_ERROR("Failed to cache rectGray_vpiL CUDA pointer for slot %d (err=%d)",
                              i, (int)st);
                    return false;
                }
                slots_[i].rectGray_L_gpu.data = d.buffer.pitch.planes[0].data;
                slots_[i].rectGray_L_gpu.pitchBytes = d.buffer.pitch.planes[0].pitchBytes;
                vpiImageUnlock(slots_[i].rectGray_vpiL);
                if (!slots_[i].rectGray_L_gpu.data ||
                    slots_[i].rectGray_L_gpu.pitchBytes <= 0) {
                    LOG_ERROR("Invalid rectGray_vpiL CUDA pointer for slot %d", i);
                    return false;
                }
            }
        }
    }

    // 9. 初始化立体匹配 (根据策略选择)
    if (config_.disparity_strategy == DisparityStrategy::ROI_ONLY) {
        const auto& P1 = calibration_->getProjectionLeft();
        float focal = static_cast<float>(P1.at<double>(0, 0));
        float cx    = static_cast<float>(P1.at<double>(0, 2));
        float cy    = static_cast<float>(P1.at<double>(1, 2));

        const bool dual_yolo_depth_only =
            config_.dual_yolo.enabled &&
            config_.dual_yolo.use_for_depth &&
            !config_.dual_yolo.fallback_to_roi_match;
        const bool need_roi_matcher =
            config_.tracker.enabled || !dual_yolo_depth_only;
        if (need_roi_matcher) {
            // ROI 模式: 多点块匹配, 不需要全帧视差
            LOG_INFO("Initializing ROI Stereo Matcher (maxDisp=%d, patchR=%d)...",
                     config_.max_disparity, 5);
            roi_matcher_ = std::make_unique<ROIStereoMatcher>();
            ROIMatchConfig roi_cfg;
            roi_cfg.maxDisparity    = config_.max_disparity;
            roi_cfg.patchRadius     = 5;
            roi_cfg.minDepth        = config_.depth.min_depth;
            roi_cfg.maxDepth        = config_.depth.max_depth;
            roi_cfg.objectDiameter  = config_.depth.object_diameter;
            roi_cfg.useCircleFit    = true;
            roi_matcher_->init(focal, calibration_->getBaseline(), cx, cy, roi_cfg);
        } else {
            LOG_INFO("Skipping ROI Stereo Matcher: dual YOLO depth path has ROI fallback disabled");
        }

        const float required_disp = focal * calibration_->getBaseline() /
                                    std::max(0.01f, config_.depth.min_depth);
        if (required_disp > config_.max_disparity) {
            LOG_WARN("max_disparity=%d is below f*B/min_depth=%.0f px; "
                     "long-baseline near objects may be rejected",
                     config_.max_disparity, required_disp);
        }

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

    const bool dualYoloDepthOnly =
        config_.disparity_strategy == DisparityStrategy::ROI_ONLY &&
        config_.dual_yolo.enabled &&
        config_.dual_yolo.use_for_depth &&
        !config_.dual_yolo.fallback_to_roi_match;
    const std::string strategyStr = dualYoloDepthOnly
        ? ("Dual YOLO " + config_.dual_yolo.depth_solver)
        : ((config_.disparity_strategy == DisparityStrategy::ROI_ONLY)
            ? "ROI Multi-Point" : (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION
            ? "Half Resolution" : "Full Frame"));

    LOG_INFO("========================================");
    LOG_INFO("Pipeline initialized successfully");
    LOG_INFO("  Raw resolution:  %dx%d", config_.camera.width, config_.camera.height);
    LOG_INFO("  Rect resolution: %dx%d", config_.rect_width, config_.rect_height);
    LOG_INFO("  Trigger: %d Hz", config_.trigger_freq_hz);
    LOG_INFO("  Detect: %s (DLA=%d)", config_.engine_file.c_str(), config_.use_dla);
    LOG_INFO("  Disparity: %s", strategyStr.c_str());
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
        } else {
            LOG_WARN("[Pipeline] grabLoop VPI raw lock failed: L=%d R=%d",
                     (int)stL, (int)stR);
        }

        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rawL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rawR);

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
                VPIImage vizImg = leftDetectorUsesBGR()
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

void Pipeline::stage0_grab_and_rectify(FrameSlot& slot, bool grab_preloaded) {
    NVTX_RANGE("Stage0_GrabRect");

#ifdef HIK_CAMERA_ENABLED
    if (camera_ && !grab_preloaded) {
        // 同步采集: 直接在 pipeline 线程阻塞抓取 (pipelineLoop 全帧模式使用)
        VPIImageData imgDataL, imgDataR;
        VPIStatus stL = vpiImageLockData(
            slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataL);
        VPIStatus stR = vpiImageLockData(
            slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataR);

        GrabResult resL, resR;
        bool grab_ok = false;
        if (stL == VPI_SUCCESS && stR == VPI_SUCCESS) {
            grab_ok = camera_->grabFramePair(
                static_cast<uint8_t*>(imgDataL.buffer.pitch.planes[0].data),
                static_cast<uint8_t*>(imgDataR.buffer.pitch.planes[0].data),
                imgDataL.buffer.pitch.planes[0].pitchBytes,
                imgDataR.buffer.pitch.planes[0].pitchBytes,
                1000, resL, resR);
        } else {
            LOG_WARN("[Pipeline] stage0 raw lock failed: L=%d R=%d",
                     (int)stL, (int)stR);
        }

        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rawL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rawR);

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
    if (colorPipelineEnabled()) {
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

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                LOG_ERROR("BayerToBGR kernel launch failed: %s",
                          cudaGetErrorString(err));
                slot.grab_failed = true;
                NVTX_RANGE_POP();
                return;
            }
            err = cudaStreamSynchronize(streams_.cudaStreamBGR);
            if (err != cudaSuccess) {
                LOG_ERROR("BayerToBGR stream sync failed: %s",
                          cudaGetErrorString(err));
                slot.grab_failed = true;
                NVTX_RANGE_POP();
                return;
            }
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
// Detector helpers
// ===================================================================

TRTDetector* Pipeline::getDetector(int /*frame_id*/) const {
    return detector_.get();
}

cudaStream_t Pipeline::getDLAStream(int /*frame_id*/) const {
    return streams_.cudaStreamDLA;
}

TRTDetector* Pipeline::getRightDetector() const {
    return detector_right_.get();
}

cudaStream_t Pipeline::getRightDLAStream(int /*frame_id*/) const {
    return streams_.cudaStreamDLA_R;
}

bool Pipeline::dualYoloEnabled() const {
    return config_.dual_yolo.enabled && detector_right_;
}

namespace {
bool isBGRFormat(std::string fmt) {
    std::transform(fmt.begin(), fmt.end(), fmt.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return fmt == "bgr";
}

bool isROISubpixelDepthSolver(std::string solver) {
    std::transform(solver.begin(), solver.end(), solver.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return solver == "roi_subpixel_match" ||
           solver == "subpixel" ||
           solver == "multi_point";
}

struct CircleFit2D {
    float cx = 0.0f;
    float cy = 0.0f;
    float radius = 0.0f;
    float confidence = 0.0f;
    bool valid = false;
};

struct CircleFitOptions {
    bool denoise = true;
    int max_roi_pixels = 18000;
    float min_radius_ratio = 0.35f;
    float max_radius_ratio = 1.65f;
    float max_center_shift = 0.0f;
};

struct SubpixelSampleOffset {
    float dx = 0.0f;
    float dy = 0.0f;
};

struct SubpixelDisparityResult {
    bool valid = false;
    bool low_confidence = false;
    float disparity = 0.0f;
    float confidence = 0.0f;
    float stddev = 0.0f;
    int support = 0;
    int attempted = 0;
};

inline float sampleGrayCPU(const uint8_t* img, int pitch, int x, int y, bool denoise)
{
    const uint8_t* row = img + y * pitch;
    if (!denoise) return static_cast<float>(row[x]);

    const uint8_t* prev = img + (y - 1) * pitch;
    const uint8_t* next = img + (y + 1) * pitch;
    const int v =
        4 * static_cast<int>(row[x]) +
        2 * (static_cast<int>(row[x - 1]) + static_cast<int>(row[x + 1]) +
             static_cast<int>(prev[x]) + static_cast<int>(next[x])) +
        static_cast<int>(prev[x - 1]) + static_cast<int>(prev[x + 1]) +
        static_cast<int>(next[x - 1]) + static_cast<int>(next[x + 1]);
    return static_cast<float>(v) * (1.0f / 16.0f);
}

bool solve3x3CPU(
    double A00, double A01, double A02,
    double A10, double A11, double A12,
    double A20, double A21, double A22,
    double b0,  double b1,  double b2,
    double& x0, double& x1, double& x2)
{
    double det = A00 * (A11 * A22 - A12 * A21)
               - A01 * (A10 * A22 - A12 * A20)
               + A02 * (A10 * A21 - A11 * A20);
    if (std::abs(det) < 1e-12) return false;

    const double inv_det = 1.0 / det;
    x0 = (b0 * (A11 * A22 - A12 * A21)
        - A01 * (b1 * A22 - A12 * b2)
        + A02 * (b1 * A21 - A11 * b2)) * inv_det;
    x1 = (A00 * (b1 * A22 - A12 * b2)
        - b0 * (A10 * A22 - A12 * A20)
        + A02 * (A10 * b2 - b1 * A20)) * inv_det;
    x2 = (A00 * (A11 * b2 - b1 * A21)
        - A01 * (A10 * b2 - b1 * A20)
        + b0 * (A10 * A21 - A11 * A20)) * inv_det;
    return true;
}

CircleFit2D fitCircleInRegionCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    int x1, int y1, int x2, int y2,
    float expected_cx, float expected_cy, float expected_radius,
    const CircleFitOptions& options)
{
    CircleFit2D out;
    if (!img || pitch <= 0 || img_w <= 0 || img_h <= 0 ||
        expected_radius < 4.0f) {
        return out;
    }

    const int border = options.denoise ? 2 : 1;
    x1 = std::max(border, x1);
    y1 = std::max(border, y1);
    x2 = std::min(img_w - 1 - border, x2);
    y2 = std::min(img_h - 1 - border, y2);
    if (x2 - x1 < 8 || y2 - y1 < 8) return out;

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int area = roi_w * roi_h;
    const int max_pixels = std::max(256, options.max_roi_pixels);
    const int stride = std::max(
        1, static_cast<int>(std::ceil(std::sqrt(
               static_cast<float>(area) / static_cast<float>(max_pixels)))));

    float max_grad = 0.0f;
    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, options.denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, options.denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, options.denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, options.denoise);
            max_grad = std::max(max_grad, std::sqrt(gx * gx + gy * gy));
        }
    }
    if (max_grad < 8.0f) return out;

    const float grad_thresh = std::max(10.0f, max_grad * 0.25f);
    double sw = 0.0, swx = 0.0, swy = 0.0;
    double swxx = 0.0, swyy = 0.0, swxy = 0.0;
    double swxz = 0.0, swyz = 0.0, swz = 0.0;
    int edge_count = 0;

    for (int y = y1; y <= y2; y += stride) {
        for (int x = x1; x <= x2; x += stride) {
            const float gx = sampleGrayCPU(img, pitch, x + 1, y, options.denoise) -
                             sampleGrayCPU(img, pitch, x - 1, y, options.denoise);
            const float gy = sampleGrayCPU(img, pitch, x, y + 1, options.denoise) -
                             sampleGrayCPU(img, pitch, x, y - 1, options.denoise);
            const float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < grad_thresh) continue;

            const double w = static_cast<double>(mag);
            const double dx = static_cast<double>(x);
            const double dy = static_cast<double>(y);
            const double z = dx * dx + dy * dy;
            sw += w;
            swx += w * dx;
            swy += w * dy;
            swxx += w * dx * dx;
            swyy += w * dy * dy;
            swxy += w * dx * dy;
            swxz += w * dx * z;
            swyz += w * dy * z;
            swz += w * z;
            ++edge_count;
        }
    }

    if (edge_count < 8 || sw <= 0.0) return out;

    double a = 0.0, b = 0.0, c = 0.0;
    if (!solve3x3CPU(swxx, swxy, swx,
                     swxy, swyy, swy,
                     swx,  swy,  sw,
                     swxz, swyz, swz,
                     a, b, c)) {
        return out;
    }

    const float cx = static_cast<float>(a * 0.5);
    const float cy = static_cast<float>(b * 0.5);
    const float r2 = static_cast<float>(c + static_cast<double>(cx) * cx +
                                        static_cast<double>(cy) * cy);
    if (r2 <= 0.0f) return out;
    const float radius = std::sqrt(r2);

    const float min_r = std::max(4.0f, expected_radius * options.min_radius_ratio);
    const float max_r = std::max(min_r + 1.0f, expected_radius * options.max_radius_ratio);
    const float center_dist = std::sqrt((cx - expected_cx) * (cx - expected_cx) +
                                        (cy - expected_cy) * (cy - expected_cy));
    const float max_center_shift = options.max_center_shift > 0.0f
        ? options.max_center_shift
        : std::max(roi_w, roi_h) * 0.5f;
    if (radius < min_r || radius > max_r || center_dist > max_center_shift) return out;

    out.cx = cx;
    out.cy = cy;
    out.radius = radius;
    const float dense_edge_count = static_cast<float>(edge_count * stride * stride);
    const float edge_conf = std::min(1.0f, dense_edge_count / 80.0f);
    const float center_conf = std::max(0.2f, 1.0f - center_dist / std::max(1.0f, max_center_shift));
    out.confidence = edge_conf * center_conf;
    out.valid = true;
    return out;
}

CircleFit2D fitCircleInBBoxCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const Detection& det, bool denoise, int max_roi_pixels)
{
    if (det.width < 8.0f || det.height < 8.0f) return {};

    const int x1 = static_cast<int>(std::floor(det.cx - det.width * 0.55f));
    const int y1 = static_cast<int>(std::floor(det.cy - det.height * 0.55f));
    const int x2 = static_cast<int>(std::ceil(det.cx + det.width * 0.55f));
    const int y2 = static_cast<int>(std::ceil(det.cy + det.height * 0.55f));

    CircleFitOptions options;
    options.denoise = denoise;
    options.max_roi_pixels = max_roi_pixels;
    options.max_center_shift = std::max(det.width, det.height) * 0.65f;
    return fitCircleInRegionCPU(img, pitch, img_w, img_h,
                                x1, y1, x2, y2,
                                det.cx, det.cy,
                                std::max(det.width, det.height) * 0.5f,
                                options);
}

CircleFit2D circleFromDetectionCPU(const Detection& det)
{
    CircleFit2D circle;
    if (det.width < 2.0f || det.height < 2.0f) return circle;
    circle.cx = det.cx;
    circle.cy = det.cy;
    circle.radius = std::max(det.width, det.height) * 0.5f;
    circle.confidence = 1.0f;
    circle.valid = true;
    return circle;
}

Detection detectionFromCircleCPU(const CircleFit2D& circle, const Detection& source)
{
    Detection det;
    det.cx = circle.cx;
    det.cy = circle.cy;
    det.width = std::max(2.0f, circle.radius * 2.0f);
    det.height = det.width;
    det.confidence = source.confidence * std::max(0.2f, circle.confidence);
    det.class_id = source.class_id;
    return det;
}

Detection detectionWithCircleCenterCPU(const CircleFit2D& circle, const Detection& source)
{
    Detection det = source;
    det.cx = circle.cx;
    det.cy = circle.cy;
    det.confidence = source.confidence * std::max(0.2f, circle.confidence);
    return det;
}

float estimateDisparityFromBBoxCPU(
    const Detection& det, float baseline,
    const HybridDepthConfig& depth_cfg, int max_disparity)
{
    if (det.width <= 1.0f || depth_cfg.object_diameter <= 0.01f ||
        baseline <= 0.0f || max_disparity <= 0) {
        return -1.0f;
    }

    const float disp = baseline * det.width * depth_cfg.bbox_scale /
                       depth_cfg.object_diameter;
    return std::clamp(disp, 1.0f, static_cast<float>(max_disparity));
}

CircleFit2D searchCircleOnEpipolarCPU(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const CircleFit2D& source_circle,
    float predicted_cx, float predicted_cy,
    float y_tolerance,
    const PipelineConfig::DualYoloConfig& dual_cfg)
{
    if (!img || pitch <= 0 || !source_circle.valid) return {};

    const float expected_radius = std::max(4.0f, source_circle.radius);
    const float max_width = std::max(32.0f, static_cast<float>(dual_cfg.fallback_max_width_px));
    const float max_roi_half_x = max_width * 0.5f;
    const float radius_pad = expected_radius * 1.05f;
    const float margin = std::max(4.0f, static_cast<float>(dual_cfg.fallback_search_margin_px));
    const float center_half_x = std::min(margin, std::max(4.0f, max_roi_half_x - radius_pad));
    const float roi_half_x = std::min(max_roi_half_x, center_half_x + radius_pad);
    const float roi_half_y = radius_pad + y_tolerance + 2.0f;

    const int x1 = static_cast<int>(std::floor(predicted_cx - roi_half_x));
    const int x2 = static_cast<int>(std::ceil(predicted_cx + roi_half_x));
    const int y1 = static_cast<int>(std::floor(predicted_cy - roi_half_y));
    const int y2 = static_cast<int>(std::ceil(predicted_cy + roi_half_y));

    CircleFitOptions options;
    options.denoise = dual_cfg.roi_denoise;
    options.max_roi_pixels = dual_cfg.circle_max_roi_pixels;
    options.min_radius_ratio = 0.45f;
    options.max_radius_ratio = 1.70f;
    options.max_center_shift = std::sqrt(center_half_x * center_half_x +
                                         y_tolerance * y_tolerance) + 2.0f;

    CircleFit2D circle = fitCircleInRegionCPU(
        img, pitch, img_w, img_h,
        x1, y1, x2, y2,
        predicted_cx, predicted_cy,
        expected_radius,
        options);
    if (!circle.valid) return circle;
    if (std::abs(circle.cx - predicted_cx) > center_half_x ||
        std::abs(circle.cy - predicted_cy) > y_tolerance) {
        return {};
    }
    return circle;
}

bool patchInsideCPU(int img_w, int img_h, int x, int y, int radius, bool denoise)
{
    const int border = denoise ? 1 : 0;
    return x - radius - border >= 0 &&
           y - radius - border >= 0 &&
           x + radius + border < img_w &&
           y + radius + border < img_h;
}

float znccPatchCPU(
    const uint8_t* left, int left_pitch,
    const uint8_t* right, int right_pitch,
    int x_left, int y_left,
    int x_right, int y_right,
    int radius,
    bool denoise)
{
    double sum_l = 0.0;
    double sum_r = 0.0;
    int n = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            sum_l += sampleGrayCPU(left, left_pitch, x_left + dx, y_left + dy, denoise);
            sum_r += sampleGrayCPU(right, right_pitch, x_right + dx, y_right + dy, denoise);
            ++n;
        }
    }
    if (n <= 1) return -2.0f;

    const double mean_l = sum_l / static_cast<double>(n);
    const double mean_r = sum_r / static_cast<double>(n);
    double cov = 0.0;
    double var_l = 0.0;
    double var_r = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const double lv = sampleGrayCPU(left, left_pitch,
                                            x_left + dx, y_left + dy, denoise) - mean_l;
            const double rv = sampleGrayCPU(right, right_pitch,
                                            x_right + dx, y_right + dy, denoise) - mean_r;
            cov += lv * rv;
            var_l += lv * lv;
            var_r += rv * rv;
        }
    }

    const double denom = std::sqrt(var_l * var_r);
    if (denom < 1e-6) return -2.0f;
    return static_cast<float>(cov / denom);
}

std::vector<SubpixelSampleOffset> makeSubpixelSampleOffsetsCPU(
    float radius,
    int max_points,
    int patch_radius)
{
    std::vector<SubpixelSampleOffset> offsets;
    max_points = std::clamp(max_points, 1, 64);
    offsets.reserve(static_cast<size_t>(max_points));
    offsets.push_back({});

    const float usable_radius = std::max(static_cast<float>(patch_radius + 2),
                                         radius * 0.70f);
    const float ring_fracs[] = {0.28f, 0.48f, 0.66f};
    const int angle_count = max_points <= 12 ? 4 : 8;
    constexpr float kPi = 3.14159265358979323846f;

    for (float frac : ring_fracs) {
        const float r = usable_radius * frac;
        for (int i = 0; i < angle_count; ++i) {
            if (static_cast<int>(offsets.size()) >= max_points) return offsets;
            const float angle = 2.0f * kPi * static_cast<float>(i) /
                                static_cast<float>(angle_count);
            offsets.push_back({r * std::cos(angle), r * std::sin(angle)});
        }
    }
    return offsets;
}

float medianOfSortedCPU(const std::vector<float>& values)
{
    if (values.empty()) return 0.0f;
    const size_t mid = values.size() / 2;
    if ((values.size() & 1U) != 0U) return values[mid];
    return 0.5f * (values[mid - 1] + values[mid]);
}

SubpixelDisparityResult refineDisparityByROIMultiPointCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity)
{
    SubpixelDisparityResult result;
    if (!left_img || !right_img || left_pitch <= 0 || right_pitch <= 0 ||
        !left_circle.valid || !right_circle.valid || max_disparity <= 0) {
        return result;
    }

    const float initial_disp = left_circle.cx - right_circle.cx;
    if (!std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity)) {
        return result;
    }

    const int patch_radius = std::clamp(dual_cfg.subpixel_patch_radius, 2, 12);
    const int search_radius = std::max(1, dual_cfg.subpixel_search_radius_px);
    const int max_points = std::clamp(dual_cfg.subpixel_max_points, 1, 64);
    const int min_points = std::clamp(dual_cfg.subpixel_min_points, 1, max_points);
    const float max_delta = std::max(0.5f, dual_cfg.subpixel_max_disp_delta_px);
    const float max_stddev = std::max(0.05f, dual_cfg.subpixel_max_stddev_px);
    const float min_score = std::max(0.10f, dual_cfg.subpixel_min_confidence * 0.60f);
    const float sample_radius = std::min(left_circle.radius, right_circle.radius);
    const auto offsets = makeSubpixelSampleOffsetsCPU(sample_radius,
                                                      max_points,
                                                      patch_radius);

    std::vector<float> disparities;
    std::vector<float> scores;
    disparities.reserve(offsets.size());
    scores.reserve(offsets.size());

    const int d_start = std::max(1, static_cast<int>(std::floor(initial_disp)) - search_radius);
    const int d_end = std::min(max_disparity,
                               static_cast<int>(std::ceil(initial_disp)) + search_radius);
    if (d_start >= d_end) return result;

    const auto score_at = [&](int x_left, int y_left,
                              int y_right,
                              int disparity) -> float {
        const int x_right = static_cast<int>(std::lround(
            static_cast<float>(x_left) - static_cast<float>(disparity)));
        if (!patchInsideCPU(img_w, img_h, x_right, y_right,
                            patch_radius, dual_cfg.roi_denoise)) {
            return -2.0f;
        }
        return znccPatchCPU(left_img, left_pitch, right_img, right_pitch,
                            x_left, y_left, x_right, y_right,
                            patch_radius, dual_cfg.roi_denoise);
    };

    for (const auto& offset : offsets) {
        const int x_left = static_cast<int>(std::lround(left_circle.cx + offset.dx));
        const int y_left = static_cast<int>(std::lround(left_circle.cy + offset.dy));
        const int y_right = y_left;

        if (!patchInsideCPU(img_w, img_h, x_left, y_left,
                            patch_radius, dual_cfg.roi_denoise) ||
            !patchInsideCPU(img_w, img_h,
                            static_cast<int>(std::lround(right_circle.cx + offset.dx)),
                            y_right, patch_radius, dual_cfg.roi_denoise)) {
            continue;
        }

        ++result.attempted;
        float best_score = -2.0f;
        float second_score = -2.0f;
        int best_disp = -1;
        for (int disp = d_start; disp <= d_end; ++disp) {
            const float score = score_at(x_left, y_left, y_right, disp);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_disp = disp;
            } else if (score > second_score) {
                second_score = score;
            }
        }

        if (best_disp < 0 || best_score < min_score) continue;

        float sub_disp = static_cast<float>(best_disp);
        if (best_disp > d_start && best_disp < d_end) {
            const float s_minus = score_at(x_left, y_left, y_right, best_disp - 1);
            const float s_plus = score_at(x_left, y_left, y_right, best_disp + 1);
            const float denom = s_minus - 2.0f * best_score + s_plus;
            if (s_minus > -1.5f && s_plus > -1.5f && denom < -1e-5f) {
                const float delta = std::clamp(
                    0.5f * (s_minus - s_plus) / denom,
                    -1.0f, 1.0f);
                sub_disp += delta;
            }
        }

        const float uniqueness_margin =
            second_score > -1.5f ? best_score - second_score : 1.0f;
        if (uniqueness_margin < 0.01f && best_score < 0.75f) continue;
        if (std::abs(sub_disp - initial_disp) > max_delta) continue;

        disparities.push_back(sub_disp);
        scores.push_back(best_score);
    }

    if (static_cast<int>(disparities.size()) < min_points) {
        result.low_confidence = true;
        return result;
    }

    std::vector<float> sorted = disparities;
    std::sort(sorted.begin(), sorted.end());
    const float median = medianOfSortedCPU(sorted);

    std::vector<float> abs_dev;
    abs_dev.reserve(sorted.size());
    for (float d : disparities) {
        abs_dev.push_back(std::abs(d - median));
    }
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = medianOfSortedCPU(abs_dev);
    const float inlier_gate = std::max(0.60f, mad * 2.5f);

    double sum_disp = 0.0;
    double sum_score = 0.0;
    int inliers = 0;
    for (size_t i = 0; i < disparities.size(); ++i) {
        if (std::abs(disparities[i] - median) > inlier_gate) continue;
        sum_disp += disparities[i];
        sum_score += scores[i];
        ++inliers;
    }
    if (inliers < min_points) {
        result.low_confidence = true;
        return result;
    }

    const float refined_disp = static_cast<float>(sum_disp / static_cast<double>(inliers));
    double var = 0.0;
    for (float d : disparities) {
        if (std::abs(d - median) > inlier_gate) continue;
        const double diff = static_cast<double>(d - refined_disp);
        var += diff * diff;
    }
    result.stddev = static_cast<float>(
        std::sqrt(var / std::max(1.0, static_cast<double>(inliers))));
    result.support = inliers;
    result.disparity = refined_disp;

    if (result.stddev > max_stddev ||
        std::abs(result.disparity - initial_disp) > max_delta ||
        result.disparity <= 0.5f ||
        result.disparity > static_cast<float>(max_disparity)) {
        result.low_confidence = true;
        return result;
    }

    const float support_ratio = static_cast<float>(inliers) /
                                static_cast<float>(std::max(1, max_points));
    const float mean_score = static_cast<float>(sum_score / static_cast<double>(inliers));
    const float score_conf = std::clamp((mean_score - 0.10f) / 0.80f, 0.0f, 1.0f);
    const float consistency = std::clamp(1.0f / (1.0f + result.stddev),
                                         0.0f, 1.0f);
    const float delta_conf = 1.0f -
        std::min(1.0f, std::abs(result.disparity - initial_disp) / max_delta);
    result.confidence = std::clamp(0.35f * support_ratio +
                                   0.35f * score_conf +
                                   0.20f * consistency +
                                   0.10f * delta_conf,
                                   0.0f, 1.0f);
    if (result.confidence < dual_cfg.subpixel_min_confidence) {
        result.low_confidence = true;
        return result;
    }

    result.valid = true;
    return result;
}
}  // namespace

bool Pipeline::leftDetectorUsesBGR() const {
    return isBGRFormat(config_.detector_input_format);
}

bool Pipeline::rightDetectorUsesBGR() const {
    const std::string fmt = config_.dual_yolo.right_input_format.empty()
        ? config_.detector_input_format : config_.dual_yolo.right_input_format;
    return isBGRFormat(fmt);
}

bool Pipeline::colorPipelineEnabled() const {
    return leftDetectorUsesBGR() ||
           (config_.dual_yolo.enabled && rightDetectorUsesBGR());
}

void Pipeline::recordDetectDoneEvents(FrameSlot& slot) const {
    // 左目 lock/enqueue 失败时也要刷新 event，否则下游可能等待到旧 slot 事件。
    cudaEventRecord(slot.evtDetectDone,
                    slot.detection_submitted
                        ? getDLAStream(slot.frame_id)
                        : streams_.cudaStreamGPU);
    if (dualYoloEnabled() && slot.is_detect_frame && slot.right_detection_submitted) {
        cudaEventRecord(slot.evtDetectRightDone, getRightDLAStream(slot.frame_id));
    }
}

void Pipeline::waitDetectDone(cudaStream_t stream, const FrameSlot& slot) const {
    cudaStreamWaitEvent(stream, slot.evtDetectDone, 0);
    if (dualYoloEnabled() && slot.is_detect_frame && slot.right_detection_submitted) {
        cudaStreamWaitEvent(stream, slot.evtDetectRightDone, 0);
    }
}

void Pipeline::collectRightDetections(FrameSlot& slot, int slot_index) {
    slot.detections_right.clear();
    if (!dualYoloEnabled() || !slot.is_detect_frame ||
        !slot.right_detection_submitted) {
        return;
    }

    slot.detections_right = detector_right_->collect(slot_index,
                                                     config_.rect_width,
                                                     config_.rect_height);
}

Pipeline::DualYoloMatchOutput Pipeline::matchDualYoloDetections(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const uint8_t* left_cpu, int left_pitch,
    const uint8_t* right_cpu, int right_pitch,
    int img_width, int img_height,
    DualYoloMatchStats* stats) const
{
    DualYoloMatchStats local_stats;
    local_stats.left_count = static_cast<int>(left_detections.size());
    local_stats.right_count = static_cast<int>(right_detections.size());
    if (right_detections.empty()) {
        local_stats.right_missing = static_cast<int>(left_detections.size());
    }
    if (left_detections.empty()) {
        local_stats.left_missing = static_cast<int>(right_detections.size());
    }

    DualYoloMatchOutput output;
    output.detections = left_detections;
    output.results.resize(output.detections.size());
    for (size_t i = 0; i < output.results.size(); ++i) {
        output.results[i].class_id = output.detections[i].class_id;
        output.results[i].z = -1.0f;
    }

    if ((left_detections.empty() && right_detections.empty()) || !calibration_) {
        if (!calibration_) {
            local_stats.no_candidate =
                static_cast<int>(left_detections.size() + right_detections.size());
        }
        if (stats) *stats = local_stats;
        return output;
    }

    const bool image_available = left_cpu && right_cpu &&
                                 left_pitch > 0 && right_pitch > 0;
    const bool subpixel_depth_enabled =
        config_.dual_yolo.subpixel_enabled &&
        isROISubpixelDepthSolver(config_.dual_yolo.depth_solver);
    const bool use_subpixel_depth =
        subpixel_depth_enabled && image_available;
    if (!image_available &&
        (config_.dual_yolo.center_refine ||
         config_.dual_yolo.fallback_epipolar_search ||
         subpixel_depth_enabled)) {
        local_stats.image_lock_fail =
            static_cast<int>(left_detections.size() + right_detections.size());
    }

    const auto& P1 = calibration_->getProjectionLeft();
    const float focal = static_cast<float>(P1.at<double>(0, 0));
    const float cx0 = static_cast<float>(P1.at<double>(0, 2));
    const float cy0 = static_cast<float>(P1.at<double>(1, 2));
    const float baseline = calibration_->getBaseline();
    const float y_tol = std::max(1.0f, config_.dual_yolo.epipolar_y_tolerance);
    const float max_ratio = std::max(1.0f, config_.dual_yolo.max_size_ratio);

    std::vector<bool> right_used(right_detections.size(), false);
    std::vector<bool> right_blocked_by_left(right_detections.size(), false);
    std::vector<bool> left_has_stereo(left_detections.size(), false);

    auto refine_detection = [&](const uint8_t* img, int pitch,
                                const Detection& det) -> CircleFit2D {
        if (!config_.dual_yolo.center_refine) {
            return circleFromDetectionCPU(det);
        }
        return fitCircleInBBoxCPU(img, pitch, img_width, img_height, det,
                                  config_.dual_yolo.roi_denoise,
                                  config_.dual_yolo.circle_max_roi_pixels);
    };

    auto build_object = [&](const Detection& left_det,
                            const CircleFit2D& left_circle,
                            const CircleFit2D& right_circle,
                            float semantic_conf,
                            Object3D& obj) -> bool {
        const float refined_dy = std::abs(left_circle.cy - right_circle.cy);
        if (refined_dy > y_tol) {
            ++local_stats.epipolar_reject;
            return false;
        }

        const float radius_ratio = std::max(left_circle.radius / right_circle.radius,
                                            right_circle.radius / left_circle.radius);
        if (radius_ratio > max_ratio) {
            ++local_stats.size_reject;
            return false;
        }

        float disparity = left_circle.cx - right_circle.cx;
        if (disparity <= 0.0f) {
            ++local_stats.nonpositive_disparity;
            return false;
        }
        if (disparity > config_.max_disparity) {
            ++local_stats.over_max_disparity;
            return false;
        }

        float disparity_conf = 1.0f;
        if (use_subpixel_depth) {
            ++local_stats.subpixel_attempted;
            const SubpixelDisparityResult refined =
                refineDisparityByROIMultiPointCPU(
                    left_cpu, left_pitch, right_cpu, right_pitch,
                    img_width, img_height,
                    left_circle, right_circle,
                    config_.dual_yolo,
                    config_.max_disparity);
            if (refined.valid) {
                disparity = refined.disparity;
                disparity_conf = std::clamp(0.70f + 0.30f * refined.confidence,
                                            0.0f, 1.0f);
                ++local_stats.subpixel_refined;
            } else {
                ++local_stats.subpixel_rejected;
                if (refined.low_confidence) {
                    ++local_stats.subpixel_low_conf;
                }
            }
        }

        const float z = focal * baseline / disparity;
        if (z < config_.depth.min_depth || z > config_.depth.max_depth) {
            ++local_stats.depth_reject;
            return false;
        }

        const float dy_norm = std::min(1.0f, refined_dy / y_tol);
        const float geom_conf = std::max(0.2f, 1.0f - 0.5f * dy_norm);
        obj.x = (left_circle.cx - cx0) * z / focal;
        obj.y = (left_circle.cy - cy0) * z / focal;
        obj.z = z;
        obj.z_stereo = z;
        obj.confidence = semantic_conf *
                         std::sqrt(left_circle.confidence * right_circle.confidence) *
                         geom_conf *
                         disparity_conf;
        obj.class_id = left_det.class_id;
        obj.depth_method = 1;
        return true;
    };

    auto mark_right_detection_near = [&](const CircleFit2D& right_circle,
                                         int class_id) {
        for (size_t ri = 0; ri < right_detections.size(); ++ri) {
            if (right_used[ri]) continue;
            const Detection& right = right_detections[ri];
            if (right.class_id != class_id) continue;
            const float dx = std::abs(right.cx - right_circle.cx);
            const float dy = std::abs(right.cy - right_circle.cy);
            const float x_tol = std::max(right.width * 0.75f,
                                         right_circle.radius * 1.25f);
            if (dx <= x_tol && dy <= y_tol) {
                right_used[ri] = true;
                right_blocked_by_left[ri] = true;
            }
        }
    };

    auto find_left_detection_near = [&](const CircleFit2D& left_circle,
                                        int class_id) -> int {
        int best_idx = -1;
        float best_dist2 = std::numeric_limits<float>::max();
        for (size_t li = 0; li < left_detections.size(); ++li) {
            if (left_has_stereo[li]) continue;
            const Detection& left = left_detections[li];
            if (left.class_id != class_id) continue;
            const float dx = std::abs(left.cx - left_circle.cx);
            const float dy = std::abs(left.cy - left_circle.cy);
            const float x_tol = std::max(left.width * 0.75f,
                                         left_circle.radius * 1.25f);
            const float y_merge_tol = std::max(y_tol, std::max(left.height * 0.75f,
                                                               left_circle.radius * 1.25f));
            if (dx <= x_tol && dy <= y_merge_tol) {
                const float dist2 = dx * dx + dy * dy;
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_idx = static_cast<int>(li);
                }
            }
        }
        return best_idx;
    };

    auto estimate_fallback_disparity = [&](const Detection& det,
                                           bool allow_track_depth) -> float {
        if (hybrid_depth_) {
            const float z_prior = allow_track_depth
                ? hybrid_depth_->predictDepthForDetection(det)
                : hybrid_depth_->predictPrimaryDepth();
            if (z_prior >= config_.depth.min_depth &&
                z_prior <= config_.depth.max_depth) {
                ++local_stats.fallback_prior_depth;
                return std::clamp(focal * baseline / z_prior,
                                  1.0f,
                                  static_cast<float>(config_.max_disparity));
            }
        }
        return estimateDisparityFromBBoxCPU(
            det, baseline, config_.depth, config_.max_disparity);
    };

    for (size_t li = 0; li < left_detections.size(); ++li) {
        const auto& left = left_detections[li];
        output.results[li].class_id = left.class_id;
        output.results[li].z = -1.0f;

        int best_idx = -1;
        float best_score = 1e9f;

        for (size_t ri = 0; ri < right_detections.size(); ++ri) {
            if (right_used[ri]) continue;
            const auto& right = right_detections[ri];
            if (left.class_id != right.class_id) {
                ++local_stats.class_mismatch;
                continue;
            }
            if (left.width <= 1.0f || left.height <= 1.0f ||
                right.width <= 1.0f || right.height <= 1.0f) {
                ++local_stats.invalid_box;
                continue;
            }

            const float disparity = left.cx - right.cx;
            if (disparity <= 0.0f) {
                ++local_stats.nonpositive_disparity;
                continue;
            }
            if (disparity > config_.max_disparity) {
                ++local_stats.over_max_disparity;
                continue;
            }

            const float dy = std::abs(left.cy - right.cy);
            const float candidate_y_tol =
                std::max(y_tol, 0.35f * std::max(left.height, right.height));
            if (dy > candidate_y_tol) {
                ++local_stats.epipolar_reject;
                continue;
            }

            const float w_ratio = std::max(left.width / right.width,
                                           right.width / left.width);
            const float h_ratio = std::max(left.height / right.height,
                                           right.height / left.height);
            if (w_ratio > max_ratio || h_ratio > max_ratio) {
                ++local_stats.size_reject;
                continue;
            }

            const float size_cost = std::abs(std::log(w_ratio)) +
                                    std::abs(std::log(h_ratio));
            const float score = dy / candidate_y_tol + size_cost -
                                0.25f * right.confidence;
            if (score < best_score) {
                best_score = score;
                best_idx = static_cast<int>(ri);
            }
        }

        if (best_idx < 0) {
            ++local_stats.no_candidate;
            continue;
        }
        const auto& right = right_detections[best_idx];
        CircleFit2D left_circle = refine_detection(left_cpu, left_pitch, left);
        CircleFit2D right_circle = refine_detection(right_cpu, right_pitch, right);
        if (!left_circle.valid || !right_circle.valid) {
            ++local_stats.circle_fit_fail;
            continue;
        }

        Object3D obj;
        const float semantic_conf = std::sqrt(left.confidence * right.confidence);
        if (!build_object(left, left_circle, right_circle, semantic_conf, obj)) {
            continue;
        }

        output.detections[li] = detectionWithCircleCenterCPU(left_circle, left);
        output.results[li] = obj;
        right_used[best_idx] = true;
        left_has_stereo[li] = true;
        ++local_stats.matched;
    }

    if (config_.dual_yolo.fallback_epipolar_search && image_available) {
        for (size_t li = 0; li < left_detections.size(); ++li) {
            if (left_has_stereo[li]) continue;

            const Detection& left = left_detections[li];
            ++local_stats.fallback_attempted;
            ++local_stats.fallback_left_to_right;

            CircleFit2D left_circle = refine_detection(left_cpu, left_pitch, left);
            if (!left_circle.valid) {
                ++local_stats.circle_fit_fail;
                ++local_stats.fallback_failed;
                continue;
            }

            const float expected_disp = estimate_fallback_disparity(left, true);
            if (expected_disp <= 0.0f) {
                ++local_stats.invalid_box;
                ++local_stats.fallback_failed;
                continue;
            }

            CircleFit2D right_circle = searchCircleOnEpipolarCPU(
                right_cpu, right_pitch, img_width, img_height,
                left_circle,
                left_circle.cx - expected_disp,
                left_circle.cy,
                y_tol,
                config_.dual_yolo);
            if (!right_circle.valid) {
                ++local_stats.fallback_failed;
                continue;
            }

            Object3D obj;
            if (!build_object(left, left_circle, right_circle, left.confidence, obj)) {
                ++local_stats.fallback_failed;
                continue;
            }

            output.detections[li] = detectionWithCircleCenterCPU(left_circle, left);
            output.results[li] = obj;
            left_has_stereo[li] = true;
            mark_right_detection_near(right_circle, left.class_id);
            ++local_stats.matched;
            ++local_stats.fallback_matched;
        }

        for (size_t ri = 0; ri < right_detections.size(); ++ri) {
            if (right_used[ri] || right_blocked_by_left[ri]) continue;

            const Detection& right = right_detections[ri];
            ++local_stats.fallback_attempted;
            ++local_stats.fallback_right_to_left;

            CircleFit2D right_circle = refine_detection(right_cpu, right_pitch, right);
            if (!right_circle.valid) {
                ++local_stats.circle_fit_fail;
                ++local_stats.fallback_failed;
                continue;
            }

            const float expected_disp = estimate_fallback_disparity(right, false);
            if (expected_disp <= 0.0f) {
                ++local_stats.invalid_box;
                ++local_stats.fallback_failed;
                continue;
            }

            CircleFit2D left_circle = searchCircleOnEpipolarCPU(
                left_cpu, left_pitch, img_width, img_height,
                right_circle,
                right_circle.cx + expected_disp,
                right_circle.cy,
                y_tol,
                config_.dual_yolo);
            if (!left_circle.valid) {
                ++local_stats.fallback_failed;
                continue;
            }

            Detection left_proxy = detectionFromCircleCPU(left_circle, right);
            Object3D obj;
            if (!build_object(left_proxy, left_circle, right_circle,
                              right.confidence, obj)) {
                ++local_stats.fallback_failed;
                continue;
            }

            int left_idx = find_left_detection_near(left_circle, right.class_id);
            if (left_idx >= 0) {
                const Detection& left_source = left_detections[left_idx];
                output.detections[left_idx] =
                    detectionWithCircleCenterCPU(left_circle, left_source);
                output.results[left_idx] = obj;
                left_has_stereo[left_idx] = true;
            } else if (!left_detections.empty()) {
                ++local_stats.fallback_failed;
                continue;
            } else {
                output.detections.push_back(left_proxy);
                output.results.push_back(obj);
            }
            right_used[ri] = true;
            ++local_stats.matched;
            ++local_stats.fallback_matched;
        }
    }

    if (stats) *stats = local_stats;
    return output;
}

void Pipeline::stage1_detect(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage1_Detect");
    slot.detection_submitted = false;
    slot.right_detection_submitted = false;

    auto* det = getDetector(slot.frame_id);
    auto stream = getDLAStream(slot.frame_id);
    cudaStreamWaitEvent(stream, slot.evtRectDone, 0);

    // 从 VPI Image 获取 GPU 指针，传给 TensorRT
    // BGR 模式: 使用校正后 BGR 图像; Gray 模式: 使用校正后灰度图
    VPIImage detectImg = leftDetectorUsesBGR()
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
    const bool submitted = det->enqueue(slot_index, gpu_ptr, pitch,
                                        config_.rect_width, config_.rect_height,
                                        stream);

    vpiImageUnlock(detectImg);
    if (!submitted) {
        LOG_WARN("stage1_detect: left TRT enqueue failed");
        NVTX_RANGE_POP();
        return;
    }
    slot.detection_submitted = true;

    if (dualYoloEnabled() && slot.is_detect_frame) {
        auto* detR = getRightDetector();
        auto streamR = getRightDLAStream(slot.frame_id);
        cudaStreamWaitEvent(streamR, slot.evtRectDone, 0);

        VPIImage detectImgR = rightDetectorUsesBGR()
                             ? slot.rectBGR_vpiR : slot.rectGray_vpiR;
        VPIImageData imgDataR;
        VPIStatus stR = vpiImageLockData(detectImgR, VPI_LOCK_READ,
                                         VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,
                                         &imgDataR);
        if (stR != VPI_SUCCESS) {
            LOG_WARN("stage1_detect: right vpiImageLockData failed (%d)", (int)stR);
            NVTX_RANGE_POP();
            return;
        }

        void* gpu_ptr_r = imgDataR.buffer.pitch.planes[0].data;
        int pitch_r = imgDataR.buffer.pitch.planes[0].pitchBytes;

        const bool submittedR = detR->enqueue(slot_index, gpu_ptr_r, pitch_r,
                                              config_.rect_width,
                                              config_.rect_height,
                                              streamR);

        vpiImageUnlock(detectImgR);
        if (!submittedR) {
            LOG_WARN("stage1_detect: right TRT enqueue failed");
            NVTX_RANGE_POP();
            return;
        }
        slot.right_detection_submitted = true;
    }
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
    if (slot.detection_submitted) {
        slot.detections = getDetector(slot.frame_id)->collect(
            slot_index, config_.rect_width, config_.rect_height);
    } else {
        slot.detections.clear();
    }
    collectRightDetections(slot, slot_index);

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
    if (slot.detection_submitted) {
        slot.detections = det->collect(slot_index,
                                       config_.rect_width, config_.rect_height);
    } else {
        slot.detections.clear();
    }
    collectRightDetections(slot, slot_index);

    const bool can_try_right_only =
        dualYoloEnabled() && !slot.detections_right.empty() &&
        config_.dual_yolo.fallback_epipolar_search;
    if (slot.detections.empty() && !can_try_right_only) {
        // 无检测: 仅 Kalman 预测
        if (hybrid_depth_) {
            slot.results = hybrid_depth_->predictOnly();
        }
        NVTX_RANGE_POP();
        return;
    }

    auto has_valid_stereo = [this](const Object3D& obj) {
        return obj.z > 0.0f && obj.confidence > config_.depth.min_confidence;
    };

    auto count_valid = [&](const std::vector<Object3D>& results) {
        int n = 0;
        for (const auto& obj : results) {
            if (has_valid_stereo(obj)) ++n;
        }
        return n;
    };

    std::vector<stereo3d::Object3D> roi_results;
    std::vector<Detection> fusion_detections = slot.detections;
    bool need_roi_texture_match = true;

    if (dualYoloEnabled()) {
        ScopedTimer tdual("Stage2_DualYoloMatch");
        DualYoloMatchStats match_stats;
        DualYoloMatchOutput semantic_match;

        const bool need_host_images =
            config_.dual_yolo.center_refine ||
            config_.dual_yolo.fallback_epipolar_search ||
            (config_.dual_yolo.subpixel_enabled &&
             isROISubpixelDepthSolver(config_.dual_yolo.depth_solver));
        if (need_host_images) {
            VPIImageData hostDataL, hostDataR;
            VPIStatus stL = vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ,
                                             VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                             &hostDataL);
            VPIStatus stR = vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ,
                                             VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                             &hostDataR);
            if (stL == VPI_SUCCESS && stR == VPI_SUCCESS) {
                const uint8_t* leftCPU = static_cast<const uint8_t*>(
                    hostDataL.buffer.pitch.planes[0].data);
                const uint8_t* rightCPU = static_cast<const uint8_t*>(
                    hostDataR.buffer.pitch.planes[0].data);
                int leftPitch = hostDataL.buffer.pitch.planes[0].pitchBytes;
                int rightPitch = hostDataR.buffer.pitch.planes[0].pitchBytes;
                semantic_match = matchDualYoloDetections(
                    slot.detections, slot.detections_right,
                    leftCPU, leftPitch, rightCPU, rightPitch,
                    config_.rect_width, config_.rect_height,
                    &match_stats);
            } else {
                semantic_match = matchDualYoloDetections(
                    slot.detections, slot.detections_right,
                    nullptr, 0, nullptr, 0,
                    config_.rect_width, config_.rect_height,
                    &match_stats);
            }
            if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiL);
            if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiR);
        } else {
            semantic_match = matchDualYoloDetections(
                slot.detections, slot.detections_right,
                nullptr, 0, nullptr, 0,
                config_.rect_width, config_.rect_height,
                &match_stats);
        }

        int semantic_valid = count_valid(semantic_match.results);
        globalPerf().record("Stage2_DualYoloMatch", tdual.elapsedMs());

        if (config_.dual_yolo.log_matches &&
            config_.stats_interval > 0 &&
            slot.frame_id % config_.stats_interval == 0) {
            LOG_INFO("[DualYOLO] frame=%d left=%d right=%d matches=%d valid=%d "
                     "missL=%d missR=%d fb=%d/%d fail=%d prior=%d l2r=%d r2l=%d "
                     "noCand=%d cls=%d badBox=%d d<=0=%d dMax=%d epi=%d "
                     "size=%d circle=%d subpx=%d/%d rej=%d low=%d depth=%d lock=%d",
                     slot.frame_id,
                     match_stats.left_count,
                     match_stats.right_count,
                     match_stats.matched,
                     semantic_valid,
                     match_stats.left_missing,
                     match_stats.right_missing,
                     match_stats.fallback_matched,
                     match_stats.fallback_attempted,
                     match_stats.fallback_failed,
                     match_stats.fallback_prior_depth,
                     match_stats.fallback_left_to_right,
                     match_stats.fallback_right_to_left,
                     match_stats.no_candidate,
                     match_stats.class_mismatch,
                     match_stats.invalid_box,
                     match_stats.nonpositive_disparity,
                     match_stats.over_max_disparity,
                     match_stats.epipolar_reject,
                     match_stats.size_reject,
                     match_stats.circle_fit_fail,
                     match_stats.subpixel_refined,
                     match_stats.subpixel_attempted,
                     match_stats.subpixel_rejected,
                     match_stats.subpixel_low_conf,
                     match_stats.depth_reject,
                     match_stats.image_lock_fail);
        }

        if (config_.dual_yolo.use_for_depth) {
            fusion_detections = std::move(semantic_match.detections);
            roi_results = std::move(semantic_match.results);
            need_roi_texture_match =
                config_.dual_yolo.fallback_to_roi_match &&
                semantic_valid < static_cast<int>(fusion_detections.size());

            if (!config_.dual_yolo.fallback_to_roi_match &&
                semantic_valid < static_cast<int>(fusion_detections.size())) {
                // 双路 YOLO/极线 fallback 找不到可靠视差时，只输出预测，不用单目 bbox 更新深度。
                std::vector<Detection> valid_detections;
                std::vector<Object3D> valid_results;
                const size_t n = std::min(fusion_detections.size(), roi_results.size());
                valid_detections.reserve(n);
                valid_results.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    if (!has_valid_stereo(roi_results[i])) continue;
                    valid_detections.push_back(fusion_detections[i]);
                    valid_results.push_back(roi_results[i]);
                }
                fusion_detections = std::move(valid_detections);
                roi_results = std::move(valid_results);
            }
        }
    }

    if (need_roi_texture_match) {
        if (!roi_matcher_) {
            LOG_ERROR("ROI texture match requested but ROIStereoMatcher is not initialized");
            if (hybrid_depth_) slot.results = hybrid_depth_->predictOnly();
            NVTX_RANGE_POP();
            return;
        }

        // 获取校正后灰度左右图 GPU 指针 (color pipeline → rectGray)
        VPIImageData imgDataL, imgDataR;
        VPIStatus stL, stR;
        {
            ScopedTimer tvl("Stage2_VPILock");
            stL = vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ,
                                   VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
            stR = vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ,
                                   VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);
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

        std::vector<stereo3d::Object3D> texture_results;
        {
            ScopedTimer troi("Stage2_ROIMatch");
            texture_results = roi_matcher_->match(
                leftPtr, leftPitch, rightPtr, rightPitch,
                config_.rect_width, config_.rect_height,
                fusion_detections, streams_.cudaStreamFuse);
            globalPerf().record("Stage2_ROIMatch", troi.elapsedMs());
        }

        vpiImageUnlock(slot.rectGray_vpiL);
        vpiImageUnlock(slot.rectGray_vpiR);

        if (config_.dual_yolo.use_for_depth && !roi_results.empty()) {
            const size_t n = std::min(roi_results.size(), texture_results.size());
            for (size_t i = 0; i < n; ++i) {
                if ((roi_results[i].z <= 0.0f ||
                     roi_results[i].confidence <= config_.depth.min_confidence) &&
                    texture_results[i].z > 0.0f &&
                    texture_results[i].confidence > config_.depth.min_confidence) {
                    roi_results[i] = texture_results[i];
                }
            }
        } else {
            roi_results = std::move(texture_results);
        }
    }

    slot.detections = std::move(fusion_detections);
    if (slot.detections.empty()) {
        if (hybrid_depth_) {
            slot.results = hybrid_depth_->predictOnly();
        }
        NVTX_RANGE_POP();
        return;
    }

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

    if (!roi_matcher_) {
        LOG_ERROR("Tracker ROI fuse requested but ROIStereoMatcher is not initialized");
        if (hybrid_depth_) slot.results = hybrid_depth_->predictOnly();
        NVTX_RANGE_POP();
        return;
    }

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
                VPIImage vizImg = leftDetectorUsesBGR()
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
        if (result_callback_) result_callback_(slot.frame_id, slot.results);
        next_fuse_frame++;
    }

    LOG_INFO("ROI Pipeline loop exited");
}

void Pipeline::printPerfReport() const {
    globalPerf().printReport();
}

}  // namespace stereo3d
