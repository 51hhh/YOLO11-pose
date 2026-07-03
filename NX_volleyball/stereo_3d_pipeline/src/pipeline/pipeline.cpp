/**
 * @file pipeline.cpp
 * @brief Pipeline initialization and stage implementations.
 *
 * Source layout:
 *   - pipeline_loops.cpp: full-frame and ROI pipeline schedulers.
 *   - pipeline_async_roi.cpp: async ROI Stage2 snapshot, worker, and expiry.
 *   - pipeline.cpp: init/start/stop, Stage0/1/2/3 bodies, ROI matching, and
 *     result callbacks.
 *
 * CUDA/VPI synchronization is expressed with slot-level events:
 *   evtRectDone        -> Stage0 rectification gate for detector/ROI work.
 *   evtDetectDone      -> left YOLO completion gate.
 *   evtDetectRightDone -> right YOLO completion gate when dual YOLO is active.
 */

#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "../stereo/depth_candidate_builder.h"
#include "../stereo/depth_match_contract.h"
#include "../stereo/neural_feature_matcher.h"
#include "../stereo/roi_feature_match_cpu.h"
#include "../stereo/roi_feature_match_gpu.h"
#include "../stereo/roi_feature_validation.h"
#include "../stereo/roi_geometry_cpu.h"
#include "../stereo/roi_patch_match_cpu.h"
#include "../track/nanotrack_trt.h"
#include "../track/mixformer_trt.h"
#include "../utils/logger.h"
#include <vpi/algo/ConvertImageFormat.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <filesystem>
#include <fstream>
#include <sstream>

// 自定义 CUDA Bayer→BGR8 kernel (bilinear 插值, 在 detect_preprocess.cu 中)
extern "C" void launchBayerToBGR8(const unsigned char* bayer, unsigned char* bgr,
                                   int width, int height,
                                   int bayer_pitch, int bgr_pitch,
                                   cudaStream_t stream);

namespace stereo3d {

namespace {

void limitDetectionsByConfidence(std::vector<Detection>& detections,
                                 int max_detections) {
    if (max_detections <= 0 ||
        detections.size() <= static_cast<size_t>(max_detections)) {
        return;
    }
    auto by_confidence = [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    };
    const auto keep_end = detections.begin() + max_detections;
    std::nth_element(detections.begin(), keep_end, detections.end(), by_confidence);
    detections.resize(static_cast<size_t>(max_detections));
    std::sort(detections.begin(), detections.end(), by_confidence);
}

StereoRoiPairGateConfig makeStereoRoiPairGateConfig(
    const PipelineConfig& config) {
    StereoRoiPairGateConfig gate;
    gate.max_disparity = config.max_disparity;
    gate.epipolar_y_tolerance = config.dual_yolo.epipolar_y_tolerance;
    gate.max_size_ratio = config.dual_yolo.max_size_ratio;
    gate.adaptive_y_ratio = 0.35f;
    gate.min_shifted_iou = config.dual_yolo.min_shifted_iou;
    return gate;
}

}  // namespace

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
    destroyAsyncRoiStage2();
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
    config_.tracker.detect_interval = std::max(1, config_.tracker.detect_interval);

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

    // 5. 分配 VPI Images (CPU 写入, CUDA/VPI 后续复用的 host-mapped buffers)
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
                !cachePtr("tempBGR_R", slots_[i].tempBGR_R, slots_[i].tempBGR_R_gpu) ||
                !cachePtr("rectBGR_vpiL", slots_[i].rectBGR_vpiL, slots_[i].rectBGR_L_gpu) ||
                !cachePtr("rectBGR_vpiR", slots_[i].rectBGR_vpiR, slots_[i].rectBGR_R_gpu)) {
                return false;
            }
        }
    }

    // 双目 ROI/GPU 热路径直接使用校正灰度 CUDA 指针，避免 Stage2 每帧 VPI lock/unlock。
    LOG_INFO("Caching CUDA pointers for rectified gray images...");
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
            if (!out.data || out.pitchBytes <= 0) {
                LOG_ERROR("Invalid %s CUDA pointer for slot %d", name, i);
                return false;
            }
            return true;
        };
        if (!cachePtr("rectGray_vpiL", slots_[i].rectGray_vpiL,
                      slots_[i].rectGray_L_gpu) ||
            !cachePtr("rectGray_vpiR", slots_[i].rectGray_vpiR,
                      slots_[i].rectGray_R_gpu)) {
            return false;
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
                 "modes[bbox=%d bboxEdge=%d circle=%d circleEdge=%d roiCent=%d "
                 "radial=%d edgePair=%d cornerPts=%d texturePts=%d binaryPts=%d "
                 "orbPts=%d briskPts=%d akazePts=%d siftPts=%d colorPatch=%d "
                 "colorEdge=%d cudaTmpl=%d cudaBM=%d cudaSGM=%d centerPatch=%d "
                 "subpx=%d fallback=%d tmpl=%d featFb=%d], "
                 "epipolar_fallback=%d, gpu=%d, center_refine=%d, roi_denoise=%d, "
                 "depth_solver=%s, subpx=%d patch=%d search=%d pts=%d "
                 "delta=%.2fpx ratio=%.3f depthJump=%.2fm budget=%.2fms",
                 right_engine.c_str(),
                 config_.dual_yolo.use_for_depth,
                 config_.dual_yolo.fallback_to_roi_match,
                 config_.dual_yolo.depth_bbox_pair,
                 config_.dual_yolo.depth_bbox_edges,
                 config_.dual_yolo.depth_circle_center,
                 config_.dual_yolo.depth_circle_edges,
                 config_.dual_yolo.depth_roi_edge_centroid,
                 config_.dual_yolo.depth_roi_radial_center,
                 config_.dual_yolo.depth_roi_edge_pair_center,
                 config_.dual_yolo.depth_roi_corner_points,
                 config_.dual_yolo.depth_roi_texture_points,
                 config_.dual_yolo.depth_roi_binary_points,
                 config_.dual_yolo.depth_roi_orb_points,
                 config_.dual_yolo.depth_roi_brisk_points,
                 config_.dual_yolo.depth_roi_akaze_points,
                 config_.dual_yolo.depth_roi_sift_points,
                 config_.dual_yolo.depth_roi_iou_region_color_patch,
                 config_.dual_yolo.depth_roi_patch_iou_color_edge,
                 config_.dual_yolo.depth_roi_cuda_template_match,
                 config_.dual_yolo.depth_roi_cuda_stereo_bm,
                 config_.dual_yolo.depth_roi_cuda_stereo_sgm,
                 config_.dual_yolo.depth_roi_center_patch,
                 config_.dual_yolo.depth_roi_subpixel,
                 config_.dual_yolo.depth_epipolar_fallback,
                 config_.dual_yolo.depth_fallback_template,
                 config_.dual_yolo.depth_fallback_feature_points,
                 config_.dual_yolo.fallback_epipolar_search,
                 config_.dual_yolo.gpu_candidate_refine,
                 config_.dual_yolo.center_refine,
                 config_.dual_yolo.roi_denoise,
                 config_.dual_yolo.depth_solver.c_str(),
                 config_.dual_yolo.subpixel_enabled,
                 config_.dual_yolo.subpixel_patch_radius,
                 config_.dual_yolo.subpixel_search_radius_px,
                 config_.dual_yolo.subpixel_max_points,
                 config_.dual_yolo.subpixel_max_disp_delta_px,
                 config_.dual_yolo.subpixel_max_disp_delta_ratio,
                 config_.dual_yolo.subpixel_max_depth_delta_m,
                 config_.dual_yolo.subpixel_time_budget_ms);
    }

    if (config_.neural_features.enabled) {
        const bool has_neural_engine =
            !config_.neural_features.extractor_engine_path.empty() ||
            !config_.neural_features.fused_engine_path.empty();
        if (!has_neural_engine) {
            LOG_WARN("neural_feature_matching.enabled=true but no extractor_engine_path "
                     "or fused_engine_path is configured; disabling neural matcher for this run");
            config_.neural_features.enabled = false;
        } else {
            const auto& P1 = calibration_->getProjectionLeft();
            const float focal = static_cast<float>(P1.at<double>(0, 0));
            const float baseline = calibration_->getBaseline();
            neural_feature_matcher_ = std::make_unique<NeuralFeatureMatcher>();
            if (!neural_feature_matcher_->init(config_.neural_features,
                                               focal,
                                               baseline,
                                               config_.max_disparity)) {
                LOG_ERROR("NeuralFeatureMatcher init failed");
                return false;
            }
            if (neural_feature_matcher_->requiresBgrInput() &&
                !colorPipelineEnabled()) {
                LOG_WARN("NeuralFeatureMatcher engine requires 3/6-channel BGR input, "
                         "but the current detector input format keeps the pipeline in "
                         "gray-only mode; roi_neural_feature will report unsupported "
                         "input until BGR input is enabled");
            }
            LOG_INFO("NeuralFeatureMatcher engines loaded; TensorRT ROI matches "
                     "will be used as roi_neural_feature candidates");
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

    }

    // 9. 初始化立体匹配 (根据策略选择)
    if (config_.disparity_strategy == DisparityStrategy::ROI_ONLY &&
        config_.detection_only) {
        LOG_INFO("Detection-only mode: skipping ROI stereo/depth matcher initialization");
    } else if (config_.disparity_strategy == DisparityStrategy::ROI_ONLY) {
        const auto& P1 = calibration_->getProjectionLeft();
        float focal = static_cast<float>(P1.at<double>(0, 0));
        float cx    = static_cast<float>(P1.at<double>(0, 2));
        float cy    = static_cast<float>(P1.at<double>(1, 2));

        auto is_subpixel_solver = [](std::string solver) {
            std::transform(solver.begin(), solver.end(), solver.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return solver == "roi_subpixel_match" ||
                   solver == "subpixel" ||
                   solver == "multi_point";
        };
        const bool dual_yolo_depth_modes_enabled =
            config_.dual_yolo.depth_bbox_pair ||
            config_.dual_yolo.depth_bbox_edges ||
            (config_.dual_yolo.depth_circle_center &&
             config_.dual_yolo.center_refine) ||
            (config_.dual_yolo.depth_circle_edges &&
             config_.dual_yolo.center_refine) ||
            (config_.dual_yolo.depth_roi_edge_centroid &&
             config_.dual_yolo.center_refine) ||
            config_.dual_yolo.depth_roi_radial_center ||
            config_.dual_yolo.depth_roi_edge_pair_center ||
            config_.dual_yolo.depth_roi_corner_points ||
            config_.dual_yolo.depth_roi_texture_points ||
            config_.dual_yolo.depth_roi_binary_points ||
            config_.dual_yolo.depth_roi_orb_points ||
            config_.dual_yolo.depth_roi_brisk_points ||
            config_.dual_yolo.depth_roi_akaze_points ||
            config_.dual_yolo.depth_roi_sift_points ||
            config_.dual_yolo.depth_roi_iou_region_color_patch ||
            config_.dual_yolo.depth_roi_patch_iou_color_edge ||
            config_.dual_yolo.depth_roi_cuda_template_match ||
            config_.dual_yolo.depth_roi_cuda_stereo_bm ||
            config_.dual_yolo.depth_roi_cuda_stereo_sgm ||
            config_.neural_features.enabled ||
            (config_.dual_yolo.depth_roi_center_patch &&
             config_.dual_yolo.center_refine) ||
            (config_.dual_yolo.depth_roi_subpixel &&
             config_.dual_yolo.subpixel_enabled &&
             is_subpixel_solver(config_.dual_yolo.depth_solver)) ||
            (config_.dual_yolo.depth_epipolar_fallback &&
             config_.dual_yolo.fallback_epipolar_search) ||
            (config_.dual_yolo.depth_fallback_template &&
             config_.dual_yolo.fallback_epipolar_search) ||
            (config_.dual_yolo.depth_fallback_feature_points &&
             config_.dual_yolo.fallback_epipolar_search);
        const bool dual_yolo_depth_only =
            config_.dual_yolo.enabled &&
            config_.dual_yolo.use_for_depth &&
            dual_yolo_depth_modes_enabled &&
            !config_.dual_yolo.fallback_to_roi_match;
        const bool dual_yolo_gpu_batch_modes_enabled =
            (config_.dual_yolo.depth_circle_center &&
             config_.dual_yolo.center_refine) ||
            (config_.dual_yolo.depth_circle_edges &&
             config_.dual_yolo.center_refine) ||
            (config_.dual_yolo.depth_roi_edge_centroid &&
             config_.dual_yolo.center_refine) ||
            config_.dual_yolo.depth_roi_radial_center ||
            config_.dual_yolo.depth_roi_edge_pair_center ||
            config_.dual_yolo.depth_roi_corner_points ||
            config_.dual_yolo.depth_roi_texture_points ||
            config_.dual_yolo.depth_roi_binary_points ||
            config_.dual_yolo.depth_roi_iou_region_color_patch ||
            config_.dual_yolo.depth_roi_patch_iou_color_edge ||
            (config_.dual_yolo.depth_roi_center_patch &&
             config_.dual_yolo.center_refine) ||
            (config_.dual_yolo.depth_roi_subpixel &&
             config_.dual_yolo.subpixel_enabled &&
             is_subpixel_solver(config_.dual_yolo.depth_solver));
        if (config_.dual_yolo.enabled &&
            config_.dual_yolo.use_for_depth &&
            dual_yolo_gpu_batch_modes_enabled &&
            config_.dual_yolo.gpu_candidate_refine) {
            DualYoloDepthGpuConfig gpu_cfg;
            gpu_cfg.max_disparity = config_.max_disparity;
            gpu_cfg.patch_radius = config_.dual_yolo.subpixel_patch_radius;
            gpu_cfg.search_radius_px = config_.dual_yolo.subpixel_search_radius_px;
            gpu_cfg.max_points = config_.dual_yolo.subpixel_max_points;
            gpu_cfg.min_points = config_.dual_yolo.subpixel_min_points;
            gpu_cfg.circle_max_roi_pixels = config_.dual_yolo.circle_max_roi_pixels;
            gpu_cfg.min_confidence = config_.dual_yolo.subpixel_min_confidence;
            gpu_cfg.max_disp_delta_px = config_.dual_yolo.subpixel_max_disp_delta_px;
            gpu_cfg.max_disp_delta_ratio = config_.dual_yolo.subpixel_max_disp_delta_ratio;
            gpu_cfg.max_depth_delta_m = config_.dual_yolo.subpixel_max_depth_delta_m;
            gpu_cfg.max_stddev_px = config_.dual_yolo.subpixel_max_stddev_px;
            gpu_cfg.epipolar_y_tolerance = config_.dual_yolo.epipolar_y_tolerance;
            gpu_cfg.feature_y_tolerance_px = config_.dual_yolo.feature_y_tolerance_px;
            gpu_cfg.feature_y_slope = config_.dual_yolo.feature_y_slope;
            gpu_cfg.feature_y_offset_px = config_.dual_yolo.feature_y_offset_px;
            gpu_cfg.feature_reverse_check_px = config_.dual_yolo.feature_reverse_check_px;
            gpu_cfg.feature_overlap_scale = config_.dual_yolo.feature_overlap_scale;
            gpu_cfg.feature_mad_scale = config_.dual_yolo.feature_mad_scale;
            gpu_cfg.feature_ransac_gate_px = config_.dual_yolo.feature_ransac_gate_px;
            gpu_cfg.feature_sphere_radius_m =
                std::max(0.0f, config_.depth.object_diameter * 0.5f);
            gpu_cfg.feature_sphere_radius_scale =
                config_.dual_yolo.feature_sphere_radius_scale;
            gpu_cfg.feature_sphere_margin_m = config_.dual_yolo.feature_sphere_margin_m;
            gpu_cfg.min_depth = config_.depth.min_depth;
            gpu_cfg.max_depth = config_.depth.max_depth;
            gpu_cfg.compute_center_patch =
                config_.dual_yolo.depth_roi_center_patch &&
                config_.dual_yolo.center_refine;
            gpu_cfg.compute_multi_point =
                config_.dual_yolo.depth_roi_subpixel &&
                config_.dual_yolo.subpixel_enabled &&
                is_subpixel_solver(config_.dual_yolo.depth_solver);
            gpu_cfg.compute_corner_points = config_.dual_yolo.depth_roi_corner_points;
            gpu_cfg.compute_texture_points = config_.dual_yolo.depth_roi_texture_points;
            gpu_cfg.compute_binary_points = config_.dual_yolo.depth_roi_binary_points;
            gpu_cfg.compute_orb_points = false;
            gpu_cfg.compute_brisk_points = false;
            gpu_cfg.compute_akaze_points = false;
            gpu_cfg.compute_sift_points = false;
            gpu_cfg.compute_iou_region_color_patch =
                config_.dual_yolo.depth_roi_iou_region_color_patch;
            gpu_cfg.compute_patch_iou_color_edge =
                config_.dual_yolo.depth_roi_patch_iou_color_edge;
            gpu_cfg.compute_geometry =
                (config_.dual_yolo.center_refine &&
                 (config_.dual_yolo.depth_circle_center ||
                  config_.dual_yolo.depth_circle_edges ||
                  config_.dual_yolo.depth_roi_edge_centroid ||
                  gpu_cfg.compute_center_patch ||
                  gpu_cfg.compute_multi_point)) ||
                config_.dual_yolo.depth_roi_radial_center ||
                config_.dual_yolo.depth_roi_edge_pair_center ||
                gpu_cfg.compute_corner_points ||
                gpu_cfg.compute_texture_points ||
                gpu_cfg.compute_binary_points ||
                gpu_cfg.compute_orb_points ||
                gpu_cfg.compute_brisk_points ||
                gpu_cfg.compute_akaze_points ||
                gpu_cfg.compute_sift_points ||
                gpu_cfg.compute_iou_region_color_patch ||
                gpu_cfg.compute_patch_iou_color_edge;
            const bool gpu_any_mode =
                gpu_cfg.compute_geometry ||
                gpu_cfg.compute_center_patch ||
                gpu_cfg.compute_multi_point ||
                gpu_cfg.compute_corner_points ||
                gpu_cfg.compute_texture_points ||
                gpu_cfg.compute_binary_points ||
                gpu_cfg.compute_orb_points ||
                gpu_cfg.compute_brisk_points ||
                gpu_cfg.compute_akaze_points ||
                gpu_cfg.compute_sift_points ||
                gpu_cfg.compute_iou_region_color_patch ||
                gpu_cfg.compute_patch_iou_color_edge;
            if (!gpu_any_mode) {
                LOG_INFO("Skipping dual YOLO GPU depth candidates: no direct GPU mode enabled");
            } else {
                const int max_pairs = std::clamp(
                    config_.max_detections * config_.max_detections,
                    16, 128);
                dual_yolo_depth_gpu_ = std::make_unique<DualYoloDepthGpuMatcher>();
                if (!dual_yolo_depth_gpu_->init(focal, calibration_->getBaseline(),
                                                 cx, cy, gpu_cfg, max_pairs)) {
                    LOG_ERROR("Failed to initialize dual YOLO GPU depth candidates");
                    return false;
                }
            }
        }
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
            if (!roi_matcher_->ready()) {
                LOG_ERROR("ROI Stereo Matcher failed to initialize");
                return false;
            }
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

    if (config_.async_roi_stage2 &&
        config_.tracker.enabled &&
        config_.tracker.detect_interval > 1) {
        LOG_WARN("Async ROI Stage2 disabled because tracker detect_interval=%d "
                 "would allow out-of-order HybridDepth updates; use "
                 "detect_interval=1 or disable tracker for async ROI",
                 config_.tracker.detect_interval);
    }

    if (asyncRoiStage2Configured() && !initAsyncRoiStage2()) {
        LOG_ERROR("Failed to initialize async ROI Stage2 resources");
        return false;
    }

    const bool dualYoloDepthOnly =
        config_.disparity_strategy == DisparityStrategy::ROI_ONLY &&
        config_.dual_yolo.enabled &&
        config_.dual_yolo.use_for_depth &&
        (config_.dual_yolo.depth_bbox_pair ||
         config_.dual_yolo.depth_bbox_edges ||
         (config_.dual_yolo.center_refine &&
          (config_.dual_yolo.depth_circle_center ||
           config_.dual_yolo.depth_circle_edges ||
           config_.dual_yolo.depth_roi_edge_centroid ||
           config_.dual_yolo.depth_roi_center_patch)) ||
         config_.dual_yolo.depth_roi_radial_center ||
         config_.dual_yolo.depth_roi_edge_pair_center ||
         config_.dual_yolo.depth_roi_corner_points ||
         config_.dual_yolo.depth_roi_texture_points ||
         config_.dual_yolo.depth_roi_binary_points ||
         config_.dual_yolo.depth_roi_orb_points ||
         config_.dual_yolo.depth_roi_brisk_points ||
         config_.dual_yolo.depth_roi_akaze_points ||
         config_.dual_yolo.depth_roi_sift_points ||
         config_.dual_yolo.depth_roi_iou_region_color_patch ||
         config_.dual_yolo.depth_roi_patch_iou_color_edge ||
         config_.dual_yolo.depth_roi_cuda_template_match ||
         config_.dual_yolo.depth_roi_cuda_stereo_bm ||
         config_.dual_yolo.depth_roi_cuda_stereo_sgm ||
         config_.neural_features.enabled ||
         (config_.dual_yolo.depth_roi_subpixel &&
          config_.dual_yolo.subpixel_enabled) ||
         (config_.dual_yolo.depth_epipolar_fallback &&
          config_.dual_yolo.fallback_epipolar_search) ||
         (config_.dual_yolo.depth_fallback_template &&
          config_.dual_yolo.fallback_epipolar_search) ||
         (config_.dual_yolo.depth_fallback_feature_points &&
          config_.dual_yolo.fallback_epipolar_search)) &&
        !config_.dual_yolo.fallback_to_roi_match;
    const std::string strategyStr = config_.detection_only
        ? "Detection Only"
        : (dualYoloDepthOnly
        ? ("Dual YOLO " + config_.dual_yolo.depth_solver)
        : ((config_.disparity_strategy == DisparityStrategy::ROI_ONLY)
            ? "ROI Multi-Point" : (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION
            ? "Half Resolution" : "Full Frame")));

    LOG_INFO("========================================");
    LOG_INFO("Pipeline initialized successfully");
    LOG_INFO("  Raw resolution:  %dx%d", config_.camera.width, config_.camera.height);
    LOG_INFO("  Rect resolution: %dx%d", config_.rect_width, config_.rect_height);
    LOG_INFO("  Trigger: %d Hz", config_.trigger_freq_hz);
    LOG_INFO("  Detect: %s (DLA=%d)", config_.engine_file.c_str(), config_.use_dla);
    LOG_INFO("  Disparity: %s", strategyStr.c_str());
    LOG_INFO("  Drop stale ROI frames: %d", config_.drop_stale_roi_frames ? 1 : 0);
    LOG_INFO("  Async ROI Stage2: %d (buffers=%d deadline=%.1fms)",
             async_roi_ready_ ? 1 : 0,
             config_.async_roi_buffers,
             config_.async_roi_deadline_ms);
    LOG_INFO("========================================");

    return true;
}

void Pipeline::start() {
    if (running_.exchange(true)) return;

    if (!startAsyncRoiStage2()) {
        running_ = false;
        return;
    }

#ifdef HIK_CAMERA_ENABLED
    // 先启动相机采集, 再启动 PWM 触发
    if (camera_ && !camera_->startGrabbing()) {
        LOG_ERROR("Failed to start camera grabbing");
        running_ = false;
        shutdownAsyncRoiStage2();
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

    shutdownAsyncRoiStage2();

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
// 异步相机采集线程 (按需模式)
//
// 工作流:
//   1. pipeline 调用 requestGrab(slot) → 采集线程唤醒
//   2. 采集线程: lock VPI host buffer → grabFramePair memcpy → unlock
//   3. 采集线程: signal grab_done → pipeline 端 waitGrab() 返回
//
// 关键优化:
//   - 直接写入 VPI Image host-mapped buffer, 不再经过额外 staging buffer
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

        // 直接锁定 VPI Image host buffer 并写入，后续 VPI/CUDA 复用同一图像资源。
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
            slot.left_timestamp_us = resL.timestamp_us;
            slot.right_timestamp_us = resR.timestamp_us;
            slot.left_frame_number = resL.frame_number;
            slot.right_frame_number = resR.frame_number;
            slot.left_frame_counter = resL.frame_counter;
            slot.right_frame_counter = resR.frame_counter;
            slot.left_trigger_index = resL.trigger_index;
            slot.right_trigger_index = resR.trigger_index;
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
    // grab_preloaded == true: 异步采集线程已将数据写入 rawL/rawR 的 host-mapped buffer
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

CircleFit2D circleFromGpuCandidate(const DualYoloGpuCircle& in,
                                   const Detection& fallback) {
    if (in.valid) {
        CircleFit2D out;
        out.cx = in.cx;
        out.cy = in.cy;
        out.radius = in.radius;
        out.confidence = in.confidence;
        out.source = in.source;
        out.valid = true;
        return out;
    }
    return circleFromDetectionCPU(fallback);
}

PointMeasure2D pointFromGpuCandidate(const DualYoloGpuPointMeasure& in) {
    PointMeasure2D out;
    if (in.valid) {
        out.cx = in.cx;
        out.cy = in.cy;
        out.confidence = in.confidence;
        out.valid = true;
    }
    return out;
}

DualYoloGpuDetection makeGpuDetection(const Detection& det) {
    DualYoloGpuDetection out;
    out.cx = det.cx;
    out.cy = det.cy;
    out.width = det.width;
    out.height = det.height;
    out.confidence = det.confidence;
    out.class_id = det.class_id;
    return out;
}

struct SubpixelSampleOffset {
    float dx = 0.0f;
    float dy = 0.0f;
};

SubpixelDisparityResult subpixelFromGpuCandidate(const DualYoloGpuDisparity& in) {
    SubpixelDisparityResult out;
    out.valid = in.valid != 0;
    out.low_confidence = in.low_confidence != 0;
    out.disparity = in.disparity;
    out.confidence = in.confidence;
    out.stddev = in.stddev;
    out.delta_gate_px = in.delta_gate_px;
    out.support = in.support;
    out.attempted = in.attempted;
    return out;
}

SparseFeatureDisparityResult sparseFromGpuCandidate(const DualYoloGpuDisparity& in) {
    SparseFeatureDisparityResult out;
    out.valid = in.valid != 0;
    out.low_confidence = in.low_confidence != 0;
    out.disparity = in.disparity;
    out.confidence = in.confidence;
    out.stddev = in.stddev;
    out.anchor_cx = in.anchor_cx;
    out.anchor_cy = in.anchor_cy;
    out.support = in.support;
    out.attempted = in.attempted;
    return out;
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

float bboxDisparityConsistencyPenaltyCPU(
    const Detection& left,
    const Detection& right,
    float pair_disparity,
    float baseline,
    const HybridDepthConfig& depth_cfg,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity)
{
    if (!std::isfinite(pair_disparity) || pair_disparity <= 0.0f) {
        return 0.0f;
    }
    const float left_expected =
        estimateDisparityFromBBoxCPU(left, baseline, depth_cfg, max_disparity);
    const float right_expected =
        estimateDisparityFromBBoxCPU(right, baseline, depth_cfg, max_disparity);

    float expected = -1.0f;
    if (left_expected > 0.0f && right_expected > 0.0f) {
        expected = 0.5f * (left_expected + right_expected);
    } else if (left_expected > 0.0f) {
        expected = left_expected;
    } else if (right_expected > 0.0f) {
        expected = right_expected;
    }
    if (expected <= 0.0f) return 0.0f;

    const float ratio_tol =
        std::max(0.05f, dual_cfg.bbox_disparity_consistency_ratio);
    const float abs_tol =
        std::max(5.0f, dual_cfg.bbox_disparity_consistency_min_px);
    const float tolerance = std::max(abs_tol, expected * ratio_tol);
    const float excess = std::abs(pair_disparity - expected) - tolerance;
    if (excess <= 0.0f) return 0.0f;

    const float scale = std::max(0.0f, dual_cfg.bbox_disparity_penalty_scale);
    return scale * excess / std::max(1.0f, tolerance);
}

CircleFit2D searchTemplateOnEpipolarCPU(
    const uint8_t* source_img, int source_pitch,
    const uint8_t* target_img, int target_pitch,
    int img_w, int img_h,
    const CircleFit2D& source_circle,
    float predicted_cx, float predicted_cy,
    float y_tolerance,
    const PipelineConfig::DualYoloConfig& dual_cfg)
{
    CircleFit2D out;
    if (!source_img || !target_img || source_pitch <= 0 || target_pitch <= 0 ||
        !source_circle.valid) {
        return out;
    }

    const int patch_radius = std::clamp(
        static_cast<int>(std::lround(std::min(source_circle.radius * 0.35f, 10.0f))),
        4, 10);
    const int source_x = static_cast<int>(std::lround(source_circle.cx));
    const int source_y = static_cast<int>(std::lround(source_circle.cy));
    if (!patchInsideCPU(img_w, img_h, source_x, source_y,
                        patch_radius, dual_cfg.roi_denoise)) {
        return out;
    }

    const float max_width = std::max(32.0f, static_cast<float>(dual_cfg.fallback_max_width_px));
    const float margin = std::min(
        std::max(4.0f, static_cast<float>(dual_cfg.fallback_search_margin_px)),
        max_width * 0.5f);
    const int x_start = std::max(patch_radius + 1,
        static_cast<int>(std::floor(predicted_cx - margin)));
    const int x_end = std::min(img_w - patch_radius - 2,
        static_cast<int>(std::ceil(predicted_cx + margin)));
    const int y_start = std::max(patch_radius + 1,
        static_cast<int>(std::floor(predicted_cy - y_tolerance)));
    const int y_end = std::min(img_h - patch_radius - 2,
        static_cast<int>(std::ceil(predicted_cy + y_tolerance)));
    if (x_start >= x_end || y_start >= y_end) return out;

    auto score_at = [&](int x, int y) -> float {
        if (!patchInsideCPU(img_w, img_h, x, y, patch_radius, dual_cfg.roi_denoise)) {
            return -2.0f;
        }
        return znccPatchCPU(source_img, source_pitch, target_img, target_pitch,
                            source_x, source_y, x, y,
                            patch_radius, dual_cfg.roi_denoise);
    };

    float best_score = -2.0f;
    float second_score = -2.0f;
    int best_x = -1;
    int best_y = -1;
    const int coarse_step = (x_end - x_start) > 64 ? 2 : 1;
    for (int y = y_start; y <= y_end; y += coarse_step) {
        for (int x = x_start; x <= x_end; x += coarse_step) {
            const float score = score_at(x, y);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_x = x;
                best_y = y;
            } else if (score > second_score) {
                second_score = score;
            }
        }
    }
    if (best_x < 0 || best_y < 0) return out;

    const int refine_x1 = std::max(x_start, best_x - coarse_step);
    const int refine_x2 = std::min(x_end, best_x + coarse_step);
    const int refine_y1 = std::max(y_start, best_y - coarse_step);
    const int refine_y2 = std::min(y_end, best_y + coarse_step);
    for (int y = refine_y1; y <= refine_y2; ++y) {
        for (int x = refine_x1; x <= refine_x2; ++x) {
            const float score = score_at(x, y);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_x = x;
                best_y = y;
            } else if (score > second_score && (x != best_x || y != best_y)) {
                second_score = score;
            }
        }
    }

    const float min_score = std::max(0.12f, dual_cfg.subpixel_min_confidence * 0.55f);
    const float uniqueness = second_score > -1.5f ? best_score - second_score : 1.0f;
    if (best_score < min_score || (uniqueness < 0.005f && best_score < 0.70f)) {
        return out;
    }

    float sub_x = static_cast<float>(best_x);
    if (best_x > x_start && best_x < x_end) {
        const float s_minus = score_at(best_x - 1, best_y);
        const float s_plus = score_at(best_x + 1, best_y);
        const float denom = s_minus - 2.0f * best_score + s_plus;
        if (s_minus > -1.5f && s_plus > -1.5f && denom < -1e-5f) {
            sub_x += std::clamp(0.5f * (s_minus - s_plus) / denom, -1.0f, 1.0f);
        }
    }

    if (std::abs(sub_x - predicted_cx) > margin ||
        std::abs(static_cast<float>(best_y) - predicted_cy) > y_tolerance) {
        return out;
    }

    out.cx = sub_x;
    out.cy = static_cast<float>(best_y);
    out.radius = source_circle.radius;
    out.confidence = std::max(0.2f,
        std::clamp((best_score - min_score) / std::max(0.01f, 1.0f - min_score),
                   0.0f, 1.0f));
    out.source = kCircleSourceTemplateSearch;
    out.valid = true;
    return out;
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

SubpixelDisparityResult refineDisparityByROICenterPatchCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    float focal,
    float baseline)
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
    const float max_delta = computeSubpixelDispDeltaGateCPU(
        initial_disp, focal, baseline,
        dual_cfg.subpixel_max_disp_delta_px,
        dual_cfg.subpixel_max_disp_delta_ratio,
        dual_cfg.subpixel_max_depth_delta_m);
    result.delta_gate_px = max_delta;

    const int x_left = static_cast<int>(std::lround(left_circle.cx));
    const int y_left = static_cast<int>(std::lround(left_circle.cy));
    const int y_right = y_left;
    if (!patchInsideCPU(img_w, img_h, x_left, y_left,
                        patch_radius, dual_cfg.roi_denoise)) {
        return result;
    }

    const int d_start = std::max(1, static_cast<int>(std::floor(initial_disp)) - search_radius);
    const int d_end = std::min(max_disparity,
                               static_cast<int>(std::ceil(initial_disp)) + search_radius);
    if (d_start >= d_end) return result;

    auto score_at = [&](int disparity) -> float {
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

    float best_score = -2.0f;
    float second_score = -2.0f;
    int best_disp = -1;
    for (int disp = d_start; disp <= d_end; ++disp) {
        ++result.attempted;
        const float score = score_at(disp);
        if (score > best_score) {
            second_score = best_score;
            best_score = score;
            best_disp = disp;
        } else if (score > second_score) {
            second_score = score;
        }
    }
    const float min_score = std::max(0.10f, dual_cfg.subpixel_min_confidence * 0.60f);
    if (best_disp < 0 || best_score < min_score) {
        result.low_confidence = true;
        return result;
    }

    float sub_disp = static_cast<float>(best_disp);
    if (best_disp > d_start && best_disp < d_end) {
        const float s_minus = score_at(best_disp - 1);
        const float s_plus = score_at(best_disp + 1);
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
    if ((uniqueness_margin < 0.01f && best_score < 0.75f) ||
        std::abs(sub_disp - initial_disp) > max_delta ||
        sub_disp <= 0.5f ||
        sub_disp > static_cast<float>(max_disparity)) {
        result.low_confidence = true;
        return result;
    }

    result.valid = true;
    result.disparity = sub_disp;
    result.support = 1;
    result.stddev = 0.0f;
    result.confidence = std::clamp((best_score - 0.10f) / 0.80f, 0.0f, 1.0f);
    if (result.confidence < dual_cfg.subpixel_min_confidence) {
        result.valid = false;
        result.low_confidence = true;
    }
    return result;
}

SubpixelDisparityResult refineDisparityByROIMultiPointCPU(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const CircleFit2D& left_circle,
    const CircleFit2D& right_circle,
    const PipelineConfig::DualYoloConfig& dual_cfg,
    int max_disparity,
    float focal,
    float baseline)
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
    const float max_delta = computeSubpixelDispDeltaGateCPU(
        initial_disp, focal, baseline,
        dual_cfg.subpixel_max_disp_delta_px,
        dual_cfg.subpixel_max_disp_delta_ratio,
        dual_cfg.subpixel_max_depth_delta_m);
    result.delta_gate_px = max_delta;
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

void stampFrameMetadata(FrameSlot& slot)
{
    const int64_t frame_counter_delta =
        static_cast<int64_t>(slot.left_frame_counter) -
        static_cast<int64_t>(slot.right_frame_counter);
    const int64_t frame_number_delta =
        static_cast<int64_t>(slot.left_frame_number) -
        static_cast<int64_t>(slot.right_frame_number);
    const int64_t timestamp_delta_raw =
        static_cast<int64_t>(slot.left_timestamp_us) -
        static_cast<int64_t>(slot.right_timestamp_us);
    for (auto& obj : slot.results) {
        obj.left_timestamp_us = slot.left_timestamp_us;
        obj.right_timestamp_us = slot.right_timestamp_us;
        obj.left_frame_number = slot.left_frame_number;
        obj.right_frame_number = slot.right_frame_number;
        obj.left_frame_counter = slot.left_frame_counter;
        obj.right_frame_counter = slot.right_frame_counter;
        obj.left_trigger_index = slot.left_trigger_index;
        obj.right_trigger_index = slot.right_trigger_index;
        obj.frame_counter_delta = frame_counter_delta;
        obj.frame_number_delta = frame_number_delta;
        obj.timestamp_delta_us = timestamp_delta_raw / 1000;
    }
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

bool Pipeline::debugFeatureMatchesOnce(const std::string& output_dir) {
#ifndef HIK_CAMERA_ENABLED
    (void)output_dir;
    LOG_ERROR("Feature match debug capture requires Hikvision camera support");
    return false;
#else
    if (running_.load()) {
        LOG_ERROR("Feature match debug capture must run before Pipeline::start()");
        return false;
    }
    if (!camera_) {
        LOG_ERROR("Feature match debug capture requires initialized cameras");
        return false;
    }
    if (!dualYoloEnabled()) {
        LOG_ERROR("Feature match debug capture requires dual_yolo.enabled=true");
        return false;
    }
    if (!calibration_) {
        LOG_ERROR("Feature match debug capture requires stereo calibration");
        return false;
    }

    namespace fs = std::filesystem;
    const fs::path output_path(output_dir);
    if (output_path.empty()) {
        LOG_ERROR("Feature match debug output dir is empty");
        return false;
    }
    std::error_code ec;
    fs::create_directories(output_path, ec);
    if (ec) {
        LOG_ERROR("Failed to create debug output dir %s: %s",
                  output_dir.c_str(), ec.message().c_str());
        return false;
    }
    ec.clear();
    if (!fs::exists(output_path, ec) || ec) {
        LOG_ERROR("Feature match debug output dir does not exist: %s",
                  output_dir.c_str());
        return false;
    }
    ec.clear();
    if (!fs::is_directory(output_path, ec) || ec) {
        LOG_ERROR("Feature match debug output path is not a directory: %s",
                  output_dir.c_str());
        return false;
    }

    bool output_ok = true;
    auto write_debug_image = [&](const std::string& filename,
                                 const cv::Mat& image) {
        const fs::path image_path = output_path / filename;
        if (image.empty()) {
            LOG_ERROR("Feature match debug: refusing to write empty image %s",
                      image_path.string().c_str());
            output_ok = false;
            return;
        }
        try {
            if (!cv::imwrite(image_path.string(), image)) {
                LOG_ERROR("Feature match debug: failed to write %s",
                          image_path.string().c_str());
                output_ok = false;
            }
        } catch (const cv::Exception& e) {
            LOG_ERROR("Feature match debug: failed to write %s: %s",
                      image_path.string().c_str(), e.what());
            output_ok = false;
        }
    };

    bool grabbing_started = false;
    bool pwm_started = false;
    auto cleanup = [&]() {
        streams_.syncAll();
        if (pwm_started && pwm_trigger_) pwm_trigger_->stop();
        if (grabbing_started && camera_) camera_->stopGrabbing();
    };

    if (!camera_->startGrabbing()) {
        LOG_ERROR("Feature match debug: failed to start camera grabbing");
        return false;
    }
    grabbing_started = true;
    if (pwm_trigger_) {
        pwm_started = pwm_trigger_->start();
        if (!pwm_started) {
            LOG_WARN("Feature match debug: PWM trigger did not start; external trigger may be active");
        }
    }

    FrameSlot& slot = slots_[0];
    bool captured = false;
    constexpr int kMaxWarmupFrames = 12;
    for (int attempt = 0; attempt < kMaxWarmupFrames; ++attempt) {
        slot.reset();
        slot.frame_id = attempt;
        slot.is_detect_frame = true;

        stage0_grab_and_rectify(slot, false);
        VPIStatus vst = vpiStreamSync(streams_.vpiStreamPVA);
        if (vst != VPI_SUCCESS) {
            LOG_ERROR("Feature match debug: rectification sync failed: %d", (int)vst);
            cleanup();
            return false;
        }
        if (!slot.grab_failed) {
            captured = true;
            break;
        }
        LOG_WARN("Feature match debug: warmup/sync frame %d skipped", attempt);
    }
    if (!captured) {
        LOG_ERROR("Feature match debug: grab/rectify failed after %d attempts",
                  kMaxWarmupFrames);
        cleanup();
        return false;
    }
    cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);

    stage1_detect(slot, 0);
    recordDetectDoneEvents(slot);
    cudaStreamSynchronize(getDLAStream(slot.frame_id));
    if (slot.right_detection_submitted) {
        cudaStreamSynchronize(getRightDLAStream(slot.frame_id));
    }

    if (slot.detection_submitted) {
        slot.detections = getDetector(slot.frame_id)->collect(
            0, config_.rect_width, config_.rect_height);
        limitDetectionsByConfidence(slot.detections, config_.max_detections);
    }
    collectRightDetections(slot, 0);

    auto lock_gray_copy = [](VPIImage img, cv::Mat& out) -> bool {
        VPIImageData data;
        VPIStatus st = vpiImageLockData(img, VPI_LOCK_READ,
                                        VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                        &data);
        if (st != VPI_SUCCESS) return false;
        try {
            const int w = data.buffer.pitch.planes[0].width;
            const int h = data.buffer.pitch.planes[0].height;
            const int pitch = data.buffer.pitch.planes[0].pitchBytes;
            cv::Mat view(h, w, CV_8UC1, data.buffer.pitch.planes[0].data, pitch);
            view.copyTo(out);
        } catch (const cv::Exception&) {
            vpiImageUnlock(img);
            return false;
        }
        vpiImageUnlock(img);
        return true;
    };
    auto lock_bgr_copy = [](VPIImage img, cv::Mat& out) -> bool {
        VPIImageData data;
        VPIStatus st = vpiImageLockData(img, VPI_LOCK_READ,
                                        VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,
                                        &data);
        if (st != VPI_SUCCESS) return false;
        const int plane_width = data.buffer.pitch.planes[0].width;
        const int h = data.buffer.pitch.planes[0].height;
        const int pitch = data.buffer.pitch.planes[0].pitchBytes;
        const void* src = data.buffer.pitch.planes[0].data;
        if (plane_width <= 0 || h <= 0 || pitch <= 0 || !src) {
            vpiImageUnlock(img);
            return false;
        }
        int pixel_width = plane_width;
        size_t row_bytes = static_cast<size_t>(pixel_width) * 3U;
        if (row_bytes > static_cast<size_t>(pitch)) {
            if (plane_width % 3 == 0 &&
                static_cast<size_t>(plane_width) <= static_cast<size_t>(pitch)) {
                pixel_width = plane_width / 3;
                row_bytes = static_cast<size_t>(plane_width);
            } else {
                vpiImageUnlock(img);
                return false;
            }
        }
        out.create(h, pixel_width, CV_8UC3);
        const cudaError_t err = cudaMemcpy2D(out.data, out.step[0],
                                             src, static_cast<size_t>(pitch),
                                             row_bytes, static_cast<size_t>(h),
                                             cudaMemcpyDeviceToHost);
        vpiImageUnlock(img);
        if (err != cudaSuccess) {
            out.release();
            return false;
        }
        return true;
    };

    cv::Mat left_gray;
    cv::Mat right_gray;
    if (!lock_gray_copy(slot.rectGray_vpiL, left_gray) ||
        !lock_gray_copy(slot.rectGray_vpiR, right_gray)) {
        LOG_ERROR("Feature match debug: failed to copy rectified gray images");
        cleanup();
        return false;
    }
    cv::Mat left_viz;
    cv::Mat right_viz;
    if (colorPipelineEnabled() &&
        lock_bgr_copy(slot.rectBGR_vpiL, left_viz) &&
        lock_bgr_copy(slot.rectBGR_vpiR, right_viz)) {
        write_debug_image("left_rect_bgr.png", left_viz);
        write_debug_image("right_rect_bgr.png", right_viz);
    } else {
        cv::cvtColor(left_gray, left_viz, cv::COLOR_GRAY2BGR);
        cv::cvtColor(right_gray, right_viz, cv::COLOR_GRAY2BGR);
    }

    auto draw_detection_overlay = [](const cv::Mat& image,
                                     const std::vector<Detection>& detections,
                                     const std::string& title) {
        cv::Mat out;
        if (image.channels() == 1) {
            cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
        } else {
            image.copyTo(out);
        }
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            const int x = static_cast<int>(std::round(d.cx - d.width * 0.5f));
            const int y = static_cast<int>(std::round(d.cy - d.height * 0.5f));
            const int w = static_cast<int>(std::round(d.width));
            const int h = static_cast<int>(std::round(d.height));
            cv::rectangle(out, cv::Rect(x, y, w, h) &
                                cv::Rect(0, 0, out.cols, out.rows),
                          cv::Scalar(0, 255, 0), 2);
            char label[96];
            std::snprintf(label, sizeof(label), "#%zu c%d %.2f",
                          i, d.class_id, d.confidence);
            cv::putText(out, label, cv::Point(std::max(0, x), std::max(18, y - 6)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 255), 2);
        }
        cv::putText(out, title, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 180, 255), 2);
        return out;
    };

    write_debug_image("left_rect_gray.png", left_gray);
    write_debug_image("right_rect_gray.png", right_gray);
    write_debug_image("left_detections.png",
                      draw_detection_overlay(left_viz, slot.detections, "left"));
    write_debug_image("right_detections.png",
                      draw_detection_overlay(right_viz, slot.detections_right, "right"));

    StereoRoiPair debug_pair;
    if (!findBestStereoRoiPair(slot.detections,
                               slot.detections_right,
                               makeStereoRoiPairGateConfig(config_),
                               &debug_pair)) {
        LOG_ERROR("Feature match debug: no valid left/right YOLO pair (left=%zu right=%zu)",
                  slot.detections.size(), slot.detections_right.size());
        cleanup();
        return false;
    }

    const Detection& left_det = slot.detections[debug_pair.left_index];
    const Detection& right_det = slot.detections_right[debug_pair.right_index];
    const float initial_disp = debug_pair.initial_disparity;
    const auto& P1 = calibration_->getProjectionLeft();
    const float focal = static_cast<float>(P1.at<double>(0, 0));
    const float baseline = calibration_->getBaseline();
    const ROIFeatureMatchConfig feature_cfg =
        makeROIFeatureMatchConfig(config_.dual_yolo, config_.depth);

    std::vector<DebugFeatureMatchResult> results;
    results.push_back(makeDebugSparseFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        SparseFeatureMode::CORNER));
    results.push_back(makeDebugSparseFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        SparseFeatureMode::TEXTURE));
    results.push_back(makeDebugSparseFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        SparseFeatureMode::BINARY));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::ORB));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::BRISK));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::AKAZE));
    results.push_back(makeDebugOpenCVFeatureMatchesCPU(
        left_gray, right_gray, left_det, right_det, initial_disp,
        feature_cfg, config_.max_disparity, focal, baseline,
        OpenCVFeatureMode::SIFT));

    cv::Mat left_color = left_viz.clone();
    cv::Mat right_color = right_viz.clone();
    auto draw_selected_bbox = [](cv::Mat& img, const Detection& d,
                                 const cv::Scalar& color) {
        const int x = static_cast<int>(std::round(d.cx - d.width * 0.5f));
        const int y = static_cast<int>(std::round(d.cy - d.height * 0.5f));
        const int w = static_cast<int>(std::round(d.width));
        const int h = static_cast<int>(std::round(d.height));
        cv::rectangle(img, cv::Rect(x, y, w, h) & cv::Rect(0, 0, img.cols, img.rows),
                      color, 2);
    };
    draw_selected_bbox(left_color, left_det, cv::Scalar(0, 255, 0));
    draw_selected_bbox(right_color, right_det, cv::Scalar(0, 255, 0));
    auto draw_feature_debug_panel = [](const cv::Mat& left_base,
                                       const cv::Mat& right_base,
                                       const DebugFeatureMatchResult& r) {
        cv::Mat left_panel = left_base.clone();
        cv::Mat right_panel = right_base.clone();
        std::vector<cv::Mat> side_by_side{left_panel, right_panel};
        cv::Mat canvas;
        cv::hconcat(side_by_side, canvas);
        const int x_offset = left_panel.cols;

        for (const auto& kp : r.left_keypoints) {
            cv::circle(canvas, kp.pt, 4, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
        for (const auto& kp : r.right_keypoints) {
            cv::Point2f p(kp.pt.x + static_cast<float>(x_offset), kp.pt.y);
            cv::circle(canvas, p, 4, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
        }
        for (const auto& m : r.matches) {
            if (m.queryIdx < 0 || m.trainIdx < 0 ||
                m.queryIdx >= static_cast<int>(r.left_keypoints.size()) ||
                m.trainIdx >= static_cast<int>(r.right_keypoints.size())) {
                continue;
            }
            const cv::Point2f p1 = r.left_keypoints[m.queryIdx].pt;
            const cv::Point2f p2(
                r.right_keypoints[m.trainIdx].pt.x + static_cast<float>(x_offset),
                r.right_keypoints[m.trainIdx].pt.y);
            cv::line(canvas, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::circle(canvas, p1, 5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::circle(canvas, p2, 5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        char title[224];
        std::snprintf(title, sizeof(title),
                      "%s Lkp=%zu Rkp=%zu cand=%d matches=%zu disp=%.2f std=%.2f conf=%.2f",
                      r.name.c_str(), r.left_keypoints.size(), r.right_keypoints.size(),
                      r.attempted_matches, r.matches.size(), r.disparity,
                      r.stddev, r.confidence);
        cv::putText(canvas, title, cv::Point(16, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.68,
                    cv::Scalar(0, 180, 255), 2);
        cv::putText(canvas, "left keypoints", cv::Point(16, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 0), 1);
        cv::putText(canvas, "right candidates", cv::Point(x_offset + 16, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 0, 255), 1);
        return canvas;
    };
    cv::Mat selected_pair;
    {
        std::vector<cv::Mat> side_by_side{left_color, right_color};
        cv::hconcat(side_by_side, selected_pair);
    }
    write_debug_image("selected_pair.png", selected_pair);

    std::vector<cv::Mat> panels;
    std::ofstream summary((output_path / "summary.txt").string());
    if (!summary.is_open()) {
        LOG_ERROR("Feature match debug: failed to open %s",
                  (output_path / "summary.txt").string().c_str());
        output_ok = false;
    } else {
        summary << "left_count=" << slot.detections.size()
                << " right_count=" << slot.detections_right.size() << "\n";
        summary << "selected_left=" << debug_pair.left_index
                << " selected_right=" << debug_pair.right_index
                << " initial_disp=" << initial_disp
                << " dy=" << debug_pair.epipolar_dy
                << " shifted_iou=" << debug_pair.shifted_bbox_iou
                << " pair_score=" << debug_pair.score
                << " baseline_m=" << baseline
                << " focal_px=" << focal << "\n";
        summary << "frame_counter_delta="
                << (static_cast<int64_t>(slot.left_frame_counter) -
                    static_cast<int64_t>(slot.right_frame_counter))
                << " frame_number_delta="
                << (static_cast<int64_t>(slot.left_frame_number) -
                    static_cast<int64_t>(slot.right_frame_number))
                << " trigger_delta="
                << (static_cast<int64_t>(slot.left_trigger_index) -
                    static_cast<int64_t>(slot.right_trigger_index))
                << "\n";
    }

    for (const auto& r : results) {
        cv::Mat canvas = draw_feature_debug_panel(left_color, right_color, r);
        write_debug_image(r.name + "_matches.png", canvas);
        panels.push_back(canvas);
        if (summary.is_open()) {
            summary << r.name
                    << " left_keypoints=" << r.left_keypoints.size()
                    << " right_keypoints=" << r.right_keypoints.size()
                    << " candidates=" << r.attempted_matches
                    << " matches=" << r.matches.size()
                    << " disparity=" << r.disparity
                    << " std=" << r.stddev
                    << " confidence=" << r.confidence << "\n";
        }
    }

    if (!panels.empty()) {
        const int target_w = panels.front().cols;
        std::vector<cv::Mat> resized;
        resized.reserve(panels.size());
        for (const auto& p : panels) {
            if (p.cols == target_w) {
                resized.push_back(p);
            } else {
                cv::Mat tmp;
                const double scale = static_cast<double>(target_w) /
                                     static_cast<double>(std::max(1, p.cols));
                cv::resize(p, tmp, cv::Size(target_w,
                    std::max(1, static_cast<int>(std::round(p.rows * scale)))));
                resized.push_back(tmp);
            }
        }
        cv::Mat contact;
        cv::vconcat(resized, contact);
        write_debug_image("feature_match_contact_sheet.png", contact);
    }

    if (summary.is_open()) {
        summary.flush();
        if (!summary.good()) {
            LOG_ERROR("Feature match debug: failed while writing %s",
                      (output_path / "summary.txt").string().c_str());
            output_ok = false;
        }
    }
    if (!output_ok) {
        cleanup();
        return false;
    }

    LOG_INFO("Feature match debug saved to %s", output_dir.c_str());
    cleanup();
    return true;
#endif
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

bool Pipeline::detectEventsReady(const FrameSlot& slot) const {
    if (!slot.is_detect_frame || !slot.detection_submitted) {
        return false;
    }

    cudaError_t err = cudaEventQuery(slot.evtDetectDone);
    if (err == cudaErrorNotReady) {
        return false;
    }
    if (err != cudaSuccess) {
        LOG_WARN("[Pipeline] left detect event query failed: frame=%d err=%s",
                 slot.frame_id, cudaGetErrorString(err));
        return false;
    }

    if (dualYoloEnabled()) {
        if (!slot.right_detection_submitted) {
            return false;
        }
        err = cudaEventQuery(slot.evtDetectRightDone);
        if (err == cudaErrorNotReady) {
            return false;
        }
        if (err != cudaSuccess) {
            LOG_WARN("[Pipeline] right detect event query failed: frame=%d err=%s",
                     slot.frame_id, cudaGetErrorString(err));
            return false;
        }
    }
    return true;
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
    limitDetectionsByConfidence(slot.detections_right, config_.max_detections);
}

void Pipeline::collectRoiDetections(FrameSlot& slot, int slot_index) {
    auto* det = getDetector(slot.frame_id);
    if (slot.detection_submitted) {
        slot.detections = det->collect(slot_index,
                                       config_.rect_width,
                                       config_.rect_height);
        limitDetectionsByConfidence(slot.detections, config_.max_detections);
    } else {
        slot.detections.clear();
    }
    collectRightDetections(slot, slot_index);
}

bool Pipeline::roiStage2NeedsHostImages(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections) const {
    if (left_detections.empty() && right_detections.empty()) {
        return false;
    }
    const bool fallback_enabled =
        dualYoloCpuFallbackSearchEnabled(config_.dual_yolo);

    auto fallback_may_need_host = [&]() -> bool {
        if (!fallback_enabled) {
            return false;
        }
        if (left_detections.empty() || right_detections.empty()) {
            return true;
        }
        if (left_detections.size() != 1 || right_detections.size() != 1) {
            return true;
        }

        const StereoRoiPairGateConfig roi_pair_gate =
            makeStereoRoiPairGateConfig(config_);
        std::vector<bool> right_used(right_detections.size(), false);
        int matched_left = 0;
        for (size_t li = 0; li < left_detections.size(); ++li) {
            int best_idx = -1;
            float best_score = std::numeric_limits<float>::max();
            for (size_t ri = 0; ri < right_detections.size(); ++ri) {
                if (right_used[ri]) continue;
                StereoRoiPair candidate_pair;
                if (!evaluateStereoRoiPair(left_detections[li],
                                           right_detections[ri],
                                           static_cast<int>(li),
                                           static_cast<int>(ri),
                                           roi_pair_gate,
                                           &candidate_pair,
                                           nullptr)) {
                    continue;
                }
                if (candidate_pair.score < best_score) {
                    best_score = candidate_pair.score;
                    best_idx = static_cast<int>(ri);
                }
            }
            if (best_idx >= 0) {
                right_used[static_cast<size_t>(best_idx)] = true;
                ++matched_left;
            }
        }

        if (matched_left < static_cast<int>(left_detections.size())) {
            return true;
        }
        for (bool used : right_used) {
            if (!used) return true;
        }
        return false;
    };

    const bool has_stereo_detections =
        !left_detections.empty() && !right_detections.empty();
    const bool opencv_descriptor_cpu_possible =
        dualYoloOpenCVCpuDescriptorDepthEnabled(config_.dual_yolo);

    if (config_.dual_yolo.gpu_candidate_refine) {
        return (has_stereo_detections && opencv_descriptor_cpu_possible) ||
               fallback_may_need_host();
    }

    return (has_stereo_detections &&
            dualYoloNeedsHostImages(config_.dual_yolo)) ||
           fallback_may_need_host();
}

Pipeline::DualYoloMatchOutput Pipeline::matchDualYoloDetections(
    const std::vector<Detection>& left_detections,
    const std::vector<Detection>& right_detections,
    const uint8_t* left_cpu, int left_pitch,
    const uint8_t* right_cpu, int right_pitch,
    const uint8_t* left_gpu, int left_gpu_pitch,
    const uint8_t* right_gpu, int right_gpu_pitch,
    const uint8_t* left_bgr_gpu, int left_bgr_pitch,
    const uint8_t* right_bgr_gpu, int right_bgr_pitch,
    int img_width, int img_height,
    cudaStream_t stream,
    DualYoloMatchStats* stats)
{
    using Clock = std::chrono::high_resolution_clock;
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
    const bool gpu_image_available = left_gpu && right_gpu &&
                                     left_gpu_pitch > 0 && right_gpu_pitch > 0;
    const bool gpu_candidate_refine_requested =
        config_.dual_yolo.gpu_candidate_refine &&
        dual_yolo_depth_gpu_ &&
        dual_yolo_depth_gpu_->ready() &&
        gpu_image_available &&
        stream != nullptr;
    const bool bbox_depth_enabled =
        dualYoloBBoxDepthEnabled(config_.dual_yolo);
    const bool bbox_edges_depth_enabled =
        dualYoloBBoxEdgesDepthEnabled(config_.dual_yolo);
    const bool circle_depth_enabled =
        dualYoloCircleDepthEnabled(config_.dual_yolo);
    const bool circle_edges_depth_enabled =
        dualYoloCircleEdgesDepthEnabled(config_.dual_yolo);
    const bool roi_edge_centroid_depth_enabled =
        dualYoloROIEdgeCentroidDepthEnabled(config_.dual_yolo);
    const bool roi_radial_center_depth_enabled =
        dualYoloROIRadialCenterDepthEnabled(config_.dual_yolo);
    const bool roi_edge_pair_center_depth_enabled =
        dualYoloROIEdgePairCenterDepthEnabled(config_.dual_yolo);
    const bool roi_corner_points_depth_enabled =
        dualYoloROICornerPointsDepthEnabled(config_.dual_yolo);
    const bool roi_texture_points_depth_enabled =
        dualYoloROITexturePointsDepthEnabled(config_.dual_yolo);
    const bool roi_binary_points_depth_enabled =
        dualYoloROIBinaryPointsDepthEnabled(config_.dual_yolo);
    const bool roi_orb_points_depth_enabled =
        dualYoloROIORBPointsDepthEnabled(config_.dual_yolo);
    const bool roi_brisk_points_depth_enabled =
        dualYoloROIBRISKPointsDepthEnabled(config_.dual_yolo);
    const bool roi_akaze_points_depth_enabled =
        dualYoloROIAKAZEPointsDepthEnabled(config_.dual_yolo);
    const bool roi_sift_points_depth_enabled =
        dualYoloROISIFTPointsDepthEnabled(config_.dual_yolo);
    const bool roi_iou_region_color_patch_depth_enabled =
        dualYoloROIIoURegionColorPatchDepthEnabled(config_.dual_yolo);
    const bool roi_patch_iou_color_edge_depth_enabled =
        dualYoloROIPatchIoUColorEdgeDepthEnabled(config_.dual_yolo);
    const bool roi_cuda_template_match_depth_enabled =
        dualYoloROICudaTemplateMatchDepthEnabled(config_.dual_yolo);
    const bool roi_cuda_stereo_bm_depth_enabled =
        dualYoloROICudaStereoBMDepthEnabled(config_.dual_yolo);
    const bool roi_cuda_stereo_sgm_depth_enabled =
        dualYoloROICudaStereoSGMDepthEnabled(config_.dual_yolo);
    const bool neural_feature_depth_enabled =
        config_.neural_features.enabled &&
        neural_feature_matcher_ &&
        neural_feature_matcher_->isReady();
    const bool roi_center_patch_depth_enabled =
        dualYoloROICenterPatchDepthEnabled(config_.dual_yolo);
    const bool subpixel_depth_enabled =
        dualYoloSubpixelDepthEnabled(config_.dual_yolo);
    const bool epipolar_fallback_enabled =
        dualYoloEpipolarFallbackEnabled(config_.dual_yolo);
    const bool fallback_template_enabled =
        dualYoloFallbackTemplateEnabled(config_.dual_yolo);
    const bool fallback_feature_points_enabled =
        dualYoloFallbackFeaturePointsEnabled(config_.dual_yolo);
    const bool circle_seed_refine_enabled =
        dualYoloNeedsCircleSeedRefine(config_.dual_yolo);
    const bool direct_circle_seed_refine_enabled =
        config_.dual_yolo.center_refine &&
        (circle_depth_enabled ||
         circle_edges_depth_enabled ||
         roi_edge_centroid_depth_enabled ||
         roi_center_patch_depth_enabled ||
         subpixel_depth_enabled);
    const bool direct_pixel_modes_enabled =
        direct_circle_seed_refine_enabled ||
        roi_radial_center_depth_enabled ||
        roi_edge_pair_center_depth_enabled ||
        roi_corner_points_depth_enabled ||
        roi_texture_points_depth_enabled ||
        roi_binary_points_depth_enabled ||
        roi_orb_points_depth_enabled ||
        roi_brisk_points_depth_enabled ||
        roi_akaze_points_depth_enabled ||
        roi_sift_points_depth_enabled ||
        roi_iou_region_color_patch_depth_enabled ||
        roi_patch_iou_color_edge_depth_enabled ||
        roi_center_patch_depth_enabled ||
        subpixel_depth_enabled;
    const bool gpu_candidate_refine_enabled =
        gpu_candidate_refine_requested && direct_pixel_modes_enabled;
    const bool use_subpixel_depth =
        subpixel_depth_enabled && (image_available || gpu_candidate_refine_enabled);
    const bool use_circle_refine =
        circle_seed_refine_enabled && (image_available || gpu_candidate_refine_enabled);
    const bool direct_depth_without_circle_enabled =
        bbox_depth_enabled ||
        bbox_edges_depth_enabled ||
        roi_edge_centroid_depth_enabled ||
        roi_radial_center_depth_enabled ||
        roi_edge_pair_center_depth_enabled ||
        roi_corner_points_depth_enabled ||
        roi_texture_points_depth_enabled ||
        roi_binary_points_depth_enabled ||
        roi_orb_points_depth_enabled ||
        roi_brisk_points_depth_enabled ||
        roi_akaze_points_depth_enabled ||
        roi_sift_points_depth_enabled ||
        roi_iou_region_color_patch_depth_enabled ||
        roi_patch_iou_color_edge_depth_enabled ||
        roi_cuda_template_match_depth_enabled ||
        roi_cuda_stereo_bm_depth_enabled ||
        roi_cuda_stereo_sgm_depth_enabled ||
        neural_feature_depth_enabled ||
        roi_center_patch_depth_enabled ||
        use_subpixel_depth;
    const bool fallback_cpu_modes_enabled =
        (epipolar_fallback_enabled ||
         fallback_template_enabled ||
         fallback_feature_points_enabled) &&
        (!config_.dual_yolo.gpu_candidate_refine ||
         left_detections.empty() ||
         right_detections.empty());
    const bool cpu_pixel_modes_enabled =
        (!gpu_candidate_refine_enabled && direct_pixel_modes_enabled) ||
        fallback_cpu_modes_enabled;
    if (!image_available && cpu_pixel_modes_enabled) {
        local_stats.image_lock_fail =
            static_cast<int>(left_detections.size() + right_detections.size());
        globalPerf().record("Stage2_CPUHostImageUnavailable", 0.0);
    } else if (image_available && cpu_pixel_modes_enabled) {
        globalPerf().record("Stage2_CPUHostImageUsed", 0.0);
    }

    const auto& P1 = calibration_->getProjectionLeft();
    const float focal = static_cast<float>(P1.at<double>(0, 0));
    const float cx0 = static_cast<float>(P1.at<double>(0, 2));
    const float cy0 = static_cast<float>(P1.at<double>(1, 2));
    const float baseline = calibration_->getBaseline();
    const ROIFeatureMatchConfig feature_cfg =
        makeROIFeatureMatchConfig(config_.dual_yolo, config_.depth);
    const ROICircleSearchConfig circle_search_cfg =
        makeROICircleSearchConfig(config_.dual_yolo);
    const float y_tol = std::max(1.0f, config_.dual_yolo.epipolar_y_tolerance);
    const float max_ratio = std::max(1.0f, config_.dual_yolo.max_size_ratio);
    const StereoRoiPairGateConfig roi_pair_gate =
        makeStereoRoiPairGateConfig(config_);

    auto record_pair_reject = [&](StereoRoiPairRejectReason reason) {
        switch (reason) {
        case StereoRoiPairRejectReason::NONE:
            break;
        case StereoRoiPairRejectReason::CLASS_MISMATCH:
            ++local_stats.class_mismatch;
            break;
        case StereoRoiPairRejectReason::INVALID_BOX:
            ++local_stats.invalid_box;
            break;
        case StereoRoiPairRejectReason::NONPOSITIVE_DISPARITY:
            ++local_stats.nonpositive_disparity;
            break;
        case StereoRoiPairRejectReason::OVER_MAX_DISPARITY:
            ++local_stats.over_max_disparity;
            break;
        case StereoRoiPairRejectReason::EPIPOLAR_REJECT:
            ++local_stats.epipolar_reject;
            break;
        case StereoRoiPairRejectReason::SIZE_REJECT:
            ++local_stats.size_reject;
            break;
        case StereoRoiPairRejectReason::LOW_IOU:
            ++local_stats.low_iou;
            break;
        }
    };

    std::vector<bool> right_used(right_detections.size(), false);
    std::vector<bool> right_blocked_by_left(right_detections.size(), false);
    std::vector<bool> left_has_stereo(left_detections.size(), false);

    std::vector<DualYoloGpuCandidate> gpu_candidates;
    if (gpu_candidate_refine_enabled &&
        !left_detections.empty() &&
        !right_detections.empty()) {
        std::vector<StereoRoiPair> roi_pairs =
            collectStereoRoiPairCandidates(
                left_detections,
                right_detections,
                roi_pair_gate,
                left_detections.size() * right_detections.size());
        for (auto& roi_pair : roi_pairs) {
            roi_pair.score += bboxDisparityConsistencyPenaltyCPU(
                roi_pair.left,
                roi_pair.right,
                roi_pair.initial_disparity,
                baseline,
                config_.depth,
                config_.dual_yolo,
                config_.max_disparity);
        }
        std::sort(roi_pairs.begin(), roi_pairs.end(),
                  [](const StereoRoiPair& a, const StereoRoiPair& b) {
                      return a.score < b.score;
                  });
        const std::size_t max_gpu_pairs =
            static_cast<std::size_t>(dual_yolo_depth_gpu_->maxPairs());
        if (roi_pairs.size() > max_gpu_pairs) {
            roi_pairs.resize(max_gpu_pairs);
        }
        std::vector<DualYoloGpuDetectionPair> gpu_pairs;
        gpu_pairs.reserve(roi_pairs.size());
        for (const auto& roi_pair : roi_pairs) {
            DualYoloGpuDetectionPair pair;
            pair.left = makeGpuDetection(roi_pair.left);
            pair.right = makeGpuDetection(roi_pair.right);
            pair.left_index = roi_pair.left_index;
            pair.right_index = roi_pair.right_index;
            gpu_pairs.push_back(pair);
        }
        if (!gpu_pairs.empty()) {
            const auto gpu_start = Clock::now();
            gpu_candidates = dual_yolo_depth_gpu_->matchPairs(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                left_bgr_gpu, left_bgr_pitch,
                right_bgr_gpu, right_bgr_pitch,
                img_width, img_height,
                gpu_pairs,
                stream);
            const double gpu_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - gpu_start).count();
            globalPerf().record("Stage2_DualYoloGpuCandidates", gpu_ms);
        }
    }

    auto find_gpu_candidate = [&](int left_index,
                                  int right_index) -> const DualYoloGpuCandidate* {
        for (const auto& candidate : gpu_candidates) {
            if (candidate.left_index == left_index &&
                candidate.right_index == right_index) {
                return &candidate;
            }
        }
        return nullptr;
    };

    auto refine_detection = [&](const uint8_t* img, int pitch,
                                const Detection& det) -> CircleFit2D {
        if (!use_circle_refine || !img || pitch <= 0) {
            return circleFromDetectionCPU(det);
        }
        return fitCircleInBBoxCPU(img, pitch, img_width, img_height, det,
                                  config_.dual_yolo.roi_denoise,
                                  config_.dual_yolo.circle_max_roi_pixels);
    };

    auto edge_centroid = [&](const uint8_t* img, int pitch,
                             const Detection& det) -> PointMeasure2D {
        return edgeCentroidInBBoxCPU(img, pitch, img_width, img_height, det,
                                     config_.dual_yolo.roi_denoise,
                                     config_.dual_yolo.circle_max_roi_pixels);
    };

    auto radial_center = [&](const uint8_t* img, int pitch,
                             const Detection& det) -> PointMeasure2D {
        return radialCenterInBBoxCPU(img, pitch, img_width, img_height, det,
                                     config_.dual_yolo.roi_denoise,
                                     config_.dual_yolo.circle_max_roi_pixels);
    };

    auto edge_pair_center = [&](const uint8_t* img, int pitch,
                                const Detection& det) -> PointMeasure2D {
        return edgePairCenterInBBoxCPU(img, pitch, img_width, img_height, det,
                                       config_.dual_yolo.roi_denoise,
                                       config_.dual_yolo.circle_max_roi_pixels);
    };

    auto build_object = [&](const Detection& left_det,
                            const CircleFit2D& left_circle,
                            const CircleFit2D& right_circle,
                            float semantic_conf,
                            int match_source,
                            float yolo_disparity,
                            const Detection* right_det,
                            const StereoRoiPair* pair_info,
                            const DualYoloGpuCandidate* gpu_candidate,
                            Object3D& obj) -> bool {
        const float fb = focal * baseline;
        auto depth_from_disparity = [&](float disp) -> float {
            if (!std::isfinite(disp) || disp <= 0.0f ||
                disp > static_cast<float>(config_.max_disparity)) {
                return -1.0f;
            }
            const float z_candidate = fb / disp;
            if (z_candidate < config_.depth.min_depth ||
                z_candidate > config_.depth.max_depth) {
                return -1.0f;
            }
            return z_candidate;
        };

        const bool direct_yolo_match = match_source == 1;
        const bool is_fallback_match = (match_source == 2 || match_source == 3);
        const float z_yolo =
            (bbox_depth_enabled && direct_yolo_match)
                ? depth_from_disparity(yolo_disparity)
                : -1.0f;
        float z_bbox_left_edge = -1.0f;
        float z_bbox_right_edge = -1.0f;
        float disparity_bbox_left_edge = -1.0f;
        float disparity_bbox_right_edge = -1.0f;
        if (bbox_edges_depth_enabled && direct_yolo_match && right_det) {
            disparity_bbox_left_edge =
                (left_det.cx - left_det.width * 0.5f) -
                (right_det->cx - right_det->width * 0.5f);
            disparity_bbox_right_edge =
                (left_det.cx + left_det.width * 0.5f) -
                (right_det->cx + right_det->width * 0.5f);
            z_bbox_left_edge = depth_from_disparity(disparity_bbox_left_edge);
            z_bbox_right_edge = depth_from_disparity(disparity_bbox_right_edge);
        }
        float disparity_bbox_edge_final = -1.0f;
        float z_bbox_edge_final = -1.0f;
        if (z_bbox_left_edge > 0.0f && z_bbox_right_edge > 0.0f) {
            disparity_bbox_edge_final =
                0.5f * (disparity_bbox_left_edge + disparity_bbox_right_edge);
            z_bbox_edge_final = depth_from_disparity(disparity_bbox_edge_final);
        } else if (z_bbox_left_edge > 0.0f) {
            disparity_bbox_edge_final = disparity_bbox_left_edge;
            z_bbox_edge_final = z_bbox_left_edge;
        } else if (z_bbox_right_edge > 0.0f) {
            disparity_bbox_edge_final = disparity_bbox_right_edge;
            z_bbox_edge_final = z_bbox_right_edge;
        }

        PointMeasure2D left_edge_centroid_measure;
        PointMeasure2D right_edge_centroid_measure;
        float disparity_roi_edge_centroid = -1.0f;
        float z_roi_edge_centroid = -1.0f;
        if (roi_edge_centroid_depth_enabled && direct_yolo_match &&
            right_det) {
            if (gpu_candidate) {
                left_edge_centroid_measure =
                    pointFromGpuCandidate(gpu_candidate->left_edge_centroid);
                right_edge_centroid_measure =
                    pointFromGpuCandidate(gpu_candidate->right_edge_centroid);
            } else if (image_available) {
                left_edge_centroid_measure =
                    edge_centroid(left_cpu, left_pitch, left_det);
                right_edge_centroid_measure =
                    edge_centroid(right_cpu, right_pitch, *right_det);
            }
            if (left_edge_centroid_measure.valid && right_edge_centroid_measure.valid &&
                std::abs(left_edge_centroid_measure.cy -
                         right_edge_centroid_measure.cy) <= y_tol) {
                disparity_roi_edge_centroid =
                    left_edge_centroid_measure.cx - right_edge_centroid_measure.cx;
                z_roi_edge_centroid = depth_from_disparity(disparity_roi_edge_centroid);
            }
        }

        PointMeasure2D left_radial_measure;
        PointMeasure2D right_radial_measure;
        float disparity_roi_radial_center = -1.0f;
        float z_roi_radial_center = -1.0f;
        if (roi_radial_center_depth_enabled && direct_yolo_match &&
            right_det) {
            if (gpu_candidate) {
                left_radial_measure =
                    pointFromGpuCandidate(gpu_candidate->left_radial_center);
                right_radial_measure =
                    pointFromGpuCandidate(gpu_candidate->right_radial_center);
            } else if (image_available) {
                left_radial_measure =
                    radial_center(left_cpu, left_pitch, left_det);
                right_radial_measure =
                    radial_center(right_cpu, right_pitch, *right_det);
            }
            if (left_radial_measure.valid && right_radial_measure.valid &&
                std::abs(left_radial_measure.cy - right_radial_measure.cy) <= y_tol) {
                disparity_roi_radial_center =
                    left_radial_measure.cx - right_radial_measure.cx;
                z_roi_radial_center = depth_from_disparity(disparity_roi_radial_center);
            }
        }

        PointMeasure2D left_edge_pair_measure;
        PointMeasure2D right_edge_pair_measure;
        float disparity_roi_edge_pair_center = -1.0f;
        float z_roi_edge_pair_center = -1.0f;
        if (roi_edge_pair_center_depth_enabled && direct_yolo_match &&
            right_det) {
            if (gpu_candidate) {
                left_edge_pair_measure =
                    pointFromGpuCandidate(gpu_candidate->left_edge_pair_center);
                right_edge_pair_measure =
                    pointFromGpuCandidate(gpu_candidate->right_edge_pair_center);
            } else if (image_available) {
                left_edge_pair_measure =
                    edge_pair_center(left_cpu, left_pitch, left_det);
                right_edge_pair_measure =
                    edge_pair_center(right_cpu, right_pitch, *right_det);
            }
            if (left_edge_pair_measure.valid && right_edge_pair_measure.valid &&
                std::abs(left_edge_pair_measure.cy - right_edge_pair_measure.cy) <= y_tol) {
                disparity_roi_edge_pair_center =
                    left_edge_pair_measure.cx - right_edge_pair_measure.cx;
                z_roi_edge_pair_center = depth_from_disparity(disparity_roi_edge_pair_center);
            }
        }

        const float refined_dy = std::abs(left_circle.cy - right_circle.cy);
        bool circle_geometry_valid = true;
        bool epipolar_bad = false;
        bool size_bad = false;
        bool nonpositive_disp_bad = false;
        bool over_max_disp_bad = false;
        if (refined_dy > y_tol) {
            epipolar_bad = true;
            circle_geometry_valid = false;
        }

        const float radius_ratio = std::max(left_circle.radius / right_circle.radius,
                                            right_circle.radius / left_circle.radius);
        if (radius_ratio > max_ratio) {
            size_bad = true;
            circle_geometry_valid = false;
        }

        const float circle_disparity = left_circle.cx - right_circle.cx;
        const bool circle_disp_positive =
            circle_disparity > 0.0f &&
            circle_disparity <= static_cast<float>(config_.max_disparity);
        const bool has_feature_proxy_circle =
            left_circle.source == kCircleSourceFeatureProxy ||
            right_circle.source == kCircleSourceFeatureProxy;
        const float feature_initial_disparity =
            (right_det && yolo_disparity > 0.0f &&
             yolo_disparity <= static_cast<float>(config_.max_disparity))
                ? yolo_disparity
                : (circle_disp_positive ? circle_disparity : -1.0f);
        auto record_cpu_feature_elapsed =
            [](const char* aggregate_metric,
               const char* mode_metric,
               const Clock::time_point& start) {
                const double ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - start).count();
                if (aggregate_metric) {
                    globalPerf().record(aggregate_metric, ms);
                }
                if (mode_metric) {
                    globalPerf().record(mode_metric, ms);
                }
            };
        auto match_sparse_feature_cpu_timed =
            [&](const Detection& left_feature_det,
                const Detection& right_feature_det,
                bool source_left,
                SparseFeatureMode mode,
                const char* aggregate_metric,
                const char* mode_metric) -> SparseFeatureDisparityResult {
                const auto start = Clock::now();
                SparseFeatureDisparityResult result =
                    matchSparseFeatureDisparityCPU(
                        left_cpu, left_pitch, right_cpu, right_pitch,
                        img_width, img_height,
                        left_feature_det, right_feature_det,
                        source_left,
                        feature_initial_disparity,
                        feature_cfg,
                        config_.max_disparity,
                        focal,
                        baseline,
                        mode);
                record_cpu_feature_elapsed(aggregate_metric, mode_metric, start);
                return result;
            };
        auto match_opencv_feature_cpu_timed =
            [&](const Detection& left_feature_det,
                const Detection& right_feature_det,
                bool source_left,
                OpenCVFeatureMode mode,
                const char* aggregate_metric,
                const char* mode_metric) -> SparseFeatureDisparityResult {
                const auto start = Clock::now();
                SparseFeatureDisparityResult result =
                    matchOpenCVFeatureDisparityCPU(
                        left_cpu, left_pitch, right_cpu, right_pitch,
                        img_width, img_height,
                        left_feature_det, right_feature_det,
                        source_left,
                        feature_initial_disparity,
                        feature_cfg,
                        config_.max_disparity,
                        focal,
                        baseline,
                        mode);
                record_cpu_feature_elapsed(aggregate_metric, mode_metric, start);
                return result;
            };

        SparseFeatureDisparityResult corner_points_result;
        float z_roi_corner_points = -1.0f;
        if (roi_corner_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (gpu_candidate) {
                corner_points_result =
                    sparseFromGpuCandidate(gpu_candidate->corner_points);
                if (!validateSparseFeatureGeometry(
                        corner_points_result, left_det, *right_det,
                        feature_initial_disparity, feature_cfg,
                        focal, baseline)) {
                    corner_points_result = SparseFeatureDisparityResult{};
                }
            }
            if (!corner_points_result.valid && image_available) {
                corner_points_result = match_sparse_feature_cpu_timed(
                    left_det, *right_det, true,
                    SparseFeatureMode::CORNER,
                    "Stage2_CPUFeatureSparse",
                    "Stage2_CPUFeatureSparseCorner");
            }
            if (corner_points_result.valid) {
                z_roi_corner_points =
                    depth_from_disparity(corner_points_result.disparity);
                corner_points_result.valid = z_roi_corner_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult texture_points_result;
        float z_roi_texture_points = -1.0f;
        if (roi_texture_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (gpu_candidate) {
                texture_points_result =
                    sparseFromGpuCandidate(gpu_candidate->texture_points);
                if (!validateSparseFeatureGeometry(
                        texture_points_result, left_det, *right_det,
                        feature_initial_disparity, feature_cfg,
                        focal, baseline)) {
                    texture_points_result = SparseFeatureDisparityResult{};
                }
            }
            if (!texture_points_result.valid && image_available) {
                texture_points_result = match_sparse_feature_cpu_timed(
                    left_det, *right_det, true,
                    SparseFeatureMode::TEXTURE,
                    "Stage2_CPUFeatureSparse",
                    "Stage2_CPUFeatureSparseTexture");
            }
            if (texture_points_result.valid) {
                z_roi_texture_points =
                    depth_from_disparity(texture_points_result.disparity);
                texture_points_result.valid = z_roi_texture_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult binary_points_result;
        float z_roi_binary_points = -1.0f;
        if (roi_binary_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (gpu_candidate) {
                binary_points_result =
                    sparseFromGpuCandidate(gpu_candidate->binary_points);
                if (!validateSparseFeatureGeometry(
                        binary_points_result, left_det, *right_det,
                        feature_initial_disparity, feature_cfg,
                        focal, baseline)) {
                    binary_points_result = SparseFeatureDisparityResult{};
                }
            }
            if (!binary_points_result.valid && image_available) {
                binary_points_result = match_sparse_feature_cpu_timed(
                    left_det, *right_det, true,
                    SparseFeatureMode::BINARY,
                    "Stage2_CPUFeatureSparse",
                    "Stage2_CPUFeatureSparseBinary");
            }
            if (binary_points_result.valid) {
                z_roi_binary_points =
                    depth_from_disparity(binary_points_result.disparity);
                binary_points_result.valid = z_roi_binary_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult orb_points_result;
        float z_roi_orb_points = -1.0f;
        if (roi_orb_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (gpu_image_available && stream != nullptr) {
                const auto orb_gpu_start = Clock::now();
                orb_points_result = matchOpenCVORBDisparityGPU(
                    left_gpu, left_gpu_pitch,
                    right_gpu, right_gpu_pitch,
                    img_width, img_height,
                    left_det, *right_det,
                    feature_initial_disparity,
                    feature_cfg,
                    config_.max_disparity,
                    focal,
                    baseline,
                    stream);
                const double orb_gpu_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - orb_gpu_start).count();
                globalPerf().record("Stage2_OpenCVCudaORB", orb_gpu_ms);
            }
            if (!orb_points_result.valid && image_available) {
                orb_points_result = match_opencv_feature_cpu_timed(
                    left_det, *right_det, true,
                    OpenCVFeatureMode::ORB,
                    "Stage2_CPUFeatureOpenCV",
                    "Stage2_CPUFeatureOpenCVORB");
            }
            if (orb_points_result.valid) {
                z_roi_orb_points =
                    depth_from_disparity(orb_points_result.disparity);
                orb_points_result.valid = z_roi_orb_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult brisk_points_result;
        float z_roi_brisk_points = -1.0f;
        if (roi_brisk_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (image_available) {
                brisk_points_result = match_opencv_feature_cpu_timed(
                    left_det, *right_det, true,
                    OpenCVFeatureMode::BRISK,
                    "Stage2_CPUFeatureOpenCV",
                    "Stage2_CPUFeatureOpenCVBRISK");
            }
            if (brisk_points_result.valid) {
                z_roi_brisk_points =
                    depth_from_disparity(brisk_points_result.disparity);
                brisk_points_result.valid = z_roi_brisk_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult akaze_points_result;
        float z_roi_akaze_points = -1.0f;
        if (roi_akaze_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (image_available) {
                akaze_points_result = match_opencv_feature_cpu_timed(
                    left_det, *right_det, true,
                    OpenCVFeatureMode::AKAZE,
                    "Stage2_CPUFeatureOpenCV",
                    "Stage2_CPUFeatureOpenCVAKAZE");
            }
            if (akaze_points_result.valid) {
                z_roi_akaze_points =
                    depth_from_disparity(akaze_points_result.disparity);
                akaze_points_result.valid = z_roi_akaze_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult sift_points_result;
        float z_roi_sift_points = -1.0f;
        if (roi_sift_points_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f) {
            if (image_available) {
                sift_points_result = match_opencv_feature_cpu_timed(
                    left_det, *right_det, true,
                    OpenCVFeatureMode::SIFT,
                    "Stage2_CPUFeatureOpenCV",
                    "Stage2_CPUFeatureOpenCVSIFT");
            }
            if (sift_points_result.valid) {
                z_roi_sift_points =
                    depth_from_disparity(sift_points_result.disparity);
                sift_points_result.valid = z_roi_sift_points > 0.0f;
            }
        }

        SparseFeatureDisparityResult iou_region_color_patch_result;
        float z_roi_iou_region_color_patch = -1.0f;
        if (roi_iou_region_color_patch_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f && gpu_candidate) {
            iou_region_color_patch_result =
                sparseFromGpuCandidate(gpu_candidate->iou_region_color_patch);
            if (!validateSparseFeatureGeometry(
                    iou_region_color_patch_result, left_det, *right_det,
                    feature_initial_disparity, feature_cfg,
                    focal, baseline)) {
                iou_region_color_patch_result = SparseFeatureDisparityResult{};
            }
            if (iou_region_color_patch_result.valid) {
                z_roi_iou_region_color_patch =
                    depth_from_disparity(iou_region_color_patch_result.disparity);
                iou_region_color_patch_result.valid =
                    z_roi_iou_region_color_patch > 0.0f;
            }
        }

        SparseFeatureDisparityResult patch_iou_color_edge_result;
        float z_roi_patch_iou_color_edge = -1.0f;
        if (roi_patch_iou_color_edge_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f && gpu_candidate) {
            patch_iou_color_edge_result =
                sparseFromGpuCandidate(gpu_candidate->patch_iou_color_edge);
            if (!validateSparseFeatureGeometry(
                    patch_iou_color_edge_result, left_det, *right_det,
                    feature_initial_disparity, feature_cfg,
                    focal, baseline)) {
                patch_iou_color_edge_result = SparseFeatureDisparityResult{};
            }
            if (patch_iou_color_edge_result.valid) {
                z_roi_patch_iou_color_edge =
                    depth_from_disparity(patch_iou_color_edge_result.disparity);
                patch_iou_color_edge_result.valid =
                    z_roi_patch_iou_color_edge > 0.0f;
            }
        }

        SparseFeatureDisparityResult cuda_template_match_result;
        float z_roi_cuda_template_match = -1.0f;
        if (roi_cuda_template_match_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f &&
            gpu_image_available && stream != nullptr) {
            const auto cuda_template_start = Clock::now();
            cuda_template_match_result = matchCudaTemplateDisparityGPU(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                img_width, img_height,
                left_det, *right_det,
                feature_initial_disparity,
                feature_cfg,
                config_.max_disparity,
                focal,
                baseline,
                stream);
            const double cuda_template_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - cuda_template_start).count();
            globalPerf().record("Stage2_OpenCVCudaTemplateMatch",
                                cuda_template_ms);
            if (cuda_template_match_result.valid) {
                z_roi_cuda_template_match =
                    depth_from_disparity(cuda_template_match_result.disparity);
                cuda_template_match_result.valid =
                    z_roi_cuda_template_match > 0.0f;
            }
        }

        SparseFeatureDisparityResult cuda_stereo_bm_result;
        float z_roi_cuda_stereo_bm = -1.0f;
        if (roi_cuda_stereo_bm_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f &&
            gpu_image_available && stream != nullptr) {
            const auto cuda_bm_start = Clock::now();
            cuda_stereo_bm_result = matchCudaStereoBMDisparityGPU(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                img_width, img_height,
                left_det, *right_det,
                feature_initial_disparity,
                feature_cfg,
                config_.max_disparity,
                focal,
                baseline,
                stream);
            const double cuda_bm_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - cuda_bm_start).count();
            globalPerf().record("Stage2_OpenCVCudaStereoBM", cuda_bm_ms);
            if (cuda_stereo_bm_result.valid) {
                z_roi_cuda_stereo_bm =
                    depth_from_disparity(cuda_stereo_bm_result.disparity);
                cuda_stereo_bm_result.valid =
                    z_roi_cuda_stereo_bm > 0.0f;
            }
        }

        SparseFeatureDisparityResult cuda_stereo_sgm_result;
        float z_roi_cuda_stereo_sgm = -1.0f;
        if (roi_cuda_stereo_sgm_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f &&
            gpu_image_available && stream != nullptr) {
            const auto cuda_sgm_start = Clock::now();
            cuda_stereo_sgm_result = matchCudaStereoSGMDisparityGPU(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                img_width, img_height,
                left_det, *right_det,
                feature_initial_disparity,
                feature_cfg,
                config_.max_disparity,
                focal,
                baseline,
                stream);
            const double cuda_sgm_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - cuda_sgm_start).count();
            globalPerf().record("Stage2_OpenCVCudaStereoSGM", cuda_sgm_ms);
            if (cuda_stereo_sgm_result.valid) {
                z_roi_cuda_stereo_sgm =
                    depth_from_disparity(cuda_stereo_sgm_result.disparity);
                cuda_stereo_sgm_result.valid =
                    z_roi_cuda_stereo_sgm > 0.0f;
            }
        }

        SparseFeatureDisparityResult neural_feature_result;
        float z_roi_neural_feature = -1.0f;
        if (neural_feature_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f &&
            gpu_image_available && stream != nullptr) {
            const NeuralFeatureMatchResult neural =
                neural_feature_matcher_->matchGpuRoi(
                    left_gpu, left_gpu_pitch,
                    right_gpu, right_gpu_pitch,
                    left_bgr_gpu, left_bgr_pitch,
                    right_bgr_gpu, right_bgr_pitch,
                    img_width, img_height,
                    left_det, *right_det,
                    feature_initial_disparity,
                    stream);
            if (neural.inference_ms > 0.0f) {
                globalPerf().record("Stage2_NeuralFeatureMatch",
                                    neural.inference_ms);
            }
            if (neural.valid) {
                neural_feature_result.valid = true;
                neural_feature_result.disparity = neural.disparity;
                neural_feature_result.stddev = neural.stddev_px;
                neural_feature_result.confidence = neural.confidence;
                neural_feature_result.support =
                    static_cast<int>(neural.matches.size());
                float sx = 0.0f;
                float sy = 0.0f;
                float rx = 0.0f;
                float ry = 0.0f;
                for (const auto& m : neural.matches) {
                    sx += m.left_x;
                    sy += m.left_y;
                    rx += m.right_x;
                    ry += m.right_y;
                }
                const float inv = 1.0f /
                    static_cast<float>(std::max(1, neural_feature_result.support));
                neural_feature_result.anchor_cx = sx * inv;
                neural_feature_result.anchor_cy = sy * inv;
                neural_feature_result.right_anchor_cx = rx * inv;
                neural_feature_result.right_anchor_cy = ry * inv;
                if (!validateSparseFeatureGeometry(
                        neural_feature_result, left_det, *right_det,
                        feature_initial_disparity, feature_cfg,
                        focal, baseline)) {
                    neural_feature_result = SparseFeatureDisparityResult{};
                }
            }
            if (neural_feature_result.valid) {
                z_roi_neural_feature =
                    depth_from_disparity(neural_feature_result.disparity);
                neural_feature_result.valid = z_roi_neural_feature > 0.0f;
            }
        }

        SparseFeatureDisparityResult fallback_feature_result;
        float z_fallback_feature_points = -1.0f;
        if (fallback_feature_points_enabled && image_available &&
            is_fallback_match && feature_initial_disparity > 0.0f) {
            const bool source_left = match_source == 2;
            const Detection right_proxy =
                right_det ? *right_det : detectionFromCircleCPU(right_circle, left_det);
            const Detection& right_for_feature = right_det ? *right_det : right_proxy;
            SparseFeatureDisparityResult corner_fb =
                match_sparse_feature_cpu_timed(
                    left_det, right_for_feature, source_left,
                    SparseFeatureMode::CORNER,
                    "Stage2_CPUFallbackFeatureSparse",
                    "Stage2_CPUFallbackFeatureSparseCorner");
            SparseFeatureDisparityResult texture_fb =
                match_sparse_feature_cpu_timed(
                    left_det, right_for_feature, source_left,
                    SparseFeatureMode::TEXTURE,
                    "Stage2_CPUFallbackFeatureSparse",
                    "Stage2_CPUFallbackFeatureSparseTexture");
            SparseFeatureDisparityResult binary_fb =
                match_sparse_feature_cpu_timed(
                    left_det, right_for_feature, source_left,
                    SparseFeatureMode::BINARY,
                    "Stage2_CPUFallbackFeatureSparse",
                    "Stage2_CPUFallbackFeatureSparseBinary");
            fallback_feature_result = corner_fb;
            if (!fallback_feature_result.valid ||
                (texture_fb.valid &&
                 texture_fb.confidence > fallback_feature_result.confidence)) {
                fallback_feature_result = texture_fb;
            }
            if (!fallback_feature_result.valid ||
                (binary_fb.valid &&
                 binary_fb.confidence > fallback_feature_result.confidence)) {
                fallback_feature_result = binary_fb;
            }
            if (roi_orb_points_depth_enabled) {
                SparseFeatureDisparityResult orb_fb =
                    match_opencv_feature_cpu_timed(
                        left_det, right_for_feature, source_left,
                        OpenCVFeatureMode::ORB,
                        "Stage2_CPUFallbackFeatureOpenCV",
                        "Stage2_CPUFallbackFeatureOpenCVORB");
                if (!fallback_feature_result.valid ||
                    (orb_fb.valid &&
                     orb_fb.confidence > fallback_feature_result.confidence)) {
                    fallback_feature_result = orb_fb;
                }
            }
            if (roi_brisk_points_depth_enabled) {
                SparseFeatureDisparityResult brisk_fb =
                    match_opencv_feature_cpu_timed(
                        left_det, right_for_feature, source_left,
                        OpenCVFeatureMode::BRISK,
                        "Stage2_CPUFallbackFeatureOpenCV",
                        "Stage2_CPUFallbackFeatureOpenCVBRISK");
                if (!fallback_feature_result.valid ||
                    (brisk_fb.valid &&
                     brisk_fb.confidence > fallback_feature_result.confidence)) {
                    fallback_feature_result = brisk_fb;
                }
            }
            if (roi_akaze_points_depth_enabled) {
                SparseFeatureDisparityResult akaze_fb =
                    match_opencv_feature_cpu_timed(
                        left_det, right_for_feature, source_left,
                        OpenCVFeatureMode::AKAZE,
                        "Stage2_CPUFallbackFeatureOpenCV",
                        "Stage2_CPUFallbackFeatureOpenCVAKAZE");
                if (!fallback_feature_result.valid ||
                    (akaze_fb.valid &&
                     akaze_fb.confidence > fallback_feature_result.confidence)) {
                    fallback_feature_result = akaze_fb;
                }
            }
            if (roi_sift_points_depth_enabled) {
                SparseFeatureDisparityResult sift_fb =
                    match_opencv_feature_cpu_timed(
                        left_det, right_for_feature, source_left,
                        OpenCVFeatureMode::SIFT,
                        "Stage2_CPUFallbackFeatureOpenCV",
                        "Stage2_CPUFallbackFeatureOpenCVSIFT");
                if (!fallback_feature_result.valid ||
                    (sift_fb.valid &&
                     sift_fb.confidence > fallback_feature_result.confidence)) {
                    fallback_feature_result = sift_fb;
                }
            }
            if (fallback_feature_result.valid) {
                z_fallback_feature_points =
                    depth_from_disparity(fallback_feature_result.disparity);
                fallback_feature_result.valid = z_fallback_feature_points > 0.0f;
            }
        }

        if (circle_disparity <= 0.0f) {
            nonpositive_disp_bad = true;
            circle_geometry_valid = false;
        }
        if (circle_disparity > config_.max_disparity) {
            over_max_disp_bad = true;
            circle_geometry_valid = false;
        }
        if (!circle_geometry_valid && z_yolo <= 0.0f &&
            z_bbox_edge_final <= 0.0f &&
            z_roi_edge_centroid <= 0.0f &&
            z_roi_radial_center <= 0.0f &&
            z_roi_edge_pair_center <= 0.0f &&
            z_roi_corner_points <= 0.0f &&
            z_roi_texture_points <= 0.0f &&
            z_roi_binary_points <= 0.0f &&
            z_roi_orb_points <= 0.0f &&
            z_roi_brisk_points <= 0.0f &&
            z_roi_akaze_points <= 0.0f &&
            z_roi_sift_points <= 0.0f &&
            z_roi_iou_region_color_patch <= 0.0f &&
            z_roi_patch_iou_color_edge <= 0.0f &&
            z_roi_cuda_template_match <= 0.0f &&
            z_roi_cuda_stereo_bm <= 0.0f &&
            z_roi_cuda_stereo_sgm <= 0.0f &&
            z_roi_neural_feature <= 0.0f &&
            z_fallback_feature_points <= 0.0f) {
            if (epipolar_bad) ++local_stats.epipolar_reject;
            if (size_bad) ++local_stats.size_reject;
            if (nonpositive_disp_bad) ++local_stats.nonpositive_disparity;
            if (over_max_disp_bad) ++local_stats.over_max_disparity;
            return false;
        }

        const bool measured_circle_geometry_valid =
            circle_geometry_valid && !has_feature_proxy_circle;
        const float z_circle_raw =
            measured_circle_geometry_valid ? (fb / circle_disparity) : -1.0f;
        const bool circle_depth_valid =
            z_circle_raw >= config_.depth.min_depth &&
            z_circle_raw <= config_.depth.max_depth;
        const bool circle_candidate_valid =
            circle_depth_enabled &&
            left_circle.source == kCircleSourceRoiFit &&
            right_circle.source == kCircleSourceRoiFit &&
            circle_depth_valid;
        const bool epipolar_fallback_depth_valid =
            is_fallback_match &&
            epipolar_fallback_enabled &&
            circle_depth_valid &&
            (left_circle.source == kCircleSourceEpipolarSearch ||
             right_circle.source == kCircleSourceEpipolarSearch);
        const bool fallback_template_depth_valid =
            is_fallback_match &&
            fallback_template_enabled &&
            circle_depth_valid &&
            (left_circle.source == kCircleSourceTemplateSearch ||
             right_circle.source == kCircleSourceTemplateSearch);
        const bool any_fallback_depth_valid =
            epipolar_fallback_depth_valid || fallback_template_depth_valid;

        float disparity_circle_left_edge = -1.0f;
        float disparity_circle_right_edge = -1.0f;
        float z_circle_left_edge = -1.0f;
        float z_circle_right_edge = -1.0f;
        if (circle_edges_depth_enabled &&
            measured_circle_geometry_valid &&
            left_circle.source == kCircleSourceRoiFit &&
            right_circle.source == kCircleSourceRoiFit) {
            disparity_circle_left_edge =
                (left_circle.cx - left_circle.radius) -
                (right_circle.cx - right_circle.radius);
            disparity_circle_right_edge =
                (left_circle.cx + left_circle.radius) -
                (right_circle.cx + right_circle.radius);
            z_circle_left_edge = depth_from_disparity(disparity_circle_left_edge);
            z_circle_right_edge = depth_from_disparity(disparity_circle_right_edge);
        }

        SubpixelDisparityResult center_patch_result;
        bool center_patch_valid_for_obj = false;
        if (roi_center_patch_depth_enabled &&
            measured_circle_geometry_valid) {
            if (gpu_candidate) {
                center_patch_result =
                    subpixelFromGpuCandidate(gpu_candidate->center_patch);
            } else if (image_available) {
                center_patch_result = refineDisparityByROICenterPatchCPU(
                    left_cpu, left_pitch, right_cpu, right_pitch,
                    img_width, img_height,
                    left_circle, right_circle,
                    config_.dual_yolo,
                    config_.max_disparity,
                    focal,
                    baseline);
            }
            center_patch_valid_for_obj = center_patch_result.valid;
        }
        const float z_roi_center_patch =
            center_patch_valid_for_obj
                ? depth_from_disparity(center_patch_result.disparity)
                : -1.0f;
        center_patch_valid_for_obj = z_roi_center_patch > 0.0f;

        float disparity = -1.0f;
        int depth_source = 0;
        float disparity_conf = 1.0f;
        bool subpixel_attempted_for_obj = false;
        bool subpixel_valid_for_obj = false;
        SubpixelDisparityResult subpixel_result;
        const float subpixel_budget_ms =
            std::max(0.0f, config_.dual_yolo.subpixel_time_budget_ms);
        if (use_subpixel_depth &&
            measured_circle_geometry_valid &&
            (subpixel_budget_ms <= 0.0f ||
             local_stats.subpixel_time_ms < static_cast<double>(subpixel_budget_ms))) {
            ++local_stats.subpixel_attempted;
            subpixel_attempted_for_obj = true;
            const auto subpixel_start = Clock::now();
            const SubpixelDisparityResult refined = gpu_candidate
                ? subpixelFromGpuCandidate(gpu_candidate->multi_point)
                : refineDisparityByROIMultiPointCPU(
                      left_cpu, left_pitch, right_cpu, right_pitch,
                      img_width, img_height,
                      left_circle, right_circle,
                      config_.dual_yolo,
                      config_.max_disparity,
                      focal,
                      baseline);
            const double subpixel_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - subpixel_start).count();
            local_stats.subpixel_time_ms += subpixel_ms;
            local_stats.subpixel_max_time_ms =
                std::max(local_stats.subpixel_max_time_ms, subpixel_ms);
            if (refined.delta_gate_px > 0.0f) {
                if (local_stats.subpixel_gate_min_px <= 0.0f ||
                    refined.delta_gate_px < local_stats.subpixel_gate_min_px) {
                    local_stats.subpixel_gate_min_px = refined.delta_gate_px;
                }
                local_stats.subpixel_gate_max_px =
                    std::max(local_stats.subpixel_gate_max_px,
                             refined.delta_gate_px);
            }
            globalPerf().record("Stage2_SubpixelMatch", subpixel_ms);
            subpixel_result = refined;
            if (refined.valid) {
                disparity = refined.disparity;
                subpixel_valid_for_obj = true;
                disparity_conf = std::clamp(0.70f + 0.30f * refined.confidence,
                                            0.0f, 1.0f);
                local_stats.subpixel_support_sum += refined.support;
                local_stats.subpixel_support_max =
                    std::max(local_stats.subpixel_support_max, refined.support);
                ++local_stats.subpixel_refined;
            } else {
                ++local_stats.subpixel_rejected;
                if (refined.low_confidence) {
                    ++local_stats.subpixel_low_conf;
                }
            }
        } else if (use_subpixel_depth && measured_circle_geometry_valid) {
            ++local_stats.subpixel_budget_skip;
        }

        const float z_subpixel =
            subpixel_valid_for_obj ? fb / subpixel_result.disparity : -1.0f;

        DepthCandidateBuilderInput depth_candidate_input;
        depth_candidate_input.left_detection = left_det;
        depth_candidate_input.left_circle = left_circle;
        depth_candidate_input.left_edge_centroid_measure =
            left_edge_centroid_measure;
        depth_candidate_input.left_radial_measure = left_radial_measure;
        depth_candidate_input.left_edge_pair_measure = left_edge_pair_measure;
        depth_candidate_input.subpixel_valid = subpixel_valid_for_obj;
        depth_candidate_input.subpixel_result = subpixel_result;
        depth_candidate_input.z_subpixel = z_subpixel;
        depth_candidate_input.fallback_feature_result = fallback_feature_result;
        depth_candidate_input.z_fallback_feature_points =
            z_fallback_feature_points;
        depth_candidate_input.fallback_template_depth_valid =
            fallback_template_depth_valid;
        depth_candidate_input.epipolar_fallback_depth_valid =
            epipolar_fallback_depth_valid;
        depth_candidate_input.circle_candidate_valid = circle_candidate_valid;
        depth_candidate_input.circle_disparity = circle_disparity;
        depth_candidate_input.z_circle_raw = z_circle_raw;
        depth_candidate_input.circle_confidence =
            std::min(left_circle.confidence, right_circle.confidence);
        depth_candidate_input.disparity_circle_left_edge =
            disparity_circle_left_edge;
        depth_candidate_input.z_circle_left_edge = z_circle_left_edge;
        depth_candidate_input.disparity_circle_right_edge =
            disparity_circle_right_edge;
        depth_candidate_input.z_circle_right_edge = z_circle_right_edge;
        depth_candidate_input.center_patch_valid = center_patch_valid_for_obj;
        depth_candidate_input.center_patch_result = center_patch_result;
        depth_candidate_input.z_roi_center_patch = z_roi_center_patch;
        depth_candidate_input.iou_region_color_patch_result =
            iou_region_color_patch_result;
        depth_candidate_input.z_roi_iou_region_color_patch =
            z_roi_iou_region_color_patch;
        depth_candidate_input.patch_iou_color_edge_result =
            patch_iou_color_edge_result;
        depth_candidate_input.z_roi_patch_iou_color_edge =
            z_roi_patch_iou_color_edge;
        depth_candidate_input.cuda_template_match_result =
            cuda_template_match_result;
        depth_candidate_input.z_roi_cuda_template_match =
            z_roi_cuda_template_match;
        depth_candidate_input.cuda_stereo_bm_result =
            cuda_stereo_bm_result;
        depth_candidate_input.z_roi_cuda_stereo_bm =
            z_roi_cuda_stereo_bm;
        depth_candidate_input.cuda_stereo_sgm_result =
            cuda_stereo_sgm_result;
        depth_candidate_input.z_roi_cuda_stereo_sgm =
            z_roi_cuda_stereo_sgm;
        depth_candidate_input.neural_feature_result = neural_feature_result;
        depth_candidate_input.z_roi_neural_feature = z_roi_neural_feature;
        depth_candidate_input.corner_points_result = corner_points_result;
        depth_candidate_input.z_roi_corner_points = z_roi_corner_points;
        depth_candidate_input.texture_points_result = texture_points_result;
        depth_candidate_input.z_roi_texture_points = z_roi_texture_points;
        depth_candidate_input.binary_points_result = binary_points_result;
        depth_candidate_input.z_roi_binary_points = z_roi_binary_points;
        depth_candidate_input.orb_points_result = orb_points_result;
        depth_candidate_input.z_roi_orb_points = z_roi_orb_points;
        depth_candidate_input.brisk_points_result = brisk_points_result;
        depth_candidate_input.z_roi_brisk_points = z_roi_brisk_points;
        depth_candidate_input.akaze_points_result = akaze_points_result;
        depth_candidate_input.z_roi_akaze_points = z_roi_akaze_points;
        depth_candidate_input.sift_points_result = sift_points_result;
        depth_candidate_input.z_roi_sift_points = z_roi_sift_points;
        depth_candidate_input.disparity_roi_radial_center =
            disparity_roi_radial_center;
        depth_candidate_input.z_roi_radial_center = z_roi_radial_center;
        depth_candidate_input.disparity_roi_edge_pair_center =
            disparity_roi_edge_pair_center;
        depth_candidate_input.z_roi_edge_pair_center = z_roi_edge_pair_center;
        depth_candidate_input.disparity_roi_edge_centroid =
            disparity_roi_edge_centroid;
        depth_candidate_input.z_roi_edge_centroid = z_roi_edge_centroid;
        depth_candidate_input.yolo_disparity = yolo_disparity;
        depth_candidate_input.z_yolo = z_yolo;
        depth_candidate_input.disparity_bbox_edge_final =
            disparity_bbox_edge_final;
        depth_candidate_input.z_bbox_edge_final = z_bbox_edge_final;
        depth_candidate_input.disparity_bbox_left_edge =
            disparity_bbox_left_edge;
        depth_candidate_input.z_bbox_left_edge = z_bbox_left_edge;
        depth_candidate_input.disparity_bbox_right_edge =
            disparity_bbox_right_edge;
        depth_candidate_input.z_bbox_right_edge = z_bbox_right_edge;

        const DepthCandidateBuildResult depth_candidate_build =
            buildDepthCandidateObservations(depth_candidate_input);
        const DepthCandidateSelection& depth_selection =
            depth_candidate_build.selection;
        if (depth_selection.valid) {
            disparity = depth_selection.observation.disparity_px;
            depth_source = depth_selection.observation.stereo_depth_source;
            disparity_conf = depth_selection.observation.fusion_confidence;
        } else {
            ++local_stats.depth_reject;
            return false;
        }

        const float z = depth_selection.observation.depth_m;
        if (z < config_.depth.min_depth || z > config_.depth.max_depth) {
            ++local_stats.depth_reject;
            return false;
        }

        const float dy_norm = std::min(1.0f, refined_dy / y_tol);
        const float geom_conf = circle_geometry_valid
            ? std::max(0.2f, 1.0f - 0.5f * dy_norm)
            : 0.45f;
        float anchor_cx = depth_selection.observation.anchor_left_x;
        float anchor_cy = depth_selection.observation.anchor_left_y;
        if (!std::isfinite(anchor_cx) || !std::isfinite(anchor_cy) ||
            (anchor_cx == 0.0f && anchor_cy == 0.0f)) {
            anchor_cx = left_det.cx;
            anchor_cy = left_det.cy;
        }
        obj.x = (anchor_cx - cx0) * z / focal;
        obj.y = (anchor_cy - cy0) * z / focal;
        obj.z = z;
        obj.raw_x = obj.x;
        obj.raw_y = obj.y;
        obj.raw_z = obj.z;
        obj.raw_observation_valid = 1;
        obj.z_stereo = z;
        obj.z_bbox_center = z_yolo;
        obj.z_bbox_left_edge = z_bbox_left_edge;
        obj.z_bbox_right_edge = z_bbox_right_edge;
        obj.z_circle_center = circle_candidate_valid ? z_circle_raw : -1.0f;
        obj.z_circle_left_edge = z_circle_left_edge;
        obj.z_circle_right_edge = z_circle_right_edge;
        obj.z_roi_edge_centroid = z_roi_edge_centroid;
        obj.z_roi_radial_center = z_roi_radial_center;
        obj.z_roi_edge_pair_center = z_roi_edge_pair_center;
        obj.z_roi_corner_points = z_roi_corner_points;
        obj.z_roi_texture_points = z_roi_texture_points;
        obj.z_roi_binary_points = z_roi_binary_points;
        obj.z_roi_orb_points = z_roi_orb_points;
        obj.z_roi_brisk_points = z_roi_brisk_points;
        obj.z_roi_akaze_points = z_roi_akaze_points;
        obj.z_roi_sift_points = z_roi_sift_points;
        obj.z_roi_iou_region_color_patch = z_roi_iou_region_color_patch;
        obj.z_roi_patch_iou_color_edge = z_roi_patch_iou_color_edge;
        obj.z_roi_cuda_template_match = z_roi_cuda_template_match;
        obj.z_roi_cuda_stereo_bm = z_roi_cuda_stereo_bm;
        obj.z_roi_cuda_stereo_sgm = z_roi_cuda_stereo_sgm;
        obj.z_roi_neural_feature = z_roi_neural_feature;
        obj.z_roi_center_patch = z_roi_center_patch;
        obj.z_roi_multi_point = z_subpixel;
        obj.z_yolo_bbox_pair = z_yolo;
        obj.z_circle = circle_candidate_valid ? z_circle_raw : -1.0f;
        obj.z_subpixel = z_subpixel;
        obj.z_fallback = z_fallback_feature_points > 0.0f
            ? z_fallback_feature_points
            : (any_fallback_depth_valid ? z_circle_raw : -1.0f);
        obj.z_fallback_template = fallback_template_depth_valid ? z_circle_raw : -1.0f;
        obj.z_fallback_feature_points = z_fallback_feature_points;
        obj.disparity_bbox_center = (z_yolo > 0.0f) ? yolo_disparity : -1.0f;
        obj.disparity_bbox_left_edge =
            (z_bbox_left_edge > 0.0f) ? disparity_bbox_left_edge : -1.0f;
        obj.disparity_bbox_right_edge =
            (z_bbox_right_edge > 0.0f) ? disparity_bbox_right_edge : -1.0f;
        obj.disparity_circle_center =
            circle_candidate_valid ? circle_disparity : -1.0f;
        obj.disparity_circle_left_edge =
            (z_circle_left_edge > 0.0f) ? disparity_circle_left_edge : -1.0f;
        obj.disparity_circle_right_edge =
            (z_circle_right_edge > 0.0f) ? disparity_circle_right_edge : -1.0f;
        obj.disparity_roi_edge_centroid =
            (z_roi_edge_centroid > 0.0f) ? disparity_roi_edge_centroid : -1.0f;
        obj.disparity_roi_radial_center =
            (z_roi_radial_center > 0.0f) ? disparity_roi_radial_center : -1.0f;
        obj.disparity_roi_edge_pair_center =
            (z_roi_edge_pair_center > 0.0f) ? disparity_roi_edge_pair_center : -1.0f;
        obj.disparity_roi_corner_points =
            corner_points_result.valid ? corner_points_result.disparity : -1.0f;
        obj.disparity_roi_texture_points =
            texture_points_result.valid ? texture_points_result.disparity : -1.0f;
        obj.disparity_roi_binary_points =
            binary_points_result.valid ? binary_points_result.disparity : -1.0f;
        obj.disparity_roi_orb_points =
            orb_points_result.valid ? orb_points_result.disparity : -1.0f;
        obj.disparity_roi_brisk_points =
            brisk_points_result.valid ? brisk_points_result.disparity : -1.0f;
        obj.disparity_roi_akaze_points =
            akaze_points_result.valid ? akaze_points_result.disparity : -1.0f;
        obj.disparity_roi_sift_points =
            sift_points_result.valid ? sift_points_result.disparity : -1.0f;
        obj.disparity_roi_iou_region_color_patch =
            iou_region_color_patch_result.valid
                ? iou_region_color_patch_result.disparity
                : -1.0f;
        obj.disparity_roi_patch_iou_color_edge =
            patch_iou_color_edge_result.valid
                ? patch_iou_color_edge_result.disparity
                : -1.0f;
        obj.disparity_roi_cuda_template_match =
            cuda_template_match_result.valid
                ? cuda_template_match_result.disparity
                : -1.0f;
        obj.disparity_roi_cuda_stereo_bm =
            cuda_stereo_bm_result.valid
                ? cuda_stereo_bm_result.disparity
                : -1.0f;
        obj.disparity_roi_cuda_stereo_sgm =
            cuda_stereo_sgm_result.valid
                ? cuda_stereo_sgm_result.disparity
                : -1.0f;
        obj.disparity_roi_neural_feature =
            neural_feature_result.valid
                ? neural_feature_result.disparity
                : -1.0f;
        obj.disparity_roi_center_patch =
            center_patch_valid_for_obj ? center_patch_result.disparity : -1.0f;
        obj.disparity_roi_multi_point =
            subpixel_valid_for_obj ? subpixel_result.disparity : -1.0f;
        obj.disparity_fallback_template =
            fallback_template_depth_valid ? circle_disparity : -1.0f;
        obj.disparity_fallback_feature_points =
            fallback_feature_result.valid ? fallback_feature_result.disparity : -1.0f;
        obj.disparity_yolo = (z_yolo > 0.0f) ? yolo_disparity : -1.0f;
        obj.disparity_circle =
            circle_candidate_valid ? circle_disparity : -1.0f;
        obj.disparity_subpixel =
            subpixel_valid_for_obj ? subpixel_result.disparity : -1.0f;
        obj.left_bbox_cx = left_det.cx;
        obj.left_bbox_cy = left_det.cy;
        obj.left_bbox_w = left_det.width;
        obj.left_bbox_h = left_det.height;
        obj.left_bbox_conf = left_det.confidence;
        if (right_det) {
            obj.right_bbox_cx = right_det->cx;
            obj.right_bbox_cy = right_det->cy;
            obj.right_bbox_w = right_det->width;
            obj.right_bbox_h = right_det->height;
            obj.right_bbox_conf = right_det->confidence;
        }
        obj.left_circle_cx = left_circle.cx;
        obj.left_circle_cy = left_circle.cy;
        obj.left_circle_r = left_circle.radius;
        obj.right_circle_cx = right_circle.cx;
        obj.right_circle_cy = right_circle.cy;
        obj.right_circle_r = right_circle.radius;
        obj.left_circle_source = left_circle.source;
        obj.right_circle_source = right_circle.source;
        obj.epipolar_dy = refined_dy;
        obj.size_ratio = radius_ratio;
        obj.left_circle_conf = left_circle.confidence;
        obj.right_circle_conf = right_circle.confidence;
        obj.subpixel_valid = subpixel_valid_for_obj ? 1 : 0;
        obj.subpixel_attempted = subpixel_attempted_for_obj ? 1 : 0;
        obj.subpixel_support = subpixel_result.support;
        obj.subpixel_std_px =
            subpixel_attempted_for_obj ? subpixel_result.stddev : -1.0f;
        obj.subpixel_confidence = subpixel_result.confidence;
        obj.subpixel_gate_px = subpixel_result.delta_gate_px;
        obj.roi_corner_points_support = corner_points_result.support;
        obj.roi_corner_points_std_px =
            corner_points_result.valid ? corner_points_result.stddev : -1.0f;
        obj.roi_corner_points_confidence = corner_points_result.confidence;
        obj.roi_texture_points_support = texture_points_result.support;
        obj.roi_texture_points_std_px =
            texture_points_result.valid ? texture_points_result.stddev : -1.0f;
        obj.roi_texture_points_confidence = texture_points_result.confidence;
        obj.roi_binary_points_support = binary_points_result.support;
        obj.roi_binary_points_std_px =
            binary_points_result.valid ? binary_points_result.stddev : -1.0f;
        obj.roi_binary_points_confidence = binary_points_result.confidence;
        obj.roi_orb_points_support = orb_points_result.support;
        obj.roi_orb_points_std_px =
            orb_points_result.valid ? orb_points_result.stddev : -1.0f;
        obj.roi_orb_points_confidence = orb_points_result.confidence;
        obj.roi_brisk_points_support = brisk_points_result.support;
        obj.roi_brisk_points_std_px =
            brisk_points_result.valid ? brisk_points_result.stddev : -1.0f;
        obj.roi_brisk_points_confidence = brisk_points_result.confidence;
        obj.roi_akaze_points_support = akaze_points_result.support;
        obj.roi_akaze_points_std_px =
            akaze_points_result.valid ? akaze_points_result.stddev : -1.0f;
        obj.roi_akaze_points_confidence = akaze_points_result.confidence;
        obj.roi_sift_points_support = sift_points_result.support;
        obj.roi_sift_points_std_px =
            sift_points_result.valid ? sift_points_result.stddev : -1.0f;
        obj.roi_sift_points_confidence = sift_points_result.confidence;
        obj.roi_iou_region_color_patch_support =
            iou_region_color_patch_result.support;
        obj.roi_iou_region_color_patch_std_px =
            iou_region_color_patch_result.valid
                ? iou_region_color_patch_result.stddev
                : -1.0f;
        obj.roi_iou_region_color_patch_confidence =
            iou_region_color_patch_result.confidence;
        obj.roi_patch_iou_color_edge_support =
            patch_iou_color_edge_result.support;
        obj.roi_patch_iou_color_edge_std_px =
            patch_iou_color_edge_result.valid
                ? patch_iou_color_edge_result.stddev
                : -1.0f;
        obj.roi_patch_iou_color_edge_confidence =
            patch_iou_color_edge_result.confidence;
        obj.roi_cuda_template_match_support =
            cuda_template_match_result.support;
        obj.roi_cuda_template_match_std_px =
            cuda_template_match_result.valid
                ? cuda_template_match_result.stddev
                : -1.0f;
        obj.roi_cuda_template_match_confidence =
            cuda_template_match_result.confidence;
        obj.roi_cuda_stereo_bm_support =
            cuda_stereo_bm_result.support;
        obj.roi_cuda_stereo_bm_std_px =
            cuda_stereo_bm_result.valid
                ? cuda_stereo_bm_result.stddev
                : -1.0f;
        obj.roi_cuda_stereo_bm_confidence =
            cuda_stereo_bm_result.confidence;
        obj.roi_cuda_stereo_sgm_support =
            cuda_stereo_sgm_result.support;
        obj.roi_cuda_stereo_sgm_std_px =
            cuda_stereo_sgm_result.valid
                ? cuda_stereo_sgm_result.stddev
                : -1.0f;
        obj.roi_cuda_stereo_sgm_confidence =
            cuda_stereo_sgm_result.confidence;
        obj.roi_neural_feature_support = neural_feature_result.support;
        obj.roi_neural_feature_std_px =
            neural_feature_result.valid ? neural_feature_result.stddev : -1.0f;
        obj.roi_neural_feature_confidence = neural_feature_result.confidence;
        obj.fallback_feature_points_support = fallback_feature_result.support;
        obj.fallback_feature_points_std_px =
            fallback_feature_result.valid ? fallback_feature_result.stddev : -1.0f;
        obj.fallback_feature_points_confidence = fallback_feature_result.confidence;
        if (pair_info && direct_yolo_match && right_det) {
            obj.pair_initial_disparity = pair_info->initial_disparity;
            obj.pair_epipolar_dy = pair_info->epipolar_dy;
            obj.pair_y_tolerance = pair_info->y_tolerance;
            obj.pair_size_ratio = pair_info->size_ratio;
            obj.pair_shifted_iou = pair_info->shifted_bbox_iou;
            obj.pair_score = pair_info->score;
            obj.pair_bbox_prior_penalty =
                bboxDisparityConsistencyPenaltyCPU(
                    left_det,
                    *right_det,
                    pair_info->initial_disparity,
                    baseline,
                    config_.depth,
                    config_.dual_yolo,
                    config_.max_disparity);
            obj.pair_positive_disparity =
                (pair_info->initial_disparity > 0.0f &&
                 pair_info->initial_disparity <=
                     static_cast<float>(config_.max_disparity))
                    ? 1
                    : 0;
        }
        obj.stereo_match_source = match_source;
        obj.stereo_depth_source = depth_source;
        obj.confidence = semantic_conf *
                         std::sqrt(left_circle.confidence * right_circle.confidence) *
                         geom_conf *
                         disparity_conf;
        obj.class_id = left_det.class_id;
        obj.depth_method = 1;
        return true;
    };

    auto mark_right_detection_near = [&](const CircleFit2D& right_circle,
                                         int class_id) -> int {
        int best_idx = -1;
        float best_dist2 = std::numeric_limits<float>::max();
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
                const float dist2 = dx * dx + dy * dy;
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_idx = static_cast<int>(ri);
                }
            }
        }
        return best_idx;
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
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
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
    }

    std::vector<bool> left_has_pair_candidate(left_detections.size(), false);
    std::vector<StereoRoiPair> direct_pairs;
    direct_pairs.reserve(left_detections.size() * right_detections.size());
    for (size_t li = 0; li < left_detections.size(); ++li) {
        const auto& left = left_detections[li];
        for (size_t ri = 0; ri < right_detections.size(); ++ri) {
            StereoRoiPair candidate_pair;
            StereoRoiPairRejectReason reject_reason =
                StereoRoiPairRejectReason::NONE;
            if (!evaluateStereoRoiPair(left,
                                       right_detections[ri],
                                       static_cast<int>(li),
                                       static_cast<int>(ri),
                                       roi_pair_gate,
                                       &candidate_pair,
                                       &reject_reason)) {
                record_pair_reject(reject_reason);
                continue;
            }
            left_has_pair_candidate[li] = true;

            candidate_pair.score += bboxDisparityConsistencyPenaltyCPU(
                left,
                right_detections[ri],
                candidate_pair.initial_disparity,
                baseline,
                config_.depth,
                config_.dual_yolo,
                config_.max_disparity);
            direct_pairs.push_back(candidate_pair);
        }
    }
    std::sort(direct_pairs.begin(), direct_pairs.end(),
              [](const StereoRoiPair& a, const StereoRoiPair& b) {
                  return a.score < b.score;
              });

    // Assign after global scoring so an early false detection cannot reserve
    // the only valid detection on the opposite camera.
    for (const auto& best_pair : direct_pairs) {
        const int li = best_pair.left_index;
        const int best_idx = best_pair.right_index;
        if (li < 0 || best_idx < 0 ||
            li >= static_cast<int>(left_detections.size()) ||
            best_idx >= static_cast<int>(right_detections.size())) {
            continue;
        }
        if (left_has_stereo[li] || right_used[best_idx]) continue;

        const auto& left = left_detections[li];
        const auto& right = right_detections[best_idx];
        const DualYoloGpuCandidate* gpu_candidate =
            find_gpu_candidate(li, best_idx);
        if (gpu_candidate) {
            local_stats.iou_color_support_max =
                std::max(local_stats.iou_color_support_max,
                         gpu_candidate->iou_region_color_patch.support);
            local_stats.iou_color_attempted_max =
                std::max(local_stats.iou_color_attempted_max,
                         gpu_candidate->iou_region_color_patch.attempted);
            local_stats.iou_edge_support_max =
                std::max(local_stats.iou_edge_support_max,
                         gpu_candidate->patch_iou_color_edge.support);
            local_stats.iou_edge_attempted_max =
                std::max(local_stats.iou_edge_attempted_max,
                         gpu_candidate->patch_iou_color_edge.attempted);
        }
        CircleFit2D left_circle = gpu_candidate
            ? circleFromGpuCandidate(gpu_candidate->left_circle, left)
            : refine_detection(left_cpu, left_pitch, left);
        CircleFit2D right_circle = gpu_candidate
            ? circleFromGpuCandidate(gpu_candidate->right_circle, right)
            : refine_detection(right_cpu, right_pitch, right);
        if (!left_circle.valid || !right_circle.valid) {
            ++local_stats.circle_fit_fail;
            if (!direct_depth_without_circle_enabled) {
                continue;
            }
            left_circle = circleFromDetectionCPU(left);
            right_circle = circleFromDetectionCPU(right);
            if (!left_circle.valid || !right_circle.valid) {
                continue;
            }
        }

        Object3D obj;
        const float semantic_conf = best_pair.semantic_confidence;
        const float bbox_disparity = best_pair.initial_disparity;
        if (!build_object(left, left_circle, right_circle, semantic_conf,
                          1, bbox_disparity, &right, &best_pair,
                          gpu_candidate, obj)) {
            continue;
        }

        output.detections[li] = (obj.stereo_depth_source == 3 ||
                                 obj.stereo_depth_source == 6)
            ? left
            : detectionWithCircleCenterCPU(left_circle, left);
        output.results[li] = obj;
        right_used[best_idx] = true;
        left_has_stereo[li] = true;
        ++local_stats.matched;
    }
    for (size_t li = 0; li < left_detections.size(); ++li) {
        if (!left_has_stereo[li] && !left_has_pair_candidate[li]) {
            ++local_stats.no_candidate;
        }
    }

    if ((epipolar_fallback_enabled ||
         fallback_template_enabled ||
         fallback_feature_points_enabled) &&
        image_available) {
        const int fallback_attempted_before = local_stats.fallback_attempted;
        const auto fallback_search_start = Clock::now();
        for (size_t li = 0; li < left_detections.size(); ++li) {
            if (left_has_stereo[li]) continue;

            const Detection& left = left_detections[li];
            ++local_stats.fallback_attempted;
            ++local_stats.fallback_left_to_right;

            CircleFit2D left_circle = refine_detection(left_cpu, left_pitch, left);
            if (!left_circle.valid) {
                ++local_stats.circle_fit_fail;
                left_circle = circleFromDetectionCPU(left);
                if (!left_circle.valid) {
                    ++local_stats.fallback_failed;
                    continue;
                }
            }

            const float expected_disp = estimate_fallback_disparity(left, true);
            if (expected_disp <= 0.0f) {
                ++local_stats.invalid_box;
                ++local_stats.fallback_failed;
                continue;
            }

            const float predicted_right_cx = left_circle.cx - expected_disp;
            CircleFit2D right_circle;
            if (epipolar_fallback_enabled) {
                right_circle = searchCircleOnEpipolarCPU(
                    right_cpu, right_pitch, img_width, img_height,
                    left_circle,
                    predicted_right_cx,
                    left_circle.cy,
                    y_tol,
                    circle_search_cfg);
            }
            if (fallback_template_enabled) {
                const CircleFit2D template_circle = searchTemplateOnEpipolarCPU(
                    left_cpu, left_pitch, right_cpu, right_pitch,
                    img_width, img_height,
                    left_circle,
                    predicted_right_cx,
                    left_circle.cy,
                    y_tol,
                    config_.dual_yolo);
                if (template_circle.valid &&
                    (!right_circle.valid ||
                     template_circle.confidence >= right_circle.confidence)) {
                    right_circle = template_circle;
                }
            }
            if (!right_circle.valid && fallback_feature_points_enabled) {
                right_circle = left_circle;
                right_circle.cx = predicted_right_cx;
                right_circle.cy = left_circle.cy;
                right_circle.confidence = std::max(0.2f, left_circle.confidence * 0.6f);
                right_circle.source = kCircleSourceFeatureProxy;
                right_circle.valid = true;
            }
            if (!right_circle.valid) {
                ++local_stats.fallback_failed;
                continue;
            }

            Object3D obj;
            if (!build_object(left, left_circle, right_circle, left.confidence,
                              2, -1.0f, nullptr, nullptr, nullptr, obj)) {
                ++local_stats.fallback_failed;
                continue;
            }

            const int right_idx = mark_right_detection_near(right_circle, left.class_id);
            if (right_idx >= 0) {
                const Detection& right_source = right_detections[right_idx];
                obj.right_bbox_cx = right_source.cx;
                obj.right_bbox_cy = right_source.cy;
                obj.right_bbox_w = right_source.width;
                obj.right_bbox_h = right_source.height;
                obj.right_bbox_conf = right_source.confidence;
            }
            output.detections[li] = detectionWithCircleCenterCPU(left_circle, left);
            output.results[li] = obj;
            left_has_stereo[li] = true;
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
                right_circle = circleFromDetectionCPU(right);
                if (!right_circle.valid) {
                    ++local_stats.fallback_failed;
                    continue;
                }
            }

            const float expected_disp = estimate_fallback_disparity(right, false);
            if (expected_disp <= 0.0f) {
                ++local_stats.invalid_box;
                ++local_stats.fallback_failed;
                continue;
            }

            const float predicted_left_cx = right_circle.cx + expected_disp;
            CircleFit2D left_circle;
            if (epipolar_fallback_enabled) {
                left_circle = searchCircleOnEpipolarCPU(
                    left_cpu, left_pitch, img_width, img_height,
                    right_circle,
                    predicted_left_cx,
                    right_circle.cy,
                    y_tol,
                    circle_search_cfg);
            }
            if (fallback_template_enabled) {
                const CircleFit2D template_circle = searchTemplateOnEpipolarCPU(
                    right_cpu, right_pitch, left_cpu, left_pitch,
                    img_width, img_height,
                    right_circle,
                    predicted_left_cx,
                    right_circle.cy,
                    y_tol,
                    config_.dual_yolo);
                if (template_circle.valid &&
                    (!left_circle.valid ||
                     template_circle.confidence >= left_circle.confidence)) {
                    left_circle = template_circle;
                }
            }
            if (!left_circle.valid && fallback_feature_points_enabled) {
                left_circle = right_circle;
                left_circle.cx = predicted_left_cx;
                left_circle.cy = right_circle.cy;
                left_circle.confidence = std::max(0.2f, right_circle.confidence * 0.6f);
                left_circle.source = kCircleSourceFeatureProxy;
                left_circle.valid = true;
            }
            if (!left_circle.valid) {
                ++local_stats.fallback_failed;
                continue;
            }

            Detection left_proxy = detectionFromCircleCPU(left_circle, right);
            Object3D obj;
            if (!build_object(left_proxy, left_circle, right_circle,
                              right.confidence, 3, -1.0f, &right, nullptr,
                              nullptr, obj)) {
                ++local_stats.fallback_failed;
                continue;
            }

            int left_idx = find_left_detection_near(left_circle, right.class_id);
            if (left_idx >= 0) {
                const Detection& left_source = left_detections[left_idx];
                output.detections[left_idx] =
                    detectionWithCircleCenterCPU(left_circle, left_source);
                obj.left_bbox_cx = left_source.cx;
                obj.left_bbox_cy = left_source.cy;
                obj.left_bbox_w = left_source.width;
                obj.left_bbox_h = left_source.height;
                obj.left_bbox_conf = left_source.confidence;
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
        if (local_stats.fallback_attempted > fallback_attempted_before) {
            const double fallback_search_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - fallback_search_start).count();
            globalPerf().record("Stage2_CPUFallbackSearch",
                                fallback_search_ms);
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

    // BGR 模式使用校正后 BGR 图像，Gray 模式使用校正后灰度图。
    // CUDA 指针在 init() 时缓存，避免每帧 VPI lock/unlock。
    const auto& leftDetectGpu = leftDetectorUsesBGR()
        ? slot.rectBGR_L_gpu
        : slot.rectGray_L_gpu;
    if (!leftDetectGpu.data || leftDetectGpu.pitchBytes <= 0) {
        LOG_WARN("stage1_detect: invalid left detect CUDA pointer");
        NVTX_RANGE_POP();
        return;
    }

    // 异步推理提交: 仅 enqueue，不在此处同步
    const bool submitted = det->enqueue(slot_index, leftDetectGpu.data,
                                        leftDetectGpu.pitchBytes,
                                        config_.rect_width, config_.rect_height,
                                        stream);

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

        const auto& rightDetectGpu = rightDetectorUsesBGR()
            ? slot.rectBGR_R_gpu
            : slot.rectGray_R_gpu;
        if (!rightDetectGpu.data || rightDetectGpu.pitchBytes <= 0) {
            LOG_WARN("stage1_detect: invalid right detect CUDA pointer");
            NVTX_RANGE_POP();
            return;
        }

        const bool submittedR = detR->enqueue(slot_index, rightDetectGpu.data,
                                              rightDetectGpu.pitchBytes,
                                              config_.rect_width,
                                              config_.rect_height,
                                              streamR);

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
        limitDetectionsByConfidence(slot.detections, config_.max_detections);
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
    stampFrameMetadata(slot);

    vpiImageUnlock(slot.disparityMap);
    NVTX_RANGE_POP();
}

// ===================================================================
// ROI 模式: Stage 2 — 检测后 ROI 多点匹配 + 三角测距 (一步到位)
// ===================================================================

Pipeline::RoiStage2Output Pipeline::runRoiStage2Core(
    const RoiStage2Input& input) {
    RoiStage2Output output;
    output.detections = input.left_detections;

    if (config_.detection_only) {
        output.detection_only = true;
        return output;
    }

    const bool use_dual_yolo_depth =
        dualYoloEnabled() &&
        config_.dual_yolo.use_for_depth &&
        (dualYoloAnyDepthModeEnabled(config_.dual_yolo) ||
         config_.neural_features.enabled);
    const bool can_try_right_only =
        use_dual_yolo_depth &&
        !input.right_detections.empty() &&
        dualYoloEpipolarFallbackEnabled(config_.dual_yolo);
    if (input.left_detections.empty() && !can_try_right_only) {
        output.predict_only = true;
        return output;
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
    std::vector<Detection> fusion_detections = input.left_detections;
    bool need_roi_texture_match = true;

    if (use_dual_yolo_depth) {
        ScopedTimer tdual("Stage2_DualYoloMatch");
        DualYoloMatchStats match_stats;
        DualYoloMatchOutput semantic_match;

        semantic_match = matchDualYoloDetections(
            input.left_detections, input.right_detections,
            input.left_cpu, input.left_cpu_pitch,
            input.right_cpu, input.right_cpu_pitch,
            input.left_gray_gpu, input.left_gray_pitch,
            input.right_gray_gpu, input.right_gray_pitch,
            input.left_bgr_gpu, input.left_bgr_pitch,
            input.right_bgr_gpu, input.right_bgr_pitch,
            input.width, input.height,
            input.stream,
            &match_stats);

        int semantic_valid = count_valid(semantic_match.results);
        globalPerf().record("Stage2_DualYoloMatch", tdual.elapsedMs());

        if (config_.dual_yolo.log_matches &&
            config_.stats_interval > 0 &&
            input.frame_id % config_.stats_interval == 0) {
            const float subpixel_avg_support =
                match_stats.subpixel_refined > 0
                    ? static_cast<float>(match_stats.subpixel_support_sum) /
                          static_cast<float>(match_stats.subpixel_refined)
                    : 0.0f;
            LOG_INFO("[DualYOLO] frame=%d left=%d right=%d matches=%d valid=%d "
                     "missL=%d missR=%d fb=%d/%d fail=%d prior=%d l2r=%d r2l=%d "
                     "noCand=%d cls=%d badBox=%d d<=0=%d dMax=%d epi=%d "
                     "size=%d iou=%d circle=%d subpx=%d/%d rej=%d low=%d skip=%d "
                     "subMs=%.2f/%.2f sup=%.1f/%d gate=%.2f-%.2f depth=%d lock=%d "
                     "iouSup=%d/%d iouEdgeSup=%d/%d",
                     input.frame_id,
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
                     match_stats.low_iou,
                     match_stats.circle_fit_fail,
                     match_stats.subpixel_refined,
                     match_stats.subpixel_attempted,
                     match_stats.subpixel_rejected,
                     match_stats.subpixel_low_conf,
                     match_stats.subpixel_budget_skip,
                     match_stats.subpixel_time_ms,
                     match_stats.subpixel_max_time_ms,
                     subpixel_avg_support,
                     match_stats.subpixel_support_max,
                     match_stats.subpixel_gate_min_px,
                     match_stats.subpixel_gate_max_px,
                     match_stats.depth_reject,
                     match_stats.image_lock_fail,
                     match_stats.iou_color_support_max,
                     match_stats.iou_color_attempted_max,
                     match_stats.iou_edge_support_max,
                     match_stats.iou_edge_attempted_max);
        }

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

    if (need_roi_texture_match) {
        if (!roi_matcher_) {
            LOG_ERROR("ROI texture match requested but ROIStereoMatcher is not initialized");
            output.detections = std::move(fusion_detections);
            output.predict_only = true;
            return output;
        }

        const uint8_t* leftPtr = input.left_gray_gpu;
        const uint8_t* rightPtr = input.right_gray_gpu;
        const int leftPitch = input.left_gray_pitch;
        const int rightPitch = input.right_gray_pitch;
        if (!leftPtr || !rightPtr || leftPitch <= 0 || rightPitch <= 0) {
            output.detections = std::move(fusion_detections);
            output.predict_only = true;
            return output;
        }

        std::vector<stereo3d::Object3D> texture_results;
        {
            ScopedTimer troi("Stage2_ROIMatch");
            texture_results = roi_matcher_->match(
                leftPtr, leftPitch, rightPtr, rightPitch,
                input.width, input.height,
                fusion_detections, input.stream);
            globalPerf().record("Stage2_ROIMatch", troi.elapsedMs());
        }

        if (use_dual_yolo_depth && !roi_results.empty()) {
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

    output.detections = std::move(fusion_detections);
    output.roi_results = std::move(roi_results);
    if (output.detections.empty()) {
        output.predict_only = true;
    }
    return output;
}

void Pipeline::applyRoiStage2Output(FrameSlot& slot,
                                    RoiStage2Output&& output) {
    slot.detections = std::move(output.detections);
    if (output.detection_only) {
        slot.results.clear();
        return;
    }

    if (output.predict_only || slot.detections.empty()) {
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly();
            stampFrameMetadata(slot);
        } else {
            slot.results.clear();
        }
        return;
    }

    if (hybrid_depth_) {
        std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
        auto now = std::chrono::steady_clock::now();
        double dt = 0.01;
        if (last_fuse_time_.time_since_epoch().count() > 0) {
            dt = std::chrono::duration<double>(now - last_fuse_time_).count();
            dt = std::clamp(dt, 0.002, 0.1);
        }
        last_fuse_time_ = now;
        ScopedTimer thd("Stage2_HybridDepth");
        slot.results = hybrid_depth_->estimate(slot.detections,
                                               output.roi_results,
                                               dt);
        stampFrameMetadata(slot);
        globalPerf().record("Stage2_HybridDepth", thd.elapsedMs());
    } else {
        slot.results = std::move(output.roi_results);
        stampFrameMetadata(slot);
    }
}

void Pipeline::publishRoiResultCallback(FrameSlot& slot) {
    if (result_callback_) {
        ScopedTimer trc("Stage2_ResultCB");
        result_callback_(slot.frame_id, slot.results);
        globalPerf().record("Stage2_ResultCB", trc.elapsedMs());
    }
}

void Pipeline::publishRoiFrameCallbacks(FrameSlot& slot) {
    publishRoiResultCallback(slot);
    if (frame_callback_) {
        ScopedTimer tfc("Stage2_FrameCB");
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
        globalPerf().record("Stage2_FrameCB", tfc.elapsedMs());
    }
}

void Pipeline::stage2_roi_match_fuse(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage2_ROIMatchFuse");

    slot.results.clear();
    collectRoiDetections(slot, slot_index);

    RoiStage2Input input;
    input.frame_id = slot.frame_id;
    input.left_detections = slot.detections;
    input.right_detections = slot.detections_right;
    input.left_gray_gpu = static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    input.left_gray_pitch = slot.rectGray_L_gpu.pitchBytes;
    input.right_gray_gpu = static_cast<const uint8_t*>(slot.rectGray_R_gpu.data);
    input.right_gray_pitch = slot.rectGray_R_gpu.pitchBytes;
    if (colorPipelineEnabled()) {
        input.left_bgr_gpu = static_cast<const uint8_t*>(slot.rectBGR_L_gpu.data);
        input.left_bgr_pitch = slot.rectBGR_L_gpu.pitchBytes;
        input.right_bgr_gpu = static_cast<const uint8_t*>(slot.rectBGR_R_gpu.data);
        input.right_bgr_pitch = slot.rectBGR_R_gpu.pitchBytes;
    }
    input.width = config_.rect_width;
    input.height = config_.rect_height;
    input.stream = streams_.cudaStreamFuse;

    const bool need_host_images =
        roiStage2NeedsHostImages(slot.detections, slot.detections_right);
    VPIImageData hostDataL, hostDataR;
    bool lockedL = false;
    bool lockedR = false;
    if (need_host_images) {
        const VPIStatus stL =
            vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ,
                             VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                             &hostDataL);
        const VPIStatus stR =
            vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ,
                             VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                             &hostDataR);
        lockedL = stL == VPI_SUCCESS;
        lockedR = stR == VPI_SUCCESS;
        if (stL == VPI_SUCCESS && stR == VPI_SUCCESS) {
            input.left_cpu = static_cast<const uint8_t*>(
                hostDataL.buffer.pitch.planes[0].data);
            input.left_cpu_pitch = hostDataL.buffer.pitch.planes[0].pitchBytes;
            input.right_cpu = static_cast<const uint8_t*>(
                hostDataR.buffer.pitch.planes[0].data);
            input.right_cpu_pitch = hostDataR.buffer.pitch.planes[0].pitchBytes;
        }
    }

    RoiStage2Output output;
    {
        std::lock_guard<std::mutex> post_lock(roi_postprocess_mutex_);
        output = runRoiStage2Core(input);
    }

    if (lockedL) vpiImageUnlock(slot.rectGray_vpiL);
    if (lockedR) vpiImageUnlock(slot.rectGray_vpiR);

    applyRoiStage2Output(slot, std::move(output));

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
    std::lock_guard<std::mutex> post_lock(roi_postprocess_mutex_);
    slot.results.clear();

    if (slot.bbox_source != BboxSource::TRACKER || !slot.sot_bbox_result.valid) {
        // 无有效 tracker 结果: 仅 Kalman 预测
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly();
            stampFrameMetadata(slot);
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
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly();
            stampFrameMetadata(slot);
        }
        NVTX_RANGE_POP();
        return;
    }

    const uint8_t* leftPtr =
        static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    const uint8_t* rightPtr =
        static_cast<const uint8_t*>(slot.rectGray_R_gpu.data);
    const int leftPitch = slot.rectGray_L_gpu.pitchBytes;
    const int rightPitch = slot.rectGray_R_gpu.pitchBytes;
    if (!leftPtr || !rightPtr || leftPitch <= 0 || rightPitch <= 0) {
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly();
            stampFrameMetadata(slot);
        }
        NVTX_RANGE_POP();
        return;
    }

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

    // 4. 混合深度估计
    if (hybrid_depth_) {
        std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
        auto now = std::chrono::steady_clock::now();
        double dt = 0.01;
        if (last_fuse_time_.time_since_epoch().count() > 0) {
            dt = std::chrono::duration<double>(now - last_fuse_time_).count();
            dt = std::clamp(dt, 0.002, 0.1);
        }
        last_fuse_time_ = now;
        slot.results = hybrid_depth_->estimate(slot.detections, roi_results, dt);
        stampFrameMetadata(slot);
    } else {
        slot.results = std::move(roi_results);
        stampFrameMetadata(slot);
    }

    NVTX_RANGE_POP();
}

}  // namespace stereo3d
