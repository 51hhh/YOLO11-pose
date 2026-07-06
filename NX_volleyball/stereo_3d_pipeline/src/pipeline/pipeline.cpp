/**
 * @file pipeline.cpp
 * @brief Pipeline initialization.
 *
 * Source layout:
 *   - pipeline_loops.cpp: full-frame and ROI pipeline schedulers.
 *   - pipeline_async_roi.cpp: async ROI Stage2 snapshot, worker, and expiry.
 *   - pipeline_lifecycle.cpp: construction, start/stop, and camera grab loop.
 *   - pipeline_dual_yolo_match.cpp: dual-YOLO pairing and ROI depth candidates.
 *   - pipeline.cpp: init and resource construction.
 *
 * CUDA/VPI synchronization is expressed with slot-level events:
 *   evtRectDone        -> Stage0 rectification gate for detector/ROI work.
 *   evtDetectDone      -> left YOLO completion gate.
 *   evtDetectRightDone -> right YOLO completion gate when dual YOLO is active.
 */

#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "../stereo/neural_feature_matcher.h"
#include "../track/nanotrack_trt.h"
#include "../track/mixformer_trt.h"
#include "../utils/logger.h"
#include <vpi/algo/ConvertImageFormat.h>
#include <algorithm>
#include <cctype>
#include <cstdint>

namespace stereo3d {

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
                 "colorEdge=%d cudaTmpl=%d cudaBM=%d cudaSGM=%d ringEdge=%d centerPatch=%d "
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
                 config_.dual_yolo.depth_roi_ring_edge_profile,
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

    if (pipelineP2DepthModesEnabled(config_) ||
        config_.p2_feature_job_scaffold_enabled ||
        config_.p2_diagnostic_lane_decision_enabled) {
        LOG_INFO("  P2 lanes: modes=0x%x feature_jobs=%d realtime=%d diagnostic=%d "
                 "selective=%d pair_quality=%d no_valid_pair=%d "
                 "diag_stride=%d diag_inflight=%d deadline_rt=%.2fms "
                 "deadline_diag=%.2fms",
                 static_cast<unsigned int>(p2FeatureDepthModeMask(config_)),
                 config_.p2_feature_job_scaffold_enabled,
                 config_.p2_realtime_lane_decision_enabled,
                 config_.p2_diagnostic_lane_decision_enabled,
                 config_.p2_selective_trigger,
                 config_.p2_trigger_on_pair_quality,
                 config_.p2_trigger_on_no_valid_direct_pair,
                 config_.p2_diagnostic_stride,
                 config_.p2_diagnostic_max_in_flight,
                 config_.p2_realtime_deadline_ms,
                 config_.p2_diagnostic_deadline_ms);
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

    // Split neural matchers: XFeat and SuperPoint can run in the same frame,
    // each writing its own z_roi_neural_{xfeat,superpoint} candidate fields.
    // Each matcher owns an independent TensorRT execution context and GPU
    // workspace; the inline P2 path dispatches them on independent streams.
    {
        const auto& P1 = calibration_->getProjectionLeft();
        const float focal = static_cast<float>(P1.at<double>(0, 0));
        const float baseline = calibration_->getBaseline();
        auto init_split_neural =
            [&](NeuralFeatureConfig& ncfg,
                std::unique_ptr<NeuralFeatureMatcher>& matcher,
                const char* label) -> bool {
            if (!ncfg.enabled) {
                return true;
            }
            const bool has_engine =
                !ncfg.extractor_engine_path.empty() ||
                !ncfg.fused_engine_path.empty();
            if (!has_engine) {
                LOG_WARN("%s.enabled=true but no extractor_engine_path or "
                         "fused_engine_path configured; disabling", label);
                ncfg.enabled = false;
                return true;
            }
            matcher = std::make_unique<NeuralFeatureMatcher>();
            if (!matcher->init(ncfg, focal, baseline, config_.max_disparity)) {
                LOG_ERROR("%s init failed", label);
                matcher.reset();
                return false;
            }
            if (matcher->requiresBgrInput() && !colorPipelineEnabled()) {
                LOG_WARN("%s engine requires BGR input but pipeline is gray-only; "
                         "candidate will report unsupported until BGR is enabled",
                         label);
            }
            LOG_INFO("%s ready: roi=%d top_k=%d", label,
                     ncfg.roi_size, ncfg.top_k);
            return true;
        };
        if (!init_split_neural(config_.neural_xfeat,
                               neural_xfeat_matcher_,
                               "neural_feature_matching_xfeat")) {
            return false;
        }
        if (!init_split_neural(config_.neural_superpoint,
                               neural_superpoint_matcher_,
                               "neural_feature_matching_superpoint")) {
            return false;
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
            dualYoloCircleEdgesDepthEnabled(config_.dual_yolo) ||
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
            config_.dual_yolo.depth_roi_ring_edge_profile ||
            config_.neural_features.enabled ||
            config_.neural_xfeat.enabled ||
            config_.neural_superpoint.enabled ||
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
            dualYoloCircleEdgesDepthEnabled(config_.dual_yolo) ||
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
         config_.dual_yolo.depth_roi_ring_edge_profile ||
         config_.neural_features.enabled ||
         config_.neural_xfeat.enabled ||
         config_.neural_superpoint.enabled ||
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


}  // namespace stereo3d
