/**
 * @file pipeline_dual_yolo_match.cpp
 * @brief Dual-YOLO ROI pairing and depth candidate construction.
 */

#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "pipeline_roi_match_helpers.h"
#include "../stereo/depth_candidate_builder.h"
#include "../stereo/depth_match_contract.h"
#include "../stereo/neural_feature_matcher.h"
#include "../stereo/roi_feature_match_common.h"
#include "../stereo/roi_feature_match_cpu.h"
#include "../stereo/roi_feature_match_gpu.h"
#include "../stereo/roi_feature_validation.h"
#include "../stereo/roi_geometry_cpu.h"
#include "../stereo/roi_patch_match_cpu.h"
#include "../utils/logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>

namespace stereo3d {

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
    bool p2_inline_feature_jobs_enabled,
    int frame_id,
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
    p2_inline_feature_jobs_enabled =
        p2_inline_feature_jobs_enabled &&
        !(config_.p2_feature_job_scaffold_enabled &&
          config_.p2_diagnostic_lane_decision_enabled &&
          !config_.p2_realtime_lane_decision_enabled);
    const bool roi_corner_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROICornerPointsDepthEnabled(config_.dual_yolo);
    const bool roi_texture_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROITexturePointsDepthEnabled(config_.dual_yolo);
    const bool roi_binary_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIBinaryPointsDepthEnabled(config_.dual_yolo);
    const bool roi_orb_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIORBPointsDepthEnabled(config_.dual_yolo);
    const bool roi_brisk_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIBRISKPointsDepthEnabled(config_.dual_yolo);
    const bool roi_akaze_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIAKAZEPointsDepthEnabled(config_.dual_yolo);
    const bool roi_sift_points_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROISIFTPointsDepthEnabled(config_.dual_yolo);
    const bool roi_iou_region_color_patch_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIIoURegionColorPatchDepthEnabled(config_.dual_yolo);
    const bool roi_patch_iou_color_edge_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIPatchIoUColorEdgeDepthEnabled(config_.dual_yolo);
    const bool roi_cuda_template_match_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROICudaTemplateMatchDepthEnabled(config_.dual_yolo);
    const bool roi_cuda_stereo_bm_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROICudaStereoBMDepthEnabled(config_.dual_yolo);
    const bool roi_cuda_stereo_sgm_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROICudaStereoSGMDepthEnabled(config_.dual_yolo);
    const bool roi_ring_edge_profile_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        dualYoloROIRingEdgeProfileDepthEnabled(config_.dual_yolo);
    const bool neural_feature_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        config_.neural_features.enabled &&
        neural_feature_matcher_ &&
        neural_feature_matcher_->isReady();
    const bool neural_xfeat_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        config_.neural_xfeat.enabled &&
        neural_xfeat_matcher_ &&
        neural_xfeat_matcher_->isReady();
    const bool neural_superpoint_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        config_.neural_superpoint.enabled &&
        neural_superpoint_matcher_ &&
        neural_superpoint_matcher_->isReady();
    const bool neural_aliked_depth_enabled =
        p2_inline_feature_jobs_enabled &&
        config_.neural_aliked.enabled &&
        neural_aliked_matcher_ &&
        neural_aliked_matcher_->isReady();
    const int neural_superpoint_realtime_stride =
        std::max(1, config_.neural_superpoint.realtime_stride);
    const bool neural_superpoint_realtime_due =
        neural_superpoint_depth_enabled &&
        (frame_id % neural_superpoint_realtime_stride) == 0;
    if (neural_superpoint_depth_enabled &&
        !neural_superpoint_realtime_due) {
        globalPerf().record("Stage2_NeuralSuperPointRealtimeStrideSkip", 0.0);
    }
    const int neural_aliked_realtime_stride =
        std::max(1, config_.neural_aliked.realtime_stride);
    const bool neural_aliked_realtime_due =
        neural_aliked_depth_enabled &&
        (frame_id % neural_aliked_realtime_stride) == 0;
    if (neural_aliked_depth_enabled && !neural_aliked_realtime_due) {
        globalPerf().record("Stage2_NeuralAlikedRealtimeStrideSkip", 0.0);
    }
    const bool roi_center_patch_depth_enabled =
        dualYoloROICenterPatchDepthEnabled(config_.dual_yolo);
    const bool subpixel_depth_enabled =
        dualYoloSubpixelDepthEnabled(config_.dual_yolo);
    const bool epipolar_fallback_enabled =
        dualYoloEpipolarFallbackEnabled(config_.dual_yolo);
    const bool fallback_template_enabled =
        dualYoloFallbackTemplateEnabled(config_.dual_yolo);
    const bool fallback_feature_points_enabled =
        p2_inline_feature_jobs_enabled &&
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
        roi_ring_edge_profile_depth_enabled ||
        neural_feature_depth_enabled ||
        neural_xfeat_depth_enabled ||
        neural_superpoint_depth_enabled ||
        neural_aliked_depth_enabled ||
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
    ROIFeatureMatchConfig feature_cfg =
        makeROIFeatureMatchConfig(config_.dual_yolo, config_.depth);
    feature_cfg.debug_patch_enabled = config_.p2_diagnostic_artifacts_enabled;
    const ROICircleSearchConfig circle_search_cfg =
        makeROICircleSearchConfig(config_.dual_yolo);
    const float y_tol = std::max(1.0f, config_.dual_yolo.epipolar_y_tolerance);
    const float max_ratio = std::max(1.0f, config_.dual_yolo.max_size_ratio);
    const StereoRoiPairGateConfig roi_pair_gate =
        makeStereoRoiPairGateConfig(config_);

    struct CudaTemplateOutcome {
        SparseFeatureDisparityResult result;
        double elapsed_ms = 0.0;
    };
    struct NeuralOutcome {
        NeuralFeatureMatchResult neural;
        double elapsed_ms = 0.0;
    };
    struct P2EarlyFeatureResult {
        int left_index = -1;
        int right_index = -1;
        float initial_disparity = -1.0f;
        bool completed = false;
        bool cuda_template_due = false;
        bool neural_xfeat_due = false;
        bool neural_superpoint_due = false;
        bool neural_aliked_due = false;
        CudaTemplateOutcome cuda_template;
        NeuralOutcome neural_xfeat;
        NeuralOutcome neural_superpoint;
        NeuralOutcome neural_aliked;
    };

    auto wait_p2_ready_event = [&](cudaStream_t algo_stream,
                                   cudaEvent_t ready_event,
                                   const char* label) -> bool {
        if (!ready_event) {
            return true;
        }
        if (!algo_stream) {
            LOG_WARN("P2 inline %s stream unavailable", label);
            return false;
        }
        const cudaError_t err = cudaStreamWaitEvent(algo_stream, ready_event, 0);
        if (err != cudaSuccess) {
            LOG_WARN("P2 inline %s stream wait failed: %s",
                     label, cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    auto run_cuda_template_candidate =
        [&](const Detection& left_det,
            const Detection& right_det,
            float initial_disparity,
            cudaStream_t algo_stream,
            cudaEvent_t ready_event,
            const char* label) {
        CudaTemplateOutcome outcome;
        const auto algo_start = Clock::now();
        if (!wait_p2_ready_event(algo_stream, ready_event, label)) {
            outcome.result.low_confidence = true;
        } else {
            outcome.result = matchCudaTemplateDisparityGPU(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                img_width, img_height,
                left_det, right_det,
                initial_disparity,
                feature_cfg,
                config_.max_disparity,
                focal,
                baseline,
                algo_stream);
        }
        outcome.elapsed_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - algo_start).count();
        return outcome;
    };

    auto run_neural_candidate =
        [&](NeuralFeatureMatcher* matcher,
            const Detection& left_det,
            const Detection& right_det,
            float initial_disparity,
            cudaStream_t algo_stream,
            cudaEvent_t ready_event,
            const char* label) {
        NeuralOutcome outcome;
        const auto algo_start = Clock::now();
        if (!matcher || !matcher->isReady()) {
            outcome.neural.status = "unavailable";
        } else if (!wait_p2_ready_event(algo_stream, ready_event, label)) {
            outcome.neural.status = "stream_wait_failed";
        } else {
            outcome.neural = matcher->matchGpuRoi(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                left_bgr_gpu, left_bgr_pitch,
                right_bgr_gpu, right_bgr_pitch,
                img_width, img_height,
                left_det, right_det,
                initial_disparity,
                algo_stream);
        }
        outcome.elapsed_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - algo_start).count();
        return outcome;
    };

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

    struct P2InputReadyEventGuard {
        cudaEvent_t event = nullptr;
        ~P2InputReadyEventGuard() {
            if (event) {
                cudaEventDestroy(event);
            }
        }
    };
    P2InputReadyEventGuard p2_input_ready;
    bool p2_input_ready_recorded = false;
    if (p2_inline_feature_jobs_enabled && gpu_image_available && stream != nullptr) {
        cudaError_t err =
            cudaEventCreateWithFlags(&p2_input_ready.event,
                                     cudaEventDisableTiming);
        if (err == cudaSuccess) {
            err = cudaEventRecord(p2_input_ready.event, stream);
        }
        if (err == cudaSuccess) {
            p2_input_ready_recorded = true;
        } else {
            LOG_WARN("P2 early: failed to create/record input ready event: %s",
                     cudaGetErrorString(err));
            globalPerf().record("Stage2_P2EarlyReadyEventFallback", 0.0);
            if (p2_input_ready.event) {
                cudaEventDestroy(p2_input_ready.event);
                p2_input_ready.event = nullptr;
            }
        }
    }
    const cudaEvent_t p2_input_ready_event =
        p2_input_ready_recorded ? p2_input_ready.event : nullptr;

    std::vector<DualYoloGpuCandidate> gpu_candidates;
    bool gpu_candidates_pending = false;
    Clock::time_point gpu_candidates_start{};
    if (gpu_candidate_refine_enabled &&
        !left_detections.empty() &&
        !right_detections.empty()) {
        std::vector<DualYoloGpuDetectionPair> gpu_pairs =
            buildGpuDetectionPairsForRefine(
                left_detections,
                right_detections,
                roi_pair_gate,
                baseline,
                config_.depth,
                config_.dual_yolo,
                config_.max_disparity,
                static_cast<std::size_t>(dual_yolo_depth_gpu_->maxPairs()));
        if (!gpu_pairs.empty()) {
            gpu_candidates_start = Clock::now();
            gpu_candidates_pending = dual_yolo_depth_gpu_->submitPairs(
                left_gpu, left_gpu_pitch,
                right_gpu, right_gpu_pitch,
                left_bgr_gpu, left_bgr_pitch,
                right_bgr_gpu, right_bgr_pitch,
                img_width, img_height,
                gpu_pairs,
                stream);
            globalPerf().record("Stage2_DualYoloGpuCandidatesSubmit",
                                std::chrono::duration<double, std::milli>(
                                    Clock::now() - gpu_candidates_start).count());
            if (!gpu_candidates_pending) {
                gpu_candidates = dual_yolo_depth_gpu_->matchPairs(
                    left_gpu, left_gpu_pitch,
                    right_gpu, right_gpu_pitch,
                    left_bgr_gpu, left_bgr_pitch,
                    right_bgr_gpu, right_bgr_pitch,
                    img_width, img_height,
                    gpu_pairs,
                    stream);
                const double gpu_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - gpu_candidates_start).count();
                globalPerf().record("Stage2_DualYoloGpuCandidates", gpu_ms);
            }
        }
    }

    auto collect_gpu_candidates = [&]() {
        if (!gpu_candidates_pending) {
            return;
        }
        const auto collect_start = Clock::now();
        gpu_candidates = dual_yolo_depth_gpu_->collectPairs();
        globalPerf().record("Stage2_DualYoloGpuCandidatesCollect",
                            std::chrono::duration<double, std::milli>(
                                Clock::now() - collect_start).count());
        const double gpu_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - gpu_candidates_start).count();
        globalPerf().record("Stage2_DualYoloGpuCandidates", gpu_ms);
        gpu_candidates_pending = false;
    };

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
                            const P2EarlyFeatureResult* p2_early,
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
        auto append_p2_artifact = [&](const char* mode,
                                      const SparseFeatureDisparityResult& result,
                                      float initial_disp) {
            const bool collect_debug_row =
                config_.p2_diagnostic_results_enabled ||
                config_.p2_diagnostic_artifacts_enabled ||
                config_.p2_diagnostic_point_debug_enabled;
            if (!collect_debug_row ||
                !mode || !right_det ||
                (result.debug_match_count <= 0 &&
                 result.debug_point_count <= 0 &&
                 !result.debug_patch.valid)) {
                return;
            }
            P2FeatureDiagnosticResultRow row;
            row.frame_id = frame_id;
            row.lane = "inline";
            row.mode = mode;
            row.status = result.valid
                ? "valid"
                : (result.unsupported ? "unsupported" : "invalid");
            row.valid = result.valid;
            row.low_confidence = result.low_confidence;
            row.disparity = result.disparity;
            row.z_m = result.valid
                ? depth_from_disparity(result.disparity)
                : std::numeric_limits<float>::quiet_NaN();
            row.confidence = result.confidence;
            row.stddev = result.stddev;
            row.support = result.support;
            row.attempted = result.attempted;
            row.initial_disparity = initial_disp;
            row.fb = fb;
            row.left_det = left_det;
            row.right_det = *right_det;
            row.anchor_cx = result.anchor_cx;
            row.anchor_cy = result.anchor_cy;
            row.right_anchor_cx = result.right_anchor_cx;
            row.right_anchor_cy = result.right_anchor_cy;
            row.debug_match_count = std::clamp(
                result.debug_match_count,
                0,
                kMaxSparseFeatureDebugMatches);
            for (int i = 0; i < row.debug_match_count; ++i) {
                row.debug_matches[static_cast<size_t>(i)] =
                    result.debug_matches[static_cast<size_t>(i)];
            }
            row.debug_patch = result.debug_patch;
            row.debug_point_count = std::clamp(
                result.debug_point_count,
                0,
                kMaxSparseFeatureDebugPoints);
            for (int i = 0; i < row.debug_point_count; ++i) {
                row.debug_points[static_cast<size_t>(i)] =
                    result.debug_points[static_cast<size_t>(i)];
            }
            output.p2_artifact_rows.push_back(std::move(row));
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
        const float circle_top_y_delta =
            std::abs((left_circle.cy - left_circle.radius) -
                     (right_circle.cy - right_circle.radius));
        const float circle_bottom_y_delta =
            std::abs((left_circle.cy + left_circle.radius) -
                     (right_circle.cy + right_circle.radius));
        const bool circle_vertical_axis_ok =
            circle_top_y_delta <= y_tol && circle_bottom_y_delta <= y_tol;
        if (!circle_vertical_axis_ok) {
            ++local_stats.circle_axis_reject;
            globalPerf().record("Stage2_CircleAxisInvalid", 0.0);
            circle_geometry_valid = false;
        }
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
        SparseFeatureDisparityResult circle_match_result;
        float z_circle_match = -1.0f;
        const bool circle_match_due =
            p2_inline_feature_jobs_enabled &&
            circle_depth_enabled &&
            direct_yolo_match &&
            right_det;
        if (circle_match_due) {
            const auto circle_match_start = Clock::now();
            circle_match_result.attempted = 3;
            circle_match_result.support =
                (left_circle.valid && right_circle.valid) ? 3 : 0;
            circle_match_result.disparity = circle_disparity;
            circle_match_result.anchor_cx = left_circle.cx;
            circle_match_result.anchor_cy = left_circle.cy;
            circle_match_result.right_anchor_cx = right_circle.cx;
            circle_match_result.right_anchor_cy = right_circle.cy;
            const float circle_axis_residual =
                std::max(refined_dy,
                         0.5f * (circle_top_y_delta + circle_bottom_y_delta));
            circle_match_result.stddev =
                std::max(std::abs(left_circle.radius - right_circle.radius),
                         circle_axis_residual);
            circle_match_result.confidence =
                std::sqrt(std::max(0.0f, left_circle.confidence) *
                          std::max(0.0f, right_circle.confidence));
            if (left_circle.valid && right_circle.valid) {
                auto set_circle_axis_match =
                    [&](int index, float left_y, float right_y) {
                        auto& match = circle_match_result.debug_matches[
                            static_cast<size_t>(index)];
                        match.left_x = left_circle.cx;
                        match.left_y = left_y;
                        match.right_x = right_circle.cx;
                        match.right_y = right_y;
                        match.disparity = circle_disparity;
                        match.score = circle_match_result.confidence;
                    };
                circle_match_result.debug_match_count = 3;
                set_circle_axis_match(0, left_circle.cy, right_circle.cy);
                set_circle_axis_match(1,
                                      left_circle.cy - left_circle.radius,
                                      right_circle.cy - right_circle.radius);
                set_circle_axis_match(2,
                                      left_circle.cy + left_circle.radius,
                                      right_circle.cy + right_circle.radius);
            }
            const bool circle_sources_ok =
                left_circle.source == kCircleSourceRoiFit &&
                right_circle.source == kCircleSourceRoiFit;
            z_circle_match = depth_from_disparity(circle_disparity);
            circle_match_result.valid =
                circle_sources_ok &&
                !has_feature_proxy_circle &&
                circle_geometry_valid &&
                circle_vertical_axis_ok &&
                circle_disp_positive &&
                z_circle_match > 0.0f;
            circle_match_result.low_confidence = !circle_match_result.valid;
            const double circle_match_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - circle_match_start).count();
            globalPerf().record("Stage2_CircleMatch", circle_match_ms);
            globalPerf().record(circle_match_result.valid
                                    ? "Stage2_CircleMatchValid"
                                    : "Stage2_CircleMatchInvalid",
                                0.0);
            append_p2_artifact("circle_match",
                               circle_match_result,
                               feature_initial_disparity);
        }
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
            append_p2_artifact("iou_region_color_patch",
                               iou_region_color_patch_result,
                               feature_initial_disparity);
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
            append_p2_artifact("patch_iou_color_edge",
                               patch_iou_color_edge_result,
                               feature_initial_disparity);
        }

        SparseFeatureDisparityResult cuda_template_match_result;
        float z_roi_cuda_template_match = -1.0f;

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
            append_p2_artifact("cuda_stereo_bm",
                               cuda_stereo_bm_result,
                               feature_initial_disparity);
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
            append_p2_artifact("cuda_stereo_sgm",
                               cuda_stereo_sgm_result,
                               feature_initial_disparity);
        }

        SparseFeatureDisparityResult ring_edge_profile_result;
        float z_roi_ring_edge_profile = -1.0f;
        if (roi_ring_edge_profile_depth_enabled && direct_yolo_match &&
            right_det && feature_initial_disparity > 0.0f &&
            gpu_image_available && stream != nullptr) {
            const auto profile_start = Clock::now();
            ring_edge_profile_result = matchCudaRingEdgeProfileDisparityGPU(
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
            const double profile_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - profile_start).count();
            globalPerf().record("Stage2_CudaRingEdgeProfileMatch",
                                profile_ms);
            if (ring_edge_profile_result.valid) {
                z_roi_ring_edge_profile =
                    depth_from_disparity(ring_edge_profile_result.disparity);
                ring_edge_profile_result.valid =
                    z_roi_ring_edge_profile > 0.0f;
            }
            append_p2_artifact("cuda_ring_edge_profile",
                               ring_edge_profile_result,
                               feature_initial_disparity);
        }

        // P2 inline feature candidates: NCC, XFeat, SuperPoint and ALIKED are
        // independent algorithms. When more than one is enabled, dispatch them
        // on independent CUDA streams and collect on the caller thread so their
        // result fields stay deterministic and geometry gates remain centralized.
        SparseFeatureDisparityResult neural_xfeat_result;
        SparseFeatureDisparityResult neural_superpoint_result;
        SparseFeatureDisparityResult neural_aliked_result;
        SparseFeatureDisparityResult neural_feature_result;
        float z_roi_neural_feature = -1.0f;
        float z_roi_neural_xfeat = -1.0f;
        float z_roi_neural_superpoint = -1.0f;
        float z_roi_neural_aliked = -1.0f;
        const auto p2_inline_feature_start = Clock::now();
        const bool p2_inline_gpu_common_gate =
            direct_yolo_match && right_det &&
            feature_initial_disparity > 0.0f &&
            gpu_image_available && stream != nullptr;
        const bool cuda_template_due =
            roi_cuda_template_match_depth_enabled && p2_inline_gpu_common_gate;
        const bool neural_feature_due =
            neural_feature_depth_enabled && p2_inline_gpu_common_gate;
        const bool neural_xfeat_due =
            neural_xfeat_depth_enabled && p2_inline_gpu_common_gate;
        const bool neural_superpoint_due =
            neural_superpoint_realtime_due && p2_inline_gpu_common_gate;
        const bool neural_aliked_due =
            neural_aliked_realtime_due && p2_inline_gpu_common_gate;
        const int p2_parallel_algo_count =
            (cuda_template_due ? 1 : 0) +
            (neural_xfeat_due ? 1 : 0) +
            (neural_superpoint_due ? 1 : 0) +
            (neural_aliked_due ? 1 : 0);
        const int p2_inline_algo_count =
            (neural_feature_due ? 1 : 0) +
            p2_parallel_algo_count;
        if (!p2_early && p2_inline_algo_count > 0) {
            globalPerf().record("Stage2_P2InlineFeatureAlgoCount",
                                static_cast<double>(p2_inline_algo_count));
        }

        struct CudaEventGuard {
            cudaEvent_t event = nullptr;
            ~CudaEventGuard() {
                if (event) {
                    cudaEventDestroy(event);
                }
            }
        };
        CudaEventGuard p2_ready;
        bool p2_parallel_enabled = false;
        if (p2_parallel_algo_count > 1 && !p2_early) {
            cudaError_t err =
                cudaEventCreateWithFlags(&p2_ready.event,
                                         cudaEventDisableTiming);
            if (err == cudaSuccess) {
                err = cudaEventRecord(p2_ready.event, stream);
            }
            if (err == cudaSuccess) {
                p2_parallel_enabled = true;
            } else {
                LOG_WARN("P2 inline: failed to create/record ready event, "
                         "falling back to sequential feature candidates: %s",
                         cudaGetErrorString(err));
                globalPerf().record("Stage2_P2InlineReadyEventFallback",
                                    0.0);
                if (p2_ready.event) {
                    cudaEventDestroy(p2_ready.event);
                    p2_ready.event = nullptr;
                }
            }
        }
        const cudaEvent_t ready_event =
            p2_parallel_enabled ? p2_ready.event : nullptr;

        auto finish_neural =
            [&](const NeuralOutcome& outcome,
                const NeuralFeatureConfig& neural_cfg,
                const char* perf_name,
                const char* perf_valid,
                const char* perf_invalid,
                const char* artifact_name,
                SparseFeatureDisparityResult& result) -> float {
            float z_out = -1.0f;
            const NeuralFeatureMatchResult& neural = outcome.neural;
            const bool is_superpoint =
                artifact_name &&
                std::strcmp(artifact_name, "neural_superpoint") == 0;
            auto record_superpoint_status = [&](const std::string& status) {
                if (!is_superpoint) {
                    return;
                }
                if (status == "ok_gpu_b2") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusOkGpuB2", 0.0);
                } else if (status == "ok_gpu") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusOkGpu", 0.0);
                } else if (status == "not_enough_matches") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusNotEnoughMatches", 0.0);
                } else if (status == "extractor_failed") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusExtractorFailed", 0.0);
                } else if (status == "stream_wait_failed") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusStreamWaitFailed", 0.0);
                } else if (status == "unavailable") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusUnavailable", 0.0);
                } else if (status == "unsupported_input_schema" ||
                           status == "unsupported_direct_extractor_schema" ||
                           status == "unsupported_split_matcher_schema") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusUnsupportedSchema", 0.0);
                } else if (status == "requires_bgr") {
                    globalPerf().record("Stage2_NeuralSuperPointStatusRequiresBgr", 0.0);
                } else {
                    globalPerf().record("Stage2_NeuralSuperPointStatusOther", 0.0);
                }
            };
            auto record_superpoint_final_reject =
                [&](SparseFeatureRejectReason reason) {
                if (!is_superpoint) {
                    return;
                }
                switch (reason) {
                case SparseFeatureRejectReason::BAD_DISPARITY:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectBadDisparity", 0.0);
                    break;
                case SparseFeatureRejectReason::SUPPORT:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectSupport", 0.0);
                    break;
                case SparseFeatureRejectReason::STDDEV:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectStddev", 0.0);
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectStddevPx",
                                        static_cast<double>(result.stddev));
                    break;
                case SparseFeatureRejectReason::Y_RESIDUAL:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectYResidual", 0.0);
                    break;
                case SparseFeatureRejectReason::OVERLAP:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectOverlap", 0.0);
                    break;
                case SparseFeatureRejectReason::SPHERE:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectSphere", 0.0);
                    break;
                default:
                    globalPerf().record("Stage2_NeuralSuperPointFinalRejectOther", 0.0);
                    break;
                }
            };
            record_superpoint_status(neural.status);
            globalPerf().record(perf_name,
                                neural.inference_ms > 0.0f
                                    ? static_cast<double>(neural.inference_ms)
                                    : outcome.elapsed_ms);
            result.low_confidence = !neural.valid;
            result.unsupported = neural.status == "unsupported_input_schema" ||
                                 neural.status == "requires_bgr";
            result.attempted =
                static_cast<int>(neural.debug_points.size());
            result.debug_point_count = std::min(
                static_cast<int>(neural.debug_points.size()),
                kMaxSparseFeatureDebugPoints);
            for (int i = 0; i < result.debug_point_count; ++i) {
                result.debug_points[static_cast<size_t>(i)] =
                    neural.debug_points[static_cast<size_t>(i)];
            }
            if (neural.valid) {
                result.valid = true;
                result.disparity = neural.disparity;
                result.stddev = neural.stddev_px;
                result.confidence = neural.confidence;
                result.support = static_cast<int>(neural.matches.size());
                result.attempted = static_cast<int>(neural.matches.size());
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
                    static_cast<float>(std::max(1, result.support));
                result.anchor_cx = sx * inv;
                result.anchor_cy = sy * inv;
                result.right_anchor_cx = rx * inv;
                result.right_anchor_cy = ry * inv;
                result.debug_match_count = std::min(
                    static_cast<int>(neural.matches.size()),
                    kMaxSparseFeatureDebugMatches);
                for (int i = 0; i < result.debug_match_count; ++i) {
                    const auto& src = neural.matches[static_cast<size_t>(i)];
                    auto& dst = result.debug_matches[static_cast<size_t>(i)];
                    dst.left_x = src.left_x;
                    dst.left_y = src.left_y;
                    dst.right_x = src.right_x;
                    dst.right_y = src.right_y;
                    dst.disparity = src.disparity;
                    dst.score = src.score;
                }
                const SparseFeatureRejectReason final_reason =
                    neural_cfg.final_geometry_gate_enabled
                        ? sparseFeatureGeometryRejectReason(
                              result, left_det, *right_det,
                              feature_initial_disparity, feature_cfg,
                              focal, baseline)
                        : SparseFeatureRejectReason::NONE;
                if (final_reason != SparseFeatureRejectReason::NONE) {
                    record_superpoint_final_reject(final_reason);
                    RobustMatchSample final_sample;
                    final_sample.left_x = result.anchor_cx;
                    final_sample.left_y = result.anchor_cy;
                    final_sample.right_x =
                        std::isfinite(result.right_anchor_cx)
                            ? result.right_anchor_cx
                            : result.anchor_cx - result.disparity;
                    final_sample.right_y =
                        std::isfinite(result.right_anchor_cy)
                            ? result.right_anchor_cy
                            : result.anchor_cy;
                    final_sample.disparity = result.disparity;
                    final_sample.score = result.confidence;
                    appendDebugPoint(result,
                                     final_sample,
                                     SparseFeatureDebugStage::GEOMETRY,
                                     final_reason,
                                     feature_initial_disparity,
                                     left_det,
                                     feature_cfg);
                    result.valid = false;
                    result.low_confidence = true;
                }
            }
            if (result.valid) {
                z_out = depth_from_disparity(result.disparity);
                result.valid = z_out > 0.0f;
                result.low_confidence = !result.valid;
                if (is_superpoint && !result.valid) {
                    globalPerf().record(
                        "Stage2_NeuralSuperPointFinalRejectDepthRange", 0.0);
                }
            }
            globalPerf().record(result.valid ? perf_valid : perf_invalid, 0.0);
            append_p2_artifact(artifact_name, result,
                               feature_initial_disparity);
            return z_out;
        };

        CudaTemplateOutcome cuda_template_outcome;
        NeuralOutcome neural_feature_outcome;
        NeuralOutcome neural_xfeat_outcome;
        NeuralOutcome neural_superpoint_outcome;
        NeuralOutcome neural_aliked_outcome;
        bool cuda_template_submitted = false;
        bool neural_xfeat_submitted = false;
        bool neural_superpoint_submitted = false;
        bool neural_aliked_submitted = false;

        auto record_submit_fallback = [](const char* perf_name) {
            globalPerf().record("Stage2_P2InlineSubmitFallback", 0.0);
            globalPerf().record(perf_name, 0.0);
        };

        const auto p2_parallel_start = Clock::now();
        if (p2_early) {
            cuda_template_outcome = p2_early->cuda_template;
            neural_xfeat_outcome = p2_early->neural_xfeat;
            neural_superpoint_outcome = p2_early->neural_superpoint;
            neural_aliked_outcome = p2_early->neural_aliked;
            globalPerf().record("Stage2_P2EarlyFeatureReused", 0.0);
        } else if (cuda_template_due) {
            const cudaStream_t algo_stream = p2_parallel_enabled
                ? streams_.cudaStreamP2Ncc
                : stream;
            if (p2_parallel_enabled) {
                cuda_template_submitted = submitP2InlineFeatureTask(
                    p2_inline_ncc_worker_,
                    [&, algo_stream] {
                        cuda_template_outcome =
                            run_cuda_template_candidate(
                                left_det, *right_det,
                                feature_initial_disparity,
                                algo_stream,
                                ready_event,
                                "NCC");
                    });
            }
            if (!cuda_template_submitted) {
                if (p2_parallel_enabled) {
                    record_submit_fallback(
                        "Stage2_P2InlineSubmitFallbackNcc");
                }
                const cudaStream_t fallback_stream =
                    p2_parallel_enabled ? stream : algo_stream;
                cuda_template_outcome =
                    run_cuda_template_candidate(
                        left_det, *right_det,
                        feature_initial_disparity,
                        fallback_stream,
                        ready_event,
                        "NCC");
            }
        }
        if (!p2_early && neural_xfeat_due) {
            const cudaStream_t algo_stream = p2_parallel_enabled
                ? streams_.cudaStreamP2XFeat
                : stream;
            if (p2_parallel_enabled) {
                neural_xfeat_submitted = submitP2InlineFeatureTask(
                    p2_inline_xfeat_worker_,
                    [&, algo_stream] {
                        neural_xfeat_outcome =
                            run_neural_candidate(
                                neural_xfeat_matcher_.get(),
                                left_det, *right_det,
                                feature_initial_disparity,
                                algo_stream,
                                ready_event,
                                "XFeat");
                    });
            }
            if (!neural_xfeat_submitted) {
                if (p2_parallel_enabled) {
                    record_submit_fallback(
                        "Stage2_P2InlineSubmitFallbackXFeat");
                }
                const cudaStream_t fallback_stream =
                    p2_parallel_enabled ? stream : algo_stream;
                neural_xfeat_outcome =
                    run_neural_candidate(
                        neural_xfeat_matcher_.get(),
                        left_det, *right_det,
                        feature_initial_disparity,
                        fallback_stream,
                        ready_event,
                        "XFeat");
            }
        }
        if (!p2_early && neural_superpoint_due) {
            const cudaStream_t algo_stream = p2_parallel_enabled
                ? streams_.cudaStreamP2SuperPoint
                : stream;
            if (p2_parallel_enabled) {
                neural_superpoint_submitted = submitP2InlineFeatureTask(
                    p2_inline_superpoint_worker_,
                    [&, algo_stream] {
                        neural_superpoint_outcome =
                            run_neural_candidate(
                                neural_superpoint_matcher_.get(),
                                left_det, *right_det,
                                feature_initial_disparity,
                                algo_stream,
                                ready_event,
                                "SuperPoint");
                    });
            }
            if (!neural_superpoint_submitted) {
                if (p2_parallel_enabled) {
                    record_submit_fallback(
                        "Stage2_P2InlineSubmitFallbackSuperPoint");
                }
                const cudaStream_t fallback_stream =
                    p2_parallel_enabled ? stream : algo_stream;
                neural_superpoint_outcome =
                    run_neural_candidate(
                        neural_superpoint_matcher_.get(),
                        left_det, *right_det,
                        feature_initial_disparity,
                        fallback_stream,
                        ready_event,
                        "SuperPoint");
            }
        }
        if (!p2_early && neural_aliked_due) {
            const cudaStream_t algo_stream = p2_parallel_enabled
                ? streams_.cudaStreamP2Aliked
                : stream;
            if (p2_parallel_enabled) {
                neural_aliked_submitted = submitP2InlineFeatureTask(
                    p2_inline_aliked_worker_,
                    [&, algo_stream] {
                        neural_aliked_outcome =
                            run_neural_candidate(
                                neural_aliked_matcher_.get(),
                                left_det, *right_det,
                                feature_initial_disparity,
                                algo_stream,
                                ready_event,
                                "ALIKED");
                    });
            }
            if (!neural_aliked_submitted) {
                if (p2_parallel_enabled) {
                    record_submit_fallback(
                        "Stage2_P2InlineSubmitFallbackAliked");
                }
                const cudaStream_t fallback_stream =
                    p2_parallel_enabled ? stream : algo_stream;
                neural_aliked_outcome =
                    run_neural_candidate(
                        neural_aliked_matcher_.get(),
                        left_det, *right_det,
                        feature_initial_disparity,
                        fallback_stream,
                        ready_event,
                        "ALIKED");
            }
        }

        if (cuda_template_submitted) {
            waitP2InlineFeatureTask(p2_inline_ncc_worker_);
        }
        if (neural_xfeat_submitted) {
            waitP2InlineFeatureTask(p2_inline_xfeat_worker_);
        }
        if (neural_superpoint_submitted) {
            waitP2InlineFeatureTask(p2_inline_superpoint_worker_);
        }
        if (neural_aliked_submitted) {
            waitP2InlineFeatureTask(p2_inline_aliked_worker_);
        }
        if (cuda_template_submitted ||
            neural_xfeat_submitted ||
            neural_superpoint_submitted ||
            neural_aliked_submitted) {
            const double p2_parallel_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - p2_parallel_start).count();
            globalPerf().record("Stage2_P2InlineFeatureParallelJoin",
                                p2_parallel_ms);
        }

        if (cuda_template_due) {
            cuda_template_match_result = cuda_template_outcome.result;
            globalPerf().record("Stage2_CudaTemplateNccMatch",
                                cuda_template_outcome.elapsed_ms);
            if (cuda_template_match_result.valid) {
                z_roi_cuda_template_match =
                    depth_from_disparity(cuda_template_match_result.disparity);
                cuda_template_match_result.valid =
                    z_roi_cuda_template_match > 0.0f;
                cuda_template_match_result.low_confidence =
                    !cuda_template_match_result.valid;
            }
            append_p2_artifact("cuda_template",
                               cuda_template_match_result,
                               feature_initial_disparity);
        }

        if (neural_feature_due) {
            neural_feature_outcome =
                run_neural_candidate(
                    neural_feature_matcher_.get(),
                    left_det, *right_det,
                    feature_initial_disparity,
                    stream,
                    ready_event,
                    "neural_feature");
            z_roi_neural_feature = finish_neural(
                neural_feature_outcome,
                config_.neural_features,
                "Stage2_NeuralFeatureMatch",
                "Stage2_NeuralFeatureMatchValid",
                "Stage2_NeuralFeatureMatchInvalid",
                "neural_feature",
                neural_feature_result);
        }
        if (neural_xfeat_due) {
            z_roi_neural_xfeat = finish_neural(
                neural_xfeat_outcome,
                config_.neural_xfeat,
                "Stage2_NeuralXFeatMatch",
                "Stage2_NeuralXFeatMatchValid",
                "Stage2_NeuralXFeatMatchInvalid",
                "neural_xfeat",
                neural_xfeat_result);
        }
        if (neural_superpoint_due) {
            z_roi_neural_superpoint = finish_neural(
                neural_superpoint_outcome,
                config_.neural_superpoint,
                "Stage2_NeuralSuperPointMatch",
                "Stage2_NeuralSuperPointMatchValid",
                "Stage2_NeuralSuperPointMatchInvalid",
                "neural_superpoint",
                neural_superpoint_result);
        }
        if (neural_aliked_due) {
            z_roi_neural_aliked = finish_neural(
                neural_aliked_outcome,
                config_.neural_aliked,
                "Stage2_NeuralAlikedMatch",
                "Stage2_NeuralAlikedMatchValid",
                "Stage2_NeuralAlikedMatchInvalid",
                "neural_aliked",
                neural_aliked_result);
        }
        // Legacy compatibility: keep z_roi_neural_feature populated. If the
        // legacy matcher is disabled, use XFeat first, then ALIKED, then
        // SuperPoint.
        if (!neural_feature_result.valid) {
            neural_feature_result = neural_xfeat_result.valid
                ? neural_xfeat_result
                : (neural_aliked_result.valid
                    ? neural_aliked_result
                    : neural_superpoint_result);
        }
        if (z_roi_neural_feature <= 0.0f) {
            z_roi_neural_feature =
                z_roi_neural_xfeat > 0.0f ? z_roi_neural_xfeat
                                          : (z_roi_neural_aliked > 0.0f
                                             ? z_roi_neural_aliked
                                             : z_roi_neural_superpoint);
        }
        if (!p2_early &&
            (cuda_template_due || neural_feature_due ||
             neural_xfeat_due || neural_superpoint_due ||
             neural_aliked_due)) {
            const double p2_inline_feature_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - p2_inline_feature_start).count();
            globalPerf().record("Stage2_P2InlineFeatureEndToEnd",
                                p2_inline_feature_ms);
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
            z_roi_ring_edge_profile <= 0.0f &&
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
        bool epipolar_fallback_depth_valid =
            is_fallback_match &&
            epipolar_fallback_enabled &&
            circle_depth_valid &&
            (left_circle.source == kCircleSourceEpipolarSearch ||
             right_circle.source == kCircleSourceEpipolarSearch);
        bool fallback_template_depth_valid =
            is_fallback_match &&
            fallback_template_enabled &&
            circle_depth_valid &&
            (left_circle.source == kCircleSourceTemplateSearch ||
             right_circle.source == kCircleSourceTemplateSearch);
        bool any_fallback_depth_valid =
            epipolar_fallback_depth_valid || fallback_template_depth_valid;

        float disparity_circle_left_edge = -1.0f;
        float disparity_circle_right_edge = -1.0f;
        float z_circle_left_edge = -1.0f;
        float z_circle_right_edge = -1.0f;
        // Sphere silhouette left/right edges are not the same physical surface
        // point across cameras. Keep legacy CSV fields invalid; circle_match
        // logs center/top/bottom axis points for diagnostics.

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
        float z_roi_center_patch =
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

        float z_subpixel =
            subpixel_valid_for_obj ? fb / subpixel_result.disparity : -1.0f;

        if (is_fallback_match && any_fallback_depth_valid) {
            const float consistency_gate_px = computeSubpixelDispDeltaGateCPU(
                circle_disparity,
                focal,
                baseline,
                config_.dual_yolo.subpixel_max_disp_delta_px,
                config_.dual_yolo.subpixel_max_disp_delta_ratio,
                config_.dual_yolo.subpixel_max_depth_delta_m);
            bool has_consistency_check = false;
            bool any_consistency_passed = false;

            auto check_fallback_disparity = [&](bool* valid,
                                                const SubpixelDisparityResult& result) {
                if (!*valid || !result.valid) return;
                has_consistency_check = true;
                const bool consistent =
                    std::abs(result.disparity - circle_disparity) <= consistency_gate_px;
                if (consistent) {
                    any_consistency_passed = true;
                } else {
                    *valid = false;
                }
            };

            check_fallback_disparity(&center_patch_valid_for_obj, center_patch_result);
            check_fallback_disparity(&subpixel_valid_for_obj, subpixel_result);
            if (!center_patch_valid_for_obj) {
                z_roi_center_patch = -1.0f;
            }
            if (!subpixel_valid_for_obj) {
                z_subpixel = -1.0f;
                disparity = -1.0f;
                disparity_conf = 1.0f;
            }
            if (has_consistency_check && !any_consistency_passed) {
                epipolar_fallback_depth_valid = false;
                fallback_template_depth_valid = false;
                any_fallback_depth_valid = false;
            }
        }

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
        depth_candidate_input.ring_edge_profile_result =
            ring_edge_profile_result;
        depth_candidate_input.z_roi_ring_edge_profile =
            z_roi_ring_edge_profile;
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
        const bool has_legacy_depth = depth_selection.valid;
        const bool emit_record_only_candidate =
            direct_yolo_match && direct_depth_without_circle_enabled;
        if (has_legacy_depth) {
            disparity = depth_selection.observation.disparity_px;
            depth_source = depth_selection.observation.stereo_depth_source;
            disparity_conf = depth_selection.observation.fusion_confidence;
        } else if (emit_record_only_candidate) {
            disparity = -1.0f;
            depth_source = 0;
            disparity_conf = 0.0f;
        } else {
            ++local_stats.depth_reject;
            return false;
        }

        const float z = has_legacy_depth ? depth_selection.observation.depth_m : -1.0f;
        if (has_legacy_depth &&
            (z < config_.depth.min_depth || z > config_.depth.max_depth)) {
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
        if (has_legacy_depth) {
            obj.x = (anchor_cx - cx0) * z / focal;
            obj.y = (anchor_cy - cy0) * z / focal;
            obj.z = z;
            obj.raw_x = obj.x;
            obj.raw_y = obj.y;
            obj.raw_z = obj.z;
            obj.raw_observation_valid = 1;
            obj.z_stereo = z;
        } else {
            obj.x = 0.0f;
            obj.y = 0.0f;
            obj.z = -1.0f;
            obj.raw_x = 0.0f;
            obj.raw_y = 0.0f;
            obj.raw_z = -1.0f;
            obj.raw_observation_valid = 0;
            obj.z_stereo = -1.0f;
        }
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
        obj.z_roi_ring_edge_profile = z_roi_ring_edge_profile;
        obj.z_roi_neural_feature = z_roi_neural_feature;
        obj.z_roi_neural_xfeat = z_roi_neural_xfeat;
        obj.z_roi_neural_superpoint = z_roi_neural_superpoint;
        obj.z_roi_neural_aliked = z_roi_neural_aliked;
        obj.z_roi_center_patch = z_roi_center_patch;
        obj.z_roi_multi_point = z_subpixel;
        obj.z_yolo_bbox_pair = z_yolo;
        obj.z_circle = circle_candidate_valid ? z_circle_raw : -1.0f;
        obj.z_subpixel = z_subpixel;
        obj.z_fallback_epipolar =
            epipolar_fallback_depth_valid ? z_circle_raw : -1.0f;
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
        obj.disparity_roi_ring_edge_profile =
            ring_edge_profile_result.valid
                ? ring_edge_profile_result.disparity
                : -1.0f;
        obj.disparity_roi_neural_feature =
            neural_feature_result.valid
                ? neural_feature_result.disparity
                : -1.0f;
        obj.disparity_roi_neural_xfeat =
            neural_xfeat_result.valid ? neural_xfeat_result.disparity : -1.0f;
        obj.disparity_roi_neural_superpoint =
            neural_superpoint_result.valid
                ? neural_superpoint_result.disparity
                : -1.0f;
        obj.disparity_roi_neural_aliked =
            neural_aliked_result.valid ? neural_aliked_result.disparity : -1.0f;
        obj.disparity_roi_center_patch =
            center_patch_valid_for_obj ? center_patch_result.disparity : -1.0f;
        obj.disparity_roi_multi_point =
            subpixel_valid_for_obj ? subpixel_result.disparity : -1.0f;
        obj.disparity_fallback_epipolar =
            epipolar_fallback_depth_valid ? circle_disparity : -1.0f;
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
        obj.roi_ring_edge_profile_support =
            ring_edge_profile_result.support;
        obj.roi_ring_edge_profile_std_px =
            ring_edge_profile_result.valid
                ? ring_edge_profile_result.stddev
                : -1.0f;
        obj.roi_ring_edge_profile_confidence =
            ring_edge_profile_result.confidence;
        obj.roi_neural_feature_support = neural_feature_result.support;
        obj.roi_neural_feature_std_px =
            neural_feature_result.valid ? neural_feature_result.stddev : -1.0f;
        obj.roi_neural_feature_confidence = neural_feature_result.confidence;
        obj.roi_neural_xfeat_support = neural_xfeat_result.support;
        obj.roi_neural_xfeat_std_px =
            neural_xfeat_result.valid ? neural_xfeat_result.stddev : -1.0f;
        obj.roi_neural_xfeat_confidence = neural_xfeat_result.confidence;
        obj.roi_neural_superpoint_support = neural_superpoint_result.support;
        obj.roi_neural_superpoint_std_px =
            neural_superpoint_result.valid ? neural_superpoint_result.stddev
                                           : -1.0f;
        obj.roi_neural_superpoint_confidence =
            neural_superpoint_result.confidence;
        obj.roi_neural_aliked_support = neural_aliked_result.support;
        obj.roi_neural_aliked_std_px =
            neural_aliked_result.valid ? neural_aliked_result.stddev : -1.0f;
        obj.roi_neural_aliked_confidence =
            neural_aliked_result.confidence;
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

    std::vector<P2EarlyFeatureResult> p2_early_features;
    auto find_p2_early_feature =
        [&](int left_index,
            int right_index) -> const P2EarlyFeatureResult* {
        for (const auto& feature : p2_early_features) {
            if (feature.completed &&
                feature.left_index == left_index &&
                feature.right_index == right_index) {
                return &feature;
            }
        }
        return nullptr;
    };
    if (!direct_pairs.empty() && p2_inline_feature_jobs_enabled) {
        std::vector<const StereoRoiPair*> early_pairs;
        std::vector<bool> early_left_used(left_detections.size(), false);
        std::vector<bool> early_right_used(right_detections.size(), false);
        for (const auto& candidate_pair : direct_pairs) {
            const int li = candidate_pair.left_index;
            const int ri = candidate_pair.right_index;
            if (li < 0 || ri < 0 ||
                li >= static_cast<int>(left_detections.size()) ||
                ri >= static_cast<int>(right_detections.size()) ||
                early_left_used[static_cast<size_t>(li)] ||
                early_right_used[static_cast<size_t>(ri)]) {
                continue;
            }
            early_left_used[static_cast<size_t>(li)] = true;
            early_right_used[static_cast<size_t>(ri)] = true;
            early_pairs.push_back(&candidate_pair);
        }

        bool p1_collect_overlapped_with_p2 = false;
        for (const StereoRoiPair* early_pair_ptr : early_pairs) {
            const StereoRoiPair& early_pair = *early_pair_ptr;
            const Detection& early_left =
                left_detections[static_cast<size_t>(early_pair.left_index)];
            const Detection& early_right =
                right_detections[static_cast<size_t>(early_pair.right_index)];
            const float early_initial_disp = early_pair.initial_disparity;
            const bool early_common_gate =
                early_initial_disp > 0.0f &&
                gpu_image_available &&
                stream != nullptr;
            P2EarlyFeatureResult p2_early_feature;
            p2_early_feature.left_index = early_pair.left_index;
            p2_early_feature.right_index = early_pair.right_index;
            p2_early_feature.initial_disparity = early_initial_disp;
            p2_early_feature.cuda_template_due =
                roi_cuda_template_match_depth_enabled && early_common_gate;
            p2_early_feature.neural_xfeat_due =
                neural_xfeat_depth_enabled && early_common_gate;
            p2_early_feature.neural_superpoint_due =
                neural_superpoint_realtime_due && early_common_gate;
            p2_early_feature.neural_aliked_due =
                neural_aliked_realtime_due && early_common_gate;
            const int p2_early_algo_count =
                (p2_early_feature.cuda_template_due ? 1 : 0) +
                (p2_early_feature.neural_xfeat_due ? 1 : 0) +
                (p2_early_feature.neural_superpoint_due ? 1 : 0) +
                (p2_early_feature.neural_aliked_due ? 1 : 0);
            if (p2_early_algo_count > 0) {
                globalPerf().record("Stage2_P2InlineFeatureAlgoCount",
                                    static_cast<double>(p2_early_algo_count));
                globalPerf().record("Stage2_P2EarlyFeatureAlgoCount",
                                    static_cast<double>(p2_early_algo_count));

                bool cuda_template_submitted = false;
                bool neural_xfeat_submitted = false;
                bool neural_superpoint_submitted = false;
                bool neural_aliked_submitted = false;
                const auto p2_early_start = Clock::now();

                if (p2_early_feature.cuda_template_due) {
                    cuda_template_submitted = submitP2InlineFeatureTask(
                        p2_inline_ncc_worker_,
                        [&, early_left, early_right, early_initial_disp] {
                            p2_early_feature.cuda_template =
                                run_cuda_template_candidate(
                                    early_left, early_right,
                                    early_initial_disp,
                                    streams_.cudaStreamP2Ncc,
                                    p2_input_ready_event,
                                    "NCC");
                        });
                }
                if (p2_early_feature.neural_xfeat_due) {
                    neural_xfeat_submitted = submitP2InlineFeatureTask(
                        p2_inline_xfeat_worker_,
                        [&, early_left, early_right, early_initial_disp] {
                            p2_early_feature.neural_xfeat =
                                run_neural_candidate(
                                    neural_xfeat_matcher_.get(),
                                    early_left, early_right,
                                    early_initial_disp,
                                    streams_.cudaStreamP2XFeat,
                                    p2_input_ready_event,
                                    "XFeat");
                        });
                }
                if (p2_early_feature.neural_superpoint_due) {
                    neural_superpoint_submitted = submitP2InlineFeatureTask(
                        p2_inline_superpoint_worker_,
                        [&, early_left, early_right, early_initial_disp] {
                            p2_early_feature.neural_superpoint =
                                run_neural_candidate(
                                    neural_superpoint_matcher_.get(),
                                    early_left, early_right,
                                    early_initial_disp,
                                    streams_.cudaStreamP2SuperPoint,
                                    p2_input_ready_event,
                                    "SuperPoint");
                        });
                }
                if (p2_early_feature.neural_aliked_due) {
                    neural_aliked_submitted = submitP2InlineFeatureTask(
                        p2_inline_aliked_worker_,
                        [&, early_left, early_right, early_initial_disp] {
                            p2_early_feature.neural_aliked =
                                run_neural_candidate(
                                    neural_aliked_matcher_.get(),
                                    early_left, early_right,
                                    early_initial_disp,
                                    streams_.cudaStreamP2Aliked,
                                    p2_input_ready_event,
                                    "ALIKED");
                        });
                }

                const bool all_submitted =
                    (!p2_early_feature.cuda_template_due || cuda_template_submitted) &&
                    (!p2_early_feature.neural_xfeat_due || neural_xfeat_submitted) &&
                    (!p2_early_feature.neural_superpoint_due ||
                     neural_superpoint_submitted) &&
                    (!p2_early_feature.neural_aliked_due ||
                     neural_aliked_submitted);
                if (!all_submitted) {
                    globalPerf().record("Stage2_P2InlineSubmitFallback", 0.0);
                    globalPerf().record("Stage2_P2EarlySubmitFallback", 0.0);
                }

                // Let the P1 GPU candidate collect wait overlap with the first
                // early P2 batch. Later batches are for multi-object coverage.
                if (!p1_collect_overlapped_with_p2) {
                    collect_gpu_candidates();
                    p1_collect_overlapped_with_p2 = true;
                }

                const auto p2_post_p1_join_start = Clock::now();
                if (cuda_template_submitted) {
                    waitP2InlineFeatureTask(p2_inline_ncc_worker_);
                }
                if (neural_xfeat_submitted) {
                    waitP2InlineFeatureTask(p2_inline_xfeat_worker_);
                }
                if (neural_superpoint_submitted) {
                    waitP2InlineFeatureTask(p2_inline_superpoint_worker_);
                }
                if (neural_aliked_submitted) {
                    waitP2InlineFeatureTask(p2_inline_aliked_worker_);
                }
                if (cuda_template_submitted ||
                    neural_xfeat_submitted ||
                    neural_superpoint_submitted ||
                    neural_aliked_submitted) {
                    globalPerf().record(
                        "Stage2_P2EarlyPostP1JoinWait",
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - p2_post_p1_join_start).count());
                }
                const double p2_early_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - p2_early_start).count();
                if (cuda_template_submitted ||
                    neural_xfeat_submitted ||
                    neural_superpoint_submitted ||
                    neural_aliked_submitted) {
                    globalPerf().record("Stage2_P2InlineFeatureParallelJoin",
                                        p2_early_ms);
                    globalPerf().record("Stage2_P2EarlyFeatureParallelJoin",
                                        p2_early_ms);
                    globalPerf().record("Stage2_P1P2OverlapWindow",
                                        p2_early_ms);
                }
                if (all_submitted) {
                    p2_early_feature.completed = true;
                    globalPerf().record("Stage2_P2InlineFeatureEndToEnd",
                                        p2_early_ms);
                    globalPerf().record("Stage2_P2EarlyFeatureEndToEnd",
                                        p2_early_ms);
                    p2_early_features.push_back(p2_early_feature);
                }
            }
        }
    }

    collect_gpu_candidates();

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
        const auto circle_fit_start = Clock::now();
        CircleFit2D left_circle = gpu_candidate
            ? circleFromGpuCandidate(gpu_candidate->left_circle, left)
            : refine_detection(left_cpu, left_pitch, left);
        CircleFit2D right_circle = gpu_candidate
            ? circleFromGpuCandidate(gpu_candidate->right_circle, right)
            : refine_detection(right_cpu, right_pitch, right);
        globalPerf().record(
            gpu_candidate ? "Stage2_CircleMatchGpuCandidate"
                          : "Stage2_CircleMatchFit",
            std::chrono::duration<double, std::milli>(
                Clock::now() - circle_fit_start).count());
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
        const P2EarlyFeatureResult* p2_early_match =
            find_p2_early_feature(li, best_idx);
        if (!build_object(left, left_circle, right_circle, semantic_conf,
                          1, bbox_disparity, &right, &best_pair,
                          gpu_candidate,
                          p2_early_match,
                          obj)) {
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
                              2, -1.0f, nullptr, nullptr, nullptr, nullptr,
                              obj)) {
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
                              nullptr, nullptr, obj)) {
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

}  // namespace stereo3d
