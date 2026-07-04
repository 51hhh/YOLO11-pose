#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "pipeline_roi_match_helpers.h"
#include "../stereo/neural_feature_matcher.h"
#include "../utils/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <utility>
#include <vector>

namespace stereo3d {

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
    auto has_recordable_roi_result = [&](const Object3D& obj) {
        if (has_valid_stereo(obj)) return true;
        if (obj.stereo_match_source == 1 && obj.pair_positive_disparity) {
            return true;
        }
        auto valid_depth = [](float z) {
            return std::isfinite(z) && z > 0.0f;
        };
        auto observed = [&](float z, int support, float confidence) {
            return valid_depth(z) || support > 0 ||
                   (std::isfinite(confidence) && confidence > 0.0f);
        };
        return observed(obj.z_roi_multi_point,
                        obj.subpixel_support,
                        obj.subpixel_confidence) ||
               observed(obj.z_roi_center_patch,
                        obj.subpixel_support,
                        obj.subpixel_confidence) ||
               observed(obj.z_roi_corner_points,
                        obj.roi_corner_points_support,
                        obj.roi_corner_points_confidence) ||
               observed(obj.z_roi_texture_points,
                        obj.roi_texture_points_support,
                        obj.roi_texture_points_confidence) ||
               observed(obj.z_roi_binary_points,
                        obj.roi_binary_points_support,
                        obj.roi_binary_points_confidence) ||
               observed(obj.z_roi_orb_points,
                        obj.roi_orb_points_support,
                        obj.roi_orb_points_confidence) ||
               observed(obj.z_roi_brisk_points,
                        obj.roi_brisk_points_support,
                        obj.roi_brisk_points_confidence) ||
               observed(obj.z_roi_akaze_points,
                        obj.roi_akaze_points_support,
                        obj.roi_akaze_points_confidence) ||
               observed(obj.z_roi_sift_points,
                        obj.roi_sift_points_support,
                        obj.roi_sift_points_confidence) ||
               observed(obj.z_roi_iou_region_color_patch,
                        obj.roi_iou_region_color_patch_support,
                        obj.roi_iou_region_color_patch_confidence) ||
               observed(obj.z_roi_patch_iou_color_edge,
                        obj.roi_patch_iou_color_edge_support,
                        obj.roi_patch_iou_color_edge_confidence) ||
               observed(obj.z_roi_cuda_template_match,
                        obj.roi_cuda_template_match_support,
                        obj.roi_cuda_template_match_confidence) ||
               observed(obj.z_roi_cuda_stereo_bm,
                        obj.roi_cuda_stereo_bm_support,
                        obj.roi_cuda_stereo_bm_confidence) ||
               observed(obj.z_roi_cuda_stereo_sgm,
                        obj.roi_cuda_stereo_sgm_support,
                        obj.roi_cuda_stereo_sgm_confidence) ||
               observed(obj.z_roi_neural_feature,
                        obj.roi_neural_feature_support,
                        obj.roi_neural_feature_confidence);
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
            std::vector<Detection> valid_detections;
            std::vector<Object3D> valid_results;
            const size_t n = std::min(fusion_detections.size(), roi_results.size());
            valid_detections.reserve(n);
            valid_results.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (!has_recordable_roi_result(roi_results[i])) continue;
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
        result_callback_(slot.frame_id,
                         slot.results,
                         makeFrameMetadata(slot));
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
    const bool neural_needs_bgr =
        neural_feature_matcher_ && neural_feature_matcher_->requiresBgrInput();
    const bool has_stereo_detections =
        !slot.detections.empty() && !slot.detections_right.empty();
    const bool need_bgr =
        colorPipelineEnabled() &&
        has_stereo_detections &&
        (config_.dual_yolo.depth_roi_iou_region_color_patch ||
         config_.dual_yolo.depth_roi_patch_iou_color_edge ||
         neural_needs_bgr);
    const P2FeatureJobPolicy p2_policy = makeP2FeatureJobPolicy(config_);
    const P2FeatureJobDecision p2_decision = decideP2FeatureJobs(
        p2_policy,
        slot.frame_id,
        slot.detections,
        slot.detections_right,
        need_host_images,
        need_bgr);
    const std::vector<P2FeatureJobDescriptor> p2_feature_jobs =
        buildP2FeatureJobDescriptors(
            p2_policy,
            p2_decision,
            need_host_images,
            need_bgr);
    slot.p2_depth_modes_enabled = p2_decision.p2_depth_modes_enabled;
    slot.p2_depth_mode_mask = p2_decision.depth_mode_mask;
    slot.p2_feature_job_scaffold_enabled = p2_decision.split_feature_jobs;
    slot.p2_realtime_requested = p2_decision.realtime_requested;
    slot.p2_diagnostic_requested = p2_decision.diagnostic_requested;
    slot.p2_realtime_triggers = p2_decision.realtime_triggers;
    slot.p2_diagnostic_triggers = p2_decision.diagnostic_triggers;
    slot.p2_realtime_skip_reasons = p2_decision.realtime_skip_reasons;
    slot.p2_diagnostic_skip_reasons = p2_decision.diagnostic_skip_reasons;
    slot.p2_feature_job_count = static_cast<int>(p2_feature_jobs.size());
    slot.p2_left_count = p2_decision.left_count;
    slot.p2_right_count = p2_decision.right_count;
    slot.p2_valid_direct_pair_count = p2_decision.valid_direct_pair_count;
    enqueueP2FeatureDiagnosticJobs(makeFrameMetadata(slot), p2_feature_jobs);
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

}  // namespace stereo3d
