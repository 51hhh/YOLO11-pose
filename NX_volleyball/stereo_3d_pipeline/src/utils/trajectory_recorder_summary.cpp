/**
 * @file trajectory_recorder_summary.cpp
 * @brief Frame sidecar CSV summary helpers for TrajectoryRecorder.
 */

#include "trajectory_recorder_summary.h"

#include "../pipeline/frame_slot.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>

namespace stereo3d {
namespace {

bool validDepth(float z) {
    return std::isfinite(z) && z > 0.0f;
}

bool observedCandidate(float z, int support, float confidence) {
    return validDepth(z) || support > 0 ||
           (std::isfinite(confidence) && confidence > 0.0f);
}

}  // namespace

TrajectoryFrameSummaryStats summarizeTrajectoryFrame(
    const std::vector<Object3D>& results) {
    TrajectoryFrameSummaryStats stats;

    int pair_count = 0;
    int pair_score_count = 0;
    int pair_penalty_count = 0;
    double pair_iou_sum = 0.0;
    double pair_score_sum = 0.0;
    double pair_penalty_sum = 0.0;

    for (const auto& r : results) {
        if (r.track_id >= 0) ++stats.tracked_count;
        if (r.raw_observation_valid) ++stats.raw_count;
        if (r.z_stereo > 0.0f) ++stats.stereo_count;
        if (r.stereo_match_source == 1) ++stats.direct_count;
        if (r.stereo_match_source == 2) ++stats.fallback_l2r_count;
        if (r.stereo_match_source == 3) ++stats.fallback_r2l_count;
        if (r.pair_positive_disparity) ++stats.pair_positive_count;
        if (r.pair_shifted_iou >= 0.0f) {
            ++pair_count;
            stats.pair_iou_min = stats.pair_iou_min < 0.0f
                ? r.pair_shifted_iou
                : std::min(stats.pair_iou_min, r.pair_shifted_iou);
            pair_iou_sum += r.pair_shifted_iou;
            if (std::isfinite(r.pair_score)) {
                pair_score_sum += r.pair_score;
                ++pair_score_count;
            }
            if (std::isfinite(r.pair_bbox_prior_penalty)) {
                pair_penalty_sum += r.pair_bbox_prior_penalty;
                ++pair_penalty_count;
            }
            if (r.pair_epipolar_dy >= 0.0f) {
                stats.pair_epipolar_dy_max =
                    std::max(stats.pair_epipolar_dy_max,
                             r.pair_epipolar_dy);
            }
        }
        stats.iou_color_support_max = std::max(
            stats.iou_color_support_max,
            r.roi_iou_region_color_patch_support);
        stats.iou_edge_support_max = std::max(
            stats.iou_edge_support_max,
            r.roi_patch_iou_color_edge_support);
        stats.neural_support_max = std::max(
            stats.neural_support_max,
            r.roi_neural_feature_support);

        auto count_candidate = [&](float z, int support, float confidence,
                                   int* group_valid) {
            if (observedCandidate(z, support, confidence)) {
                ++stats.p2_candidate_observed_count;
            }
            if (validDepth(z)) {
                ++stats.p2_candidate_valid_count;
                if (group_valid) ++(*group_valid);
            }
        };
        count_candidate(r.z_roi_corner_points,
                        r.roi_corner_points_support,
                        r.roi_corner_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_texture_points,
                        r.roi_texture_points_support,
                        r.roi_texture_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_binary_points,
                        r.roi_binary_points_support,
                        r.roi_binary_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_orb_points,
                        r.roi_orb_points_support,
                        r.roi_orb_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_brisk_points,
                        r.roi_brisk_points_support,
                        r.roi_brisk_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_akaze_points,
                        r.roi_akaze_points_support,
                        r.roi_akaze_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_sift_points,
                        r.roi_sift_points_support,
                        r.roi_sift_points_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_iou_region_color_patch,
                        r.roi_iou_region_color_patch_support,
                        r.roi_iou_region_color_patch_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_patch_iou_color_edge,
                        r.roi_patch_iou_color_edge_support,
                        r.roi_patch_iou_color_edge_confidence,
                        &stats.p2_feature_valid_count);
        count_candidate(r.z_roi_cuda_template_match,
                        r.roi_cuda_template_match_support,
                        r.roi_cuda_template_match_confidence,
                        &stats.p2_cuda_valid_count);
        count_candidate(r.z_roi_cuda_stereo_bm,
                        r.roi_cuda_stereo_bm_support,
                        r.roi_cuda_stereo_bm_confidence,
                        &stats.p2_cuda_valid_count);
        count_candidate(r.z_roi_cuda_stereo_sgm,
                        r.roi_cuda_stereo_sgm_support,
                        r.roi_cuda_stereo_sgm_confidence,
                        &stats.p2_cuda_valid_count);
        count_candidate(r.z_roi_ring_edge_profile,
                        r.roi_ring_edge_profile_support,
                        r.roi_ring_edge_profile_confidence,
                        &stats.p2_cuda_valid_count);
        count_candidate(r.z_roi_neural_feature,
                        r.roi_neural_feature_support,
                        r.roi_neural_feature_confidence,
                        &stats.p2_neural_valid_count);
        count_candidate(r.z_fallback_feature_points,
                        r.fallback_feature_points_support,
                        r.fallback_feature_points_confidence,
                        &stats.p2_feature_valid_count);
        stats.best_confidence = std::max(stats.best_confidence, r.confidence);
    }

    stats.pair_iou_mean =
        pair_count > 0 ? pair_iou_sum / static_cast<double>(pair_count) : -1.0;
    stats.pair_score_mean =
        pair_score_count > 0
            ? pair_score_sum / static_cast<double>(pair_score_count)
            : -1.0;
    stats.pair_penalty_mean =
        pair_penalty_count > 0
            ? pair_penalty_sum / static_cast<double>(pair_penalty_count)
            : -1.0;
    return stats;
}

void writeTrajectoryFrameSummaryHeader(std::ostream& os) {
    os << "frame_id,timestamp,left_frame_counter,right_frame_counter,"
       << "frame_counter_delta,frame_number_delta,timestamp_delta_us,"
       << "grab_failed,is_detect_frame,p2_depth_modes_enabled,"
       << "p2_depth_mode_mask,p2_feature_job_scaffold_enabled,p2_realtime_requested,"
       << "p2_diagnostic_requested,p2_feature_job_count,p2_left_count,"
       << "p2_right_count,p2_realtime_triggers,p2_diagnostic_triggers,"
       << "p2_realtime_skip_reasons,p2_diagnostic_skip_reasons,"
       << "p2_valid_direct_pair_count,"
       << "result_count,tracked_count,raw_observation_count,"
       << "stereo_observation_count,direct_pair_count,fallback_l2r_count,"
       << "fallback_r2l_count,pair_positive_count,pair_shifted_iou_min,"
       << "pair_shifted_iou_mean,pair_score_mean,pair_bbox_prior_penalty_mean,"
       << "pair_epipolar_dy_max,roi_iou_region_color_patch_support_max,"
       << "roi_patch_iou_color_edge_support_max,roi_neural_feature_support_max,"
       << "p2_candidate_observed_count,p2_candidate_valid_count,"
       << "p2_feature_valid_count,p2_cuda_valid_count,p2_neural_valid_count,"
       << "best_confidence\n";
}

void writeTrajectoryFrameSummaryRow(std::ostream& os,
                                    int frame_id,
                                    double timestamp,
                                    const FrameMetadata& metadata,
                                    std::size_t result_count,
                                    const TrajectoryFrameSummaryStats& stats) {
    os << frame_id << ","
       << std::fixed << std::setprecision(6) << timestamp << ","
       << metadata.left_frame_counter << ","
       << metadata.right_frame_counter << ","
       << metadata.frame_counter_delta << ","
       << metadata.frame_number_delta << ","
       << metadata.timestamp_delta_us << ","
       << (metadata.grab_failed ? 1 : 0) << ","
       << (metadata.is_detect_frame ? 1 : 0) << ","
       << (metadata.p2_depth_modes_enabled ? 1 : 0) << ","
       << metadata.p2_depth_mode_mask << ","
       << (metadata.p2_feature_job_scaffold_enabled ? 1 : 0) << ","
       << (metadata.p2_realtime_requested ? 1 : 0) << ","
       << (metadata.p2_diagnostic_requested ? 1 : 0) << ","
       << metadata.p2_feature_job_count << ","
       << metadata.p2_left_count << ","
       << metadata.p2_right_count << ","
       << metadata.p2_realtime_triggers << ","
       << metadata.p2_diagnostic_triggers << ","
       << metadata.p2_realtime_skip_reasons << ","
       << metadata.p2_diagnostic_skip_reasons << ","
       << metadata.p2_valid_direct_pair_count << ","
       << result_count << ","
       << stats.tracked_count << ","
       << stats.raw_count << ","
       << stats.stereo_count << ","
       << stats.direct_count << ","
       << stats.fallback_l2r_count << ","
       << stats.fallback_r2l_count << ","
       << stats.pair_positive_count << ","
       << std::setprecision(4)
       << stats.pair_iou_min << ","
       << stats.pair_iou_mean << ","
       << stats.pair_score_mean << ","
       << stats.pair_penalty_mean << ","
       << stats.pair_epipolar_dy_max << ","
       << stats.iou_color_support_max << ","
       << stats.iou_edge_support_max << ","
       << stats.neural_support_max << ","
       << stats.p2_candidate_observed_count << ","
       << stats.p2_candidate_valid_count << ","
       << stats.p2_feature_valid_count << ","
       << stats.p2_cuda_valid_count << ","
       << stats.p2_neural_valid_count << ","
       << stats.best_confidence << "\n";
}

}  // namespace stereo3d
