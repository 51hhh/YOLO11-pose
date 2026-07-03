/**
 * @file trajectory_recorder_summary.h
 * @brief Frame sidecar CSV summary helpers for TrajectoryRecorder.
 */

#ifndef STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_SUMMARY_H_
#define STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_SUMMARY_H_

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace stereo3d {

struct Object3D;

struct TrajectoryFrameSummaryStats {
    int tracked_count = 0;
    int raw_count = 0;
    int stereo_count = 0;
    int direct_count = 0;
    int fallback_l2r_count = 0;
    int fallback_r2l_count = 0;
    int pair_positive_count = 0;
    float pair_iou_min = -1.0f;
    double pair_iou_mean = -1.0;
    double pair_score_mean = -1.0;
    double pair_penalty_mean = -1.0;
    float pair_epipolar_dy_max = -1.0f;
    int iou_color_support_max = 0;
    int iou_edge_support_max = 0;
    int neural_support_max = 0;
    int p2_candidate_observed_count = 0;
    int p2_candidate_valid_count = 0;
    int p2_feature_valid_count = 0;
    int p2_cuda_valid_count = 0;
    int p2_neural_valid_count = 0;
    float best_confidence = 0.0f;
};

TrajectoryFrameSummaryStats summarizeTrajectoryFrame(
    const std::vector<Object3D>& results);

void writeTrajectoryFrameSummaryHeader(std::ostream& os);

void writeTrajectoryFrameSummaryRow(std::ostream& os,
                                    int frame_id,
                                    double timestamp,
                                    std::size_t result_count,
                                    const TrajectoryFrameSummaryStats& stats);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRAJECTORY_RECORDER_SUMMARY_H_
