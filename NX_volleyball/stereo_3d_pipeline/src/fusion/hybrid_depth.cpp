/**
 * @file hybrid_depth.cpp
 * @brief 单目+双目混合测距 + 9维Kalman滤波 实现
 */

#include "hybrid_depth.h"
#include <cmath>
#include <algorithm>

namespace stereo3d {

// ============================================================
// HybridDepthEstimator
// ============================================================

void HybridDepthEstimator::init(float focal, float baseline, float cx, float cy,
                                 const HybridDepthConfig& config) {
    focal_    = focal;
    baseline_ = baseline;
    cx_       = cx;
    cy_       = cy;
    config_   = config;
}

float HybridDepthEstimator::monoDepth(const Detection& det) const {
    // Z = focal * D_real / (bbox_width * bbox_scale)
    float w = det.width * config_.bbox_scale;
    if (w < 1.0f) return config_.max_depth;
    return focal_ * config_.object_diameter / w;
}

float HybridDepthEstimator::getObsNoise(float z, int method) const {
    // 距离自适应: R(z) = R_base × max(1, z²)
    float z2 = std::max(1.0f, z * z);
    float R_m = config_.R_mono * z2;
    float R_s = config_.R_stereo * z2;
    if (method == 0) return R_m;
    if (method == 1) return R_s;
    // blend: 使用与IVW相同的权重推导混合方差  Var(z_blend) = f_m²R_m + f_s²R_s
    float lo = config_.stereo_min_z;
    float hi = config_.mono_max_z;
    float blend = std::max(0.0f, std::min(1.0f, (z - lo) / (hi - lo)));
    float w_m = 1.0f / config_.ivw_R_mono;
    float w_s = blend / config_.ivw_R_stereo;
    float w_total = w_m + w_s;
    float f_m = w_m / w_total;
    float f_s = w_s / w_total;
    return f_m * f_m * R_m + f_s * f_s * R_s;
}

// ============================================================
// IoU Matching
// ============================================================

float HybridDepthEstimator::computeIoU(
    float cx1, float cy1, float w1, float h1,
    float cx2, float cy2, float w2, float h2)
{
    // Convert center+wh to xyxy
    float x1_min = cx1 - w1 * 0.5f, y1_min = cy1 - h1 * 0.5f;
    float x1_max = cx1 + w1 * 0.5f, y1_max = cy1 + h1 * 0.5f;
    float x2_min = cx2 - w2 * 0.5f, y2_min = cy2 - h2 * 0.5f;
    float x2_max = cx2 + w2 * 0.5f, y2_max = cy2 + h2 * 0.5f;

    float inter_x1 = std::max(x1_min, x2_min);
    float inter_y1 = std::max(y1_min, y2_min);
    float inter_x2 = std::min(x1_max, x2_max);
    float inter_y2 = std::min(y1_max, y2_max);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;

    if (union_area < 1e-6f) return 0.0f;
    return inter_area / union_area;
}

std::vector<int> HybridDepthEstimator::greedyIoUMatch(
    const std::vector<Detection>& detections, float iou_threshold)
{
    int N = static_cast<int>(detections.size());
    int M = static_cast<int>(tracks_.size());

    // result[det_idx] = track_idx or -1 (unmatched)
    std::vector<int> det_to_track(N, -1);
    std::vector<bool> track_used(M, false);

    // Build IoU matrix and greedily match highest IoU pairs
    struct Match {
        float iou;
        int det_idx;
        int track_idx;
    };
    std::vector<Match> candidates;
    candidates.reserve(N * M);

    for (int d = 0; d < N; ++d) {
        const auto& det = detections[d];
        for (int t = 0; t < M; ++t) {
            const auto& trk = tracks_[t];
            if (trk.last_w < 1.0f) continue;  // track has no bbox yet

            float iou = computeIoU(
                det.cx, det.cy, det.width, det.height,
                trk.last_cx, trk.last_cy, trk.last_w, trk.last_h);

            if (iou >= iou_threshold) {
                candidates.push_back({iou, d, t});
            }
        }
    }

    // Sort by IoU descending
    std::sort(candidates.begin(), candidates.end(),
        [](const Match& a, const Match& b) { return a.iou > b.iou; });

    // Greedy assignment
    for (const auto& m : candidates) {
        if (det_to_track[m.det_idx] != -1) continue;
        if (track_used[m.track_idx]) continue;
        det_to_track[m.det_idx] = m.track_idx;
        track_used[m.track_idx] = true;
    }

    return det_to_track;
}

std::vector<Object3D> HybridDepthEstimator::estimate(
    const std::vector<Detection>& detections,
    const std::vector<Object3D>& roi_results,
    double actual_dt)
{
    std::vector<Object3D> output;
    output.reserve(detections.size());

    const float dt = (actual_dt > 0.001) ? static_cast<float>(actual_dt) : config_.dt;

    // Step 0: Predict all existing tracks
    for (auto& track : tracks_) {
        track.predict(dt, config_.process_accel);
    }

    // Step 1: IoU greedy matching
    std::vector<int> det_to_track = greedyIoUMatch(detections);
    const size_t original_track_count = tracks_.size();

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        DepthTrack* track = nullptr;

        if (det_to_track[i] >= 0) {
            track = &tracks_[det_to_track[i]];
        } else {
            DepthTrack t;
            t.track_id = next_track_id_++;
            tracks_.push_back(t);
            track = &tracks_.back();
        }

        track->age++;

        // Step 2: 计算单目深度
        float z_mono = monoDepth(det);

        // Step 3: 查找对应的双目结果 (始终记录原始值供校准)
        float z_stereo = -1.0f;
        bool has_stereo = false;
        bool stereo_is_fallback = false;
        if (i < roi_results.size()) {
            const auto& roi = roi_results[i];
            z_stereo = roi_results[i].z;  // 原始值 (可能为-1)
            has_stereo = roi_results[i].confidence > config_.min_confidence && z_stereo > 0;
            stereo_is_fallback = roi.stereo_match_source == 2 ||
                                 roi.stereo_match_source == 3;
        }

        // Step 3.5: 自适应偏差校正 — EMA 跟踪 zs/zm 比例
        float z_stereo_corrected = z_stereo;
        if (has_stereo && !stereo_is_fallback && z_mono > 0.5f) {
            float ratio = z_stereo / z_mono;
            // 合理范围内更新 EMA (排除异常帧)
            if (ratio > 0.85f && ratio < 1.15f) {
                stereo_bias_ += config_.stereo_bias_alpha * (ratio - stereo_bias_);
            }
            // 应用偏差校正
            if (stereo_bias_ > 0.1f) {
                z_stereo_corrected = z_stereo / stereo_bias_;
            }
        }

        // Step 4: 选择/融合测距方法 → 确定 z 观测值
        float z_pred = track->z() > 0.1f ? track->z() : z_mono;
        float z_obs;
        int method;

        if (!has_stereo || z_pred < config_.stereo_min_z) {
            z_obs = z_mono;
            method = 0;
        } else if (z_pred > config_.mono_max_z) {
            // 纯双目: 使用校正后的值
            z_obs = z_stereo_corrected;
            method = 1;
        } else {
            // 过渡带: IVW 逆方差加权融合 (使用独立 IVW 权重)
            float w_mono   = 1.0f / config_.ivw_R_mono;
            float w_stereo = 1.0f / config_.ivw_R_stereo;
            // 距离相关权重渐变: 近端偏单目, 远端偏双目
            float blend = (z_pred - config_.stereo_min_z) /
                          (config_.mono_max_z - config_.stereo_min_z);
            blend = std::max(0.0f, std::min(1.0f, blend));
            w_stereo *= blend;  // 近端压制双目权重
            if (stereo_is_fallback) {
                w_stereo *= std::clamp(config_.fallback_stereo_weight_scale,
                                       0.0f, 1.0f);
            }
            float w_total = w_mono + w_stereo;
            z_obs = (w_mono * z_mono + w_stereo * z_stereo_corrected) / w_total;
            method = 2;
        }

        // Step 5: 范围检查
        z_obs = std::max(config_.min_depth, std::min(config_.max_depth, z_obs));

        // Step 6: 计算 3D 观测 (像素→世界坐标)
        float obs_x = (det.cx - cx_) * z_obs / focal_;
        float obs_y = (det.cy - cy_) * z_obs / focal_;

        // 初始化新 track
        if (track->age == 1) {
            track->init(obs_x, obs_y, z_obs);
        }

        // Step 7: Kalman 更新 (9维, 3D观测)
        float Rz  = getObsNoise(z_obs, method);
        if (stereo_is_fallback && has_stereo && method != 0) {
            Rz *= std::max(1.0f, config_.fallback_obs_noise_scale);
        }
        // xy 噪声与深度关联: sigma_xy = sigma_z * z / f (误差传播)
        float Rxy = Rz * (z_obs * z_obs) / (focal_ * focal_) + 0.001f;
        const float predicted_z = track->z();
        const float prior_z_var = std::max(0.0f, track->P[2][2]);
        const float innovation_z = z_obs - predicted_z;
        const float innovation_norm =
            innovation_z / std::sqrt(std::max(1e-6f, prior_z_var + Rz));
        track->update(obs_x, obs_y, z_obs, Rxy, Rz);
        const float kalman_sigma_z =
            std::sqrt(std::max(0.0f, track->P[2][2]));
        track->method = method;
        track->updateBBox(det.cx, det.cy, det.width, det.height);

        // Step 8: 输出 3D 结果 (从 Kalman 状态读取)
        Object3D obj;
        obj.x  = track->x();    obj.y  = track->y();    obj.z  = track->z();
        obj.vx = track->vx();   obj.vy = track->vy();   obj.vz = track->vz();
        obj.ax = track->ax();   obj.ay = track->ay();   obj.az = track->az();
        obj.raw_x = obs_x;
        obj.raw_y = obs_y;
        obj.raw_z = z_obs;
        obj.raw_observation_valid = 1;
        obj.predicted_z = predicted_z;
        obj.innovation_z = innovation_z;
        obj.innovation_norm = innovation_norm;
        obj.kalman_sigma_z = kalman_sigma_z;
        obj.confidence = det.confidence * std::max(0.0f, 1.0f - track->lost_count * 0.1f);
        obj.class_id = det.class_id;
        obj.track_id = track->track_id;
        obj.z_mono = z_mono;
        obj.z_stereo = z_stereo;
        if (i < roi_results.size()) {
            const auto& roi = roi_results[i];
            obj.z_bbox_center = roi.z_bbox_center;
            obj.z_bbox_left_edge = roi.z_bbox_left_edge;
            obj.z_bbox_right_edge = roi.z_bbox_right_edge;
            obj.z_circle_center = roi.z_circle_center;
            obj.z_circle_left_edge = roi.z_circle_left_edge;
            obj.z_circle_right_edge = roi.z_circle_right_edge;
            obj.z_roi_edge_centroid = roi.z_roi_edge_centroid;
            obj.z_roi_radial_center = roi.z_roi_radial_center;
            obj.z_roi_edge_pair_center = roi.z_roi_edge_pair_center;
            obj.z_roi_corner_points = roi.z_roi_corner_points;
            obj.z_roi_texture_points = roi.z_roi_texture_points;
            obj.z_roi_binary_points = roi.z_roi_binary_points;
            obj.z_roi_orb_points = roi.z_roi_orb_points;
            obj.z_roi_brisk_points = roi.z_roi_brisk_points;
            obj.z_roi_akaze_points = roi.z_roi_akaze_points;
            obj.z_roi_sift_points = roi.z_roi_sift_points;
            obj.z_roi_iou_region_color_patch = roi.z_roi_iou_region_color_patch;
            obj.z_roi_patch_iou_color_edge = roi.z_roi_patch_iou_color_edge;
            obj.z_roi_neural_feature = roi.z_roi_neural_feature;
            obj.z_roi_center_patch = roi.z_roi_center_patch;
            obj.z_roi_multi_point = roi.z_roi_multi_point;
            obj.z_yolo_bbox_pair = roi.z_yolo_bbox_pair;
            obj.z_circle = roi.z_circle;
            obj.z_subpixel = roi.z_subpixel;
            obj.z_fallback = roi.z_fallback;
            obj.z_fallback_epipolar = roi.z_fallback_epipolar;
            obj.z_fallback_template = roi.z_fallback_template;
            obj.z_fallback_feature_points = roi.z_fallback_feature_points;
            obj.disparity_bbox_center = roi.disparity_bbox_center;
            obj.disparity_bbox_left_edge = roi.disparity_bbox_left_edge;
            obj.disparity_bbox_right_edge = roi.disparity_bbox_right_edge;
            obj.disparity_circle_center = roi.disparity_circle_center;
            obj.disparity_circle_left_edge = roi.disparity_circle_left_edge;
            obj.disparity_circle_right_edge = roi.disparity_circle_right_edge;
            obj.disparity_roi_edge_centroid = roi.disparity_roi_edge_centroid;
            obj.disparity_roi_radial_center = roi.disparity_roi_radial_center;
            obj.disparity_roi_edge_pair_center = roi.disparity_roi_edge_pair_center;
            obj.disparity_roi_corner_points = roi.disparity_roi_corner_points;
            obj.disparity_roi_texture_points = roi.disparity_roi_texture_points;
            obj.disparity_roi_binary_points = roi.disparity_roi_binary_points;
            obj.disparity_roi_orb_points = roi.disparity_roi_orb_points;
            obj.disparity_roi_brisk_points = roi.disparity_roi_brisk_points;
            obj.disparity_roi_akaze_points = roi.disparity_roi_akaze_points;
            obj.disparity_roi_sift_points = roi.disparity_roi_sift_points;
            obj.disparity_roi_iou_region_color_patch =
                roi.disparity_roi_iou_region_color_patch;
            obj.disparity_roi_patch_iou_color_edge =
                roi.disparity_roi_patch_iou_color_edge;
            obj.disparity_roi_neural_feature =
                roi.disparity_roi_neural_feature;
            obj.disparity_roi_center_patch = roi.disparity_roi_center_patch;
            obj.disparity_roi_multi_point = roi.disparity_roi_multi_point;
            obj.disparity_fallback_epipolar = roi.disparity_fallback_epipolar;
            obj.disparity_fallback_template = roi.disparity_fallback_template;
            obj.disparity_fallback_feature_points = roi.disparity_fallback_feature_points;
            obj.disparity_yolo = roi.disparity_yolo;
            obj.disparity_circle = roi.disparity_circle;
            obj.disparity_subpixel = roi.disparity_subpixel;
            obj.left_bbox_cx = roi.left_bbox_cx;
            obj.left_bbox_cy = roi.left_bbox_cy;
            obj.left_bbox_w = roi.left_bbox_w;
            obj.left_bbox_h = roi.left_bbox_h;
            obj.left_bbox_conf = roi.left_bbox_conf;
            obj.right_bbox_cx = roi.right_bbox_cx;
            obj.right_bbox_cy = roi.right_bbox_cy;
            obj.right_bbox_w = roi.right_bbox_w;
            obj.right_bbox_h = roi.right_bbox_h;
            obj.right_bbox_conf = roi.right_bbox_conf;
            obj.left_circle_cx = roi.left_circle_cx;
            obj.left_circle_cy = roi.left_circle_cy;
            obj.left_circle_r = roi.left_circle_r;
            obj.right_circle_cx = roi.right_circle_cx;
            obj.right_circle_cy = roi.right_circle_cy;
            obj.right_circle_r = roi.right_circle_r;
            obj.left_circle_source = roi.left_circle_source;
            obj.right_circle_source = roi.right_circle_source;
            obj.epipolar_dy = roi.epipolar_dy;
            obj.size_ratio = roi.size_ratio;
            obj.left_circle_conf = roi.left_circle_conf;
            obj.right_circle_conf = roi.right_circle_conf;
            obj.subpixel_valid = roi.subpixel_valid;
            obj.subpixel_attempted = roi.subpixel_attempted;
            obj.subpixel_support = roi.subpixel_support;
            obj.subpixel_std_px = roi.subpixel_std_px;
            obj.subpixel_confidence = roi.subpixel_confidence;
            obj.subpixel_gate_px = roi.subpixel_gate_px;
            obj.roi_corner_points_support = roi.roi_corner_points_support;
            obj.roi_corner_points_std_px = roi.roi_corner_points_std_px;
            obj.roi_corner_points_confidence = roi.roi_corner_points_confidence;
            obj.roi_texture_points_support = roi.roi_texture_points_support;
            obj.roi_texture_points_std_px = roi.roi_texture_points_std_px;
            obj.roi_texture_points_confidence = roi.roi_texture_points_confidence;
            obj.roi_binary_points_support = roi.roi_binary_points_support;
            obj.roi_binary_points_std_px = roi.roi_binary_points_std_px;
            obj.roi_binary_points_confidence = roi.roi_binary_points_confidence;
            obj.roi_orb_points_support = roi.roi_orb_points_support;
            obj.roi_orb_points_std_px = roi.roi_orb_points_std_px;
            obj.roi_orb_points_confidence = roi.roi_orb_points_confidence;
            obj.roi_brisk_points_support = roi.roi_brisk_points_support;
            obj.roi_brisk_points_std_px = roi.roi_brisk_points_std_px;
            obj.roi_brisk_points_confidence = roi.roi_brisk_points_confidence;
            obj.roi_akaze_points_support = roi.roi_akaze_points_support;
            obj.roi_akaze_points_std_px = roi.roi_akaze_points_std_px;
            obj.roi_akaze_points_confidence = roi.roi_akaze_points_confidence;
            obj.roi_sift_points_support = roi.roi_sift_points_support;
            obj.roi_sift_points_std_px = roi.roi_sift_points_std_px;
            obj.roi_sift_points_confidence = roi.roi_sift_points_confidence;
            obj.roi_iou_region_color_patch_support =
                roi.roi_iou_region_color_patch_support;
            obj.roi_iou_region_color_patch_std_px =
                roi.roi_iou_region_color_patch_std_px;
            obj.roi_iou_region_color_patch_confidence =
                roi.roi_iou_region_color_patch_confidence;
            obj.roi_patch_iou_color_edge_support =
                roi.roi_patch_iou_color_edge_support;
            obj.roi_patch_iou_color_edge_std_px =
                roi.roi_patch_iou_color_edge_std_px;
            obj.roi_patch_iou_color_edge_confidence =
                roi.roi_patch_iou_color_edge_confidence;
            obj.roi_neural_feature_support =
                roi.roi_neural_feature_support;
            obj.roi_neural_feature_std_px =
                roi.roi_neural_feature_std_px;
            obj.roi_neural_feature_confidence =
                roi.roi_neural_feature_confidence;
            obj.fallback_feature_points_support = roi.fallback_feature_points_support;
            obj.fallback_feature_points_std_px = roi.fallback_feature_points_std_px;
            obj.fallback_feature_points_confidence = roi.fallback_feature_points_confidence;
            obj.pair_initial_disparity = roi.pair_initial_disparity;
            obj.pair_epipolar_dy = roi.pair_epipolar_dy;
            obj.pair_y_tolerance = roi.pair_y_tolerance;
            obj.pair_size_ratio = roi.pair_size_ratio;
            obj.pair_shifted_iou = roi.pair_shifted_iou;
            obj.pair_score = roi.pair_score;
            obj.pair_bbox_prior_penalty = roi.pair_bbox_prior_penalty;
            obj.pair_positive_disparity = roi.pair_positive_disparity;
            obj.stereo_match_source = roi.stereo_match_source;
            obj.stereo_depth_source = roi.stereo_depth_source;
        }
        obj.depth_method = method;

        if (obj.z >= config_.min_depth && obj.z <= config_.max_depth) {
            output.push_back(obj);
        }
    }

    // Mark unmatched tracks as lost (only check original tracks, not newly created ones)
    std::vector<bool> track_matched(original_track_count, false);
    for (int i = 0; i < (int)detections.size(); ++i) {
        if (det_to_track[i] >= 0 && det_to_track[i] < (int)original_track_count) {
            track_matched[det_to_track[i]] = true;
        }
    }
    for (size_t i = 0; i < original_track_count; ++i) {
        if (!track_matched[i]) {
            tracks_[i].lost_count++;
        }
    }

    pruneDeadTracks();
    return output;
}

std::vector<Object3D> HybridDepthEstimator::predictOnly() {
    std::vector<Object3D> output;

    for (auto& track : tracks_) {
        track.predict(config_.dt, config_.process_accel);
        track.lost_count++;

        if (track.lost_count <= config_.lost_predict_frames && track.z() > config_.min_depth) {
            Object3D obj;
            obj.x  = track.x();    obj.y  = track.y();    obj.z  = track.z();
            obj.vx = track.vx();   obj.vy = track.vy();   obj.vz = track.vz();
            obj.ax = track.ax();   obj.ay = track.ay();   obj.az = track.az();
            obj.predicted_z = track.z();
            obj.kalman_sigma_z = std::sqrt(std::max(0.0f, track.P[2][2]));
            obj.confidence = std::max(0.0f, 0.5f - track.lost_count * 0.1f);
            obj.class_id = 0;
            obj.track_id = track.track_id;
            output.push_back(obj);
        }
    }

    pruneDeadTracks();
    return output;
}

void HybridDepthEstimator::reset() {
    tracks_.clear();
    next_track_id_ = 0;
    stereo_bias_ = 0.95f;
}

int HybridDepthEstimator::activeTrackCount() const {
    int count = 0;
    for (const auto& t : tracks_) {
        if (t.lost_count <= config_.lost_degrade_frames) count++;
    }
    return count;
}

float HybridDepthEstimator::predictDepthForDetection(
    const Detection& det,
    float iou_threshold) const
{
    float best_iou = iou_threshold;
    float best_z = -1.0f;

    for (const auto& track : tracks_) {
        if (track.lost_count > config_.lost_degrade_frames ||
            track.last_w < 1.0f ||
            track.z() < config_.min_depth ||
            track.z() > config_.max_depth) {
            continue;
        }

        const float iou = computeIoU(
            det.cx, det.cy, det.width, det.height,
            track.last_cx, track.last_cy, track.last_w, track.last_h);
        if (iou > best_iou) {
            best_iou = iou;
            best_z = track.z();
        }
    }

    return best_z;
}

float HybridDepthEstimator::predictPrimaryDepth() const
{
    float best_score = -1.0f;
    float best_z = -1.0f;

    for (const auto& track : tracks_) {
        if (track.lost_count > config_.lost_degrade_frames ||
            track.z() < config_.min_depth ||
            track.z() > config_.max_depth) {
            continue;
        }

        const float score = track.confidence +
                            0.001f * static_cast<float>(track.age) -
                            0.05f * static_cast<float>(track.lost_count);
        if (score > best_score) {
            best_score = score;
            best_z = track.z();
        }
    }

    return best_z;
}

void HybridDepthEstimator::pruneDeadTracks() {
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const DepthTrack& t) {
                return t.lost_count > config_.lost_delete_frames;
            }),
        tracks_.end()
    );
}

}  // namespace stereo3d
