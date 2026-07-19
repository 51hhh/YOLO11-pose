/**
 * @file hybrid_depth.cpp
 * @brief 单目+双目混合测距 + 9维Kalman滤波 实现
 */

#include "hybrid_depth.h"
#include "hybrid_depth_candidate_copy.h"
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
    stereo_bias_ = std::max(0.1f, config_.stereo_bias_initial);
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

    // Build gated association candidates and greedily match the best pairs.
    struct Match {
        float score;
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

            if (trk.class_id >= 0 && det.class_id != trk.class_id) continue;
            if (trk.lost_count > config_.lost_degrade_frames) continue;

            float predicted_cx = trk.last_cx;
            float predicted_cy = trk.last_cy;
            if (trk.z() > config_.min_depth && std::isfinite(trk.z())) {
                predicted_cx = cx_ + focal_ * trk.x() / trk.z();
                predicted_cy = cy_ + focal_ * trk.y() / trk.z();
            }

            float iou = computeIoU(
                det.cx, det.cy, det.width, det.height,
                predicted_cx, predicted_cy, trk.last_w, trk.last_h);

            const float scale_x = std::max(1.0f, 0.5f * (det.width + trk.last_w));
            const float scale_y = std::max(1.0f, 0.5f * (det.height + trk.last_h));
            const float dx = (det.cx - predicted_cx) / scale_x;
            const float dy = (det.cy - predicted_cy) / scale_y;
            const float center_distance = std::sqrt(dx * dx + dy * dy);

            if (iou >= iou_threshold ||
                center_distance <= config_.match_center_gate) {
                const float score = 2.0f * iou - 0.15f * center_distance +
                                    0.20f * trk.confidence -
                                    0.05f * static_cast<float>(trk.lost_count);
                candidates.push_back({score, d, t});
            }
        }
    }

    // Sort by combined predicted-IoU/center/confidence score descending.
    std::sort(candidates.begin(), candidates.end(),
        [](const Match& a, const Match& b) { return a.score > b.score; });

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
    std::vector<int> det_to_track = greedyIoUMatch(
        detections, std::max(0.0f, config_.match_iou_threshold));
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
        if (config_.stereo_bias_correction_enabled &&
            has_stereo && !stereo_is_fallback && z_mono > 0.5f) {
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

        // Step 5: 范围检查。超范围观测不能通过 clamp 污染内部速度状态。
        if (!std::isfinite(z_obs) ||
            z_obs < config_.min_depth || z_obs >= config_.max_depth) {
            if (det_to_track[i] >= 0) {
                track->lost_count++;
            } else {
                tracks_.pop_back();
            }
            continue;
        }

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
        const float innovation_x = obs_x - track->x();
        const float innovation_y = obs_y - track->y();
        const float innovation_norm_x =
            innovation_x / std::sqrt(std::max(1e-6f, track->P[0][0] + Rxy));
        const float innovation_norm_y =
            innovation_y / std::sqrt(std::max(1e-6f, track->P[1][1] + Rxy));
        const float gate_sigma = config_.innovation_gate_sigma;
        const bool innovation_rejected =
            track->age > std::max(0, config_.innovation_gate_min_age) &&
            gate_sigma > 0.0f &&
            (std::abs(innovation_norm_x) > gate_sigma ||
             std::abs(innovation_norm_y) > gate_sigma ||
             std::abs(innovation_norm) > gate_sigma);
        if (innovation_rejected) {
            track->lost_count++;
            continue;
        }
        track->update(obs_x, obs_y, z_obs, Rxy, Rz);
        const float kalman_sigma_z =
            std::sqrt(std::max(0.0f, track->P[2][2]));
        track->method = method;
        track->class_id = det.class_id;
        const float confidence_alpha =
            std::clamp(config_.track_confidence_alpha, 0.0f, 1.0f);
        track->confidence = track->age <= 1
            ? det.confidence
            : (1.0f - confidence_alpha) * track->confidence +
                  confidence_alpha * det.confidence;
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
            copyRoiCandidateFields(roi_results[i], obj);
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

std::vector<Object3D> HybridDepthEstimator::predictOnly(double actual_dt) {
    std::vector<Object3D> output;
    const float dt = (actual_dt > 0.001)
        ? static_cast<float>(actual_dt)
        : config_.dt;

    for (auto& track : tracks_) {
        track.predict(dt, config_.process_accel);
        track.lost_count++;

        if (track.lost_count <= config_.lost_predict_frames && track.z() > config_.min_depth) {
            Object3D obj;
            obj.x  = track.x();    obj.y  = track.y();    obj.z  = track.z();
            obj.vx = track.vx();   obj.vy = track.vy();   obj.vz = track.vz();
            obj.ax = track.ax();   obj.ay = track.ay();   obj.az = track.az();
            obj.predicted_z = track.z();
            obj.kalman_sigma_z = std::sqrt(std::max(0.0f, track.P[2][2]));
            obj.confidence = track.confidence *
                std::max(0.0f, 1.0f - track.lost_count * 0.15f);
            obj.class_id = track.class_id;
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
    stereo_bias_ = std::max(0.1f, config_.stereo_bias_initial);
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
    float best_score = -std::numeric_limits<float>::infinity();
    float best_z = -1.0f;

    for (const auto& track : tracks_) {
        if (track.lost_count > config_.lost_degrade_frames ||
            (track.class_id >= 0 && det.class_id != track.class_id) ||
            track.last_w < 1.0f ||
            track.z() < config_.min_depth ||
            track.z() > config_.max_depth) {
            continue;
        }

        float predicted_cx = track.last_cx;
        float predicted_cy = track.last_cy;
        if (track.z() > config_.min_depth && std::isfinite(track.z())) {
            predicted_cx = cx_ + focal_ * track.x() / track.z();
            predicted_cy = cy_ + focal_ * track.y() / track.z();
        }
        const float iou = computeIoU(
            det.cx, det.cy, det.width, det.height,
            predicted_cx, predicted_cy, track.last_w, track.last_h);
        const float scale_x = std::max(1.0f, 0.5f * (det.width + track.last_w));
        const float scale_y = std::max(1.0f, 0.5f * (det.height + track.last_h));
        const float dx = (det.cx - predicted_cx) / scale_x;
        const float dy = (det.cy - predicted_cy) / scale_y;
        const float center_distance = std::sqrt(dx * dx + dy * dy);
        if (iou < iou_threshold && center_distance > config_.match_center_gate) {
            continue;
        }
        const float score = 2.0f * iou - 0.15f * center_distance +
                            0.20f * track.confidence -
                            0.05f * static_cast<float>(track.lost_count);
        if (score > best_score) {
            best_score = score;
            best_z = track.z();
        }
    }

    return best_z;
}

float HybridDepthEstimator::predictPrimaryDepth() const
{
    float best_score = -std::numeric_limits<float>::infinity();
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

    const size_t max_tracks = static_cast<size_t>(std::max(1, config_.max_tracks));
    while (tracks_.size() > max_tracks) {
        const auto worst = std::min_element(
            tracks_.begin(), tracks_.end(),
            [](const DepthTrack& a, const DepthTrack& b) {
                const float score_a = a.confidence -
                    0.10f * static_cast<float>(a.lost_count) +
                    0.001f * static_cast<float>(a.age);
                const float score_b = b.confidence -
                    0.10f * static_cast<float>(b.lost_count) +
                    0.001f * static_cast<float>(b.age);
                return score_a < score_b;
            });
        tracks_.erase(worst);
    }
}

}  // namespace stereo3d
