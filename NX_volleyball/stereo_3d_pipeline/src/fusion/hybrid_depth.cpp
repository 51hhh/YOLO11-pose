/**
 * @file hybrid_depth.cpp
 * @brief 单目+双目混合测距 + Kalman 滤波 实现
 */

#include "hybrid_depth.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace stereo3d {

// ============================================================
// DepthTrack: Kalman 滤波
// ============================================================

void DepthTrack::predict(float dt, float sigma_a) {
    // 状态预测: z' = z + vz*dt,  vz' = vz
    z  += vz * dt;
    vz  = vz;  // 匀速模型

    // 协方差预测: P' = F*P*F^T + Q
    // F = [1, dt; 0, 1]
    // Q = sigma_a^2 * [dt^4/4, dt^3/2; dt^3/2, dt^2]
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;
    float sa2 = sigma_a * sigma_a;

    float P00 = P[0][0] + dt * P[1][0] + dt * (P[0][1] + dt * P[1][1]);
    float P01 = P[0][1] + dt * P[1][1];
    float P10 = P[1][0] + dt * P[1][1];
    float P11 = P[1][1];

    P[0][0] = P00 + sa2 * dt4 / 4.0f;
    P[0][1] = P01 + sa2 * dt3 / 2.0f;
    P[1][0] = P10 + sa2 * dt3 / 2.0f;
    P[1][1] = P11 + sa2 * dt2;
}

void DepthTrack::update(float z_obs, float R) {
    // 观测模型: H = [1, 0]
    // y = z_obs - z (innovation)
    float y = z_obs - z;

    // S = H*P*H^T + R = P[0][0] + R
    float S = P[0][0] + R;
    if (S < 1e-6f) S = 1e-6f;

    // K = P*H^T / S = [P[0][0]/S, P[1][0]/S]
    float K0 = P[0][0] / S;
    float K1 = P[1][0] / S;

    // 状态更新
    z  += K0 * y;
    vz += K1 * y;

    // 协方差更新: P = (I - K*H) * P
    float P00 = (1.0f - K0) * P[0][0];
    float P01 = (1.0f - K0) * P[0][1];
    float P10 = P[1][0] - K1 * P[0][0];
    float P11 = P[1][1] - K1 * P[0][1];

    P[0][0] = P00;
    P[0][1] = P01;
    P[1][0] = P10;
    P[1][1] = P11;

    last_raw_z = z_obs;
    lost_count = 0;
}

void DepthTrack::updateBBox(float cx, float cy, float w, float h) {
    last_cx = cx;
    last_cy = cy;
    last_w  = w;
    last_h  = h;
}

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

float HybridDepthEstimator::blendDepth(float z_mono, float z_stereo, float z_pred) const {
    // 过渡带: [stereo_min_z, mono_max_z]
    float lo = config_.stereo_min_z;  // 3m
    float hi = config_.mono_max_z;    // 5m

    // 使用 z_pred 判断在过渡带的位置
    float ref_z = z_pred > 0.1f ? z_pred : z_mono;  // 优先使用 Kalman 预测

    if (ref_z < lo) return z_mono;
    if (ref_z > hi) return z_stereo;

    // alpha: 1=纯单目, 0=纯双目
    float alpha = (hi - ref_z) / (hi - lo);
    alpha = std::max(0.0f, std::min(1.0f, alpha));

    return alpha * z_mono + (1.0f - alpha) * z_stereo;
}

float HybridDepthEstimator::getObsNoise(float z, int method) const {
    if (method == 0) return config_.R_mono;
    if (method == 1) return config_.R_stereo;
    // blend: 线性插值
    float lo = config_.stereo_min_z;
    float hi = config_.mono_max_z;
    float alpha = std::max(0.0f, std::min(1.0f, (hi - z) / (hi - lo)));
    return alpha * config_.R_mono + (1.0f - alpha) * config_.R_stereo;
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
    const std::vector<Object3D>& roi_results)
{
    std::vector<Object3D> output;
    output.reserve(detections.size());

    // Step 0: Predict all existing tracks
    for (auto& track : tracks_) {
        track.predict(config_.dt, config_.process_accel);
    }

    // Step 1: IoU greedy matching
    std::vector<int> det_to_track = greedyIoUMatch(detections);

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        DepthTrack* track = nullptr;

        if (det_to_track[i] >= 0) {
            // Matched to existing track
            track = &tracks_[det_to_track[i]];
        } else {
            // New detection: create new track
            DepthTrack t;
            t.track_id = next_track_id_++;
            tracks_.push_back(t);
            track = &tracks_.back();
        }

        track->age++;

        // Step 2: 计算单目深度
        float z_mono = monoDepth(det);

        // Step 3: 查找对应的双目结果
        float z_stereo = -1.0f;
        bool has_stereo = false;
        if (i < roi_results.size() && roi_results[i].confidence > config_.min_confidence) {
            z_stereo = roi_results[i].z;
            has_stereo = true;
        }

        // Step 4: 选择/融合测距方法
        float z_pred = track->z > 0.1f ? track->z : z_mono;
        float z_obs;
        int method;

        if (!has_stereo || z_pred < config_.stereo_min_z) {
            z_obs = z_mono;
            method = 0;
        } else if (z_pred > config_.mono_max_z) {
            z_obs = z_stereo;
            method = 1;
        } else {
            z_obs = blendDepth(z_mono, z_stereo, z_pred);
            method = 2;
        }

        // Step 5: 范围检查
        z_obs = std::max(config_.min_depth, std::min(config_.max_depth, z_obs));

        // Step 6: Kalman 更新
        float R = getObsNoise(z_obs, method);
        track->update(z_obs, R);
        track->method = method;
        track->updateBBox(det.cx, det.cy, det.width, det.height);

        // Step 7: 计算 3D 坐标
        Object3D obj;
        obj.z = track->z;
        obj.x = (det.cx - cx_) * track->z / focal_;
        obj.y = (det.cy - cy_) * track->z / focal_;
        obj.confidence = det.confidence * std::max(0.0f, 1.0f - track->lost_count * 0.1f);
        obj.class_id = det.class_id;

        if (obj.z >= config_.min_depth && obj.z <= config_.max_depth) {
            output.push_back(obj);
        }
    }

    // Mark unmatched tracks as lost
    std::vector<bool> track_matched(tracks_.size(), false);
    for (int i = 0; i < (int)detections.size(); ++i) {
        if (det_to_track[i] >= 0) {
            track_matched[det_to_track[i]] = true;
        }
    }
    for (size_t i = 0; i < tracks_.size(); ++i) {
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

        if (track.lost_count <= config_.lost_predict_frames && track.z > config_.min_depth) {
            Object3D obj;
            obj.z = track.z;
            obj.x = 0;  // 无检测无法算 X,Y
            obj.y = 0;
            obj.confidence = std::max(0.0f, 0.5f - track.lost_count * 0.1f);
            obj.class_id = 0;
            output.push_back(obj);
        }
    }

    pruneDeadTracks();
    return output;
}

void HybridDepthEstimator::reset() {
    tracks_.clear();
    next_track_id_ = 0;
}

int HybridDepthEstimator::activeTrackCount() const {
    int count = 0;
    for (const auto& t : tracks_) {
        if (t.lost_count <= config_.lost_degrade_frames) count++;
    }
    return count;
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
