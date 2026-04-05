/**
 * @file hybrid_depth.cpp
 * @brief 单目+双目混合测距 + 9维Kalman滤波 实现
 */

#include "hybrid_depth.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace stereo3d {

// ============================================================
// DepthTrack: 9维 Kalman 滤波 (恒加速模型)
// ============================================================

void DepthTrack::init(float x0, float y0, float z0) {
    std::memset(state, 0, sizeof(state));
    x() = x0; y() = y0; z() = z0;
    // P = diag(1, 1, 1, 10, 10, 10, 100, 100, 100)
    std::memset(P, 0, sizeof(P));
    P[0][0] = P[1][1] = P[2][2] = 1.0f;       // 位置
    P[3][3] = P[4][4] = P[5][5] = 10.0f;       // 速度
    P[6][6] = P[7][7] = P[8][8] = 100.0f;      // 加速度
}

void DepthTrack::predict(float dt, float sigma_a) {
    // F = [I3, dt*I3, 0.5*dt^2*I3; 0, I3, dt*I3; 0, 0, I3]
    // 状态预测: p' = p + v*dt + 0.5*a*dt^2,  v' = v + a*dt,  a' = a
    const float dt2 = dt * dt;
    const float half_dt2 = 0.5f * dt2;

    for (int i = 0; i < 3; ++i) {
        state[i]     += state[i + 3] * dt + state[i + 6] * half_dt2;
        state[i + 3] += state[i + 6] * dt;
    }

    // 协方差预测: P' = F*P*F^T + Q
    // F 是稀疏矩阵, 手动按3×3块展开 (避免9×9通用矩阵乘法)
    // 先做 T = F*P (9×9), 再做 P' = T*F^T + Q
    float T[N][N];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < N; ++c) {
            T[r][c]     = P[r][c] + dt * P[r+3][c] + half_dt2 * P[r+6][c];
            T[r+3][c]   = P[r+3][c] + dt * P[r+6][c];
            T[r+6][c]   = P[r+6][c];
        }
    }
    // P' = T * F^T: 列上的操作
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < 3; ++c) {
            P[r][c]     = T[r][c] + dt * T[r][c+3] + half_dt2 * T[r][c+6];
            P[r][c+3]   = T[r][c+3] + dt * T[r][c+6];
            P[r][c+6]   = T[r][c+6];
        }
    }

    // Q = sigma_a^2 * G*G^T, G = [0.5*dt^2*I3; dt*I3; I3]
    const float sa2 = sigma_a * sigma_a;
    const float dt3 = dt2 * dt;
    const float dt4 = dt3 * dt;
    // Q 对角块 (3×3 对角):
    // Q_pp = sa2 * dt^4/4,  Q_pv = sa2 * dt^3/2,  Q_pa = sa2 * dt^2/2
    // Q_vv = sa2 * dt^2,    Q_va = sa2 * dt
    // Q_aa = sa2
    for (int i = 0; i < 3; ++i) {
        P[i][i]         += sa2 * dt4 * 0.25f;
        P[i][i+3]       += sa2 * dt3 * 0.5f;
        P[i][i+6]       += sa2 * dt2 * 0.5f;
        P[i+3][i]       += sa2 * dt3 * 0.5f;
        P[i+3][i+3]     += sa2 * dt2;
        P[i+3][i+6]     += sa2 * dt;
        P[i+6][i]       += sa2 * dt2 * 0.5f;
        P[i+6][i+3]     += sa2 * dt;
        P[i+6][i+6]     += sa2;
    }
}

void DepthTrack::update(float obs_x, float obs_y, float obs_z,
                        float Rxy, float Rz) {
    // H = [I3, 0, 0]  (3×9), 观测 = [x, y, z]
    // R = diag(Rxy, Rxy, Rz)
    float obs[M] = {obs_x, obs_y, obs_z};
    float R_diag[M] = {Rxy, Rxy, Rz};

    // Innovation: y = obs - H*x = obs - state[0:2]
    float y_inn[M];
    for (int i = 0; i < M; ++i) y_inn[i] = obs[i] - state[i];

    // S = H*P*H^T + R  (3×3)
    // Since H selects the first 3 rows/cols: S[i][j] = P[i][j] + R_diag[i] * delta_ij
    float S[M][M];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            S[i][j] = P[i][j] + (i == j ? R_diag[i] : 0.0f);

    // S^-1 (3×3 Cramer's rule)
    float det = S[0][0]*(S[1][1]*S[2][2] - S[1][2]*S[2][1])
              - S[0][1]*(S[1][0]*S[2][2] - S[1][2]*S[2][0])
              + S[0][2]*(S[1][0]*S[2][1] - S[1][1]*S[2][0]);
    if (std::fabs(det) < 1e-12f) return;  // singular, skip update
    float inv_det = 1.0f / det;

    float Si[M][M];
    Si[0][0] = (S[1][1]*S[2][2] - S[1][2]*S[2][1]) * inv_det;
    Si[0][1] = (S[0][2]*S[2][1] - S[0][1]*S[2][2]) * inv_det;
    Si[0][2] = (S[0][1]*S[1][2] - S[0][2]*S[1][1]) * inv_det;
    Si[1][0] = (S[1][2]*S[2][0] - S[1][0]*S[2][2]) * inv_det;
    Si[1][1] = (S[0][0]*S[2][2] - S[0][2]*S[2][0]) * inv_det;
    Si[1][2] = (S[0][2]*S[1][0] - S[0][0]*S[1][2]) * inv_det;
    Si[2][0] = (S[1][0]*S[2][1] - S[1][1]*S[2][0]) * inv_det;
    Si[2][1] = (S[0][1]*S[2][0] - S[0][0]*S[2][1]) * inv_det;
    Si[2][2] = (S[0][0]*S[1][1] - S[0][1]*S[1][0]) * inv_det;

    // K = P * H^T * S^-1  (9×3)
    // P*H^T = P[:, 0:2] (first 3 columns of P)
    float K[N][M];
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < M; ++c) {
            K[r][c] = 0.0f;
            for (int k = 0; k < M; ++k)
                K[r][c] += P[r][k] * Si[k][c];
        }

    // State update: x = x + K * y
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < M; ++c)
            state[r] += K[r][c] * y_inn[c];

    // Covariance update: P = (I - K*H) * P
    // (I-KH)[r][c] = delta(r,c) - K[r][c] for c<3, else delta(r,c)
    float P_new[N][N];
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            // IKH[r][k] * P[k][c]
            for (int k = 0; k < N; ++k) {
                float IKH_rk = (r == k ? 1.0f : 0.0f);
                if (k < M) IKH_rk -= K[r][k];
                sum += IKH_rk * P[k][c];
            }
            P_new[r][c] = sum;
        }
    std::memcpy(P, P_new, sizeof(P));

    last_raw_z = obs_z;
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

        // Step 3: 查找对应的双目结果
        float z_stereo = -1.0f;
        bool has_stereo = false;
        if (i < roi_results.size() && roi_results[i].confidence > config_.min_confidence) {
            z_stereo = roi_results[i].z;
            has_stereo = true;
        }

        // Step 4: 选择/融合测距方法 → 确定 z 观测值
        float z_pred = track->z() > 0.1f ? track->z() : z_mono;
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

        // Step 6: 计算 3D 观测 (像素→世界坐标)
        float obs_x = (det.cx - cx_) * z_obs / focal_;
        float obs_y = (det.cy - cy_) * z_obs / focal_;

        // 初始化新 track
        if (track->age == 1) {
            track->init(obs_x, obs_y, z_obs);
        }

        // Step 7: Kalman 更新 (9维, 3D观测)
        float Rz  = getObsNoise(z_obs, method);
        // xy 噪声与深度关联: sigma_xy = sigma_z * z / f (误差传播)
        float Rxy = Rz * (z_obs * z_obs) / (focal_ * focal_) + 0.001f;
        track->update(obs_x, obs_y, z_obs, Rxy, Rz);
        track->method = method;
        track->updateBBox(det.cx, det.cy, det.width, det.height);

        // Step 8: 输出 3D 结果 (从 Kalman 状态读取)
        Object3D obj;
        obj.x  = track->x();    obj.y  = track->y();    obj.z  = track->z();
        obj.vx = track->vx();   obj.vy = track->vy();   obj.vz = track->vz();
        obj.ax = track->ax();   obj.ay = track->ay();   obj.az = track->az();
        obj.confidence = det.confidence * std::max(0.0f, 1.0f - track->lost_count * 0.1f);
        obj.class_id = det.class_id;
        obj.track_id = track->track_id;

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
