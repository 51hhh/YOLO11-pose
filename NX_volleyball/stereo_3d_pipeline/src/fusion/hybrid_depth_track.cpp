/**
 * @file hybrid_depth_track.cpp
 * @brief 9D constant-acceleration Kalman track implementation.
 */

#include "hybrid_depth.h"

#include <cmath>
#include <cstring>

namespace stereo3d {

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
    float obs[M] = {obs_x, obs_y, obs_z};

    // Innovation: y = obs - H*x = obs - state[0:2]
    float y_inn[M];
    for (int i = 0; i < M; ++i) y_inn[i] = obs[i] - state[i];

    // 观测噪声矩阵 R (对角)
    float R_diag[M] = {Rxy, Rxy, Rz};

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

    last_raw_x = obs_x;
    last_raw_y = obs_y;
    last_raw_z = obs_z;
    lost_count = 0;
}

void DepthTrack::updateBBox(float cx, float cy, float w, float h) {
    last_cx = cx;
    last_cy = cy;
    last_w  = w;
    last_h  = h;
}

}  // namespace stereo3d
