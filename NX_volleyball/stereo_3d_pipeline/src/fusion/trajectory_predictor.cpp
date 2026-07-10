/**
 * @file trajectory_predictor.cpp
 * @brief Realtime volleyball landing predictor
 *
 * Primary: bbox/circle observation -> Student-t EKF -> RK4 landing.
 * Backup: polynomial history fit.
 *
 * Camera frame: X=right, Y=down, Z=depth.
 * LandingPrediction.x = landing camera-X
 * LandingPrediction.y = landing camera-Z (depth)  [existing API contract]
 * LandingPrediction.z = landing camera-Y
 */

#include "trajectory_predictor.h"
#include "../utils/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace stereo3d {
namespace {

inline int idx(int r, int c) { return r * 6 + c; }

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

}  // namespace

void TrajectoryPredictor::init(const TrajectoryPredictorConfig& config) {
    cfg_ = config;
    const float A = 3.14159265358979323846f * cfg_.ball_radius * cfg_.ball_radius;
    drag_k_ = 0.5f * cfg_.drag_coeff * cfg_.air_density * A / std::max(1e-6f, cfg_.ball_mass);

    // Keep ground_y / ground_h consistent for the common Y-down case.
    if (!cfg_.use_g_hat) {
        cfg_.g_hat_x = 0.0f;
        cfg_.g_hat_y = 1.0f;
        cfg_.g_hat_z = 0.0f;
        cfg_.ground_h = -cfg_.ground_y;
    } else {
        const float n = std::sqrt(cfg_.g_hat_x * cfg_.g_hat_x +
                                  cfg_.g_hat_y * cfg_.g_hat_y +
                                  cfg_.g_hat_z * cfg_.g_hat_z);
        if (n > 1e-6f) {
            cfg_.g_hat_x /= n;
            cfg_.g_hat_y /= n;
            cfg_.g_hat_z /= n;
        }
        // If only ground_y was provided, synthesize ground_h under Y-down approx.
        if (std::abs(cfg_.ground_h + cfg_.ground_y) > 1e-3f &&
            std::abs(cfg_.g_hat_y - 1.0f) < 1e-3f &&
            std::abs(cfg_.g_hat_x) < 1e-3f &&
            std::abs(cfg_.g_hat_z) < 1e-3f) {
            cfg_.ground_h = -cfg_.ground_y;
        }
    }

    if (cfg_.fB <= 0.0f && cfg_.fx > 0.0f) {
        // leave fB as-is; geometry optional until set from calib
    }
    if (cfg_.fy <= 0.0f && cfg_.fx > 0.0f) cfg_.fy = cfg_.fx;
    cfg_.have_geometry = (cfg_.fx > 1.0f && cfg_.fy > 1.0f && cfg_.fB > 1.0f);

    elapsed_time_ = 0.0;
    tracks_.clear();

    LOG_INFO("TrajectoryPredictor: mode=%s Cd=%.3f drag_k=%.5f ground_y=%.3f "
             "d0=%.3f fB=%.1f geom=%d nu=%.1f sigma_d=%.2f",
             cfg_.use_student_t_ekf ? "student_t_ekf" : "legacy",
             cfg_.drag_coeff, drag_k_, cfg_.ground_y,
             cfg_.d0, cfg_.fB, cfg_.have_geometry ? 1 : 0,
             cfg_.student_t_nu, cfg_.sigma_d_px);
}

void TrajectoryPredictor::reset() {
    tracks_.clear();
    elapsed_time_ = 0.0;
}

TrajectoryPredictor::TrackState* TrajectoryPredictor::findTrack(int track_id) {
    for (auto& tr : tracks_) {
        if (tr.track_id == track_id) return &tr;
    }
    return nullptr;
}

TrajectoryPredictor::TrackState& TrajectoryPredictor::getOrCreateTrack(int track_id) {
    if (auto* tr = findTrack(track_id)) return *tr;
    tracks_.push_back({});
    auto& tr = tracks_.back();
    tr.track_id = track_id;
    tr.t_origin = elapsed_time_;
    tr.t = elapsed_time_;
    return tr;
}

void TrajectoryPredictor::pruneTracks(const std::vector<Object3D>& results) {
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const TrackState& tr) {
                           for (const auto& r : results) {
                               if (r.track_id == tr.track_id) return false;
                           }
                           return true;
                       }),
        tracks_.end());
}

LandingPrediction TrajectoryPredictor::getPrediction(int track_id) const {
    for (const auto& tr : tracks_) {
        if (tr.track_id == track_id) return tr.last_pred;
    }
    return LandingPrediction{};
}

float TrajectoryPredictor::heightOf(double px, double py, double pz) const {
    if (cfg_.use_g_hat) {
        return static_cast<float>(
            -(cfg_.g_hat_x * px + cfg_.g_hat_y * py + cfg_.g_hat_z * pz) - cfg_.ground_h);
    }
    // Camera Y-down: above ground when y < ground_y.
    return static_cast<float>(cfg_.ground_y - py);
}

void TrajectoryPredictor::gravityAccel(double& ax, double& ay, double& az) const {
    if (cfg_.use_g_hat) {
        ax = cfg_.gravity * cfg_.g_hat_x;
        ay = cfg_.gravity * cfg_.g_hat_y;
        az = cfg_.gravity * cfg_.g_hat_z;
    } else {
        ax = 0.0;
        ay = cfg_.gravity;
        az = 0.0;
    }
}

bool TrajectoryPredictor::reproject(float u, float v, float disparity, bool apply_d0,
                                    float& x, float& y, float& z) const {
    if (!cfg_.have_geometry) return false;
    if (!(u >= 0.0f && v >= 0.0f && disparity > 0.0f)) return false;
    const float denom = apply_d0 ? (disparity - cfg_.d0) : disparity;
    if (denom <= 1.0f) return false;
    z = cfg_.fB / denom;
    if (!(z > 0.05f) || !std::isfinite(z)) return false;
    x = (u - cfg_.cx) * z / cfg_.fx;
    y = (v - cfg_.cy) * z / cfg_.fy;
    return std::isfinite(x) && std::isfinite(y);
}

bool TrajectoryPredictor::reprojectDepth(float u, float v, float depth,
                                         float& x, float& y, float& z) const {
    if (!cfg_.have_geometry) return false;
    if (!(u >= 0.0f && v >= 0.0f && depth > 0.05f)) return false;
    z = depth;
    x = (u - cfg_.cx) * z / cfg_.fx;
    y = (v - cfg_.cy) * z / cfg_.fy;
    return std::isfinite(x) && std::isfinite(y);
}

TrajectoryPredictor::Observation
TrajectoryPredictor::formObservation(const Object3D& obj) const {
    Observation obs;

    const bool have_bbox_disp =
        obj.disparity_bbox_center > 0.0f &&
        obj.left_bbox_cx >= 0.0f && obj.left_bbox_cy >= 0.0f;
    const bool have_circle_disp =
        obj.disparity_circle_center > 0.0f &&
        obj.left_circle_cx >= 0.0f && obj.left_circle_cy >= 0.0f;
    const bool have_bbox_z =
        obj.z_bbox_center > 0.0f &&
        obj.left_bbox_cx >= 0.0f && obj.left_bbox_cy >= 0.0f;
    const bool have_circle_z =
        obj.z_circle_center > 0.0f &&
        obj.left_circle_cx >= 0.0f && obj.left_circle_cy >= 0.0f;

    float bx = 0, by = 0, bz = 0;
    float cx = 0, cy = 0, cz = 0;
    bool bbox_ok = false;
    bool circle_ok = false;

    // Prefer disparity + d0 when geometry is available. Candidate z_* fields are
    // already d0-corrected by the dual-yolo depth path when d0 is enabled.
    if (cfg_.have_geometry && have_bbox_disp) {
        bbox_ok = reproject(obj.left_bbox_cx, obj.left_bbox_cy,
                            obj.disparity_bbox_center, /*apply_d0=*/true,
                            bx, by, bz);
    } else if (cfg_.have_geometry && have_bbox_z) {
        bbox_ok = reprojectDepth(obj.left_bbox_cx, obj.left_bbox_cy,
                                 obj.z_bbox_center, bx, by, bz);
    }

    if (cfg_.have_geometry && have_circle_disp) {
        circle_ok = reproject(obj.left_circle_cx, obj.left_circle_cy,
                              obj.disparity_circle_center, /*apply_d0=*/true,
                              cx, cy, cz);
    } else if (cfg_.have_geometry && have_circle_z) {
        circle_ok = reprojectDepth(obj.left_circle_cx, obj.left_circle_cy,
                                   obj.z_circle_center, cx, cy, cz);
    }

    if (cfg_.prefer_bbox && bbox_ok) {
        obs.x = bx; obs.y = by; obs.z = bz;
        obs.source = have_bbox_disp ? 0 : 1;
        obs.trust = (obj.p0p1_bbox_center_trust > 0.0f) ? obj.p0p1_bbox_center_trust : 1.0f;
        obs.valid = true;
        if (circle_ok) {
            const float dz = std::abs(cz - bz);
            obs.consistency = std::exp(-dz / std::max(1e-3f, cfg_.circle_consistency_m));
        }
        return obs;
    }

    if (cfg_.enable_circle_fallback && circle_ok) {
        obs.x = cx; obs.y = cy; obs.z = cz;
        obs.source = 2;
        obs.trust = (obj.p0p1_circle_center_trust > 0.0f) ? obj.p0p1_circle_center_trust : 0.8f;
        obs.consistency = 0.7f;
        obs.valid = true;
        return obs;
    }

    if (cfg_.allow_raw_fallback && obj.raw_observation_valid && obj.raw_z > 0.05f) {
        obs.x = obj.raw_x; obs.y = obj.raw_y; obs.z = obj.raw_z;
        obs.source = 3;
        obs.trust = 0.7f;
        obs.consistency = 0.5f;
        obs.valid = true;
        return obs;
    }

    if (cfg_.allow_filtered_fallback && obj.z > 0.05f) {
        obs.x = obj.x; obs.y = obj.y; obs.z = obj.z;
        obs.source = 4;
        obs.trust = 0.5f;
        obs.consistency = 0.4f;
        obs.valid = true;
        return obs;
    }

    return obs;
}

void TrajectoryPredictor::ekfInit(TrackState& tr, double t, const Observation& obs) const {
    tr.ekf_ready = true;
    tr.t = t;
    tr.x = {obs.x, obs.y, obs.z, 0.0, 0.0, 0.0};
    tr.P.fill(0.0);
    // Position variance
    tr.P[idx(0, 0)] = 0.05;
    tr.P[idx(1, 1)] = 0.05;
    tr.P[idx(2, 2)] = 0.05;
    // Velocity variance (cold start)
    tr.P[idx(3, 3)] = 25.0;
    tr.P[idx(4, 4)] = 25.0;
    tr.P[idx(5, 5)] = 25.0;
    tr.last_student_w = 1.0f;
    tr.last_obs_source = obs.source;
}

bool TrajectoryPredictor::ekfUpdate(TrackState& tr, double t, const Observation& obs) {
    if (!tr.ekf_ready) {
        ekfInit(tr, t, obs);
        return false;  // need at least one more frame for velocity
    }

    double dt = t - tr.t;
    if (dt <= 0.0) {
        return true;
    }
    if (dt > cfg_.max_dt) {
        // Large gap: soft reinit position, keep high velocity uncertainty.
        tr.x[0] = obs.x; tr.x[1] = obs.y; tr.x[2] = obs.z;
        tr.P.fill(0.0);
        tr.P[idx(0, 0)] = 0.05; tr.P[idx(1, 1)] = 0.05; tr.P[idx(2, 2)] = 0.05;
        tr.P[idx(3, 3)] = 25.0; tr.P[idx(4, 4)] = 25.0; tr.P[idx(5, 5)] = 25.0;
        tr.t = t;
        tr.last_student_w = 1.0f;
        tr.last_obs_source = obs.source;
        return true;
    }

    // ---- Predict ----
    double vx = tr.x[3], vy = tr.x[4], vz = tr.x[5];
    double speed = std::sqrt(vx * vx + vy * vy + vz * vz);
    double gax, gay, gaz;
    gravityAccel(gax, gay, gaz);
    double ax = gax, ay = gay, az = gaz;
    if (speed > 1e-3) {
        const double drag = drag_k_ * speed;
        ax += -drag * vx;
        ay += -drag * vy;
        az += -drag * vz;
    }

    // State transition Jacobian F
    double F[36];
    for (int i = 0; i < 36; ++i) F[i] = (i % 7 == 0) ? 1.0 : 0.0;
    F[idx(0, 3)] = dt; F[idx(1, 4)] = dt; F[idx(2, 5)] = dt;
    if (speed > 1e-3) {
        // da/dv ≈ -k ( |v| I + v v^T / |v| )
        const double inv_s = 1.0 / speed;
        const double k = drag_k_;
        const double J[3][3] = {
            {-k * (speed + vx * vx * inv_s), -k * (vx * vy * inv_s), -k * (vx * vz * inv_s)},
            {-k * (vy * vx * inv_s), -k * (speed + vy * vy * inv_s), -k * (vy * vz * inv_s)},
            {-k * (vz * vx * inv_s), -k * (vz * vy * inv_s), -k * (speed + vz * vz * inv_s)},
        };
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                F[idx(3 + r, 3 + c)] += J[r][c] * dt;
            }
        }
    }

    // x = f(x)
    tr.x[0] += vx * dt + 0.5 * ax * dt * dt;
    tr.x[1] += vy * dt + 0.5 * ay * dt * dt;
    tr.x[2] += vz * dt + 0.5 * az * dt * dt;
    tr.x[3] += ax * dt;
    tr.x[4] += ay * dt;
    tr.x[5] += az * dt;

    // P = F P F^T + Q
    double FP[36] = {0};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double s = 0.0;
            for (int k = 0; k < 6; ++k) s += F[idx(r, k)] * tr.P[idx(k, c)];
            FP[idx(r, c)] = s;
        }
    }
    double Pn[36] = {0};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double s = 0.0;
            for (int k = 0; k < 6; ++k) s += FP[idx(r, k)] * F[idx(c, k)];
            Pn[idx(r, c)] = s;
        }
    }
    Pn[idx(0, 0)] += cfg_.q_pos * dt;
    Pn[idx(1, 1)] += cfg_.q_pos * dt;
    Pn[idx(2, 2)] += cfg_.q_pos * dt;
    Pn[idx(3, 3)] += cfg_.q_vel * dt;
    Pn[idx(4, 4)] += cfg_.q_vel * dt;
    Pn[idx(5, 5)] += cfg_.q_vel * dt;
    tr.P = {};
    for (int i = 0; i < 36; ++i) tr.P[i] = Pn[i];
    tr.t = t;

    // ---- Update with Student-t reweighted R ----
    const double zc = std::max(0.5, tr.x[2]);
    const double fB = std::max(1.0, static_cast<double>(cfg_.fB > 0.0f ? cfg_.fB : 1500.0f));
    double sz = (zc * zc / fB) * cfg_.sigma_d_px;
    double sxy = std::max(static_cast<double>(cfg_.xy_sigma_floor),
                          static_cast<double>(cfg_.xy_sigma_scale) * zc);
    double inflate = 1.0;
    const double consistency = std::min(1.0, std::max(0.05, static_cast<double>(obs.consistency)));
    inflate *= 1.0 + (cfg_.consistency_inflate - 1.0f) * (1.0 - consistency);
    if (obs.trust >= 0.0f) {
        inflate *= 1.0 + std::max(0.0, 1.0 - static_cast<double>(obs.trust));
    }
    // De-weight non-bbox sources a bit.
    if (obs.source >= 2) inflate *= 1.5;
    if (obs.source >= 3) inflate *= 1.5;
    sxy *= inflate;
    sz *= inflate;

    const double Rdiag[3] = {sxy * sxy, sxy * sxy, sz * sz};
    const double innov[3] = {
        obs.x - tr.x[0],
        obs.y - tr.x[1],
        obs.z - tr.x[2],
    };

    // S = P_pos + R
    double S[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            S[r * 3 + c] = tr.P[idx(r, c)] + ((r == c) ? Rdiag[r] : 0.0);
        }
    }
    // Invert 3x3 S
    const double det =
        S[0] * (S[4] * S[8] - S[5] * S[7]) -
        S[1] * (S[3] * S[8] - S[5] * S[6]) +
        S[2] * (S[3] * S[7] - S[4] * S[6]);
    if (std::abs(det) < 1e-12) return true;
    double Sin[9];
    Sin[0] =  (S[4] * S[8] - S[5] * S[7]) / det;
    Sin[1] = -(S[1] * S[8] - S[2] * S[7]) / det;
    Sin[2] =  (S[1] * S[5] - S[2] * S[4]) / det;
    Sin[3] = -(S[3] * S[8] - S[5] * S[6]) / det;
    Sin[4] =  (S[0] * S[8] - S[2] * S[6]) / det;
    Sin[5] = -(S[0] * S[5] - S[2] * S[3]) / det;
    Sin[6] =  (S[3] * S[7] - S[4] * S[6]) / det;
    Sin[7] = -(S[0] * S[7] - S[1] * S[6]) / det;
    Sin[8] =  (S[0] * S[4] - S[1] * S[3]) / det;

    double delta = 0.0;
    for (int r = 0; r < 3; ++r) {
        double s = 0.0;
        for (int c = 0; c < 3; ++c) s += Sin[r * 3 + c] * innov[c];
        delta += innov[r] * s;
    }
    const double nu = std::max(1.0, static_cast<double>(cfg_.student_t_nu));
    const double w = (nu + 3.0) / (nu + delta);
    tr.last_student_w = static_cast<float>(w);
    tr.last_obs_source = obs.source;

    // Effective R and S
    const double R_eff[3] = {
        Rdiag[0] / std::max(0.05, w),
        Rdiag[1] / std::max(0.05, w),
        Rdiag[2] / std::max(0.05, w),
    };
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            S[r * 3 + c] = tr.P[idx(r, c)] + ((r == c) ? R_eff[r] : 0.0);
        }
    }
    const double det2 =
        S[0] * (S[4] * S[8] - S[5] * S[7]) -
        S[1] * (S[3] * S[8] - S[5] * S[6]) +
        S[2] * (S[3] * S[7] - S[4] * S[6]);
    if (std::abs(det2) < 1e-12) return true;
    Sin[0] =  (S[4] * S[8] - S[5] * S[7]) / det2;
    Sin[1] = -(S[1] * S[8] - S[2] * S[7]) / det2;
    Sin[2] =  (S[1] * S[5] - S[2] * S[4]) / det2;
    Sin[3] = -(S[3] * S[8] - S[5] * S[6]) / det2;
    Sin[4] =  (S[0] * S[8] - S[2] * S[6]) / det2;
    Sin[5] = -(S[0] * S[5] - S[2] * S[3]) / det2;
    Sin[6] =  (S[3] * S[7] - S[4] * S[6]) / det2;
    Sin[7] = -(S[0] * S[7] - S[1] * S[6]) / det2;
    Sin[8] =  (S[0] * S[4] - S[1] * S[3]) / det2;

    // K = P H^T S^{-1}, H = [I 0]
    double K[18];  // 6x3
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += tr.P[idx(r, k)] * Sin[k * 3 + c];
            K[r * 3 + c] = s;
        }
    }

    // x = x + K innov
    for (int r = 0; r < 6; ++r) {
        tr.x[r] += K[r * 3 + 0] * innov[0] +
                   K[r * 3 + 1] * innov[1] +
                   K[r * 3 + 2] * innov[2];
    }

    // P = (I - K H) P
    double IKH[36];
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double v = (r == c) ? 1.0 : 0.0;
            if (c < 3) v -= K[r * 3 + c];
            IKH[idx(r, c)] = v;
        }
    }
    double Pnew[36] = {0};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double s = 0.0;
            for (int k = 0; k < 6; ++k) s += IKH[idx(r, k)] * tr.P[idx(k, c)];
            Pnew[idx(r, c)] = s;
        }
    }
    for (int i = 0; i < 36; ++i) tr.P[i] = Pnew[i];
    return true;
}

LandingPrediction TrajectoryPredictor::predictBallisticState(
    double px, double py, double pz,
    double vx, double vy, double vz) const {

    LandingPrediction pred;
    const float h0 = heightOf(px, py, pz);
    if (h0 <= cfg_.min_height_for_predict) return pred;

    double s[6] = {px, py, pz, vx, vy, vz};
    const double dt = cfg_.rk4_dt;
    const int max_steps = static_cast<int>(cfg_.max_predict_time / std::max(1e-4f, cfg_.rk4_dt));
    double t = 0.0;
    float h_prev = h0;

    auto deriv = [&](const double st[6], double ds[6]) {
        ds[0] = st[3];
        ds[1] = st[4];
        ds[2] = st[5];
        const double sp = std::sqrt(st[3] * st[3] + st[4] * st[4] + st[5] * st[5]);
        double gax, gay, gaz;
        gravityAccel(gax, gay, gaz);
        double drag = drag_k_ * sp;
        ds[3] = gax - drag * st[3];
        ds[4] = gay - drag * st[4];
        ds[5] = gaz - drag * st[5];
    };

    for (int step = 0; step < max_steps; ++step) {
        double s_prev[6];
        for (int i = 0; i < 6; ++i) s_prev[i] = s[i];

        double k1[6], k2[6], k3[6], k4[6], tmp[6];
        deriv(s, k1);
        for (int i = 0; i < 6; ++i) tmp[i] = s[i] + 0.5 * dt * k1[i];
        deriv(tmp, k2);
        for (int i = 0; i < 6; ++i) tmp[i] = s[i] + 0.5 * dt * k2[i];
        deriv(tmp, k3);
        for (int i = 0; i < 6; ++i) tmp[i] = s[i] + dt * k3[i];
        deriv(tmp, k4);
        for (int i = 0; i < 6; ++i) {
            s[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += dt;

        const float h = heightOf(s[0], s[1], s[2]);
        if (h <= 0.0f && h_prev > 0.0f) {
            const double frac = static_cast<double>(h_prev) /
                                std::max(1e-9, static_cast<double>(h_prev - h));
            const double lx = s_prev[0] + (s[0] - s_prev[0]) * frac;
            const double ly = s_prev[1] + (s[1] - s_prev[1]) * frac;
            const double lz = s_prev[2] + (s[2] - s_prev[2]) * frac;
            pred.x = static_cast<float>(lx);
            pred.y = static_cast<float>(lz);  // depth as y for API compatibility
            pred.z = static_cast<float>(ly);
            pred.time_to_land = static_cast<float>(t - dt + dt * frac);
            pred.method = 0;
            pred.valid = true;
            pred.confidence = 0.75f;
            return pred;
        }
        h_prev = h;
    }
    return pred;
}

LandingPrediction TrajectoryPredictor::predictFromState(const TrackState& tr) const {
    if (!tr.ekf_ready) return LandingPrediction{};
    auto pred = predictBallisticState(
        tr.x[0], tr.x[1], tr.x[2], tr.x[3], tr.x[4], tr.x[5]);
    if (pred.valid) {
        pred.student_w = tr.last_student_w;
        pred.obs_source = tr.last_obs_source;
        // Confidence: higher when Student-t weight is healthy and TTI not tiny.
        float conf = 0.55f + 0.35f * clampf(tr.last_student_w, 0.0f, 1.0f);
        if (pred.time_to_land < 0.15f) conf *= 0.9f;
        if (tr.last_obs_source >= 3) conf *= 0.8f;
        pred.confidence = clampf(conf, 0.0f, 0.95f);
    }
    return pred;
}

LandingPrediction TrajectoryPredictor::predictPolynomial(const TrackState& tr) const {
    LandingPrediction pred;
    const int n = static_cast<int>(tr.samples.size());
    if (n < cfg_.poly_min_frames) return pred;

    std::vector<double> t(n), xs(n), ys(n), zs(n);
    for (int i = 0; i < n; ++i) {
        t[i] = tr.samples[i].t;
        xs[i] = tr.samples[i].x;
        ys[i] = tr.samples[i].y;
        zs[i] = tr.samples[i].z;
    }

    // Prefer fitting height vs time when g_hat is non-axis-aligned.
    std::vector<double> hs(n);
    for (int i = 0; i < n; ++i) {
        hs[i] = heightOf(xs[i], ys[i], zs[i]);
    }

    double ha, hb, hc;
    if (!fitQuadratic(t, hs, ha, hb, hc)) return pred;
    // Height above ground should curve downward: h'' < 0 => ha < 0.
    if (ha >= 0.0) return pred;

    // Solve ha t^2 + hb t + hc = 0 for ground crossing (height=0).
    const double disc = hb * hb - 4.0 * ha * hc;
    if (disc < 0.0) return pred;
    const double sqrt_disc = std::sqrt(disc);
    const double t1 = (-hb + sqrt_disc) / (2.0 * ha);
    const double t2 = (-hb - sqrt_disc) / (2.0 * ha);
    const double t_last = t.back();
    double t_land = -1.0;
    if (t1 > t_last && t2 > t_last) t_land = std::min(t1, t2);
    else if (t1 > t_last) t_land = t1;
    else if (t2 > t_last) t_land = t2;
    else return pred;

    double xa, xb, za, zb, ya, yb;
    if (!fitLinear(t, xs, xa, xb)) return pred;
    if (!fitLinear(t, zs, za, zb)) return pred;
    if (!fitLinear(t, ys, ya, yb)) return pred;

    pred.x = static_cast<float>(xa * t_land + xb);
    pred.y = static_cast<float>(za * t_land + zb);  // depth
    pred.z = static_cast<float>(ya * t_land + yb);
    pred.time_to_land = static_cast<float>(t_land - t_last);
    pred.method = 1;
    pred.valid = true;
    pred.confidence = 0.5f;
    return pred;
}

std::vector<LandingPrediction> TrajectoryPredictor::update(
    const std::vector<Object3D>& results, double dt) {

    elapsed_time_ += std::max(0.0, dt);
    pruneTracks(results);

    std::vector<LandingPrediction> preds(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& obj = results[i];
        if (obj.track_id < 0) continue;

        auto& tr = getOrCreateTrack(obj.track_id);
        const double t_rel = elapsed_time_ - tr.t_origin;
        // History for polynomial backup uses best available position.
        float hx = obj.x, hy = obj.y, hz = obj.z;
        if (obj.raw_observation_valid && obj.raw_z > 0.05f) {
            hx = obj.raw_x; hy = obj.raw_y; hz = obj.raw_z;
        }
        tr.samples.push_back({t_rel, hx, hy, hz});
        while (static_cast<int>(tr.samples.size()) > cfg_.history_max) {
            tr.samples.pop_front();
        }

        LandingPrediction ekf_pred;
        if (cfg_.use_student_t_ekf) {
            Observation obs = formObservation(obj);
            if (obs.valid) {
                const bool can_predict = ekfUpdate(tr, elapsed_time_, obs);
                if (can_predict) {
                    const double speed = std::sqrt(
                        tr.x[3] * tr.x[3] + tr.x[4] * tr.x[4] + tr.x[5] * tr.x[5]);
                    if (speed >= cfg_.min_speed_for_predict) {
                        ekf_pred = predictFromState(tr);
                    }
                }
            }
        } else {
            // Legacy path: use HybridDepth filtered state directly.
            const float speed = std::sqrt(obj.vx * obj.vx + obj.vy * obj.vy + obj.vz * obj.vz);
            if (speed >= cfg_.min_speed_for_predict) {
                ekf_pred = predictBallisticState(obj.x, obj.y, obj.z, obj.vx, obj.vy, obj.vz);
                if (ekf_pred.valid) ekf_pred.confidence = 0.7f;
            }
        }

        LandingPrediction poly;
        if (static_cast<int>(tr.samples.size()) >= cfg_.poly_min_frames) {
            poly = predictPolynomial(tr);
        }

        if (ekf_pred.valid && poly.valid) {
            const float dx = ekf_pred.x - poly.x;
            const float dy = ekf_pred.y - poly.y;
            const float dist = std::sqrt(dx * dx + dy * dy);
            ekf_pred.confidence = (dist < 1.0f)
                                      ? std::max(ekf_pred.confidence, 0.85f)
                                      : std::min(ekf_pred.confidence, 0.7f);
            preds[i] = ekf_pred;
        } else if (ekf_pred.valid) {
            preds[i] = ekf_pred;
        } else if (poly.valid) {
            preds[i] = poly;
        }

        tr.last_pred = preds[i];
    }

    return preds;
}

}  // namespace stereo3d
