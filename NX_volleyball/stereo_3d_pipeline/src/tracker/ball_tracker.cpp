#include "ball_tracker.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

namespace stereo3d {

namespace {
constexpr int kStartupNeedHits = 3;
constexpr int kMaxMissFrames = 8;
constexpr int kHistoryMax = 12;

constexpr float kGroundEps = 0.03f;
constexpr float kImminentRelZ = 0.12f;
constexpr float kImminentT = 0.08f;
constexpr float kMinBounceVz = -1.20f;
constexpr int kGroundContactNeedHits = 3;
constexpr float kBallRadius = 0.11f;
constexpr float kBallRadiusForContact = 0.10f;
constexpr float kMaxLandingTime = 6.0f;
constexpr float kDeadBallQuietResetSec = 0.45f;
constexpr float kStartupGateX = 0.35f;
constexpr float kStartupGateY = 0.55f;
constexpr float kStartupGateZ = 0.35f;

static float median(std::vector<float>& a) {
    if (a.empty()) return std::numeric_limits<float>::quiet_NaN();
    size_t n = a.size() / 2;
    std::nth_element(a.begin(), a.begin() + n, a.end());
    float m = a[n];
    if (a.size() % 2 == 0) {
        auto it = std::max_element(a.begin(), a.begin() + n);
        m = 0.5f * (m + *it);
    }
    return m;
}

static Eigen::Vector3f compensateSurfacePointToBallCenter(const Eigen::Vector3f& surface_point,
                                                          float ball_radius) {
    const float norm = surface_point.norm();
    if (!std::isfinite(norm) || norm < 1e-4f) return surface_point;
    return surface_point + surface_point / norm * ball_radius;
}

static LandingPoint lockLanding(const LandingPoint& chosen,
                                bool has_last_landing,
                                const LandingPoint& last_landing,
                                const Eigen::Vector3f& filt_pos,
                                float ground_z) {
    LandingPoint out;
    if (chosen.valid) {
        out = chosen;
    } else if (has_last_landing && last_landing.valid) {
        out = last_landing;
    } else {
        out.valid = true;
        out.pos = Eigen::Vector3f(filt_pos.x(), filt_pos.y(), ground_z);
    }
    out.valid = true;
    out.t = 0.0f;
    out.pos.z() = ground_z;
    return out;
}

static bool startupObservationPlausible(const Eigen::Vector3f& obs,
                                        const Detection& det) {
    const float box_w = std::max(1.0f, det.bbox[2] - det.bbox[0]);
    const float box_h = std::max(1.0f, det.bbox[3] - det.bbox[1]);
    const float box_px = 0.5f * (box_w + box_h);

    if (det.confidence < 0.72f) return false;
    if (box_px < 18.0f) return false;
    if (obs.y() < 0.25f || obs.y() > 12.0f) return false;
    if (obs.z() > 2.5f || obs.z() < -1.0f) return false;
    return true;
}
}  // namespace

BallTracker::BallTracker() {
    reset();
}

void BallTracker::reset() {
    filter_ = Ball3DFilter();
    phase_ = RallyPhase::IDLE;
    miss_xyz_count_ = 0;
    startup_hits_ = 0;
    ground_contact_hits_ = 0;
    time_since_track_start_ = 0.0f;
    time_since_last_xyz_ = 0.0f;
    phase_hold_sec_ = 0.0f;
    has_last_landing_ = false;
    last_landing_ = LandingPoint();
    first_bounce_confirmed_ = false;
    rebound_seen_ = false;
    pre_bounce_vel_.setZero();
    has_startup_candidate_xyz_ = false;
    startup_candidate_xyz_.setZero();
    has_last_obs_xyz_ = false;
    last_obs_xyz_.setZero();
    clearHistory();
}

void BallTracker::clearHistory() {
    history_.clear();
}

void BallTracker::pushHistory(const Eigen::Vector3f& pos,
                              const Eigen::Vector3f& vel,
                              float dt) {
    TrackSample s;
    s.t = history_.empty() ? 0.0f : history_.back().t + dt;
    s.pos = pos;
    s.vel = vel;
    history_.push_back(s);
    while ((int)history_.size() > kHistoryMax) history_.pop_front();
}

bool BallTracker::selectBallDetection(const std::vector<Detection>& detections,
                                      Detection& out_best) const {
    if (detections.empty()) return false;

    const Detection* best = nullptr;
    float best_score = -1.0f;
    for (const auto& d : detections) {
        if (d.confidence > best_score) {
            best_score = d.confidence;
            best = &d;
        }
    }
    if (!best) return false;
    out_best = *best;
    return true;
}

float BallTracker::computeMeasurementNoise(const Eigen::Vector3f& obs,
                                           const Detection& det) const {
    float noise = 0.05f;

    if (obs.y() > 7.0f) noise += 0.08f;
    else if (obs.y() > 5.0f) noise += 0.04f;
    else if (obs.y() > 3.0f) noise += 0.02f;

    const float box_w = std::max(1.0f, det.bbox[2] - det.bbox[0]);
    const float box_h = std::max(1.0f, det.bbox[3] - det.bbox[1]);
    const float box_px = 0.5f * (box_w + box_h);

    if (box_px < 12.0f) noise += 0.08f;
    else if (box_px < 20.0f) noise += 0.04f;

    if (det.confidence < 0.35f) noise += 0.08f;
    else if (det.confidence < 0.50f) noise += 0.04f;

    noise = std::max(0.03f, std::min(noise, 0.25f));
    return noise;
}

bool BallTracker::sampleMedianXYZ(const Detection& det,
                                  const cv::Mat& xyz,
                                  const Eigen::Vector3f* pred_pos,
                                  Eigen::Vector3f& out_xyz
                                ) const {
    const bool xyz3 = xyz.type() == CV_32FC3;
    const bool xyz4 = xyz.type() == CV_32FC4;
    if (xyz.empty() || (!xyz3 && !xyz4)) return false;

    const int img_w = xyz.cols;
    const int img_h = xyz.rows;
    cv::Rect box((int)std::floor(det.bbox[0]),
                 (int)std::floor(det.bbox[1]),
                 std::max(1, (int)std::round(det.bbox[2] - det.bbox[0])),
                 std::max(1, (int)std::round(det.bbox[3] - det.bbox[1])));
    box &= cv::Rect(0, 0, img_w, img_h);
    if (box.width <= 1 || box.height <= 1) return false;

    const bool near_edge =
        (box.x < 24) || (box.x + box.width > img_w - 24) || (box.y + box.height > img_h - 24);
    const bool tracked_mode = pred_pos != nullptr;
    const float roi_w_scale = tracked_mode ? (near_edge ? 0.12f : 0.14f) : (near_edge ? 0.16f : 0.20f);
    const float roi_h_scale = tracked_mode ? (near_edge ? 0.12f : 0.14f) : (near_edge ? 0.16f : 0.20f);
    int rx = std::min(near_edge ? (tracked_mode ? 16 : 24) : (tracked_mode ? 20 : 28),
                      std::max(near_edge ? 5 : 6, (int)std::round(box.width * roi_w_scale)));
    int ry = std::min(near_edge ? (tracked_mode ? 16 : 24) : (tracked_mode ? 20 : 28),
                      std::max(near_edge ? 5 : 6, (int)std::round(box.height * roi_h_scale)));

    auto clamp_i = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    struct Candidate {
        Eigen::Vector3f xyz = Eigen::Vector3f::Zero();
        float score = 1e9f;
        int inliers = 0;
        bool valid = false;
    };

    auto eval_anchor = [&](int cx, int cy, Candidate& cand) -> bool {
        cx = clamp_i(cx, 0, img_w - 1);
        cy = clamp_i(cy, 0, img_h - 1);
        const int x1 = std::max(0, cx - rx);
        const int y1 = std::max(0, cy - ry);
        const int x2 = std::min(img_w - 1, cx + rx);
        const int y2 = std::min(img_h - 1, cy + ry);

        std::vector<float> xs, ys, zs;
        xs.reserve((x2 - x1 + 1) * (y2 - y1 + 1));
        ys.reserve(xs.capacity());
        zs.reserve(xs.capacity());

        for (int yy = y1; yy <= y2; ++yy) {
            for (int xx = x1; xx <= x2; ++xx) {
                float X = 0.0f, Y = 0.0f, Z = 0.0f;
                if (xyz3) {
                    const cv::Vec3f p = xyz.at<cv::Vec3f>(yy, xx);
                    X = p[0]; Y = p[1]; Z = p[2];
                } else {
                    const cv::Vec4f p = xyz.at<cv::Vec4f>(yy, xx);
                    X = p[0]; Y = p[1]; Z = p[2];
                }
                if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z)) continue;
                if (Y < 0.2f || Y > 60.0f) continue;
                if (Z < -3.0f || Z > 6.0f) continue;
                if (std::fabs(X) > 10.0f) continue;
                xs.push_back(X);
                ys.push_back(Y);
                zs.push_back(Z);
            }
        }

        if ((int)xs.size() < (near_edge ? 6 : 10)) return false;

        // === 双层过滤 ===
        std::vector<float> xs_in, ys_in, zs_in;
        // 第1层: pred-gate
        if (pred_pos) {
            const float pred_y = std::max(0.0f, pred_pos->y());
            float gate_x_pred = 0.16f + 0.010f * pred_y;
            float gate_y_pred = 0.25f + 0.030f * pred_y;
            float gate_z_pred = 0.16f + 0.015f * pred_y;
            if (near_edge) {
                gate_x_pred *= 1.10f;
                gate_y_pred *= 1.10f;
                gate_z_pred *= 1.10f;
            }
            for (size_t i = 0; i < ys.size(); ++i) {
                if (std::fabs(xs[i] - pred_pos->x()) <= gate_x_pred &&
                    std::fabs(ys[i] - pred_pos->y()) <= gate_y_pred &&
                    std::fabs(zs[i] - pred_pos->z()) <= gate_z_pred) {
                    xs_in.push_back(xs[i]);
                    ys_in.push_back(ys[i]);
                    zs_in.push_back(zs[i]);
                }
            }
        }

        // 第2层: 若 pred-gate 不够, 用 median(Y)/median(Z) 做邻域 gate
        if ((int)xs_in.size() < (near_edge ? 3 : 4)) {
            xs_in.clear();
            ys_in.clear();
            zs_in.clear();

            float y_med = median(ys);
            float z_med = median(zs);
            if (!std::isfinite(y_med)) return false;
            if (!std::isfinite(z_med)) return false;

            float gate_y = 0.12f + 0.015f * std::max(0.0f, y_med);
            float gate_z = 0.10f + 0.010f * std::max(0.0f, y_med);
            if (pred_pos) {
                gate_y *= 0.90f;
                gate_z *= 0.90f;
            }
            if (near_edge) {
                gate_y *= 1.10f;
                gate_z *= 1.10f;
            }

            for (size_t i = 0; i < ys.size(); ++i) {
                if (std::fabs(ys[i] - y_med) <= gate_y &&
                    std::fabs(zs[i] - z_med) <= gate_z) {
                    xs_in.push_back(xs[i]);
                    ys_in.push_back(ys[i]);
                    zs_in.push_back(zs[i]);
                }
            }
        }

        if ((int)xs_in.size() < (near_edge ? 4 : 6)) return false;

        // 真中位数
        Eigen::Vector3f xyz_out;
        xyz_out.x() = median(xs_in);
        xyz_out.y() = median(ys_in);
        xyz_out.z() = median(zs_in);

        if (!std::isfinite(xyz_out.x()) || !std::isfinite(xyz_out.y()) || !std::isfinite(xyz_out.z())) {
            return false;
        }

        xyz_out = compensateSurfacePointToBallCenter(xyz_out, kBallRadius);

        float score = -0.0015f * (float)xs_in.size();
        if (pred_pos) {
            score += 2.0f * std::fabs(xyz_out.x() - pred_pos->x());
            score += 4.5f * std::fabs(xyz_out.y() - pred_pos->y());
            score += 0.8f * std::fabs(xyz_out.z() - pred_pos->z());
        } else {
            score += 0.02f * std::fabs(xyz_out.z());
        }
        cand.xyz = xyz_out;
        cand.score = score;
        cand.inliers = (int)xs_in.size();
        cand.valid = true;
        return true;
    };

    const float bx = (float)box.x;
    const float by = (float)box.y;
    const float bw = (float)box.width;
    const float bh = (float)box.height;
    const int cx       = (int)std::round(bx + 0.50f * bw);
    const int cy_upper = (int)std::round(by + 0.25f * bh);
    const int cy_midup = (int)std::round(by + 0.40f * bh);
    const int cy_mid   = (int)std::round(by + 0.50f * bh);

    // 快路径: 框够大 + 不靠边 + 已有预测 -> 单 anchor
    const bool fast_path =
        pred_pos && !near_edge && box.width >= 16 && box.height >= 16;
    if (fast_path) {
        Candidate c_fast;
        if (eval_anchor(cx, cy_mid, c_fast)) {
            out_xyz = c_fast.xyz;
            return true;
        }
    }

    // 多-anchor 路径
    Candidate c_center, c_upper, c_midup;
    eval_anchor(cx, cy_mid,   c_center);
    eval_anchor(cx, cy_upper, c_upper);
    eval_anchor(cx, cy_midup, c_midup);

    const Candidate* best = nullptr;
    for (const Candidate* cand : {&c_center, &c_upper, &c_midup}) {
        if (!cand->valid) continue;
        if (!best) { best = cand; continue; }
        if (std::fabs(cand->score - best->score) > 1e-6f) {
            if (cand->score < best->score) best = cand;
        } else if (cand->inliers > best->inliers) {
            best = cand;
        }
    }
    if (!best) return false;
    out_xyz = best->xyz;
    return true;
}

LandingPoint BallTracker::predictLandingBallistic(const Eigen::Vector3f& pos,
                                                  const Eigen::Vector3f& vel,
                                                  float ground_z) const {
    LandingPoint out;
    constexpr float g = 9.81f;
    const float a = -0.5f * g;
    const float b = vel.z();
    const float c = (pos.z() - kBallRadius) - ground_z;
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return out;

    const float sqrt_disc = std::sqrt(disc);
    const float t1 = (-b + sqrt_disc) / (2.0f * a);
    const float t2 = (-b - sqrt_disc) / (2.0f * a);
    float t = 1e9f;
    if (t1 > 0.0f) t = std::min(t, t1);
    if (t2 > 0.0f) t = std::min(t, t2);
    if (!std::isfinite(t) || t <= 0.0f || t > kMaxLandingTime) return out;

    out.valid = true;
    out.t = t;
    out.pos.x() = pos.x() + vel.x() * t;
    out.pos.y() = pos.y() + vel.y() * t;
    out.pos.z() = ground_z;
    return out;
}

LandingPoint BallTracker::predictLandingFromHistory(float ground_z) const {
    LandingPoint out;
    if (history_.size() < 3) return out;

    const TrackSample& s2 = history_[history_.size() - 1];
    const TrackSample& s1 = history_[history_.size() - 2];
    const TrackSample& s0 = history_[history_.size() - 3];

    const float dt1 = std::max(1e-3f, s1.t - s0.t);
    const float dt2 = std::max(1e-3f, s2.t - s1.t);

    const float vz1 = (s1.pos.z() - s0.pos.z()) / dt1;
    const float vz2 = (s2.pos.z() - s1.pos.z()) / dt2;
    const float az = (vz2 - vz1) / std::max(1e-3f, 0.5f * (dt1 + dt2));

    const float g_like = (az < -1.0f) ? (-az) : 9.81f;
    const float a = -0.5f * g_like;
    const float b = s2.vel.z();
    const float c = (s2.pos.z() - kBallRadius) - ground_z;

    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return out;

    const float sqrt_disc = std::sqrt(disc);
    const float t1 = (-b + sqrt_disc) / (2.0f * a);
    const float t2 = (-b - sqrt_disc) / (2.0f * a);

    float t = 1e9f;
    if (t1 > 0.0f) t = std::min(t, t1);
    if (t2 > 0.0f) t = std::min(t, t2);
    if (!std::isfinite(t) || t <= 0.0f || t > kMaxLandingTime) return out;

    out.valid = true;
    out.t = t;
    out.pos.x() = s2.pos.x() + s2.vel.x() * t;
    out.pos.y() = s2.pos.y() + s2.vel.y() * t;
    out.pos.z() = ground_z;
    return out;
}

BallEventType BallTracker::updateStateMachine(
    const Eigen::Vector3f& filt_pos,
    const Eigen::Vector3f& filt_vel,
    bool has_xyz,
    const Eigen::Vector3f* obs_pos,
    float dt,
    float ground_z,
    LandingPoint& out_landing,
    bool& out_hold) {

    out_landing = LandingPoint();
    out_hold = false;

    auto keep_last_landing = [&]() {
        if (has_last_landing_ && last_landing_.valid) {
            out_landing = last_landing_;
            out_landing.t = 0.0f;
            out_landing.pos.z() = ground_z;
            out_hold = true;
        }
    };

    if (!filter_.initialized()) {
        phase_ = RallyPhase::IDLE;
        return BallEventType::NONE;
    }

    if (miss_xyz_count_ >= kMaxMissFrames) {
        phase_ = RallyPhase::DEAD_BALL;
        keep_last_landing();
        return BallEventType::DEAD_BALL;
    }

    const float z_contact = filt_pos.z() - kBallRadiusForContact;
    const float rel_z_contact = z_contact - ground_z;

    LandingPoint hist_landing = predictLandingFromHistory(ground_z);
    LandingPoint ball_landing = predictLandingBallistic(filt_pos, filt_vel, ground_z);

    LandingPoint chosen;
    if (hist_landing.valid) chosen = hist_landing;
    else if (ball_landing.valid) chosen = ball_landing;

    const bool descending = filt_vel.z() < -0.20f;
    const bool strong_descend = filt_vel.z() < kMinBounceVz;
    const bool near_ground = rel_z_contact <= kGroundEps;
    const bool imminent_by_height = rel_z_contact <= kImminentRelZ;
    const bool imminent_by_time = chosen.valid && chosen.t <= kImminentT;

    switch (phase_) {
        case RallyPhase::IDLE: {
            if (has_xyz) {
                phase_ = RallyPhase::TRACKING_AIRBORNE;
                ground_contact_hits_ = 0;
                first_bounce_confirmed_ = false;
                rebound_seen_ = false;
                phase_hold_sec_ = 0.0f;
                if (chosen.valid) {
                    out_landing = chosen;
                    last_landing_ = chosen;
                    has_last_landing_ = true;
                }
                return BallEventType::TRACK_STABLE;
            }
            return BallEventType::NONE;
        }

        case RallyPhase::TRACKING_AIRBORNE: {
            if (chosen.valid) {
                out_landing = chosen;
                last_landing_ = chosen;
                has_last_landing_ = true;
            }

            if (has_xyz && descending && (imminent_by_height || imminent_by_time)) {
                phase_ = RallyPhase::LANDING_IMMINENT;
                phase_hold_sec_ = 0.0f;

                if (!has_last_landing_ && chosen.valid) {
                    last_landing_ = chosen;
                    has_last_landing_ = true;
                }

                out_hold = false;
                if (has_last_landing_ && last_landing_.valid) {
                    out_landing = last_landing_;
                } else if (chosen.valid) {
                    out_landing = chosen;
                }

                return BallEventType::LANDING_IMMINENT;
            }

            return BallEventType::NONE;
        }

        case RallyPhase::LANDING_IMMINENT: {
            phase_hold_sec_ += dt;

            if (has_last_landing_ && last_landing_.valid) {
                out_landing = last_landing_;
            } else if (chosen.valid) {
                out_landing = chosen;
                last_landing_ = chosen;
                has_last_landing_ = true;
            }

            if (has_xyz && near_ground && strong_descend) {
                ground_contact_hits_++;
            } else if (has_xyz && rel_z_contact > (kGroundEps + 0.05f)) {
                ground_contact_hits_ = 0;
            }

            if (ground_contact_hits_ >= kGroundContactNeedHits) {
                phase_ = RallyPhase::FIRST_BOUNCE_CONFIRMED;
                first_bounce_confirmed_ = true;
                pre_bounce_vel_ = filt_vel;
                phase_hold_sec_ = 0.0f;

                out_landing = lockLanding(chosen, has_last_landing_, last_landing_, filt_pos, ground_z);
                last_landing_ = out_landing;
                has_last_landing_ = true;
                out_hold = false;

                return BallEventType::FIRST_BOUNCE;
            }

            if (!has_xyz && time_since_last_xyz_ > 0.20f) {
                phase_ = RallyPhase::DEAD_BALL;
                keep_last_landing();
                return BallEventType::DEAD_BALL;
            }

            // 超时保护：landing_imminent 持续过久仍未触地，强制进入 DEAD_BALL
            if (phase_hold_sec_ > 0.50f) {
                phase_ = RallyPhase::DEAD_BALL;
                keep_last_landing();
                return BallEventType::DEAD_BALL;
            }

            return BallEventType::NONE;
        }

        case RallyPhase::FIRST_BOUNCE_CONFIRMED: {
            keep_last_landing();

            if (has_xyz) {
                const float obs_contact =
                    (obs_pos ? (obs_pos->z() - kBallRadiusForContact) : z_contact) - ground_z;

                if (!rebound_seen_ &&
                    obs_contact > 0.10f &&
                    (obs_pos && obs_pos->z() > filt_pos.z() + 0.06f)) {
                    rebound_seen_ = true;
                }
            }

            if (!has_xyz && time_since_last_xyz_ > 0.20f) {
                phase_ = RallyPhase::DEAD_BALL;
                return BallEventType::DEAD_BALL;
            }

            return BallEventType::NONE;
        }

        case RallyPhase::REBOUND_TRACKING: {
            keep_last_landing();
            if (!has_xyz && time_since_last_xyz_ > 0.20f) {
                phase_ = RallyPhase::DEAD_BALL;
                return BallEventType::DEAD_BALL;
            }
            return BallEventType::NONE;
        }

        case RallyPhase::DEAD_BALL: {
            keep_last_landing();
            return BallEventType::NONE;
        }
    }

    return BallEventType::NONE;
}

BallTrackState BallTracker::update(const std::vector<Detection>& detections,
                                   const cv::Mat& xyz_cpu,
                                   float dt,
                                   float ground_z,
                                   const HybridDepthResult* hybrid) {
    BallTrackState st;
    st.reject_reason = "none";
    if (dt <= 0.0f || dt > 0.2f) dt = 1.0f / 30.0f;

    Detection best_det;
    st.has_det = selectBallDetection(detections, best_det);
    if (st.has_det) st.det = best_det;

    Eigen::Vector3f obs = Eigen::Vector3f::Zero();
    if (st.has_det) {
        // Hybrid 路径: 直接使用 HybridDepth 融合结果
        if (hybrid && hybrid->valid) {
            obs = hybrid->pos;
            st.raw_xyz_valid = true;
            st.raw_xyz = obs;
            st.has_xyz = true;
            st.xyz = obs;
        } else {
            // 经典 SDK XYZ 采样路径
            Eigen::Vector3f pred = Eigen::Vector3f::Zero();
            const Eigen::Vector3f* pp = nullptr;
            if (filter_.initialized()) {
                pred = filter_.position();
                const float speed = filter_.velocity().norm();
                if (speed >= 0.8f || !has_last_obs_xyz_ || time_since_last_xyz_ > 0.25f) {
                    pp = &pred;
                } else {
                    pp = &last_obs_xyz_;
                }
            } else if (has_last_obs_xyz_ && time_since_last_xyz_ <= 0.25f) {
                pp = &last_obs_xyz_;
            }
            bool sampled_xyz = sampleMedianXYZ(best_det, xyz_cpu, pp, obs);
            st.raw_xyz_valid = sampled_xyz;
            if (sampled_xyz) {
                st.raw_xyz = obs;
            } else {
                st.reject_reason = "sample_fail";
            }
            st.has_xyz = sampled_xyz;
            if (sampled_xyz) {
                st.xyz = obs;
            }
        }
    } else {
        st.reject_reason = "no_det";
    }

    const bool frozen_phase =
        (phase_ == RallyPhase::FIRST_BOUNCE_CONFIRMED || phase_ == RallyPhase::DEAD_BALL);

    if (st.has_xyz && frozen_phase) {
        st.has_xyz = false;
        st.reject_reason = "frozen_phase";
    }

    if (phase_ == RallyPhase::DEAD_BALL) {
        if (!st.has_det && !st.raw_xyz_valid) {
            phase_hold_sec_ += dt;
        } else {
            phase_hold_sec_ = 0.0f;
        }
        if (phase_hold_sec_ >= kDeadBallQuietResetSec) {
            reset();
            st.valid = false;
            st.phase = phase_;
            st.reject_reason = "dead_ball_reset";
            return st;
        }
    }

    if (!filter_.initialized()) {
        if (st.has_xyz) {
            if (!startupObservationPlausible(st.xyz, best_det)) {
                st.has_xyz = false;
                st.reject_reason = "startup_guard";
            }
        }

        if (st.has_xyz) {
            if (!has_startup_candidate_xyz_) {
                startup_candidate_xyz_ = st.xyz;
                has_startup_candidate_xyz_ = true;
                startup_hits_ = 1;
            } else {
                const float dx = std::fabs(st.xyz.x() - startup_candidate_xyz_.x());
                const float dy = std::fabs(st.xyz.y() - startup_candidate_xyz_.y());
                const float dz = std::fabs(st.xyz.z() - startup_candidate_xyz_.z());
                if (dx <= kStartupGateX && dy <= kStartupGateY && dz <= kStartupGateZ) {
                    startup_candidate_xyz_ = 0.6f * startup_candidate_xyz_ + 0.4f * st.xyz;
                    startup_hits_++;
                } else {
                    startup_candidate_xyz_ = st.xyz;
                    startup_hits_ = 1;
                    st.reject_reason = "startup_jump";
                }
            }
            if (startup_hits_ >= kStartupNeedHits) {
                filter_.reset();
                filter_.update(obs, computeMeasurementNoise(obs, best_det));
                miss_xyz_count_ = 0;
                time_since_track_start_ = 0.0f;
                time_since_last_xyz_ = 0.0f;
                ground_contact_hits_ = 0;
                phase_hold_sec_ = 0.0f;
                has_startup_candidate_xyz_ = false;
                clearHistory();
                pushHistory(filter_.position(), filter_.velocity(), 0.0f);
                phase_ = RallyPhase::TRACKING_AIRBORNE;
                st.event = BallEventType::THROW_START;
            }
        } else {
            startup_hits_ = 0;
            has_startup_candidate_xyz_ = false;
            if (!st.has_xyz && st.reject_reason == "none") {
                st.reject_reason = "startup_wait";
            }
        }
    } else {
        const bool frozen = frozen_phase;

        if (!frozen && st.has_xyz) {
            filter_.predict(dt, ground_z, kBallRadiusForContact);
        }

        if (!frozen && st.has_xyz) {
            const Eigen::Vector3f pred_pos = filter_.position();
            const float pred_speed = filter_.velocity().norm();

            const float dx = std::fabs(obs.x() - pred_pos.x());
            const float dy = std::fabs(obs.y() - pred_pos.y());
            const float dz = std::fabs(obs.z() - pred_pos.z());

            float dx_gate = (pred_speed > 1.5f) ? 0.60f : 0.90f;
            float dy_gate = (pred_pos.y() < 2.0f) ? 1.00f : 0.75f;
            float dz_gate = (pred_pos.y() < 2.0f) ? 0.80f : 0.60f;
            if (pred_speed < 0.8f) {
                dx_gate *= 1.20f;
                dy_gate *= 1.20f;
                dz_gate *= 1.20f;
            }

            if (phase_ == RallyPhase::LANDING_IMMINENT) {
                const float rel_pred_z_contact = (pred_pos.z() - kBallRadiusForContact) - ground_z;
                dx_gate *= 1.20f;
                dy_gate *= 1.30f;
                dz_gate *= 2.10f;

                if (rel_pred_z_contact < 0.08f) {
                    if (obs.z() > pred_pos.z() + 0.45f) {
                        st.has_xyz = false;
                        st.reject_reason = "landing_z_guard";
                    }
                }
            }

            if (st.has_xyz && (dx > dx_gate || dy > dy_gate || dz > dz_gate)) {
                st.has_xyz = false;
                if (dy > dy_gate) st.reject_reason = "gate_dy";
                else if (dx > dx_gate) st.reject_reason = "gate_dx";
                else st.reject_reason = "gate_dz";
            }
        }

        if (frozen) {
            if (st.has_xyz) {
                miss_xyz_count_ = 0;
                time_since_last_xyz_ = 0.0f;
            } else {
                miss_xyz_count_++;
                time_since_last_xyz_ += dt;
            }
            time_since_track_start_ += dt;
        } else {
            if (st.has_xyz) {
                filter_.update(obs, computeMeasurementNoise(obs, best_det));
                miss_xyz_count_ = 0;
                time_since_last_xyz_ = 0.0f;
                time_since_track_start_ += dt;
                pushHistory(filter_.position(), filter_.velocity(), dt);
            } else {
                miss_xyz_count_++;
                time_since_last_xyz_ += dt;
                time_since_track_start_ += dt;
            }
        }
    }

    if (filter_.initialized()) {
        st.valid = true;
        st.pos = filter_.position();
        st.vel = filter_.velocity();
        st.phase = phase_;

        bool landing_hold = false;
        LandingPoint lp;
        const Eigen::Vector3f* obs_ptr = st.has_xyz ? &st.xyz : nullptr;
        BallEventType ev = updateStateMachine(st.pos, st.vel, st.has_xyz, obs_ptr, dt, ground_z, lp, landing_hold);
        if (st.event == BallEventType::NONE) st.event = ev;
        st.phase = phase_;

        st.landing = lp;
        st.landing_is_hold = landing_hold;
        st.bounce_detected = (st.event == BallEventType::FIRST_BOUNCE);
        st.rebound_detected = false;

        if (phase_ == RallyPhase::FIRST_BOUNCE_CONFIRMED) {
            st.vel.setZero();
        }

        if (phase_ == RallyPhase::DEAD_BALL && miss_xyz_count_ >= kMaxMissFrames) {
            reset();
            st.valid = false;
            return st;
        }
    } else {
        st.phase = phase_;
    }

    if (st.has_xyz) {
        last_obs_xyz_ = st.xyz;
        has_last_obs_xyz_ = true;
        st.reject_reason = "accepted";
    }

    return st;
}

}  // namespace stereo3d
