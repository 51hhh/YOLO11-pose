/**
 * @file trajectory_predictor.cpp
 * @brief Volleyball landing point predictor implementation
 *
 * Camera frame convention: X=right, Y=down, Z=depth (forward)
 * Gravity acts in +Y direction (downward in camera frame)
 * Ballistic path: RK4 integration with gravity + quadratic air drag
 * Polynomial path: Least-squares quadratic fit on Y(t), linear on X(t)/Z(t)
 */

#include "trajectory_predictor.h"
#include "../utils/logger.h"
#include <cmath>
#include <algorithm>

namespace stereo3d {

// ─────────────────── Init ───────────────────

void TrajectoryPredictor::init(const TrajectoryPredictorConfig& config) {
    cfg_ = config;
    // Precompute drag factor: k = 0.5 * Cd * rho * A / m
    float A = 3.14159265f * cfg_.ball_radius * cfg_.ball_radius;
    drag_k_ = 0.5f * cfg_.drag_coeff * cfg_.air_density * A / cfg_.ball_mass;
    elapsed_time_ = 0.0;
    histories_.clear();
    LOG_INFO("TrajectoryPredictor: g=%.2f, drag_k=%.4f, ground_y=%.2f",
             cfg_.gravity, drag_k_, cfg_.ground_y);
}

// ─────────────────── Update ───────────────────

std::vector<LandingPrediction> TrajectoryPredictor::update(
    const std::vector<Object3D>& results, double dt) {

    elapsed_time_ += dt;
    pruneHistories(results);

    std::vector<LandingPrediction> preds(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& obj = results[i];
        if (obj.track_id < 0 || obj.z <= 0) continue;

        // Record history
        auto& hist = getOrCreateHistory(obj.track_id);
        double t_rel = elapsed_time_ - hist.t_origin;
        hist.samples.push_back({t_rel, obj.x, obj.y, obj.z});
        while ((int)hist.samples.size() > cfg_.history_max) {
            hist.samples.pop_front();
        }

        // Speed check
        float speed = std::sqrt(obj.vx*obj.vx + obj.vy*obj.vy + obj.vz*obj.vz);
        if (speed < cfg_.min_speed_for_predict) continue;

        // Primary: ballistic prediction
        LandingPrediction ballistic = predictBallistic(
            obj.x, obj.y, obj.z, obj.vx, obj.vy, obj.vz);

        // Backup: polynomial prediction
        LandingPrediction poly;
        if ((int)hist.samples.size() >= cfg_.poly_min_frames) {
            poly = predictPolynomial(hist);
        }

        // Choose best: prefer ballistic if valid, polynomial as backup
        if (ballistic.valid && poly.valid) {
            // Both valid: use ballistic but boost confidence if they agree
            float dx = ballistic.x - poly.x;
            float dy = ballistic.y - poly.y;
            float dist = std::sqrt(dx*dx + dy*dy);
            ballistic.confidence = (dist < 1.0f) ? 0.9f : 0.6f;
            preds[i] = ballistic;
        } else if (ballistic.valid) {
            ballistic.confidence = 0.7f;
            preds[i] = ballistic;
        } else if (poly.valid) {
            poly.confidence = 0.5f;
            preds[i] = poly;
        }

        hist.last_pred = preds[i];
    }

    return preds;
}

// ─────────────────── Ballistic (RK4 + Drag) ───────────────────
// Camera frame: X=right, Y=down, Z=depth
// Gravity: +Y direction (g positive, Y axis points down)

LandingPrediction TrajectoryPredictor::predictBallistic(
    float x, float y, float z,
    float vx, float vy, float vz) const {

    LandingPrediction pred;

    // Only predict if ball is above ground (Y < ground_y, since Y points down)
    if (y >= cfg_.ground_y) return pred;

    // RK4 state: [x, y, z, vx, vy, vz]
    double s[6] = {x, y, z, vx, vy, vz};
    const double dt = cfg_.rk4_dt;
    const double g = cfg_.gravity;
    const double k = drag_k_;

    // Derivative function: ds/dt
    // Gravity acts in +Y direction (camera Y = down)
    auto deriv = [g, k](const double s[6], double ds[6]) {
        ds[0] = s[3]; // dx/dt = vx
        ds[1] = s[4]; // dy/dt = vy
        ds[2] = s[5]; // dz/dt = vz
        double speed = std::sqrt(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]);
        double drag = k * speed; // |F_drag/m| = k * |v|
        ds[3] = -drag * s[3];           // ax = -drag*vx
        ds[4] = +g - drag * s[4];       // ay = +g - drag*vy (gravity in +Y)
        ds[5] = -drag * s[5];           // az = -drag*vz
    };

    double t = 0.0;
    int max_steps = (int)(cfg_.max_predict_time / dt);

    for (int step = 0; step < max_steps; ++step) {
        // Save pre-step state for landing interpolation
        double s_prev[6];
        for (int i = 0; i < 6; i++) s_prev[i] = s[i];

        // RK4
        double k1[6], k2[6], k3[6], k4[6], tmp[6];

        deriv(s, k1);
        for (int i = 0; i < 6; i++) tmp[i] = s[i] + 0.5*dt*k1[i];
        deriv(tmp, k2);
        for (int i = 0; i < 6; i++) tmp[i] = s[i] + 0.5*dt*k2[i];
        deriv(tmp, k3);
        for (int i = 0; i < 6; i++) tmp[i] = s[i] + dt*k3[i];
        deriv(tmp, k4);

        for (int i = 0; i < 6; i++)
            s[i] += dt/6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        t += dt;

        // Check landing: Y crossed ground plane (Y increasing = downward)
        if (s[1] >= cfg_.ground_y) {
            // Linear interpolation using pre-step state
            double frac = (cfg_.ground_y - s_prev[1]) / (s[1] - s_prev[1]);
            frac = std::clamp(frac, 0.0, 1.0);

            pred.x = (float)(s_prev[0] + (s[0] - s_prev[0]) * frac);
            pred.y = (float)(s_prev[2] + (s[2] - s_prev[2]) * frac); // report Z as "y" for landing (X,Z = horizontal)
            pred.time_to_land = (float)(t - dt + dt * frac);
            pred.method = 0;
            pred.valid = true;
            pred.confidence = 0.7f;
            return pred;
        }
    }

    return pred; // Didn't land within max_predict_time
}

// ─────────────────── Polynomial Fit ───────────────────
// Fit Y(t) = quadratic (gravity makes Y parabolic)
// Fit X(t), Z(t) = linear (horizontal)

LandingPrediction TrajectoryPredictor::predictPolynomial(
    const TrackHistory& hist) const {

    LandingPrediction pred;
    int n = (int)hist.samples.size();
    if (n < cfg_.poly_min_frames) return pred;

    std::vector<double> t(n), xs(n), ys(n), zs(n);
    for (int i = 0; i < n; i++) {
        t[i]  = hist.samples[i].t;
        xs[i] = hist.samples[i].x;
        ys[i] = hist.samples[i].y;  // Vertical axis (camera Y = down)
        zs[i] = hist.samples[i].z;  // Depth axis
    }

    // Fit y(t) = a*t^2 + b*t + c  (quadratic for gravity on Y axis)
    double ya, yb, yc;
    if (!fitQuadratic(t, ys, ya, yb, yc)) return pred;

    // Y must be convex (a > 0 for downward acceleration = gravity)
    if (ya <= 0) return pred;

    // Solve a*t^2 + b*t + (c - ground_y) = 0
    double disc = yb*yb - 4.0*ya*(yc - cfg_.ground_y);
    if (disc < 0) return pred;

    double sqrt_disc = std::sqrt(disc);
    double t1 = (-yb + sqrt_disc) / (2.0 * ya);
    double t2 = (-yb - sqrt_disc) / (2.0 * ya);

    // Take the future root that's after the latest sample
    double t_last = t.back();
    double t_land = -1;
    if (t1 > t_last && t2 > t_last) t_land = std::min(t1, t2);
    else if (t1 > t_last) t_land = t1;
    else if (t2 > t_last) t_land = t2;
    else return pred;

    // Fit x(t) and z(t) as linear
    double xa, xb, za, zb;
    if (!fitLinear(t, xs, xa, xb)) return pred;
    if (!fitLinear(t, zs, za, zb)) return pred;

    pred.x = (float)(xa * t_land + xb);       // Landing X (horizontal)
    pred.y = (float)(za * t_land + zb);        // Landing Z (depth) → reported as "y"
    pred.time_to_land = (float)(t_land - t_last);
    pred.method = 1;
    pred.valid = true;
    pred.confidence = 0.5f;
    return pred;
}

}  // namespace stereo3d
