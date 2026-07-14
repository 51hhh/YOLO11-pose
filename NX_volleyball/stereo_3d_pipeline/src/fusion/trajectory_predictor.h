/**
 * @file trajectory_predictor.h
 * @brief Volleyball landing predictor (realtime)
 *
 * Primary path (matches trajectory_fusion/landing_pipeline):
 *   bbox_center disparity (+ circle fallback/consistency)
 *     -> d0 reproject to 3D observation
 *     -> Student-t robust EKF (gravity + quadratic drag)
 *     -> RK4 rollout to ground plane
 *   Polynomial history fit remains a backup when EKF is not ready.
 *
 * LandingPrediction reports ground-plane horizontal coordinates as:
 *   x = camera X, y = camera Z (depth), matching the existing ROS/display contract.
 */

#ifndef STEREO_3D_PIPELINE_TRAJECTORY_PREDICTOR_H_
#define STEREO_3D_PIPELINE_TRAJECTORY_PREDICTOR_H_

#include "../pipeline/object3d_types.h"
#include <vector>
#include <deque>
#include <array>
#include <string>

namespace stereo3d {

struct LandingPrediction {
    float x = 0.0f;            ///< Landing X in camera frame (m)
    float y = 0.0f;            ///< Landing Z (depth) reported as y for compatibility (m)
    float z = 0.0f;            ///< Landing camera-Y (optional/debug)
    float time_to_land = 0.0f; ///< Time until landing (s)
    float confidence = 0.0f;   ///< Prediction confidence [0,1]
    float speed_mps = 0.0f;    ///< Speed of the state used for rollout
    int   method = -1;         ///< 0=student-t EKF ballistic, 1=polynomial, -1=invalid
    bool  valid = false;
    float student_w = 1.0f;    ///< Last Student-t innovation weight
    int   obs_source = -1;     ///< 0=bbox_disp,1=bbox_z,2=circle,3=raw,4=filtered

    // Realtime control-gate audit fields. The ungated prediction above is
    // always preserved; these fields describe the selected control candidate.
    int   control_gate_selected = 0;
    int   control_gate_passed = 0;
    int   control_gate_reason = 0;
    int   control_gate_stable_frames = 0;
    float control_base_x = 0.0f;
    float control_base_y = 0.0f;
};

struct TrajectoryPredictorConfig {
    bool enabled = true;          ///< false 时仅发布 NX 原始观测，由 RDK 负责弹道预测

    // Physics
    float gravity       = 9.81f;
    float air_density   = 1.225f;
    float ball_mass     = 0.270f;
    float ball_radius   = 0.105f;
    float drag_coeff    = 0.10f;   ///< Tuned volleyball Cd (was 0.47)

    // Ground plane. Default assumes camera Y-down: height = ground_y - y.
    // Equivalent general form: height = -dot(g_hat, p) - ground_h, with
    // g_hat=(0,1,0) and ground_h = -ground_y.
    float ground_y      = 2.5f;
    float ground_h      = -2.5f;
    float g_hat_x       = 0.0f;
    float g_hat_y       = 1.0f;
    float g_hat_z       = 0.0f;
    bool  use_g_hat     = false;   ///< false: legacy ground_y on camera Y

    // Camera geometry for bbox/circle reproject (from stereo calib + d0 file)
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float fB = 0.0f;           ///< fx * baseline_m
    float d0 = 0.0f;           ///< disparity zero-point offset (px)
    bool  have_geometry = false;

    // Observation policy
    bool  prefer_bbox = true;
    bool  enable_circle_fallback = true;
    float circle_consistency_m = 0.35f;
    bool  allow_raw_fallback = true;
    bool  allow_filtered_fallback = true;

    // Student-t EKF
    bool  use_student_t_ekf = true;
    float student_t_nu = 12.0f;
    float sigma_d_px = 0.4f;
    float q_pos = 1e-4f;
    float q_vel = 1.5f;
    float max_dt = 0.5f;
    float xy_sigma_scale = 0.0012f;
    float xy_sigma_floor = 0.004f;
    float consistency_inflate = 3.0f;
    float min_height_for_predict = 0.05f;

    // Prediction limits
    float max_predict_time = 3.0f;
    float rk4_dt          = 0.008f;

    // Polynomial fallback
    int   poly_min_frames = 6;
    int   history_max     = 20;

    // Confidence / gating
    float min_speed_for_predict = 0.5f; ///< slightly lower than old 1.0; EKF needs warm-up
};

/**
 * @brief Realtime landing predictor with optional Student-t EKF backbone.
 */
class TrajectoryPredictor {
public:
    TrajectoryPredictor() = default;
    ~TrajectoryPredictor() = default;

    void init(const TrajectoryPredictorConfig& config = TrajectoryPredictorConfig());

    /**
     * @brief Update with current frame Object3D list and predict landing.
     * @param results  Current fused/tracked objects
     * @param dt       Frame interval (s)
     */
    std::vector<LandingPrediction> update(
        const std::vector<Object3D>& results, double dt);

    LandingPrediction getPrediction(int track_id) const;

    void reset();

    const TrajectoryPredictorConfig& config() const { return cfg_; }

private:
    TrajectoryPredictorConfig cfg_;
    float drag_k_ = 0.0f;  ///< 0.5 * Cd * rho * A / m

    struct TrackState {
        int track_id = -1;
        bool ekf_ready = false;
        double t = 0.0;                 ///< absolute filter time (elapsed_time_)
        std::array<double, 6> x{};      ///< [p(3), v(3)]
        std::array<double, 36> P{};     ///< row-major 6x6
        float last_student_w = 1.0f;
        int last_obs_source = -1;
        struct Sample { double t; float x, y, z; };
        std::deque<Sample> samples;     ///< for polynomial backup
        double t_origin = 0.0;
        LandingPrediction last_pred;
    };

    std::vector<TrackState> tracks_;
    double elapsed_time_ = 0.0;

    TrackState* findTrack(int track_id);
    TrackState& getOrCreateTrack(int track_id);
    void pruneTracks(const std::vector<Object3D>& results);

    struct Observation {
        float x = 0, y = 0, z = 0;
        float consistency = 1.0f;
        float trust = 1.0f;
        int source = -1;
        bool valid = false;
    };

    Observation formObservation(const Object3D& obj) const;
    bool reproject(float u, float v, float disparity, bool apply_d0,
                   float& x, float& y, float& z) const;
    bool reprojectDepth(float u, float v, float depth,
                        float& x, float& y, float& z) const;

    void ekfInit(TrackState& tr, double t, const Observation& obs) const;
    bool ekfUpdate(TrackState& tr, double t, const Observation& obs);
    LandingPrediction predictFromState(const TrackState& tr) const;
    LandingPrediction predictBallisticState(
        double px, double py, double pz,
        double vx, double vy, double vz) const;
    LandingPrediction predictPolynomial(const TrackState& tr) const;

    float heightOf(double px, double py, double pz) const;
    void gravityAccel(double& ax, double& ay, double& az) const;

    // legacy helpers retained for polynomial path
    static bool fitQuadratic(const std::vector<double>& t,
                             const std::vector<double>& y,
                             double& a, double& b, double& c);
    static bool fitLinear(const std::vector<double>& t,
                          const std::vector<double>& y,
                          double& a, double& b);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRAJECTORY_PREDICTOR_H_
