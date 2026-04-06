/**
 * @file trajectory_predictor.h
 * @brief Volleyball landing point predictor
 *
 * Two prediction paths (择优):
 *   1. Ballistic: gravity + air drag, RK4 forward integration
 *   2. Polynomial: quadratic fit on recent history (backup)
 *
 * Input:  9D Kalman state [x,y,z,vx,vy,vz,ax,ay,az]
 * Output: LandingPrediction {landing_x, landing_y, time_to_land, confidence, method}
 */

#ifndef STEREO_3D_PIPELINE_TRAJECTORY_PREDICTOR_H_
#define STEREO_3D_PIPELINE_TRAJECTORY_PREDICTOR_H_

#include "../pipeline/frame_slot.h"
#include <vector>
#include <deque>

namespace stereo3d {

struct LandingPrediction {
    float x = 0.0f;           ///< Predicted landing X (m)
    float y = 0.0f;           ///< Predicted landing Y (m)
    float time_to_land = 0.0f; ///< Time until landing (s)
    float confidence = 0.0f;   ///< Prediction confidence [0,1]
    int   method = -1;         ///< 0=ballistic, 1=polynomial, -1=invalid
    bool  valid = false;
};

struct TrajectoryPredictorConfig {
    // Physics
    float gravity       = 9.81f;   ///< Gravitational acceleration (m/s^2)
    float air_density   = 1.225f;  ///< Air density (kg/m^3)
    float ball_mass     = 0.270f;  ///< Volleyball mass (kg)
    float ball_radius   = 0.110f;  ///< Volleyball radius (m)
    float drag_coeff    = 0.47f;   ///< Drag coefficient (sphere)
    float ground_y      = 2.5f;    ///< Ground Y in camera frame (m, cam Y=down, ~camera height)

    // Prediction limits
    float max_predict_time = 3.0f; ///< Max forward integration time (s)
    float rk4_dt          = 0.002f;///< RK4 step size (s)

    // Polynomial fallback
    int   poly_min_frames = 6;     ///< Minimum frames for polynomial fit
    int   history_max     = 20;    ///< Maximum history buffer size

    // Confidence
    float min_speed_for_predict = 1.0f; ///< Min speed to start prediction (m/s)
};

/**
 * @brief TrajectoryPredictor - physics-based landing prediction
 */
class TrajectoryPredictor {
public:
    TrajectoryPredictor() = default;
    ~TrajectoryPredictor() = default;

    void init(const TrajectoryPredictorConfig& config = TrajectoryPredictorConfig());

    /**
     * @brief Update with current frame's 3D results and predict landing
     * @param results  Current frame's Object3D (from HybridDepthEstimator)
     * @param dt       Frame interval (s)
     * @return Landing predictions per tracked object
     */
    std::vector<LandingPrediction> update(
        const std::vector<Object3D>& results, double dt);

    /**
     * @brief Get last prediction for a given track_id
     */
    LandingPrediction getPrediction(int track_id) const;

    void reset();

private:
    TrajectoryPredictorConfig cfg_;

    // k = 0.5 * Cd * rho * A / m  (precomputed drag factor)
    float drag_k_ = 0.0f;

    // Per-track history for polynomial fit
    struct TrackHistory {
        int track_id = -1;
        struct Sample { double t; float x, y, z; };
        std::deque<Sample> samples;
        double t_origin = 0.0;     ///< First sample timestamp
        LandingPrediction last_pred;
    };
    std::vector<TrackHistory> histories_;
    double elapsed_time_ = 0.0;

    // Core algorithms
    LandingPrediction predictBallistic(
        float x, float y, float z,
        float vx, float vy, float vz) const;

    LandingPrediction predictPolynomial(const TrackHistory& hist) const;

    TrackHistory* findHistory(int track_id);
    TrackHistory& getOrCreateHistory(int track_id);
    void pruneHistories(const std::vector<Object3D>& results);

    // Polynomial least-squares fit
    static bool fitQuadratic(const std::vector<double>& t,
                             const std::vector<double>& y,
                             double& a, double& b, double& c);
    static bool fitLinear(const std::vector<double>& t,
                          const std::vector<double>& y,
                          double& a, double& b);
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_TRAJECTORY_PREDICTOR_H_
