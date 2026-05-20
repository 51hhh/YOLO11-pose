#ifndef STEREO3D_BALL_TRACKER_H
#define STEREO3D_BALL_TRACKER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <cmath>
#include <string>

#include "Ball3DFilter.h"
#include "dataType.h"
#include "HybridDepth.h"

namespace stereo3d {

struct LandingPoint {
    bool valid = false;
    float t = 0.0f;
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
};

enum class BallEventType {
    NONE,
    THROW_START,
    TRACK_STABLE,
    LANDING_IMMINENT,
    FIRST_BOUNCE,
    REBOUND,
    TRACK_LOST,
    DEAD_BALL
};

enum class RallyPhase {
    IDLE,                    // 空闲状态
    TRACKING_AIRBORNE,       // 追踪飞行中的球
    LANDING_IMMINENT,        // 即将落地
    FIRST_BOUNCE_CONFIRMED,  // 第一次落地确认
    REBOUND_TRACKING,        // 反弹追踪（当前未启用）
    DEAD_BALL                // 死球状态
};

struct BallTrackState {
    bool has_det = false;
    bool raw_xyz_valid = false;
    bool has_xyz = false;
    bool valid = false;
    bool landing_is_hold = false;

    Detection det;
    std::string reject_reason;
    Eigen::Vector3f raw_xyz = Eigen::Vector3f::Zero();
    Eigen::Vector3f xyz = Eigen::Vector3f::Zero();
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f vel = Eigen::Vector3f::Zero();

    LandingPoint landing;

    BallEventType event = BallEventType::NONE;
    RallyPhase phase = RallyPhase::IDLE;
    bool bounce_detected = false;
    bool rebound_detected = false;
};

class BallTracker {
public:
    BallTracker();

    BallTrackState update(
        const std::vector<Detection>& detections,
        const cv::Mat& xyz_cpu,
        float dt,
        float ground_z = -0.667f,
        const HybridDepthResult* hybrid = nullptr
    );

    void reset();

private:
    struct TrackSample {
        float t = 0.0f;
        Eigen::Vector3f pos = Eigen::Vector3f::Zero();
        Eigen::Vector3f vel = Eigen::Vector3f::Zero();
    };

    Ball3DFilter filter_;
    RallyPhase phase_ = RallyPhase::IDLE;

    int miss_xyz_count_ = 0;
    int startup_hits_ = 0;
    int ground_contact_hits_ = 0;

    float time_since_track_start_ = 0.0f;
    float time_since_last_xyz_ = 0.0f;
    float phase_hold_sec_ = 0.0f;

    bool has_last_landing_ = false;
    LandingPoint last_landing_;
    bool first_bounce_confirmed_ = false;
    bool rebound_seen_ = false;
    Eigen::Vector3f pre_bounce_vel_ = Eigen::Vector3f::Zero();
    bool has_startup_candidate_xyz_ = false;
    Eigen::Vector3f startup_candidate_xyz_ = Eigen::Vector3f::Zero();

    std::deque<TrackSample> history_;
    bool has_last_obs_xyz_ = false;
    Eigen::Vector3f last_obs_xyz_ = Eigen::Vector3f::Zero();
    bool selectBallDetection(const std::vector<Detection>& detections, Detection& out_best) const;
    bool sampleMedianXYZ(const Detection& det,
                         const cv::Mat& xyz,
                         const Eigen::Vector3f* pred_pos,
                         Eigen::Vector3f& out_xyz) const;

    float computeMeasurementNoise(const Eigen::Vector3f& obs, const Detection& det) const;
    LandingPoint predictLandingBallistic(const Eigen::Vector3f& pos,
                                         const Eigen::Vector3f& vel,
                                         float ground_z) const;
    LandingPoint predictLandingFromHistory(float ground_z) const;

    BallEventType updateStateMachine(
        const Eigen::Vector3f& filt_pos,
        const Eigen::Vector3f& filt_vel,
        bool has_xyz,
        const Eigen::Vector3f* obs_pos,
        float dt,
        float ground_z,
        LandingPoint& out_landing,
        bool& out_hold);

    void pushHistory(const Eigen::Vector3f& pos, const Eigen::Vector3f& vel, float dt);
    void clearHistory();
};

}  // namespace stereo3d

#endif  // STEREO3D_BALL_TRACKER_H
