#ifndef STEREO3D_GOAL_POSE_BRIDGE_H
#define STEREO3D_GOAL_POSE_BRIDGE_H

#ifdef HAS_ROS2

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include "ros2_bridge_config.h"

namespace stereo3d {

struct PlanarPoint2D {
    double x = 0.0;
    double y = 0.0;
};

enum class ControlGoalGateReason : int {
    NOT_EVALUATED = 0,
    PASSED = 1,
    DISABLED = 2,
    NONFINITE = 3,
    QUALITY = 4,
    CAMERA_BOUNDS = 5,
    UNSTABLE = 6,
    TRANSFORM_INVALID = 7,
    ODOM_STALE = 8,
    BASE_INVALID = 9,
};

struct ControlGoalGateResult {
    bool passed = false;
    ControlGoalGateReason reason = ControlGoalGateReason::NOT_EVALUATED;
    int stable_frames = 0;
    double base_x = 0.0;
    double base_y = 0.0;
};

// 3D 轨迹采样点用于预测路径发布
struct TrajectorySample {
    double t = 0.0;  // 相对时间（秒），t=0 表示发布瞬间
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

class GoalPoseBridge {
public:
    explicit GoalPoseBridge(const std::shared_ptr<rclcpp::Node>& node);
    GoalPoseBridge(const std::shared_ptr<rclcpp::Node>& node, const Ros2BridgeConfig& cfg);

    bool enabled() const;
    bool enableGui() const;

    // 话题名 / 坐标系访问器
    const std::string& realtimeWorldTopic() const { return realtime_world_topic_; }
    const std::string& landingWorldTopic() const  { return landing_world_topic_;  }
    const std::string& predictedPathTopic() const { return predicted_path_topic_; }
    const std::string& realtimeBaseTopic() const  { return realtime_base_topic_;  }
    const std::string& landingBaseTopic() const   { return landing_base_topic_;   }
    const std::string& worldFrameId() const       { return world_frame_id_;       }
    const std::string& baseFrameId() const        { return base_frame_id_;        }
    const std::string& odomTopic() const          { return odom_topic_;           }
    const std::string& controlGoalTopic() const   { return control_goal_topic_;   }
    bool controlGoalEnabled() const               { return control_goal_enabled_; }

    // 视觉坐标 -> 世界坐标 (依赖 swap/invert/rotation/translation 参数)
    PlanarPoint2D transformVisionToWorld(double vision_x, double vision_y) const;

    // ===== world frame 发布（不依赖 /odom，AGX 本地数据 + 标定即可）=====
    void publishRealtimeWorld(double world_x, double world_y, double world_z, const rclcpp::Time& stamp);
    void publishLandingWorld(double world_x, double world_y, const rclcpp::Time& stamp);
    // samples 中坐标已是 world 系；frame_id=world_frame_id_
    void publishPredictedPath(const std::vector<TrajectorySample>& world_samples, const rclcpp::Time& stamp);
    // 实际球轨迹 (world 系), 用于 RViz 红色线条显示
    void publishActualPath(const std::vector<TrajectorySample>& world_trail, const rclcpp::Time& stamp);

    // ===== base_link frame 发布（依赖 /odom）=====
    void publishRealtimeBase(double base_x, double base_y, double base_z, const rclcpp::Time& stamp);
    void publishLandingBase(double base_x, double base_y, const rclcpp::Time& stamp);

    bool hasFreshOdom() const;
    // world 系点 -> base_link 系点
    bool tryWorldToBase(double world_x, double world_y, double& base_x, double& base_y) const;

    // 安全控制目标：预测质量、连续稳定性、视觉范围、坐标变换和 odom
    // 全部有效时才发布。多项式 fallback 默认只供可视化，不进入控制。
    // vision_x = 相机横向 X；vision_depth = 相机深度 Z。
    bool tryPublishControlGoalFromVision(
        double vision_x,
        double vision_depth,
        int track_id,
        double confidence,
        double time_to_land_s,
        double speed_mps,
        int method,
        double student_w,
        int obs_source,
        const rclcpp::Time& stamp,
        ControlGoalGateResult* gate_result = nullptr);

private:
    void loadParams();
    void validateControlConfig();
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    std::shared_ptr<rclcpp::Node> node_;

    // 发布器（world frame）
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr realtime_world_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr landing_world_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               predicted_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               actual_path_pub_;
    // 发布器（base_link frame，机器人本地坐标，可选）
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr realtime_base_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr landing_base_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr control_goal_pub_;
    // 订阅器
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // ---- 配置 ----
    bool enabled_ = true;
    bool enable_gui_ = false;  // 默认关闭可视化（headless 跑视觉端）
    std::string realtime_world_topic_ = "/nx/debug/ball/realtime";
    std::string landing_world_topic_  = "/nx/debug/ball/landing";
    std::string predicted_path_topic_ = "/nx/debug/ball/predicted_path";
    std::string actual_path_topic_    = "/nx/debug/ball/actual_path";
    std::string realtime_base_topic_  = "/nx/debug/ball/realtime_base";
    std::string landing_base_topic_   = "/nx/debug/ball/landing_base";
    std::string world_frame_id_       = "vision_world";
    std::string base_frame_id_        = "base_link";
    std::string odom_topic_           = "/odom";
    bool   swap_xy_        = false;
    bool   invert_x_       = false;
    bool   invert_y_       = false;
    double rotation_deg_   = 0.0;
    double translation_x_  = 0.0;
    double translation_y_  = 0.0;
    double odom_timeout_sec_ = 0.5;
    bool control_goal_enabled_ = false;
    std::string control_goal_topic_ = "/nx/debug/auto_goal_pose";
    double control_min_depth_m_ = 0.0;
    double control_max_depth_m_ = 14.0;
    double control_max_abs_x_m_ = 3.6;
    double control_min_confidence_ = 0.70;
    double control_min_time_to_land_s_ = 0.25;
    double control_max_time_to_land_s_ = 2.20;
    double control_min_speed_mps_ = 0.80;
    double control_min_student_w_ = 0.15;
    int control_stable_frames_ = 3;
    double control_max_stable_jump_m_ = 0.35;
    bool control_allow_polynomial_ = false;
    bool control_allow_fallback_observation_ = false;

    // ---- 运行时状态 ----
    bool has_odom_ = false;
    double odom_x_ = 0.0;
    double odom_y_ = 0.0;
    double odom_yaw_ = 0.0;
    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};
    mutable std::mutex odom_mutex_;
    int control_candidate_track_id_ = -1;
    int control_candidate_stable_frames_ = 0;
    double control_candidate_x_ = 0.0;
    double control_candidate_depth_ = 0.0;
    rclcpp::Time control_candidate_time_{0, 0, RCL_ROS_TIME};
};

}  // namespace stereo3d

#endif  // HAS_ROS2

#endif  // STEREO3D_GOAL_POSE_BRIDGE_H
