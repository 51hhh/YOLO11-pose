#ifndef STEREO3D_GOAL_POSE_BRIDGE_H
#define STEREO3D_GOAL_POSE_BRIDGE_H

#ifdef HAS_ROS2

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include "../pipeline/pipeline.h"

namespace stereo3d {

struct PlanarPoint2D {
    double x = 0.0;
    double y = 0.0;
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

private:
    void loadParams();
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
    // 订阅器
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // ---- 配置 ----
    bool enabled_ = true;
    bool enable_gui_ = false;  // 默认关闭可视化（headless 跑视觉端）
    std::string realtime_world_topic_ = "/ball/realtime";
    std::string landing_world_topic_  = "/ball/landing";
    std::string predicted_path_topic_ = "/ball/predicted_path";
    std::string actual_path_topic_    = "/ball/actual_path";
    std::string realtime_base_topic_  = "/ball/realtime_base";
    std::string landing_base_topic_   = "/ball/landing_base";
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

    // ---- 运行时状态 ----
    bool has_odom_ = false;
    double odom_x_ = 0.0;
    double odom_y_ = 0.0;
    double odom_yaw_ = 0.0;
    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};
};

}  // namespace stereo3d

#endif  // HAS_ROS2

#endif  // STEREO3D_GOAL_POSE_BRIDGE_H
