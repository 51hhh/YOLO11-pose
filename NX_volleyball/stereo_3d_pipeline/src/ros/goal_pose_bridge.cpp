#ifdef HAS_ROS2

#include "../ros/goal_pose_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>

#include <geometry_msgs/msg/pose_stamped.hpp>

namespace stereo3d {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

GoalPoseBridge::GoalPoseBridge(const std::shared_ptr<rclcpp::Node>& node)
    : node_(node) {
    loadParams();

    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 20,
        std::bind(&GoalPoseBridge::odomCallback, this, std::placeholders::_1));

    if (enabled_) {
        // QoS: 实时数据 best-effort 低延迟；落点/路径 reliable 不能丢
        auto qos_best = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        auto qos_rel  = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

        realtime_world_pub_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(realtime_world_topic_, qos_best);
        landing_world_pub_  = node_->create_publisher<geometry_msgs::msg::PointStamped>(landing_world_topic_,  qos_rel);
        predicted_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>(predicted_path_topic_, qos_best);
        actual_path_pub_    = node_->create_publisher<nav_msgs::msg::Path>(actual_path_topic_,    qos_best);
        realtime_base_pub_  = node_->create_publisher<geometry_msgs::msg::PointStamped>(realtime_base_topic_, qos_best);
        landing_base_pub_   = node_->create_publisher<geometry_msgs::msg::PointStamped>(landing_base_topic_,  qos_rel);

        RCLCPP_INFO(
            node_->get_logger(),
            "Ball bridge enabled (gui=%d) world_frame=%s base_frame=%s\n"
            "  world topics : %s | %s | %s\n"
            "  base topics  : %s | %s\n"
            "  vision->world calib: swap_xy=%d invert_x=%d invert_y=%d rot_deg=%.2f tx=%.3f ty=%.3f\n"
            "  odom_topic=%s timeout=%.2fs",
            enable_gui_,
            world_frame_id_.c_str(), base_frame_id_.c_str(),
            realtime_world_topic_.c_str(), landing_world_topic_.c_str(), predicted_path_topic_.c_str(),
            realtime_base_topic_.c_str(),  landing_base_topic_.c_str(),
            swap_xy_, invert_x_, invert_y_, rotation_deg_, translation_x_, translation_y_,
            odom_topic_.c_str(), odom_timeout_sec_);
    } else {
        RCLCPP_INFO(node_->get_logger(), "Ball bridge disabled");
    }
}

bool GoalPoseBridge::enabled() const   { return enabled_; }
bool GoalPoseBridge::enableGui() const { return enable_gui_; }

GoalPoseBridge::GoalPoseBridge(const std::shared_ptr<rclcpp::Node>& node,
                               const Ros2BridgeConfig& cfg)
    : node_(node),
      enabled_(cfg.enabled),
      enable_gui_(false),
      realtime_world_topic_(cfg.topic_realtime),
      landing_world_topic_(cfg.topic_landing),
      predicted_path_topic_(cfg.topic_predicted_path),
      actual_path_topic_(cfg.topic_actual_path),
      realtime_base_topic_(cfg.topic_realtime_base),
      landing_base_topic_(cfg.topic_landing_base),
      world_frame_id_(cfg.world_frame_id),
      base_frame_id_(cfg.base_frame_id),
      odom_topic_(cfg.odom_topic),
      swap_xy_(cfg.swap_xy),
      invert_x_(cfg.invert_x),
      invert_y_(cfg.invert_y),
      rotation_deg_(cfg.rotation_deg),
      translation_x_(cfg.translation_x),
      translation_y_(cfg.translation_y),
      odom_timeout_sec_(cfg.odom_timeout_sec)
{
    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 20,
        std::bind(&GoalPoseBridge::odomCallback, this, std::placeholders::_1));

    if (enabled_) {
        auto qos_best = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        auto qos_rel  = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

        realtime_world_pub_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(realtime_world_topic_, qos_best);
        landing_world_pub_  = node_->create_publisher<geometry_msgs::msg::PointStamped>(landing_world_topic_,  qos_rel);
        predicted_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>(predicted_path_topic_, qos_best);
        actual_path_pub_    = node_->create_publisher<nav_msgs::msg::Path>(actual_path_topic_,    qos_best);
        realtime_base_pub_  = node_->create_publisher<geometry_msgs::msg::PointStamped>(realtime_base_topic_, qos_best);
        landing_base_pub_   = node_->create_publisher<geometry_msgs::msg::PointStamped>(landing_base_topic_,  qos_rel);

        RCLCPP_INFO(node_->get_logger(),
            "Ball bridge (config ctor): world=%s base=%s realtime=%s",
            world_frame_id_.c_str(), base_frame_id_.c_str(), realtime_world_topic_.c_str());
    }
}

PlanarPoint2D GoalPoseBridge::transformVisionToWorld(double vision_x, double vision_y) const {
    double x = vision_x;
    double y = vision_y;
    if (swap_xy_) std::swap(x, y);
    if (invert_x_) x = -x;
    if (invert_y_) y = -y;

    const double theta = rotation_deg_ * kPi / 180.0;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    return PlanarPoint2D{
        c * x - s * y + translation_x_,
        s * x + c * y + translation_y_,
    };
}

void GoalPoseBridge::publishRealtimeWorld(double wx, double wy, double wz, const rclcpp::Time& stamp) {
    if (!enabled_ || !realtime_world_pub_) return;
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_id_;
    msg.point.x = wx; msg.point.y = wy; msg.point.z = wz;
    realtime_world_pub_->publish(msg);
}

void GoalPoseBridge::publishLandingWorld(double wx, double wy, const rclcpp::Time& stamp) {
    if (!enabled_ || !landing_world_pub_) return;
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_id_;
    msg.point.x = wx; msg.point.y = wy; msg.point.z = 0.0;
    landing_world_pub_->publish(msg);
}

void GoalPoseBridge::publishPredictedPath(const std::vector<TrajectorySample>& samples, const rclcpp::Time& stamp) {
    if (!enabled_ || !predicted_path_pub_ || samples.empty()) return;
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = world_frame_id_;
    path.poses.reserve(samples.size());
    for (const auto& s : samples) {
        geometry_msgs::msg::PoseStamped p;
        // 每个 pose 的 stamp = 基准时刻 + 相对 t（接收方据此对齐时间）
        p.header.stamp = rclcpp::Time(stamp) + rclcpp::Duration::from_seconds(s.t);
        p.header.frame_id = world_frame_id_;
        p.pose.position.x = s.x;
        p.pose.position.y = s.y;
        p.pose.position.z = s.z;
        p.pose.orientation.w = 1.0;
        path.poses.push_back(p);
    }
    predicted_path_pub_->publish(path);
}

void GoalPoseBridge::publishActualPath(const std::vector<TrajectorySample>& samples, const rclcpp::Time& stamp) {
    if (!enabled_ || !actual_path_pub_ || samples.empty()) return;
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = world_frame_id_;
    path.poses.reserve(samples.size());
    for (const auto& s : samples) {
        geometry_msgs::msg::PoseStamped p;
        p.header.stamp = stamp;
        p.header.frame_id = world_frame_id_;
        p.pose.position.x = s.x;
        p.pose.position.y = s.y;
        p.pose.position.z = s.z;
        p.pose.orientation.w = 1.0;
        path.poses.push_back(p);
    }
    actual_path_pub_->publish(path);
}

void GoalPoseBridge::publishRealtimeBase(double bx, double by, double bz, const rclcpp::Time& stamp) {
    if (!enabled_ || !realtime_base_pub_) return;
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = base_frame_id_;
    msg.point.x = bx; msg.point.y = by; msg.point.z = bz;
    realtime_base_pub_->publish(msg);
}

void GoalPoseBridge::publishLandingBase(double bx, double by, const rclcpp::Time& stamp) {
    if (!enabled_ || !landing_base_pub_) return;
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = base_frame_id_;
    msg.point.x = bx; msg.point.y = by; msg.point.z = 0.0;
    landing_base_pub_->publish(msg);
}

bool GoalPoseBridge::hasFreshOdom() const {
    if (!has_odom_ || !node_) return false;
    return (node_->get_clock()->now() - last_odom_time_).seconds() <= odom_timeout_sec_;
}

bool GoalPoseBridge::tryWorldToBase(double wx, double wy, double& bx, double& by) const {
    if (!hasFreshOdom()) return false;
    const double dx = wx - odom_x_;
    const double dy = wy - odom_y_;
    const double c = std::cos(odom_yaw_);
    const double s = std::sin(odom_yaw_);
    bx =  c * dx + s * dy;
    by = -s * dx + c * dy;
    return true;
}

void GoalPoseBridge::loadParams() {
    enabled_              = node_->declare_parameter<bool>("ball_bridge_enabled", true);
    enable_gui_           = node_->declare_parameter<bool>("enable_gui", false);  // 默认 headless
    realtime_world_topic_ = node_->declare_parameter<std::string>("ball_realtime_topic", realtime_world_topic_);
    landing_world_topic_  = node_->declare_parameter<std::string>("ball_landing_topic",  landing_world_topic_);
    predicted_path_topic_ = node_->declare_parameter<std::string>("ball_predicted_path_topic", predicted_path_topic_);
    realtime_base_topic_  = node_->declare_parameter<std::string>("ball_realtime_base_topic", realtime_base_topic_);
    landing_base_topic_   = node_->declare_parameter<std::string>("ball_landing_base_topic",  landing_base_topic_);
    world_frame_id_       = node_->declare_parameter<std::string>("ball_world_frame_id", world_frame_id_);
    base_frame_id_        = node_->declare_parameter<std::string>("ball_base_frame_id",  base_frame_id_);
    swap_xy_              = node_->declare_parameter<bool>("vision_to_world_swap_xy", swap_xy_);
    invert_x_             = node_->declare_parameter<bool>("vision_to_world_invert_x", invert_x_);
    invert_y_             = node_->declare_parameter<bool>("vision_to_world_invert_y", invert_y_);
    rotation_deg_         = node_->declare_parameter<double>("vision_to_world_rotation_deg", rotation_deg_);
    translation_x_        = node_->declare_parameter<double>("vision_to_world_translation_x", translation_x_);
    translation_y_        = node_->declare_parameter<double>("vision_to_world_translation_y", translation_y_);
    odom_topic_           = node_->declare_parameter<std::string>("ball_odom_topic", odom_topic_);
    odom_timeout_sec_     = node_->declare_parameter<double>("ball_odom_timeout_sec", odom_timeout_sec_);

    if (odom_timeout_sec_ <= 0.0) {
        RCLCPP_WARN(node_->get_logger(),
                    "ball_odom_timeout_sec=%.3f invalid, fallback to 0.5 s", odom_timeout_sec_);
        odom_timeout_sec_ = 0.5;
    }
}

void GoalPoseBridge::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    odom_x_ = msg->pose.pose.position.x;
    odom_y_ = msg->pose.pose.position.y;
    const auto& q = msg->pose.pose.orientation;
    odom_yaw_ = std::atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    last_odom_time_ = node_->get_clock()->now();
    has_odom_ = true;
}

}  // namespace stereo3d

#endif  // HAS_ROS2
