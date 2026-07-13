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
        if (control_goal_enabled_) {
            control_goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(control_goal_topic_, qos_rel);
        }

        RCLCPP_INFO(
            node_->get_logger(),
            "Ball bridge enabled (gui=%d) world_frame=%s base_frame=%s\n"
            "  world topics : %s | %s | %s\n"
            "  base topics  : %s | %s\n"
            "  vision->world calib: swap_xy=%d invert_x=%d invert_y=%d rot_deg=%.2f tx=%.3f ty=%.3f\n"
            "  odom_topic=%s timeout=%.2fs\n"
            "  control goal: enabled=%d topic=%s camera bounds |x|<=%.2fm depth=[%.2f,%.2f]m",
            enable_gui_,
            world_frame_id_.c_str(), base_frame_id_.c_str(),
            realtime_world_topic_.c_str(), landing_world_topic_.c_str(), predicted_path_topic_.c_str(),
            realtime_base_topic_.c_str(),  landing_base_topic_.c_str(),
            swap_xy_, invert_x_, invert_y_, rotation_deg_, translation_x_, translation_y_,
            odom_topic_.c_str(), odom_timeout_sec_,
            control_goal_enabled_, control_goal_topic_.c_str(),
            control_max_abs_x_m_, control_min_depth_m_, control_max_depth_m_);
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
      odom_timeout_sec_(cfg.odom_timeout_sec),
      control_goal_enabled_(cfg.control_goal_enabled),
      control_goal_topic_(cfg.control_goal_topic),
      control_min_depth_m_(cfg.control_min_depth_m),
      control_max_depth_m_(cfg.control_max_depth_m),
      control_max_abs_x_m_(cfg.control_max_abs_x_m),
      control_min_confidence_(cfg.control_min_confidence),
      control_min_time_to_land_s_(cfg.control_min_time_to_land_s),
      control_max_time_to_land_s_(cfg.control_max_time_to_land_s),
      control_min_speed_mps_(cfg.control_min_speed_mps),
      control_min_student_w_(cfg.control_min_student_w),
      control_stable_frames_(cfg.control_stable_frames),
      control_max_stable_jump_m_(cfg.control_max_stable_jump_m),
      control_allow_polynomial_(cfg.control_allow_polynomial),
      control_allow_fallback_observation_(cfg.control_allow_fallback_observation)
{
    validateControlConfig();
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
        if (control_goal_enabled_) {
            control_goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(control_goal_topic_, qos_rel);
        }

        RCLCPP_INFO(node_->get_logger(),
            "Ball bridge (config ctor): world=%s base=%s realtime=%s control_goal=%d topic=%s "
            "bounds(|x|<=%.2f depth=[%.2f,%.2f]) quality(conf>=%.2f tti=[%.2f,%.2f] "
            "speed>=%.2f stable=%d jump<=%.2f poly=%d fallback_obs=%d)",
            world_frame_id_.c_str(), base_frame_id_.c_str(), realtime_world_topic_.c_str(),
            control_goal_enabled_, control_goal_topic_.c_str(), control_max_abs_x_m_,
            control_min_depth_m_, control_max_depth_m_, control_min_confidence_,
            control_min_time_to_land_s_, control_max_time_to_land_s_,
            control_min_speed_mps_, control_stable_frames_,
            control_max_stable_jump_m_, control_allow_polynomial_,
            control_allow_fallback_observation_);
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
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!has_odom_ || !node_) return false;
    const double age = (node_->get_clock()->now() - last_odom_time_).seconds();
    return age >= 0.0 && age <= odom_timeout_sec_;
}

bool GoalPoseBridge::tryWorldToBase(double wx, double wy, double& bx, double& by) const {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!has_odom_ || !node_) {
        return false;
    }
    const double odom_age = (node_->get_clock()->now() - last_odom_time_).seconds();
    if (odom_age < 0.0 || odom_age > odom_timeout_sec_) {
        return false;
    }
    const double dx = wx - odom_x_;
    const double dy = wy - odom_y_;
    const double c = std::cos(odom_yaw_);
    const double s = std::sin(odom_yaw_);
    bx =  c * dx + s * dy;
    by = -s * dx + c * dy;
    return true;
}

bool GoalPoseBridge::tryPublishControlGoalFromVision(
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
    ControlGoalGateResult* gate_result) {
    ControlGoalGateResult audit;
    auto finish = [&](ControlGoalGateReason reason, bool passed = false) {
        audit.reason = reason;
        audit.passed = passed;
        if (gate_result) *gate_result = audit;
        return passed;
    };
    if (!enabled_ || !control_goal_enabled_ || !control_goal_pub_) {
        return finish(ControlGoalGateReason::DISABLED);
    }

    if (!std::isfinite(vision_x) || !std::isfinite(vision_depth) ||
        !std::isfinite(confidence) || !std::isfinite(time_to_land_s) ||
        !std::isfinite(speed_mps) || !std::isfinite(student_w)) {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Reject control goal: non-finite landing x=%.3f depth=%.3f",
            vision_x, vision_depth);
        return finish(ControlGoalGateReason::NONFINITE);
    }
    const bool quality_ok =
        confidence >= control_min_confidence_ &&
        time_to_land_s >= control_min_time_to_land_s_ &&
        time_to_land_s <= control_max_time_to_land_s_ &&
        speed_mps >= control_min_speed_mps_ &&
        student_w >= control_min_student_w_ &&
        (method == 0 || (method == 1 && control_allow_polynomial_)) &&
        (obs_source >= 0 &&
         (obs_source < 3 || control_allow_fallback_observation_));
    if (!quality_ok) {
        control_candidate_track_id_ = -1;
        control_candidate_stable_frames_ = 0;
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Reject control goal quality: track=%d method=%d conf=%.2f tti=%.2f speed=%.2f w=%.2f obs=%d",
            track_id, method, confidence, time_to_land_s, speed_mps,
            student_w, obs_source);
        return finish(ControlGoalGateReason::QUALITY);
    }
    if (std::fabs(vision_x) > control_max_abs_x_m_ ||
        vision_depth <= control_min_depth_m_ ||
        vision_depth > control_max_depth_m_) {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Reject control goal outside camera bounds: x=%.3f depth=%.3f allowed |x|<=%.3f depth=(%.3f,%.3f]",
            vision_x, vision_depth, control_max_abs_x_m_,
            control_min_depth_m_, control_max_depth_m_);
        return finish(ControlGoalGateReason::CAMERA_BOUNDS);
    }

    const double candidate_dt = control_candidate_time_.nanoseconds() == 0
        ? 1e9
        : (stamp - control_candidate_time_).seconds();
    const double candidate_jump = std::hypot(
        vision_x - control_candidate_x_, vision_depth - control_candidate_depth_);
    if (track_id == control_candidate_track_id_ && candidate_dt >= 0.0 &&
        candidate_dt <= 0.15 && candidate_jump <= control_max_stable_jump_m_) {
        ++control_candidate_stable_frames_;
    } else {
        control_candidate_track_id_ = track_id;
        control_candidate_stable_frames_ = 1;
    }
    control_candidate_x_ = vision_x;
    control_candidate_depth_ = vision_depth;
    control_candidate_time_ = stamp;
    audit.stable_frames = control_candidate_stable_frames_;
    if (control_candidate_stable_frames_ < control_stable_frames_) {
        RCLCPP_INFO_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Hold control goal for stability: track=%d frames=%d/%d jump=%.3f",
            track_id, control_candidate_stable_frames_, control_stable_frames_,
            candidate_jump);
        return finish(ControlGoalGateReason::UNSTABLE);
    }

    const PlanarPoint2D world = transformVisionToWorld(vision_x, vision_depth);
    if (!std::isfinite(world.x) || !std::isfinite(world.y)) {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Reject control goal: invalid vision-to-world transform");
        return finish(ControlGoalGateReason::TRANSFORM_INVALID);
    }

    double base_x = 0.0;
    double base_y = 0.0;
    if (!tryWorldToBase(world.x, world.y, base_x, base_y)) {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Reject control goal: odom missing or stale on %s (timeout %.3fs)",
            odom_topic_.c_str(), odom_timeout_sec_);
        return finish(ControlGoalGateReason::ODOM_STALE);
    }
    if (!std::isfinite(base_x) || !std::isfinite(base_y)) {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Reject control goal: invalid world-to-base transform");
        return finish(ControlGoalGateReason::BASE_INVALID);
    }

    geometry_msgs::msg::PoseStamped goal;
    goal.header.stamp = stamp;
    goal.header.frame_id = base_frame_id_;
    goal.pose.position.x = base_x;
    goal.pose.position.y = base_y;
    goal.pose.position.z = 0.0;
    goal.pose.orientation.w = 1.0;
    control_goal_pub_->publish(goal);
    audit.base_x = base_x;
    audit.base_y = base_y;
    RCLCPP_INFO_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 1000,
        "Publish safe control goal topic=%s frame=%s track=%d base=(%.3f, %.3f) "
        "camera=(x=%.3f depth=%.3f) conf=%.2f tti=%.2f speed=%.2f",
        control_goal_topic_.c_str(), base_frame_id_.c_str(),
        track_id, base_x, base_y, vision_x, vision_depth,
        confidence, time_to_land_s, speed_mps);
    return finish(ControlGoalGateReason::PASSED, true);
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
    control_goal_enabled_ = node_->declare_parameter<bool>("control_goal_enabled", control_goal_enabled_);
    control_goal_topic_ = node_->declare_parameter<std::string>("control_goal_topic", control_goal_topic_);
    control_min_depth_m_ = node_->declare_parameter<double>("control_goal_min_depth_m", control_min_depth_m_);
    control_max_depth_m_ = node_->declare_parameter<double>("control_goal_max_depth_m", control_max_depth_m_);
    control_max_abs_x_m_ = node_->declare_parameter<double>("control_goal_max_abs_x_m", control_max_abs_x_m_);
    control_min_confidence_ = node_->declare_parameter<double>(
        "control_goal_min_confidence", control_min_confidence_);
    control_min_time_to_land_s_ = node_->declare_parameter<double>(
        "control_goal_min_time_to_land_s", control_min_time_to_land_s_);
    control_max_time_to_land_s_ = node_->declare_parameter<double>(
        "control_goal_max_time_to_land_s", control_max_time_to_land_s_);
    control_min_speed_mps_ = node_->declare_parameter<double>(
        "control_goal_min_speed_mps", control_min_speed_mps_);
    control_min_student_w_ = node_->declare_parameter<double>(
        "control_goal_min_student_w", control_min_student_w_);
    control_stable_frames_ = node_->declare_parameter<int>(
        "control_goal_stable_frames", control_stable_frames_);
    control_max_stable_jump_m_ = node_->declare_parameter<double>(
        "control_goal_max_stable_jump_m", control_max_stable_jump_m_);
    control_allow_polynomial_ = node_->declare_parameter<bool>(
        "control_goal_allow_polynomial", control_allow_polynomial_);
    control_allow_fallback_observation_ = node_->declare_parameter<bool>(
        "control_goal_allow_fallback_observation", control_allow_fallback_observation_);

    if (odom_timeout_sec_ <= 0.0) {
        RCLCPP_WARN(node_->get_logger(),
                    "ball_odom_timeout_sec=%.3f invalid, fallback to 0.5 s", odom_timeout_sec_);
        odom_timeout_sec_ = 0.5;
    }
    validateControlConfig();
}

void GoalPoseBridge::validateControlConfig() {
    if (control_goal_topic_ == "/auto/goal_pose") {
        RCLCPP_WARN(
            node_->get_logger(),
            "NX is not allowed to publish /auto/goal_pose in the RDK joint system; "
            "disabling local control-goal publication. Use /nx/debug/auto_goal_pose "
            "for gate recording only.");
        control_goal_enabled_ = false;
        control_goal_topic_ = "/nx/debug/auto_goal_pose";
    }
    if (control_min_depth_m_ < 0.0 ||
        control_max_depth_m_ <= control_min_depth_m_ ||
        control_max_abs_x_m_ <= 0.0) {
        RCLCPP_WARN(node_->get_logger(),
                    "Invalid control goal bounds; fallback to |x|<=3.6m depth=[0,14]m");
        control_min_depth_m_ = 0.0;
        control_max_depth_m_ = 14.0;
        control_max_abs_x_m_ = 3.6;
    }
    control_min_confidence_ = std::clamp(control_min_confidence_, 0.0, 1.0);
    control_min_student_w_ = std::clamp(control_min_student_w_, 0.0, 1.0);
    if (control_min_time_to_land_s_ < 0.0 ||
        control_max_time_to_land_s_ <= control_min_time_to_land_s_) {
        RCLCPP_WARN(node_->get_logger(),
                    "Invalid control goal TTI bounds; fallback to [0.25,2.20] s");
        control_min_time_to_land_s_ = 0.25;
        control_max_time_to_land_s_ = 2.20;
    }
    if (control_min_speed_mps_ < 0.0) control_min_speed_mps_ = 0.80;
    if (control_stable_frames_ < 1) control_stable_frames_ = 1;
    if (control_max_stable_jump_m_ <= 0.0) control_max_stable_jump_m_ = 0.35;
}

void GoalPoseBridge::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    const auto& p = msg->pose.pose.position;
    const auto& q = msg->pose.pose.orientation;
    const double q_norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(q.x) || !std::isfinite(q.y) ||
        !std::isfinite(q.z) || !std::isfinite(q.w) || q_norm_sq < 1e-12) {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "Ignore invalid odom pose on %s", odom_topic_.c_str());
        return;
    }
    const double inv_q_norm = 1.0 / std::sqrt(q_norm_sq);
    const double qx = q.x * inv_q_norm;
    const double qy = q.y * inv_q_norm;
    const double qz = q.z * inv_q_norm;
    const double qw = q.w * inv_q_norm;
    std::lock_guard<std::mutex> lock(odom_mutex_);
    odom_x_ = p.x;
    odom_y_ = p.y;
    odom_yaw_ = std::atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz));
    last_odom_time_ = node_->get_clock()->now();
    has_odom_ = true;
}

}  // namespace stereo3d

#endif  // HAS_ROS2
