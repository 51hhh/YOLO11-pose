#ifdef HAS_ROS2
#include "nx_ball_observation_publisher.h"
#include "nx_observation_quality.h"
#include "source_epoch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace stereo3d {

namespace {

float clamp01(float value) {
    if (!std::isfinite(value)) return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

bool validPositive(float value) {
    return std::isfinite(value) && value > 0.0f;
}

void fillBboxXyxy(std::array<float, 4>& out,
                  float cx, float cy, float w, float h, float conf) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    out.fill(nan);
    if (!std::isfinite(cx) || !std::isfinite(cy) ||
        !validPositive(w) || !validPositive(h) ||
        !std::isfinite(conf) || conf <= 0.0f) {
        return;
    }
    out[0] = cx - 0.5f * w;
    out[1] = cy - 0.5f * h;
    out[2] = cx + 0.5f * w;
    out[3] = cy + 0.5f * h;
}

float selectedDisparity(const Object3D& obj) {
    switch (obj.stereo_depth_source) {
    case 1:  return obj.disparity_circle_center;
    case 2:  return obj.disparity_roi_multi_point;
    case 3:  return obj.disparity_bbox_center;
    case 4:  return obj.disparity_roi_center_patch;
    case 5:  return obj.disparity_roi_edge_centroid;
    case 6:
        if (validPositive(obj.disparity_bbox_left_edge) &&
            validPositive(obj.disparity_bbox_right_edge)) {
            return 0.5f * (obj.disparity_bbox_left_edge +
                           obj.disparity_bbox_right_edge);
        }
        return validPositive(obj.disparity_bbox_left_edge)
            ? obj.disparity_bbox_left_edge
            : obj.disparity_bbox_right_edge;
    case 7:  return obj.disparity_fallback_template;
    case 8:  return obj.disparity_roi_radial_center;
    case 9:  return obj.disparity_roi_edge_pair_center;
    case 10: return obj.disparity_roi_corner_points;
    case 11: return obj.disparity_roi_texture_points;
    case 12: return obj.disparity_fallback_feature_points;
    case 13: return obj.disparity_roi_binary_points;
    case 14: return obj.disparity_roi_orb_points;
    case 15: return obj.disparity_roi_brisk_points;
    case 16: return obj.disparity_roi_akaze_points;
    case 17: return obj.disparity_roi_sift_points;
    case 18: return obj.disparity_roi_iou_region_color_patch;
    case 19: return obj.disparity_roi_patch_iou_color_edge;
    case 20:
        return validPositive(obj.disparity_roi_neural_xfeat)
            ? obj.disparity_roi_neural_xfeat
            : obj.disparity_roi_neural_feature;
    case 21: return obj.disparity_roi_cuda_template_match;
    case 22: return obj.disparity_roi_cuda_stereo_bm;
    case 23: return obj.disparity_roi_cuda_stereo_sgm;
    case 24: return obj.disparity_roi_ring_edge_profile;
    default: break;
    }
    return validPositive(obj.pair_initial_disparity)
        ? obj.pair_initial_disparity
        : std::numeric_limits<float>::quiet_NaN();
}

}  // namespace

NxBallObservationPublisher::NxBallObservationPublisher(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::string& topic,
    const std::string& frame_id,
    const std::string& source_epoch_file,
    int class_id,
    double min_depth_m,
    double max_depth_m,
    double max_speed_mps,
    double reacquire_timeout_s,
    double reacquire_base_gate_m,
    bool allow_fallback)
    : node_(node),
      frame_id_(frame_id),
      class_id_(class_id),
      min_depth_m_(std::max(0.0, min_depth_m)),
      max_depth_m_(std::max(min_depth_m_ + 0.1, max_depth_m)),
      max_speed_mps_(std::max(1.0, max_speed_mps)),
      reacquire_timeout_s_(std::max(0.0, reacquire_timeout_s)),
      reacquire_base_gate_m_(std::max(0.0, reacquire_base_gate_m)),
      allow_fallback_(allow_fallback) {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                   .best_effort()
                   .durability_volatile()
                   .deadline(rclcpp::Duration::from_seconds(0.05));
    pub_ = node_->create_publisher<volleyball_interfaces::msg::NxBallObservation>(
        topic, qos);
    source_epoch_ = createSourceEpoch(source_epoch_file);
    RCLCPP_INFO(
        node_->get_logger(),
        "NX raw observation: %s epoch=%u epoch_file=%s frame=%s depth=[%.2f,%.2f) "
        "max_speed=%.1f reacquire=%.2fs fallback=%d",
        topic.c_str(), source_epoch_, source_epoch_file.c_str(), frame_id_.c_str(),
        min_depth_m_, max_depth_m_, max_speed_mps_,
        reacquire_timeout_s_, allow_fallback_ ? 1 : 0);
}

bool NxBallObservationPublisher::eligible(const Object3D& obj) const {
    if (!obj.raw_observation_valid || !std::isfinite(obj.raw_x) ||
        !std::isfinite(obj.raw_y) || !std::isfinite(obj.raw_z) ||
        obj.class_id != class_id_ ||
        obj.raw_z < min_depth_m_ || obj.raw_z >= max_depth_m_ ||
        obj.stereo_match_source <= 0 || obj.depth_method == 0 ||
        (!allow_fallback_ && obj.stereo_match_source != 1)) {
        return false;
    }
    const double speed = std::sqrt(
        static_cast<double>(obj.vx) * obj.vx +
        static_cast<double>(obj.vy) * obj.vy +
        static_cast<double>(obj.vz) * obj.vz);
    return std::isfinite(speed) && speed <= max_speed_mps_;
}

bool NxBallObservationPublisher::physicallyPlausible(
    const Object3D& obj, double stamp_s) const {
    if (!have_last_state_) return true;
    const double dt = stamp_s - last_stamp_s_;
    if (!std::isfinite(dt) || dt <= 0.0) return false;
    if (dt > reacquire_timeout_s_) return true;
    double distance_sq = 0.0;
    const double observed[3] = {obj.raw_x, obj.raw_y, obj.raw_z};
    for (int axis = 0; axis < 3; ++axis) {
        const double predicted = last_position_[axis] + last_velocity_[axis] * dt;
        const double delta = observed[axis] - predicted;
        distance_sq += delta * delta;
    }
    const double gate = reacquire_base_gate_m_ + max_speed_mps_ * dt;
    return distance_sq <= gate * gate;
}

void NxBallObservationPublisher::acceptObservation(
    const Object3D& obj, double stamp_s, bool new_logical_track) {
    if (new_logical_track || active_output_track_id_ < 0) {
        active_output_track_id_ = next_output_track_id_;
        next_output_track_id_ =
            next_output_track_id_ >= std::numeric_limits<int>::max()
                ? 1
                : next_output_track_id_ + 1;
    }
    active_source_track_id_ = obj.track_id;
    last_stamp_s_ = stamp_s;
    last_position_ = {obj.raw_x, obj.raw_y, obj.raw_z};
    last_velocity_ = {obj.vx, obj.vy, obj.vz};
    have_last_state_ = true;
}

int NxBallObservationPublisher::selectObservation(
    const std::vector<Object3D>& results, double stamp_s) {
    int same_track = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& obj = results[i];
        if (!eligible(obj) || obj.track_id != active_source_track_id_ ||
            !physicallyPlausible(obj, stamp_s)) {
            continue;
        }
        if (same_track < 0 || obj.confidence > results[same_track].confidence) {
            same_track = static_cast<int>(i);
        }
    }
    if (same_track >= 0) {
        acceptObservation(results[same_track], stamp_s, false);
        return same_track;
    }

    const double gap = have_last_state_ ? stamp_s - last_stamp_s_
                                        : std::numeric_limits<double>::infinity();
    if (have_last_state_ && gap >= 0.0 && gap <= reacquire_timeout_s_) {
        int associated = -1;
        double best_score = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& obj = results[i];
            if (!eligible(obj) || !physicallyPlausible(obj, stamp_s)) continue;
            double distance_sq = 0.0;
            const double observed[3] = {obj.raw_x, obj.raw_y, obj.raw_z};
            for (int axis = 0; axis < 3; ++axis) {
                const double predicted =
                    last_position_[axis] + last_velocity_[axis] * gap;
                const double delta = observed[axis] - predicted;
                distance_sq += delta * delta;
            }
            const double score = std::sqrt(distance_sq) -
                                 0.25 * clamp01(obj.confidence);
            if (score < best_score) {
                best_score = score;
                associated = static_cast<int>(i);
            }
        }
        if (associated >= 0) {
            acceptObservation(results[associated], stamp_s, false);
            return associated;
        }
        return -1;
    }

    int best = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& obj = results[i];
        if (eligible(obj) &&
            (best < 0 || obj.confidence > results[best].confidence)) {
            best = static_cast<int>(i);
        }
    }
    if (best >= 0) acceptObservation(results[best], stamp_s, true);
    return best;
}

void NxBallObservationPublisher::publish(
    int frame_id,
    const std::vector<Object3D>& results,
    const FrameMetadata& metadata) {
    const double stamp_s = metadata.host_capture_timestamp_ns > 0
        ? static_cast<double>(metadata.host_capture_timestamp_ns) * 1e-9
        : node_->get_clock()->now().seconds();
    const int best = selectObservation(results, stamp_s);
    if (best < 0) return;

    const auto& obj = results[best];
    volleyball_interfaces::msg::NxBallObservation msg;
    if (metadata.host_capture_timestamp_ns > 0) {
        const uint64_t ns = metadata.host_capture_timestamp_ns;
        msg.header.stamp.sec = static_cast<int32_t>(ns / 1000000000ULL);
        msg.header.stamp.nanosec = static_cast<uint32_t>(ns % 1000000000ULL);
    } else {
        msg.header.stamp = node_->get_clock()->now();
    }
    msg.header.frame_id = frame_id_;
    msg.source_epoch = source_epoch_;
    msg.frame_id = frame_id;
    msg.track_id = active_output_track_id_;
    msg.class_id = obj.class_id;
    msg.detection_confidence = obj.confidence;

    fillBboxXyxy(msg.bbox_left_xyxy, obj.left_bbox_cx, obj.left_bbox_cy,
                 obj.left_bbox_w, obj.left_bbox_h, obj.left_bbox_conf);
    fillBboxXyxy(msg.bbox_right_xyxy, obj.right_bbox_cx, obj.right_bbox_cy,
                 obj.right_bbox_w, obj.right_bbox_h, obj.right_bbox_conf);

    const float match_confidence = selectedMatchConfidence(obj);
    const double sigma_z = depthSigmaFromObservation(obj, match_confidence);
    const double sigma_xy = std::max(0.02, 0.006 * static_cast<double>(obj.raw_z));

    msg.position.x = obj.raw_x;
    msg.position.y = obj.raw_y;
    msg.position.z = obj.raw_z;
    msg.position_covariance.fill(0.0);
    msg.position_covariance[0] = sigma_xy * sigma_xy;
    msg.position_covariance[4] = sigma_xy * sigma_xy;
    msg.position_covariance[8] = sigma_z * sigma_z;
    msg.depth_m = obj.raw_z;
    msg.disparity_px = selectedDisparity(obj);
    msg.depth_sigma_m = static_cast<float>(sigma_z);
    msg.match_confidence = match_confidence;
    msg.depth_method = obj.stereo_depth_source;
    msg.detection_valid = std::isfinite(obj.confidence) && obj.confidence > 0.0f;
    msg.stereo_valid = true;
    msg.fallback_observation = obj.stereo_match_source != 1;
    msg.left_device_timestamp = obj.left_timestamp_us;
    msg.right_device_timestamp = obj.right_timestamp_us;
    msg.stereo_timestamp_delta_ns = metadata.stereo_timestamp_residual_ns;
    pub_->publish(msg);
}

}  // namespace stereo3d
#endif
