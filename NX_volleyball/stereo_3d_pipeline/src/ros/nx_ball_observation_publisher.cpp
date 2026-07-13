#ifdef HAS_ROS2
#include "nx_ball_observation_publisher.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <unistd.h>

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

float selectedMatchConfidence(const Object3D& obj) {
    switch (obj.stereo_depth_source) {
    case 2:  return clamp01(obj.subpixel_confidence);
    case 4:  return clamp01(obj.p0p1_center_patch_trust);
    case 5:  return clamp01(obj.p0p1_edge_centroid_trust);
    case 8:  return clamp01(obj.p0p1_radial_center_trust);
    case 9:  return clamp01(obj.p0p1_edge_pair_center_trust);
    case 10: return clamp01(obj.roi_corner_points_confidence);
    case 11: return clamp01(obj.roi_texture_points_confidence);
    case 12: return clamp01(obj.fallback_feature_points_confidence);
    case 13: return clamp01(obj.roi_binary_points_confidence);
    case 14: return clamp01(obj.roi_orb_points_confidence);
    case 15: return clamp01(obj.roi_brisk_points_confidence);
    case 16: return clamp01(obj.roi_akaze_points_confidence);
    case 17: return clamp01(obj.roi_sift_points_confidence);
    case 18: return clamp01(obj.roi_iou_region_color_patch_confidence);
    case 19: return clamp01(obj.roi_patch_iou_color_edge_confidence);
    case 20:
        return clamp01(std::max(obj.roi_neural_xfeat_confidence,
                                obj.roi_neural_feature_confidence));
    case 21: return clamp01(obj.roi_cuda_template_match_confidence);
    case 22: return clamp01(obj.roi_cuda_stereo_bm_confidence);
    case 23: return clamp01(obj.roi_cuda_stereo_sgm_confidence);
    case 24: return clamp01(obj.roi_ring_edge_profile_confidence);
    default: break;
    }
    return clamp01(obj.pair_score);
}

double depthSigmaFromObservation(const Object3D& obj, float confidence) {
    const double z = std::max(0.1, static_cast<double>(obj.raw_z));
    double sigma = std::max(0.05, 0.025 * z);
    const double conf = std::max(0.05, static_cast<double>(confidence));
    sigma /= std::sqrt(conf);
    if (obj.stereo_match_source != 1) sigma *= 2.0;
    if (obj.stereo_depth_source == 0) sigma *= 3.0;
    return sigma;
}

}  // namespace

NxBallObservationPublisher::NxBallObservationPublisher(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::string& topic,
    const std::string& frame_id)
    : node_(node), frame_id_(frame_id) {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    pub_ = node_->create_publisher<volleyball_interfaces::msg::NxBallObservation>(
        topic, qos);
    std::random_device rd;
    source_epoch_ = rd() ^ uint32_t(::getpid()) ^
        uint32_t(std::chrono::system_clock::now().time_since_epoch().count());
    if (source_epoch_ == 0) source_epoch_ = 1;
    RCLCPP_INFO(node_->get_logger(), "NX raw observation: %s epoch=%u frame=%s",
                topic.c_str(), source_epoch_, frame_id_.c_str());
}

void NxBallObservationPublisher::publish(
    int frame_id,
    const std::vector<Object3D>& results,
    const FrameMetadata& metadata) {
    int best = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& obj = results[i];
        const bool stereo =
            obj.raw_observation_valid &&
            validPositive(obj.raw_z) &&
            obj.stereo_match_source > 0 &&
            obj.depth_method != 0 &&
            std::isfinite(obj.raw_x) &&
            std::isfinite(obj.raw_y);
        if (stereo && (best < 0 || obj.confidence > results[best].confidence)) {
            best = static_cast<int>(i);
        }
    }
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
    msg.track_id = obj.track_id;
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
    msg.stereo_timestamp_delta_ns =
        static_cast<int64_t>(obj.left_timestamp_us) -
        static_cast<int64_t>(obj.right_timestamp_us);
    pub_->publish(msg);
}

}  // namespace stereo3d
#endif
