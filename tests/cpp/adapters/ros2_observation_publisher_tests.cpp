#include "volleyball/adapters/ros2/observation_publisher.hpp"

#include <rclcpp/rclcpp.hpp>
#include <volleyball_interfaces/msg/nx_ball_observation.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace adapter = volleyball::adapters::ros2;
namespace core = volleyball::core;
namespace ports = volleyball::ports;
using NxMessage = volleyball_interfaces::msg::NxBallObservation;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

core::Timestamp timestamp(std::int64_t value) {
    return {value, core::ClockDomainId{"nx-realtime"}};
}

core::CameraFrameRecord frame(
    core::CameraRole role,
    const char* camera,
    std::int64_t capture_ns,
    std::int64_t hardware_ns) {
    core::CameraFrameRecord output;
    output.camera_id = core::CameraId{camera};
    output.role = role;
    output.frame_index = 8;
    output.hardware_timestamp = {
        hardware_ns,
        core::ClockDomainId{
            role == core::CameraRole::left ? "left-device" : "right-device"},
    };
    output.capture_timestamp = timestamp(capture_ns);
    output.roi = {0, 0, 1440, 1080};
    output.acquisition.stereo_hardware_timestamp_residual_ns = 3;
    return output;
}

core::Detection2D detection(
    core::CameraRole role,
    const char* camera,
    core::CandidateId candidate,
    double center_x,
    double score) {
    core::Detection2D output;
    output.frame = {core::CameraId{camera}, role, 8};
    output.candidate_id = candidate;
    output.center_px = {center_x, 400.0};
    output.bounding_box = {center_x - 10.0, 390.0, 20.0, 20.0};
    output.score = score;
    output.selected = true;
    output.source_pipeline = core::PipelineId{"yolo"};
    return output;
}

ports::ObservationPublication publication() {
    ports::ObservationPublication output;
    output.left_frame =
        frame(core::CameraRole::left, "left", 3'000'000'000, 500);
    output.right_frame =
        frame(core::CameraRole::right, "right", 3'000'000'000, 497);
    output.left_detection =
        detection(core::CameraRole::left, "left", 1, 600.0, 0.9);
    output.right_detection =
        detection(core::CameraRole::right, "right", 2, 550.0, 0.85);
    output.association.pair_id = 4;
    output.association.left_frame_index = 8;
    output.association.right_frame_index = 8;
    output.association.left_candidate_id = 1;
    output.association.right_candidate_id = 2;
    output.association.left_capture = output.left_frame.capture_timestamp;
    output.association.right_capture = output.right_frame.capture_timestamp;
    output.association.delta_t_lr_ns = 0;
    output.association.disparity_px = 50.0;
    output.association.epipolar_error_px = 0.0;
    output.association.decision = core::AssociationDecision::accepted;
    output.association.reject_reason = core::AssociationRejectReason::none;

    auto& measurement = output.observation.measurement;
    measurement.observation_id = 5;
    measurement.source_pair_id = 4;
    measurement.capture_reference = timestamp(3'000'000'000);
    measurement.coordinate_frame =
        core::CoordinateFrameId{"nx_left_rectified_optical_frame"};
    measurement.position_m = {1.0, 2.0, 6.0};
    measurement.covariance_m2 = core::Covariance3D{{
        0.01, 0.0, 0.0, 0.01, 0.0, 0.09,
    }};
    measurement.confidence = 0.8;
    measurement.reprojection_error_px = 0.0;
    output.observation.status = core::ValidationStatus::accepted;
    output.estimation.method = core::StereoDepthMethodId{"roi_multi_point"};
    output.estimation.disparity_px = 49.5;
    output.estimation.support_points = 7;
    return output;
}

adapter::ObservationMessageConfig config() {
    return {
        "nx_left_rectified_optical_frame",
        core::ClockDomainId{"nx-realtime"},
        core::StereoDepthMethodId{"roi_multi_point"},
        23,
        0,
    };
}

void testPublisherUsesSensorQosAndMovesMappedFields() {
    auto node = std::make_shared<rclcpp::Node>();
    adapter::Ros2ObservationPublisher publisher(
        node, "/nx/ball/observation", config());
    check(node->last_topic == "/nx/ball/observation" &&
              node->last_qos.depth == 1 &&
              node->last_qos.best_effort_enabled &&
              node->last_qos.volatile_enabled &&
              node->last_qos.deadline_seconds == 0.05,
          "publisher must use the frozen low-latency sensor QoS");

    const auto result = publisher.publish(publication());
    check(result.success, "valid direct observation must publish");
    const auto transport = rclcpp::lastPublisher<NxMessage>();
    check(transport && transport->messages.size() == 1,
          "rclcpp publisher must receive exactly one message");
    if (!transport || transport->messages.empty()) return;
    const auto& message = transport->messages.front();
    check(message.header.stamp.sec == 3 &&
              message.header.stamp.nanosec == 0 &&
              message.header.frame_id ==
                  "nx_left_rectified_optical_frame",
          "header must retain mapped time and coordinate frame");
    check(message.source_epoch == 23 && message.frame_id == 8 &&
              message.track_id == 0 && message.depth_method == 2,
          "identity and depth method fields must preserve mapping semantics");
    check(message.position.z == 6.0 &&
              message.position_covariance[8] == 0.09 &&
              message.left_device_timestamp == 500 &&
              message.right_device_timestamp == 497 &&
              message.stereo_timestamp_delta_ns == 3,
          "position, covariance, and device timing must be copied exactly");
}

void testMappingAndTransportFailuresAreObservable() {
    auto node = std::make_shared<rclcpp::Node>();
    adapter::Ros2ObservationPublisher publisher(node, "/observation", config());
    auto invalid = publication();
    invalid.observation.measurement.covariance_m2.reset();
    const auto rejected = publisher.publish(invalid);
    check(!rejected.success && !rejected.message.empty(),
          "unrepresentable direct observation must return a useful failure");

    auto transport = rclcpp::lastPublisher<NxMessage>();
    transport->throw_on_publish = true;
    const auto failed = publisher.publish(publication());
    check(!failed.success &&
              failed.message.find("injected ROS publish failure") !=
                  std::string::npos,
          "rclcpp exception must become a PublishResult failure");
}

void testInvalidConstructionFailsBeforeRuntime() {
    bool threw = false;
    try {
        adapter::Ros2ObservationPublisher publisher(
            std::make_shared<rclcpp::Node>(), "", config());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty topic must fail during publisher construction");
}

}  // namespace

int main() {
    testPublisherUsesSensorQosAndMovesMappedFields();
    testMappingAndTransportFailuresAreObservable();
    testInvalidConstructionFailsBeforeRuntime();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "ROS2 observation publisher tests passed\n";
    return EXIT_SUCCESS;
}
