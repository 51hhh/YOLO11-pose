#pragma once
#ifdef HAS_ROS2
#include <array>
#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <volleyball_interfaces/msg/nx_ball_observation.hpp>
#include "../pipeline/frame_slot.h"
namespace stereo3d {
class NxBallObservationPublisher {
public:
    NxBallObservationPublisher(const std::shared_ptr<rclcpp::Node>& node,
                               const std::string& topic,
                               const std::string& frame_id,
                               const std::string& source_epoch_file,
                               int class_id,
                               double min_depth_m,
                               double max_depth_m,
                               double max_speed_mps,
                               double reacquire_timeout_s,
                               double reacquire_base_gate_m,
                               bool allow_fallback);
    void publish(int frame_id, const std::vector<Object3D>& results,
                 const FrameMetadata& metadata);
private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<volleyball_interfaces::msg::NxBallObservation>::SharedPtr pub_;
    std::string frame_id_;
    uint32_t source_epoch_{0};
    int class_id_{0};
    double min_depth_m_{0.8};
    double max_depth_m_{15.0};
    double max_speed_mps_{35.0};
    double reacquire_timeout_s_{3.0};
    double reacquire_base_gate_m_{0.75};
    bool allow_fallback_{false};
    int active_source_track_id_{-1};
    int active_output_track_id_{-1};
    int next_output_track_id_{1};
    bool have_last_state_{false};
    double last_stamp_s_{0.0};
    std::array<double, 3> last_position_{};
    std::array<double, 3> last_velocity_{};

    bool eligible(const Object3D& obj) const;
    bool physicallyPlausible(const Object3D& obj, double stamp_s) const;
    int selectObservation(const std::vector<Object3D>& results, double stamp_s);
    void acceptObservation(const Object3D& obj, double stamp_s,
                           bool new_logical_track);
};
}
#endif
