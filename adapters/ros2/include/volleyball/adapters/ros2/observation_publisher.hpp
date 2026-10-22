#pragma once

#include "volleyball/adapters/ros2/observation_message.hpp"
#include "volleyball/ports/observation_publisher.hpp"

#include <memory>
#include <string>

namespace rclcpp {
class Node;
}

namespace volleyball::adapters::ros2 {

class Ros2ObservationPublisher final : public ports::ObservationPublisher {
public:
    Ros2ObservationPublisher(
        std::shared_ptr<rclcpp::Node> node,
        std::string topic,
        ObservationMessageConfig config);
    ~Ros2ObservationPublisher() override;

    Ros2ObservationPublisher(const Ros2ObservationPublisher&) = delete;
    Ros2ObservationPublisher& operator=(const Ros2ObservationPublisher&) = delete;
    Ros2ObservationPublisher(Ros2ObservationPublisher&&) noexcept;
    Ros2ObservationPublisher& operator=(Ros2ObservationPublisher&&) noexcept;

    [[nodiscard]] ports::PublishResult publish(
        const ports::ObservationPublication& publication) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volleyball::adapters::ros2
