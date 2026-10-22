#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rclcpp {

struct Duration {
    double seconds{0.0};

    static Duration from_seconds(double value) { return {value}; }
};

struct KeepLast {
    explicit KeepLast(std::size_t value) : depth(value) {}
    std::size_t depth;
};

class QoS {
public:
    explicit QoS(KeepLast history) : depth(history.depth) {}

    QoS& best_effort() {
        best_effort_enabled = true;
        return *this;
    }
    QoS& durability_volatile() {
        volatile_enabled = true;
        return *this;
    }
    QoS& deadline(Duration value) {
        deadline_seconds = value.seconds;
        return *this;
    }

    std::size_t depth{0};
    bool best_effort_enabled{false};
    bool volatile_enabled{false};
    double deadline_seconds{0.0};
};

template <typename Message>
class Publisher {
public:
    using SharedPtr = std::shared_ptr<Publisher<Message>>;

    void publish(const Message& message) {
        if (throw_on_publish) {
            throw std::runtime_error("injected ROS publish failure");
        }
        messages.push_back(message);
    }

    bool throw_on_publish{false};
    std::vector<Message> messages;
};

template <typename Message>
typename Publisher<Message>::SharedPtr& lastPublisher() {
    static typename Publisher<Message>::SharedPtr value;
    return value;
}

class Node {
public:
    template <typename Message>
    typename Publisher<Message>::SharedPtr create_publisher(
        const std::string& topic,
        const QoS& qos) {
        last_topic = topic;
        last_qos = qos;
        auto output = std::make_shared<Publisher<Message>>();
        lastPublisher<Message>() = output;
        return output;
    }

    std::string last_topic;
    QoS last_qos{KeepLast{0}};
};

}  // namespace rclcpp
