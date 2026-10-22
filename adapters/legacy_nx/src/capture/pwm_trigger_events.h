/**
 * @file pwm_trigger_events.h
 * @brief 有界保存 GPIO 上升沿的多时钟时间戳，供相机帧按触发序号关联。
 */

#ifndef STEREO_3D_PIPELINE_PWM_TRIGGER_EVENTS_H_
#define STEREO_3D_PIPELINE_PWM_TRIGGER_EVENTS_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>

namespace stereo3d {

struct PWMTriggerEdgeEvent {
    std::uint64_t sequence = 0;
    std::uint64_t scheduled_monotonic_ns = 0;
    std::uint64_t edge_monotonic_ns = 0;
    std::uint64_t edge_realtime_ns = 0;
    std::uint64_t write_uncertainty_ns = 0;
};

class PWMTriggerEventHistory {
public:
    explicit PWMTriggerEventHistory(std::size_t capacity = 16384)
        : capacity_(capacity > 0 ? capacity : 1) {}

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
    }

    void push(const PWMTriggerEdgeEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!events_.empty() && event.sequence <= events_.back().sequence) return;
        events_.push_back(event);
        while (events_.size() > capacity_) events_.pop_front();
    }

    [[nodiscard]] bool find(
        std::uint64_t sequence,
        PWMTriggerEdgeEvent& output) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (events_.empty() || sequence < events_.front().sequence ||
            sequence > events_.back().sequence) {
            return false;
        }
        const std::uint64_t offset = sequence - events_.front().sequence;
        if (offset >= events_.size()) return false;
        const auto& candidate = events_[static_cast<std::size_t>(offset)];
        if (candidate.sequence != sequence) return false;
        output = candidate;
        return true;
    }

    [[nodiscard]] std::uint64_t latestSequence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.empty() ? 0 : events_.back().sequence;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<PWMTriggerEdgeEvent> events_;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PWM_TRIGGER_EVENTS_H_
