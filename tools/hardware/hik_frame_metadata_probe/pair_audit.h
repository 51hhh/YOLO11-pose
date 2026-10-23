#ifndef VOLLEYBALL_HIK_METADATA_PROBE_PAIR_AUDIT_H_
#define VOLLEYBALL_HIK_METADATA_PROBE_PAIR_AUDIT_H_

#include "camera_session.h"

#include <cstdint>
#include <optional>

namespace hik_metadata_probe {

/**
 * 汇总双目 payload 身份、设备/主机时间步长与 GPIO 映射证据。
 *
 * recordPair() 只接收已从各自 SDK buffer 复制出的元数据，不参与取流，
 * 因而判定逻辑可以在无相机主机上复现和回归测试。
 */
class PairAudit {
public:
    [[nodiscard]] bool recordPair(
        int sample_index,
        const FrameMeta& left,
        const FrameMeta& right);
    void recordAcquisitionFailure() noexcept;
    void recordPwmMapping(bool matched) noexcept;

    [[nodiscard]] int okPairs() const noexcept { return ok_pairs_; }
    [[nodiscard]] int failedPairs() const noexcept { return failed_pairs_; }
    [[nodiscard]] std::optional<std::uint64_t> currentFrameSequence() const
        noexcept {
        return current_frame_sequence_;
    }
    [[nodiscard]] bool passed(bool pwm_enabled) const noexcept;
    void printSummary(bool pwm_enabled) const;

private:
    struct RunningStats {
        int count = 0;
        double mean = 0.0;
        double squared_deviation_sum = 0.0;
        double minimum = 0.0;
        double maximum = 0.0;

        void add(double value) noexcept;
        [[nodiscard]] double standardDeviation() const noexcept;
    };

    int ok_pairs_ = 0;
    int failed_pairs_ = 0;
    int frame_delta_changes_ = 0;
    int counter_delta_changes_ = 0;
    int trigger_delta_changes_ = 0;
    int trigger_nonzero_or_changing_ = 0;
    int counter_nonzero_or_changing_ = 0;
    int pwm_sequence_matches_ = 0;
    int pwm_sequence_missing_ = 0;
    int left_frame_num_jumps_ = 0;
    int right_frame_num_jumps_ = 0;
    int left_frame_counter_jumps_ = 0;
    int right_frame_counter_jumps_ = 0;
    int left_trigger_index_jumps_ = 0;
    int right_trigger_index_jumps_ = 0;
    std::uint32_t left_max_frame_num_step_ = 0;
    std::uint32_t right_max_frame_num_step_ = 0;
    std::uint32_t left_max_frame_counter_step_ = 0;
    std::uint32_t right_max_frame_counter_step_ = 0;
    std::uint32_t left_max_trigger_index_step_ = 0;
    std::uint32_t right_max_trigger_index_step_ = 0;
    RunningStats left_device_step_;
    RunningStats right_device_step_;
    RunningStats left_host_step_;
    RunningStats right_host_step_;
    RunningStats device_delta_;
    std::int64_t first_frame_delta_ = 0;
    std::int64_t first_counter_delta_ = 0;
    std::int64_t first_trigger_delta_ = 0;
    bool have_first_ = false;
    std::optional<std::uint64_t> current_frame_sequence_;
    FrameMeta previous_left_;
    FrameMeta previous_right_;
};

}  // namespace hik_metadata_probe

#endif  // VOLLEYBALL_HIK_METADATA_PROBE_PAIR_AUDIT_H_
