#include "pair_audit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace hik_metadata_probe {
namespace {

std::uint32_t counterStep(std::uint32_t previous, std::uint32_t current) {
    return current - previous;
}

void printMetaLine(
    int sample_index,
    const FrameMeta& left,
    const FrameMeta& right) {
    if (!left.ok || !right.ok) {
        std::printf(
            "[%04d] grab fail L(ok=%d ret=0x%X) R(ok=%d ret=0x%X)\n",
            sample_index,
            left.ok ? 1 : 0,
            left.ret,
            right.ok ? 1 : 0,
            right.ret);
        return;
    }
    const std::int64_t frame_delta =
        static_cast<std::int64_t>(left.frame_num) -
        static_cast<std::int64_t>(right.frame_num);
    const std::int64_t counter_delta =
        static_cast<std::int64_t>(left.frame_counter) -
        static_cast<std::int64_t>(right.frame_counter);
    const std::int64_t trigger_delta =
        static_cast<std::int64_t>(left.trigger_index) -
        static_cast<std::int64_t>(right.trigger_index);
    const std::int64_t device_delta =
        static_cast<std::int64_t>(left.dev_ts) -
        static_cast<std::int64_t>(right.dev_ts);
    const std::int64_t host_delta = left.host_ts - right.host_ts;
    std::printf(
        "[%04d] L#%u R#%u d#=%ld  Lfc=%u Rfc=%u dfc=%ld  "
        "Ltr=%u Rtr=%u dtr=%ld  dDevTs=%ld dHostTs=%ld  "
        "Lsc/cyc/off=%u/%u/%u R=%u/%u/%u  chunk=%u/%u  "
        "first=%02X%02X%02X%02X/%02X%02X%02X%02X\n",
        sample_index,
        left.frame_num,
        right.frame_num,
        static_cast<long>(frame_delta),
        left.frame_counter,
        right.frame_counter,
        static_cast<long>(counter_delta),
        left.trigger_index,
        right.trigger_index,
        static_cast<long>(trigger_delta),
        static_cast<long>(device_delta),
        static_cast<long>(host_delta),
        left.second_count,
        left.cycle_count,
        left.cycle_offset,
        right.second_count,
        right.cycle_count,
        right.cycle_offset,
        left.unparsed_chunks,
        right.unparsed_chunks,
        left.first_bytes[0],
        left.first_bytes[1],
        left.first_bytes[2],
        left.first_bytes[3],
        right.first_bytes[0],
        right.first_bytes[1],
        right.first_bytes[2],
        right.first_bytes[3]);
}

void printIntervalStats(
    const char* label,
    int count,
    double mean,
    double minimum,
    double maximum,
    double standard_deviation) {
    if (count <= 0) {
        std::printf("  %s: no samples\n", label);
        return;
    }
    const double fps = mean > 0.0 ? 1.0e9 / mean : 0.0;
    std::printf(
        "  %s: n=%d mean=%.1fns min=%.1f max=%.1f std=%.1f fps=%.3f\n",
        label,
        count,
        mean,
        minimum,
        maximum,
        standard_deviation,
        fps);
}

}  // namespace

void PairAudit::RunningStats::add(double value) noexcept {
    if (count == 0) {
        minimum = value;
        maximum = value;
    } else {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    const double updated_delta = value - mean;
    squared_deviation_sum += delta * updated_delta;
}

double PairAudit::RunningStats::standardDeviation() const noexcept {
    return count > 1
        ? std::sqrt(squared_deviation_sum / static_cast<double>(count - 1))
        : 0.0;
}

bool PairAudit::recordPair(
    int sample_index,
    const FrameMeta& left,
    const FrameMeta& right) {
    const bool print_sample = sample_index < 30 || !left.ok || !right.ok;
    if (print_sample) printMetaLine(sample_index, left, right);
    if (!left.ok || !right.ok) {
        ++failed_pairs_;
        return false;
    }
    ++ok_pairs_;

    const bool counter_has_evidence = left.frame_counter != 0 ||
        right.frame_counter != 0 ||
        (have_first_ &&
         (left.frame_counter != previous_left_.frame_counter ||
          right.frame_counter != previous_right_.frame_counter));
    if (counter_has_evidence) ++counter_nonzero_or_changing_;
    const bool trigger_has_evidence = left.trigger_index != 0 ||
        right.trigger_index != 0 ||
        (have_first_ &&
         (left.trigger_index != previous_left_.trigger_index ||
          right.trigger_index != previous_right_.trigger_index));
    if (trigger_has_evidence) ++trigger_nonzero_or_changing_;

    if (!have_first_) {
        first_frame_delta_ =
            static_cast<std::int64_t>(left.frame_num) -
            static_cast<std::int64_t>(right.frame_num);
        first_counter_delta_ =
            static_cast<std::int64_t>(left.frame_counter) -
            static_cast<std::int64_t>(right.frame_counter);
        first_trigger_delta_ =
            static_cast<std::int64_t>(left.trigger_index) -
            static_cast<std::int64_t>(right.trigger_index);
        have_first_ = true;
        previous_left_ = left;
        previous_right_ = right;
        return true;
    }

    const std::int64_t frame_delta =
        static_cast<std::int64_t>(left.frame_num) -
        static_cast<std::int64_t>(right.frame_num);
    const std::int64_t counter_delta =
        static_cast<std::int64_t>(left.frame_counter) -
        static_cast<std::int64_t>(right.frame_counter);
    const std::int64_t trigger_delta =
        static_cast<std::int64_t>(left.trigger_index) -
        static_cast<std::int64_t>(right.trigger_index);
    bool delta_changed = false;
    if (frame_delta != first_frame_delta_) {
        ++frame_delta_changes_;
        delta_changed = true;
    }
    if (counter_delta != first_counter_delta_) {
        ++counter_delta_changes_;
        delta_changed = true;
    }
    if (trigger_delta != first_trigger_delta_) {
        ++trigger_delta_changes_;
        delta_changed = true;
    }
    if (delta_changed && !print_sample) {
        printMetaLine(sample_index, left, right);
    }

    const std::uint32_t left_frame_num_step = counterStep(
        previous_left_.frame_num, left.frame_num);
    const std::uint32_t right_frame_num_step = counterStep(
        previous_right_.frame_num, right.frame_num);
    const std::uint32_t left_frame_counter_step = counterStep(
        previous_left_.frame_counter, left.frame_counter);
    const std::uint32_t right_frame_counter_step = counterStep(
        previous_right_.frame_counter, right.frame_counter);
    const std::uint32_t left_trigger_index_step = counterStep(
        previous_left_.trigger_index, left.trigger_index);
    const std::uint32_t right_trigger_index_step = counterStep(
        previous_right_.trigger_index, right.trigger_index);
    if (left.dev_ts >= previous_left_.dev_ts) {
        left_device_step_.add(static_cast<double>(
            left.dev_ts - previous_left_.dev_ts));
    }
    if (right.dev_ts >= previous_right_.dev_ts) {
        right_device_step_.add(static_cast<double>(
            right.dev_ts - previous_right_.dev_ts));
    }
    if (left.host_ts >= previous_left_.host_ts) {
        left_host_step_.add(static_cast<double>(
            left.host_ts - previous_left_.host_ts));
    }
    if (right.host_ts >= previous_right_.host_ts) {
        right_host_step_.add(static_cast<double>(
            right.host_ts - previous_right_.host_ts));
    }
    device_delta_.add(static_cast<double>(
        static_cast<std::int64_t>(left.dev_ts) -
        static_cast<std::int64_t>(right.dev_ts)));
    left_max_frame_num_step_ = std::max(
        left_max_frame_num_step_, left_frame_num_step);
    right_max_frame_num_step_ = std::max(
        right_max_frame_num_step_, right_frame_num_step);
    left_max_frame_counter_step_ = std::max(
        left_max_frame_counter_step_, left_frame_counter_step);
    right_max_frame_counter_step_ = std::max(
        right_max_frame_counter_step_, right_frame_counter_step);
    left_max_trigger_index_step_ = std::max(
        left_max_trigger_index_step_, left_trigger_index_step);
    right_max_trigger_index_step_ = std::max(
        right_max_trigger_index_step_, right_trigger_index_step);

    bool step_anomaly = false;
    if (left_frame_num_step != 1) {
        ++left_frame_num_jumps_;
        step_anomaly = true;
    }
    if (right_frame_num_step != 1) {
        ++right_frame_num_jumps_;
        step_anomaly = true;
    }
    if (left_frame_counter_step != 1) {
        ++left_frame_counter_jumps_;
        step_anomaly = true;
    }
    if (right_frame_counter_step != 1) {
        ++right_frame_counter_jumps_;
        step_anomaly = true;
    }
    if (left_trigger_index_step != 1) {
        ++left_trigger_index_jumps_;
        step_anomaly = true;
    }
    if (right_trigger_index_step != 1) {
        ++right_trigger_index_jumps_;
        step_anomaly = true;
    }
    if (step_anomaly) {
        std::printf(
            "    step anomaly at %04d: L# +%u R# +%u  "
            "Lfc +%u Rfc +%u  Ltr +%u Rtr +%u\n",
            sample_index,
            left_frame_num_step,
            right_frame_num_step,
            left_frame_counter_step,
            right_frame_counter_step,
            left_trigger_index_step,
            right_trigger_index_step);
    }

    previous_left_ = left;
    previous_right_ = right;
    return true;
}

void PairAudit::recordAcquisitionFailure() noexcept {
    ++failed_pairs_;
}

void PairAudit::recordPwmMapping(bool matched) noexcept {
    if (matched) {
        ++pwm_sequence_matches_;
    } else {
        ++pwm_sequence_missing_;
    }
}

bool PairAudit::passed(bool pwm_enabled) const noexcept {
    const bool frame_counter_ok = ok_pairs_ > 0 && failed_pairs_ == 0 &&
        counter_nonzero_or_changing_ > 0 && first_counter_delta_ == 0 &&
        counter_delta_changes_ == 0 && left_frame_counter_jumps_ == 0 &&
        right_frame_counter_jumps_ == 0;
    const bool pwm_mapping_ok = !pwm_enabled ||
        (pwm_sequence_missing_ == 0 && pwm_sequence_matches_ == ok_pairs_);
    return frame_counter_ok && pwm_mapping_ok;
}

void PairAudit::printSummary(bool pwm_enabled) const {
    std::puts("\nSummary:");
    std::printf(
        "  ok_pairs=%d fail_pairs=%d\n", ok_pairs_, failed_pairs_);
    std::printf(
        "  first_frame_delta=%ld changes=%d\n",
        static_cast<long>(first_frame_delta_),
        frame_delta_changes_);
    std::printf(
        "  first_frame_counter_delta=%ld changes=%d "
        "nonzero_or_changing_samples=%d\n",
        static_cast<long>(first_counter_delta_),
        counter_delta_changes_,
        counter_nonzero_or_changing_);
    std::printf(
        "  first_trigger_delta=%ld changes=%d "
        "nonzero_or_changing_samples=%d\n",
        static_cast<long>(first_trigger_delta_),
        trigger_delta_changes_,
        trigger_nonzero_or_changing_);
    std::printf(
        "  per-camera jumps: frame_num L/R=%d/%d max_step=%u/%u\n",
        left_frame_num_jumps_,
        right_frame_num_jumps_,
        left_max_frame_num_step_,
        right_max_frame_num_step_);
    std::printf(
        "  per-camera jumps: frame_counter L/R=%d/%d max_step=%u/%u\n",
        left_frame_counter_jumps_,
        right_frame_counter_jumps_,
        left_max_frame_counter_step_,
        right_max_frame_counter_step_);
    std::printf(
        "  per-camera jumps: trigger_index L/R=%d/%d max_step=%u/%u\n",
        left_trigger_index_jumps_,
        right_trigger_index_jumps_,
        left_max_trigger_index_step_,
        right_max_trigger_index_step_);
    if (pwm_enabled) {
        std::printf(
            "  FrameCounter-to-GPIO mapping: matches=%d missing=%d\n",
            pwm_sequence_matches_,
            pwm_sequence_missing_);
    }
    printIntervalStats(
        "left_dev_timestamp_step",
        left_device_step_.count,
        left_device_step_.mean,
        left_device_step_.minimum,
        left_device_step_.maximum,
        left_device_step_.standardDeviation());
    printIntervalStats(
        "right_dev_timestamp_step",
        right_device_step_.count,
        right_device_step_.mean,
        right_device_step_.minimum,
        right_device_step_.maximum,
        right_device_step_.standardDeviation());
    printIntervalStats(
        "left_host_timestamp_step",
        left_host_step_.count,
        left_host_step_.mean,
        left_host_step_.minimum,
        left_host_step_.maximum,
        left_host_step_.standardDeviation());
    printIntervalStats(
        "right_host_timestamp_step",
        right_host_step_.count,
        right_host_step_.mean,
        right_host_step_.minimum,
        right_host_step_.maximum,
        right_host_step_.standardDeviation());
    printIntervalStats(
        "left_minus_right_dev_timestamp",
        device_delta_.count,
        device_delta_.mean,
        device_delta_.minimum,
        device_delta_.maximum,
        device_delta_.standardDeviation());
    std::printf(
        "  verdict: frame_counter=%s trigger_index=%s\n",
        counter_nonzero_or_changing_ > 0
            ? "usable-looking" : "not populated",
        trigger_nonzero_or_changing_ > 0
            ? "usable-looking" : "not populated");
}

}  // namespace hik_metadata_probe
