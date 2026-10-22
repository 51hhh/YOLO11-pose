#include "args.h"
#include "camera_session.h"
#include "capture/pwm_trigger.h"
#include "stats.h"
#include "../common/parallel_tasks.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace {

using hik_metadata_probe::Args;
using hik_metadata_probe::CameraSession;
using hik_metadata_probe::FrameMeta;
using hik_metadata_probe::RunningStats;
using hik_metadata_probe::changed;
using hik_metadata_probe::configureCamera;
using hik_metadata_probe::counterStep;
using hik_metadata_probe::grabMeta;
using hik_metadata_probe::layerName;
using hik_metadata_probe::modelOf;
using hik_metadata_probe::openCamera;
using hik_metadata_probe::printIntervalStats;
using hik_metadata_probe::printMetaLine;
using hik_metadata_probe::printNodeReadback;
using hik_metadata_probe::resolveCameraInfo;
using hik_metadata_probe::serialOf;

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args;
    if (!hik_metadata_probe::parseArgs(argc, argv, args)) return 2;
    if (args.help_requested) return 0;
    if (!hik_metadata_probe::validateArgs(args)) return 2;

    MV_CC_DEVICE_INFO_LIST list {};
    const int enum_status = MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE, &list);
    if (enum_status != MV_OK || list.nDeviceNum == 0) {
        std::fprintf(
            stderr,
            "EnumDevices failed ret=0x%X count=%u\n",
            enum_status,
            list.nDeviceNum);
        return 1;
    }
    std::printf("Enumerated %u devices\n", list.nDeviceNum);
    for (unsigned int index = 0; index < list.nDeviceNum; ++index) {
        std::printf(
            "  [%u] SN=%s model=%s layer=%s\n",
            index,
            serialOf(list.pDeviceInfo[index]).c_str(),
            modelOf(list.pDeviceInfo[index]).c_str(),
            layerName(list.pDeviceInfo[index]));
    }
    if (list.nDeviceNum < 2) {
        std::fprintf(
            stderr,
            "ERROR: stereo metadata probe requires two enumerated cameras, "
            "got %u.\nCheck USB connection/power, close MVS or any process "
            "holding a camera, then rerun the probe.\n",
            list.nDeviceNum);
        return 1;
    }

    const MV_CC_DEVICE_INFO* left_info = resolveCameraInfo(
        list, args.left_sn, args.left_index);
    const MV_CC_DEVICE_INFO* right_info = resolveCameraInfo(
        list, args.right_sn, args.right_index);
    if (left_info == nullptr || right_info == nullptr) {
        std::fprintf(
            stderr,
            "ERROR: cannot resolve stereo roles: left sn='%s' index=%d, "
            "right sn='%s' index=%d\n",
            args.left_sn.c_str(),
            args.left_index,
            args.right_sn.c_str(),
            args.right_index);
        return 1;
    }
    const std::string left_serial = serialOf(left_info);
    const std::string right_serial = serialOf(right_info);
    if (left_info == right_info || left_serial.empty() ||
        right_serial.empty() || left_serial == right_serial) {
        std::fprintf(
            stderr,
            "ERROR: left/right must resolve to two distinct cameras with "
            "non-empty serial numbers (left='%s', right='%s')\n",
            left_serial.c_str(),
            right_serial.c_str());
        return 1;
    }

    CameraSession left;
    CameraSession right;
    if (!openCamera(list, args.left_sn, args.left_index, "LEFT", left) ||
        !openCamera(list, args.right_sn, args.right_index, "RIGHT", right)) {
        return 1;
    }
    if (!configureCamera(left.handle(), args, "LEFT") ||
        !configureCamera(right.handle(), args, "RIGHT")) {
        return 1;
    }

    std::puts("\nNode readback after configuration:");
    printNodeReadback(left.handle(), "LEFT");
    printNodeReadback(right.handle(), "RIGHT");

    if (!left.startGrabbing("LEFT") || !right.startGrabbing("RIGHT")) {
        return 1;
    }

    std::unique_ptr<stereo3d::PWMTrigger> pwm;
    if (args.start_pwm) {
        pwm = std::make_unique<stereo3d::PWMTrigger>(
            args.trigger_chip, args.trigger_line, args.pwm_hz, 50.0);
        if (!pwm->start()) {
            std::fprintf(
                stderr,
                "PWM start failed; cannot verify FrameCounter-to-GPIO "
                "mapping\n");
            return 1;
        }
    }

    int ok_pairs = 0;
    int fail_pairs = 0;
    int frame_delta_changes = 0;
    int counter_delta_changes = 0;
    int trigger_delta_changes = 0;
    int trigger_nonzero_or_changing = 0;
    int counter_nonzero_or_changing = 0;
    int pwm_sequence_matches = 0;
    int pwm_sequence_missing = 0;
    int left_frame_num_jumps = 0;
    int right_frame_num_jumps = 0;
    int left_frame_counter_jumps = 0;
    int right_frame_counter_jumps = 0;
    int left_trigger_index_jumps = 0;
    int right_trigger_index_jumps = 0;
    std::uint32_t left_max_frame_num_step = 0;
    std::uint32_t right_max_frame_num_step = 0;
    std::uint32_t left_max_frame_counter_step = 0;
    std::uint32_t right_max_frame_counter_step = 0;
    std::uint32_t left_max_trigger_index_step = 0;
    std::uint32_t right_max_trigger_index_step = 0;
    RunningStats left_dev_ts_step_stats;
    RunningStats right_dev_ts_step_stats;
    RunningStats left_host_ts_step_stats;
    RunningStats right_host_ts_step_stats;
    RunningStats dev_ts_delta_stats;
    std::int64_t first_frame_delta = 0;
    std::int64_t first_counter_delta = 0;
    std::int64_t first_trigger_delta = 0;
    bool have_first = false;
    FrameMeta previous_left;
    FrameMeta previous_right;

    std::puts("\nFrame metadata:");
    for (int index = 0; index < args.frames; ++index) {
        FrameMeta left_meta;
        FrameMeta right_meta;
        std::string worker_error;
        const bool workers_ok = legacy_nx_tools::runIndexedTasks(
            2,
            2,
            [&](std::size_t camera_index) {
                if (camera_index == 0) {
                    left_meta = grabMeta(left.handle(), args.timeout_ms);
                } else {
                    right_meta = grabMeta(right.handle(), args.timeout_ms);
                }
            },
            worker_error);
        if (!workers_ok) {
            std::fprintf(
                stderr,
                "Frame acquisition worker failed: %s\n",
                worker_error.c_str());
            ++fail_pairs;
            break;
        }

        const bool print_this = index < 30 ||
            !left_meta.ok || !right_meta.ok;
        if (print_this) printMetaLine(index, left_meta, right_meta);

        if (!left_meta.ok || !right_meta.ok) {
            ++fail_pairs;
            continue;
        }
        ++ok_pairs;

        if (pwm) {
            stereo3d::PWMTriggerEdgeEvent edge;
            if (left_meta.frame_counter == right_meta.frame_counter &&
                pwm->eventForSequence(left_meta.frame_counter, edge) &&
                edge.sequence == left_meta.frame_counter) {
                ++pwm_sequence_matches;
            } else {
                ++pwm_sequence_missing;
                std::printf(
                    "    GPIO mapping failure at %04d: Lfc=%u Rfc=%u "
                    "latest_gpio=%llu\n",
                    index,
                    left_meta.frame_counter,
                    right_meta.frame_counter,
                    static_cast<unsigned long long>(
                        pwm->latestEventSequence()));
            }
        }

        if (!have_first) {
            first_frame_delta =
                static_cast<std::int64_t>(left_meta.frame_num) -
                static_cast<std::int64_t>(right_meta.frame_num);
            first_counter_delta =
                static_cast<std::int64_t>(left_meta.frame_counter) -
                static_cast<std::int64_t>(right_meta.frame_counter);
            first_trigger_delta =
                static_cast<std::int64_t>(left_meta.trigger_index) -
                static_cast<std::int64_t>(right_meta.trigger_index);
            have_first = true;
        } else {
            const std::int64_t frame_delta =
                static_cast<std::int64_t>(left_meta.frame_num) -
                static_cast<std::int64_t>(right_meta.frame_num);
            const std::int64_t counter_delta =
                static_cast<std::int64_t>(left_meta.frame_counter) -
                static_cast<std::int64_t>(right_meta.frame_counter);
            const std::int64_t trigger_delta =
                static_cast<std::int64_t>(left_meta.trigger_index) -
                static_cast<std::int64_t>(right_meta.trigger_index);
            if (frame_delta != first_frame_delta) {
                ++frame_delta_changes;
                printMetaLine(index, left_meta, right_meta);
            }
            if (counter_delta != first_counter_delta) {
                ++counter_delta_changes;
                printMetaLine(index, left_meta, right_meta);
            }
            if (trigger_delta != first_trigger_delta) {
                ++trigger_delta_changes;
                printMetaLine(index, left_meta, right_meta);
            }
            if (changed(
                    left_meta.trigger_index,
                    previous_left.trigger_index) ||
                changed(
                    right_meta.trigger_index,
                    previous_right.trigger_index) ||
                left_meta.trigger_index != 0 || right_meta.trigger_index != 0) {
                ++trigger_nonzero_or_changing;
            }
            if (changed(
                    left_meta.frame_counter,
                    previous_left.frame_counter) ||
                changed(
                    right_meta.frame_counter,
                    previous_right.frame_counter) ||
                left_meta.frame_counter != 0 || right_meta.frame_counter != 0) {
                ++counter_nonzero_or_changing;
            }

            const std::uint32_t left_frame_num_step = counterStep(
                previous_left.frame_num, left_meta.frame_num);
            const std::uint32_t right_frame_num_step = counterStep(
                previous_right.frame_num, right_meta.frame_num);
            const std::uint32_t left_frame_counter_step = counterStep(
                previous_left.frame_counter, left_meta.frame_counter);
            const std::uint32_t right_frame_counter_step = counterStep(
                previous_right.frame_counter, right_meta.frame_counter);
            const std::uint32_t left_trigger_index_step = counterStep(
                previous_left.trigger_index, left_meta.trigger_index);
            const std::uint32_t right_trigger_index_step = counterStep(
                previous_right.trigger_index, right_meta.trigger_index);
            if (left_meta.dev_ts >= previous_left.dev_ts) {
                left_dev_ts_step_stats.add(static_cast<double>(
                    left_meta.dev_ts - previous_left.dev_ts));
            }
            if (right_meta.dev_ts >= previous_right.dev_ts) {
                right_dev_ts_step_stats.add(static_cast<double>(
                    right_meta.dev_ts - previous_right.dev_ts));
            }
            if (left_meta.host_ts >= previous_left.host_ts) {
                left_host_ts_step_stats.add(static_cast<double>(
                    left_meta.host_ts - previous_left.host_ts));
            }
            if (right_meta.host_ts >= previous_right.host_ts) {
                right_host_ts_step_stats.add(static_cast<double>(
                    right_meta.host_ts - previous_right.host_ts));
            }
            dev_ts_delta_stats.add(static_cast<double>(
                static_cast<std::int64_t>(left_meta.dev_ts) -
                static_cast<std::int64_t>(right_meta.dev_ts)));
            left_max_frame_num_step = std::max(
                left_max_frame_num_step, left_frame_num_step);
            right_max_frame_num_step = std::max(
                right_max_frame_num_step, right_frame_num_step);
            left_max_frame_counter_step = std::max(
                left_max_frame_counter_step, left_frame_counter_step);
            right_max_frame_counter_step = std::max(
                right_max_frame_counter_step, right_frame_counter_step);
            left_max_trigger_index_step = std::max(
                left_max_trigger_index_step, left_trigger_index_step);
            right_max_trigger_index_step = std::max(
                right_max_trigger_index_step, right_trigger_index_step);
            bool step_anomaly = false;
            if (left_frame_num_step != 1) {
                ++left_frame_num_jumps;
                step_anomaly = true;
            }
            if (right_frame_num_step != 1) {
                ++right_frame_num_jumps;
                step_anomaly = true;
            }
            if (left_frame_counter_step != 1) {
                ++left_frame_counter_jumps;
                step_anomaly = true;
            }
            if (right_frame_counter_step != 1) {
                ++right_frame_counter_jumps;
                step_anomaly = true;
            }
            if (left_trigger_index_step != 1) {
                ++left_trigger_index_jumps;
                step_anomaly = true;
            }
            if (right_trigger_index_step != 1) {
                ++right_trigger_index_jumps;
                step_anomaly = true;
            }
            if (step_anomaly) {
                std::printf(
                    "    step anomaly at %04d: L# +%u R# +%u  "
                    "Lfc +%u Rfc +%u  Ltr +%u Rtr +%u\n",
                    index,
                    left_frame_num_step,
                    right_frame_num_step,
                    left_frame_counter_step,
                    right_frame_counter_step,
                    left_trigger_index_step,
                    right_trigger_index_step);
            }
        }
        previous_left = left_meta;
        previous_right = right_meta;
    }

    // 先停止产生触发边沿，再释放任一相机；提前返回也由 RAII 完成清理。
    if (pwm) pwm->stop();
    left.shutdown();
    right.shutdown();

    std::puts("\nSummary:");
    std::printf("  ok_pairs=%d fail_pairs=%d\n", ok_pairs, fail_pairs);
    std::printf(
        "  first_frame_delta=%ld changes=%d\n",
        static_cast<long>(first_frame_delta),
        frame_delta_changes);
    std::printf(
        "  first_frame_counter_delta=%ld changes=%d "
        "nonzero_or_changing_samples=%d\n",
        static_cast<long>(first_counter_delta),
        counter_delta_changes,
        counter_nonzero_or_changing);
    std::printf(
        "  first_trigger_delta=%ld changes=%d "
        "nonzero_or_changing_samples=%d\n",
        static_cast<long>(first_trigger_delta),
        trigger_delta_changes,
        trigger_nonzero_or_changing);
    std::printf(
        "  per-camera jumps: frame_num L/R=%d/%d max_step=%u/%u\n",
        left_frame_num_jumps,
        right_frame_num_jumps,
        left_max_frame_num_step,
        right_max_frame_num_step);
    std::printf(
        "  per-camera jumps: frame_counter L/R=%d/%d max_step=%u/%u\n",
        left_frame_counter_jumps,
        right_frame_counter_jumps,
        left_max_frame_counter_step,
        right_max_frame_counter_step);
    std::printf(
        "  per-camera jumps: trigger_index L/R=%d/%d max_step=%u/%u\n",
        left_trigger_index_jumps,
        right_trigger_index_jumps,
        left_max_trigger_index_step,
        right_max_trigger_index_step);
    if (args.start_pwm) {
        std::printf(
            "  FrameCounter-to-GPIO mapping: matches=%d missing=%d\n",
            pwm_sequence_matches,
            pwm_sequence_missing);
    }
    printIntervalStats("left_dev_timestamp_step", left_dev_ts_step_stats);
    printIntervalStats("right_dev_timestamp_step", right_dev_ts_step_stats);
    printIntervalStats("left_host_timestamp_step", left_host_ts_step_stats);
    printIntervalStats("right_host_timestamp_step", right_host_ts_step_stats);
    printIntervalStats("left_minus_right_dev_timestamp", dev_ts_delta_stats);
    std::printf(
        "  verdict: frame_counter=%s trigger_index=%s\n",
        counter_nonzero_or_changing > 0 ? "usable-looking" : "not populated",
        trigger_nonzero_or_changing > 0 ? "usable-looking" : "not populated");
    const bool frame_counter_ok = ok_pairs > 0 && fail_pairs == 0 &&
        first_counter_delta == 0 && counter_delta_changes == 0 &&
        left_frame_counter_jumps == 0 && right_frame_counter_jumps == 0;
    const bool pwm_mapping_ok = !args.start_pwm ||
        (pwm_sequence_missing == 0 && pwm_sequence_matches == ok_pairs);
    if (!frame_counter_ok || !pwm_mapping_ok) {
        std::fprintf(
            stderr,
            "FAIL: frame_counter_ok=%d pwm_mapping_ok=%d; "
            "do not start capture\n",
            frame_counter_ok ? 1 : 0,
            pwm_mapping_ok ? 1 : 0);
        return 1;
    }
    return 0;
}
