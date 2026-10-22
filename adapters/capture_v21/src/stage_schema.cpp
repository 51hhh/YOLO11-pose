#include "volleyball/adapters/capture_v21/stage_schema.hpp"

#include <iomanip>
#include <ostream>

namespace volleyball::adapters::capture_v21 {
namespace {

template <std::size_t Size>
void writeHeader(
    std::ostream& output,
    const std::array<std::string_view, Size>& columns) {
    for (std::size_t index = 0; index < columns.size(); ++index) {
        if (index != 0) output << ',';
        output << columns[index];
    }
    output << '\n';
}

bool isCsvToken(std::string_view value) {
    return value.find_first_of(",\r\n") == std::string_view::npos;
}

}  // namespace

void writeCaptureV21FrameHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21FrameColumns);
}

void writeCaptureV21DetectionHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21DetectionColumns);
}

void writeCaptureV21PairHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21PairColumns);
}

bool writeCaptureV21FrameRow(
    std::ostream& output, const CaptureV21FrameRow& row) {
    if (!isCsvToken(row.raw_left) || !isCsvToken(row.raw_right)) return false;
    output
        << row.frame_id << ','
        << row.host_capture_timestamp_ns << ','
        << row.callback_timestamp_ns << ','
        << row.left_receive_monotonic_ns << ','
        << row.right_receive_monotonic_ns << ','
        << row.left_receive_realtime_ns << ','
        << row.right_receive_realtime_ns << ','
        << row.left_sdk_host_timestamp_raw << ','
        << row.right_sdk_host_timestamp_raw << ','
        << row.left_device_timestamp_ns << ','
        << row.right_device_timestamp_ns << ','
        << row.stereo_timestamp_residual_ns << ','
        << row.trigger_sequence << ','
        << row.trigger_scheduled_monotonic_ns << ','
        << row.trigger_edge_monotonic_ns << ','
        << row.trigger_edge_realtime_ns << ','
        << row.trigger_write_uncertainty_ns << ','
        << row.left_frame_number << ','
        << row.right_frame_number << ','
        << row.left_frame_counter << ','
        << row.right_frame_counter << ','
        << row.left_trigger_index << ','
        << row.right_trigger_index << ','
        << row.raw_left << ','
        << row.raw_right << '\n';
    return static_cast<bool>(output);
}

bool writeCaptureV21DetectionRow(
    std::ostream& output, const CaptureV21DetectionRow& row) {
    if ((row.camera_role != "left" && row.camera_role != "right") ||
        !isCsvToken(row.camera_role)) {
        return false;
    }
    output
        << row.frame_id << ','
        << row.camera_role << ','
        << row.candidate_id << ','
        << std::setprecision(9)
        << row.cx_px << ','
        << row.cy_px << ','
        << row.width_px << ','
        << row.height_px << ','
        << row.confidence << ','
        << row.class_id << ','
        << (row.selected ? "true" : "false") << '\n';
    return static_cast<bool>(output);
}

bool writeCaptureV21PairRow(
    std::ostream& output, const CaptureV21PairRow& row) {
    if (row.accepted != row.position_rig_m.has_value() ||
        (row.accepted && !row.reject_reason.empty()) ||
        (!row.accepted && row.reject_reason.empty()) ||
        !isCsvToken(row.reject_reason)) {
        return false;
    }
    output
        << row.pair_id << ','
        << row.frame_id << ','
        << row.left_candidate_id << ','
        << row.right_candidate_id << ','
        << std::setprecision(9)
        << row.disparity_px << ','
        << row.epipolar_error_px << ','
        << row.pair_score << ','
        << row.semantic_confidence << ',';
    if (row.position_rig_m) {
        output
            << (*row.position_rig_m)[0] << ','
            << (*row.position_rig_m)[1] << ','
            << (*row.position_rig_m)[2] << ',';
    } else {
        output << ",,,";
    }
    output
        << (row.accepted ? "true" : "false") << ','
        << row.reject_reason << '\n';
    return static_cast<bool>(output);
}

}  // namespace volleyball::adapters::capture_v21
