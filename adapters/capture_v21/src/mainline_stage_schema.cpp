#include "volleyball/adapters/capture_v21/mainline_stage_schema.hpp"

#include <iomanip>
#include <optional>
#include <ostream>
#include <string_view>

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

template <typename Value>
void writeOptional(std::ostream& output, const std::optional<Value>& value) {
    if (value) output << *value;
}

void writeOptionalTimestamp(
    std::ostream& output,
    const std::optional<core::Timestamp>& value) {
    if (value) output << value->nanoseconds;
}

void writeOptionalClock(
    std::ostream& output,
    const std::optional<core::Timestamp>& value) {
    if (value) output << value->clock_domain.value();
}

std::string_view cameraRole(core::CameraRole role) noexcept {
    return role == core::CameraRole::left ? "left" : "right";
}

std::string_view pixelFormat(core::PixelFormat format) noexcept {
    switch (format) {
    case core::PixelFormat::bayer8:
        return "bayer8";
    case core::PixelFormat::gray8:
        return "gray8";
    case core::PixelFormat::bgr8:
        return "bgr8";
    case core::PixelFormat::rgb8:
        return "rgb8";
    }
    return "unknown";
}

std::string_view memoryKind(ports::MemoryKind kind) noexcept {
    switch (kind) {
    case ports::MemoryKind::host:
        return "host";
    case ports::MemoryKind::host_pinned:
        return "host_pinned";
    case ports::MemoryKind::cuda_device:
        return "cuda_device";
    case ports::MemoryKind::external:
        return "external";
    }
    return "unknown";
}

std::string_view rejection(core::AssociationRejectReason reason) noexcept {
    switch (reason) {
    case core::AssociationRejectReason::none:
        return "";
    case core::AssociationRejectReason::class_mismatch:
        return "class_mismatch";
    case core::AssociationRejectReason::nonpositive_disparity:
        return "nonpositive_disparity";
    case core::AssociationRejectReason::disparity_out_of_range:
        return "disparity_out_of_range";
    case core::AssociationRejectReason::epipolar_error:
        return "epipolar_error";
    case core::AssociationRejectReason::size_ratio:
        return "size_ratio";
    case core::AssociationRejectReason::shifted_iou:
        return "shifted_iou";
    case core::AssociationRejectReason::temporal_mismatch:
        return "temporal_mismatch";
    case core::AssociationRejectReason::not_selected:
        return "not_selected";
    case core::AssociationRejectReason::triangulation_failure:
        return "triangulation_failure";
    case core::AssociationRejectReason::quality_gate:
        return "quality_gate";
    }
    return "unknown";
}

bool validOptionalClock(const std::optional<core::Timestamp>& value) {
    return !value || isCsvToken(value->clock_domain.value());
}

bool validFrameTokens(
    const ports::CapturedFrame& frame,
    std::string_view raw_image) {
    const auto& metadata = frame.metadata;
    const auto& acquisition = metadata.acquisition;
    return !raw_image.empty() && isCsvToken(raw_image) &&
           isCsvToken(metadata.camera_id.value()) &&
           isCsvToken(metadata.hardware_timestamp.clock_domain.value()) &&
           isCsvToken(metadata.capture_timestamp.clock_domain.value()) &&
           (!metadata.drop_reason || isCsvToken(*metadata.drop_reason)) &&
           validOptionalClock(metadata.exposure_start) &&
           validOptionalClock(metadata.receive_timestamp) &&
           validOptionalClock(metadata.inference_done_timestamp) &&
           validOptionalClock(acquisition.receive_monotonic_timestamp) &&
           validOptionalClock(acquisition.trigger_scheduled_monotonic) &&
           validOptionalClock(acquisition.trigger_edge_monotonic) &&
           validOptionalClock(acquisition.trigger_edge_realtime);
}

}  // namespace

void writeCaptureV21MainlineFrameHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21MainlineFrameColumns);
}

void writeCaptureV21MainlineDetectionHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21MainlineDetectionColumns);
}

void writeCaptureV21MainlineAssociationHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21MainlineAssociationColumns);
}

void writeCaptureV21MainlineObservationHeader(std::ostream& output) {
    writeHeader(output, kCaptureV21MainlineObservationColumns);
}

bool writeCaptureV21MainlineFrameRow(
    std::ostream& output,
    const ports::CapturedFrame& frame,
    std::string_view raw_image) {
    if (!frame.image || !validFrameTokens(frame, raw_image)) return false;
    const auto& value = frame.metadata;
    const auto& acquisition = value.acquisition;
    output << std::setprecision(17)
           << value.camera_id.value() << ',' << cameraRole(value.role) << ','
           << value.frame_index << ',' << value.hardware_timestamp.nanoseconds
           << ',' << value.hardware_timestamp.clock_domain.value() << ',';
    writeOptionalTimestamp(output, value.exposure_start);
    output << ',';
    writeOptionalClock(output, value.exposure_start);
    output << ',' << value.capture_timestamp.nanoseconds << ','
           << value.capture_timestamp.clock_domain.value() << ',';
    writeOptionalTimestamp(output, value.receive_timestamp);
    output << ',';
    writeOptionalClock(output, value.receive_timestamp);
    output << ',';
    writeOptionalTimestamp(output, value.inference_done_timestamp);
    output << ',';
    writeOptionalClock(output, value.inference_done_timestamp);
    output << ',';
    writeOptional(output, value.exposure_us);
    output << ',';
    writeOptional(output, value.gain_db);
    output << ',' << value.roi.x << ',' << value.roi.y << ','
           << value.roi.width << ',' << value.roi.height << ','
           << pixelFormat(value.pixel_format) << ','
           << (value.dropped ? "true" : "false") << ','
           << value.drop_reason.value_or("") << ',';
    writeOptionalTimestamp(output, acquisition.receive_monotonic_timestamp);
    output << ',';
    writeOptionalClock(output, acquisition.receive_monotonic_timestamp);
    output << ',';
    writeOptional(output, acquisition.sdk_host_timestamp_raw);
    output << ',';
    writeOptional(output, acquisition.sensor_frame_number);
    output << ',';
    writeOptional(output, acquisition.sensor_frame_counter);
    output << ',';
    writeOptional(output, acquisition.sensor_trigger_index);
    output << ',';
    writeOptional(output, acquisition.stereo_hardware_timestamp_residual_ns);
    output << ',';
    writeOptional(output, acquisition.trigger_sequence);
    output << ',';
    writeOptionalTimestamp(output, acquisition.trigger_scheduled_monotonic);
    output << ',';
    writeOptionalClock(output, acquisition.trigger_scheduled_monotonic);
    output << ',';
    writeOptionalTimestamp(output, acquisition.trigger_edge_monotonic);
    output << ',';
    writeOptionalClock(output, acquisition.trigger_edge_monotonic);
    output << ',';
    writeOptionalTimestamp(output, acquisition.trigger_edge_realtime);
    output << ',';
    writeOptionalClock(output, acquisition.trigger_edge_realtime);
    output << ',';
    writeOptional(output, acquisition.trigger_write_uncertainty_ns);
    output << ',';
    writeOptional(output, acquisition.capture_timestamp_uncertainty_ns);
    output << ','
           << (acquisition.capture_timestamp_mapping_valid ? "true" : "false")
           << ',' << frame.image->width() << ',' << frame.image->height() << ','
           << frame.image->strideBytes() << ',' << frame.image->sizeBytes() << ','
           << pixelFormat(frame.image->pixelFormat()) << ','
           << memoryKind(frame.image->memoryKind()) << ',' << raw_image << '\n';
    return static_cast<bool>(output);
}

bool writeCaptureV21MainlineDetectionRow(
    std::ostream& output,
    const core::Detection2D& value) {
    if (!isCsvToken(value.frame.camera_id.value()) ||
        !isCsvToken(value.source_pipeline.value())) {
        return false;
    }
    output << std::setprecision(17) << value.frame.camera_id.value() << ','
           << cameraRole(value.frame.role) << ',' << value.frame.frame_index << ','
           << value.candidate_id << ',' << value.center_px.x << ','
           << value.center_px.y << ',';
    writeOptional(output, value.radius_px);
    output << ',';
    if (value.bounding_box) output << value.bounding_box->x;
    output << ',';
    if (value.bounding_box) output << value.bounding_box->y;
    output << ',';
    if (value.bounding_box) output << value.bounding_box->width;
    output << ',';
    if (value.bounding_box) output << value.bounding_box->height;
    output << ',';
    if (value.pixel_velocity_pxps) output << value.pixel_velocity_pxps->x;
    output << ',';
    if (value.pixel_velocity_pxps) output << value.pixel_velocity_pxps->y;
    output << ',';
    writeOptional(output, value.score);
    output << ',';
    if (value.covariance_px2) output << value.covariance_px2->upper[0];
    output << ',';
    if (value.covariance_px2) output << value.covariance_px2->upper[1];
    output << ',';
    if (value.covariance_px2) output << value.covariance_px2->upper[2];
    output << ',' << (value.selected ? "true" : "false") << ','
           << value.source_pipeline.value() << '\n';
    return static_cast<bool>(output);
}

bool writeCaptureV21MainlineAssociationRow(
    std::ostream& output,
    const core::StereoAssociationAttempt& value) {
    const auto rejection_token = rejection(value.reject_reason);
    if (!isCsvToken(value.left_capture.clock_domain.value()) ||
        !isCsvToken(value.right_capture.clock_domain.value()) ||
        !isCsvToken(rejection_token)) {
        return false;
    }
    output << std::setprecision(17) << value.pair_id << ','
           << value.left_frame_index << ',' << value.right_frame_index << ','
           << value.left_candidate_id << ',' << value.right_candidate_id << ','
           << value.left_capture.nanoseconds << ','
           << value.left_capture.clock_domain.value() << ','
           << value.right_capture.nanoseconds << ','
           << value.right_capture.clock_domain.value() << ','
           << value.delta_t_lr_ns << ',';
    writeOptional(output, value.disparity_px);
    output << ',';
    writeOptional(output, value.epipolar_error_px);
    output << ','
           << (value.decision == core::AssociationDecision::accepted
                   ? "accepted"
                   : "rejected")
           << ',' << rejection_token << '\n';
    return static_cast<bool>(output);
}

bool writeCaptureV21MainlineObservationRow(
    std::ostream& output,
    const ports::EstimatedDirectObservation& record) {
    const auto& value = record.measurement;
    const auto& evidence = record.evidence;
    if (!isCsvToken(value.capture_reference.clock_domain.value()) ||
        !isCsvToken(value.coordinate_frame.value()) ||
        !isCsvToken(evidence.method.value())) {
        return false;
    }
    output << std::setprecision(17) << value.observation_id << ','
           << value.source_pair_id << ',' << value.capture_reference.nanoseconds
           << ',' << value.capture_reference.clock_domain.value() << ','
           << value.coordinate_frame.value() << ',' << value.position_m.x << ','
           << value.position_m.y << ',' << value.position_m.z << ',';
    for (std::size_t index = 0; index < 6; ++index) {
        if (value.covariance_m2) output << value.covariance_m2->upper[index];
        output << ',';
    }
    writeOptional(output, value.confidence);
    output << ',';
    writeOptional(output, value.reprojection_error_px);
    output << ',' << evidence.method.value() << ',' << evidence.disparity_px
           << ',';
    writeOptional(output, evidence.disparity_stddev_px);
    output << ',';
    if (evidence.support_points) output << *evidence.support_points;
    output << '\n';
    return static_cast<bool>(output);
}

}  // namespace volleyball::adapters::capture_v21
