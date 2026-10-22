#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/captured_frame.hpp"
#include "volleyball/ports/direct_observation_estimator.hpp"

#include <array>
#include <iosfwd>
#include <string_view>

namespace volleyball::adapters::capture_v21 {

// v6 为主线端口逐事实写入格式，并补齐每条直接观测实际使用的深度方法
// 证据。它只属于未发布 staging，不改变对外冻结的 capture v2.1.0 版本。
inline constexpr int kCaptureV21MainlineStageSchemaVersion = 6;

inline constexpr std::array<std::string_view, 46>
    kCaptureV21MainlineFrameColumns = {
        "camera_id", "camera_role", "frame_index",
        "hardware_timestamp_ns", "hardware_clock_domain",
        "exposure_start_ns", "exposure_start_clock_domain",
        "capture_timestamp_ns", "capture_clock_domain",
        "receive_timestamp_ns", "receive_clock_domain",
        "inference_done_timestamp_ns", "inference_done_clock_domain",
        "exposure_us", "gain_db", "roi_x", "roi_y", "roi_width",
        "roi_height", "pixel_format", "dropped", "drop_reason",
        "receive_monotonic_ns", "receive_monotonic_clock_domain",
        "sdk_host_timestamp_raw", "sensor_frame_number",
        "sensor_frame_counter", "sensor_trigger_index",
        "stereo_hardware_timestamp_residual_ns", "trigger_sequence",
        "trigger_scheduled_monotonic_ns",
        "trigger_scheduled_monotonic_clock_domain",
        "trigger_edge_monotonic_ns", "trigger_edge_monotonic_clock_domain",
        "trigger_edge_realtime_ns", "trigger_edge_realtime_clock_domain",
        "trigger_write_uncertainty_ns", "capture_timestamp_uncertainty_ns",
        "capture_timestamp_mapping_valid", "image_width", "image_height",
        "image_stride_bytes", "image_size_bytes", "image_pixel_format",
        "image_memory_kind", "raw_image",
    };

inline constexpr std::array<std::string_view, 19>
    kCaptureV21MainlineDetectionColumns = {
        "camera_id", "camera_role", "frame_index", "candidate_id",
        "u_px", "v_px", "radius_px", "bbox_x", "bbox_y", "bbox_w",
        "bbox_h", "du_dt_pxps", "dv_dt_pxps", "score", "cov_uu_px2",
        "cov_uv_px2", "cov_vv_px2", "selected", "source",
    };

inline constexpr std::array<std::string_view, 14>
    kCaptureV21MainlineAssociationColumns = {
        "pair_id", "left_frame_index", "right_frame_index",
        "left_candidate_id", "right_candidate_id", "left_capture_ns",
        "left_capture_clock_domain", "right_capture_ns",
        "right_capture_clock_domain", "delta_t_lr_ns", "disparity_px",
        "epipolar_error_px", "decision", "reject_reason",
    };

inline constexpr std::array<std::string_view, 20>
    kCaptureV21MainlineObservationColumns = {
        "observation_id", "source_pair_id", "capture_reference_ns",
        "capture_reference_clock_domain", "coordinate_frame", "x_m", "y_m",
        "z_m", "cov_xx_m2", "cov_xy_m2", "cov_xz_m2", "cov_yy_m2",
        "cov_yz_m2", "cov_zz_m2", "confidence", "reprojection_error_px",
        "depth_method", "measurement_disparity_px",
        "disparity_stddev_px", "support_points",
    };

void writeCaptureV21MainlineFrameHeader(std::ostream& output);
void writeCaptureV21MainlineDetectionHeader(std::ostream& output);
void writeCaptureV21MainlineAssociationHeader(std::ostream& output);
void writeCaptureV21MainlineObservationHeader(std::ostream& output);

[[nodiscard]] bool writeCaptureV21MainlineFrameRow(
    std::ostream& output,
    const ports::CapturedFrame& frame,
    std::string_view raw_image);
[[nodiscard]] bool writeCaptureV21MainlineDetectionRow(
    std::ostream& output,
    const core::Detection2D& detection);
[[nodiscard]] bool writeCaptureV21MainlineAssociationRow(
    std::ostream& output,
    const core::StereoAssociationAttempt& association);
[[nodiscard]] bool writeCaptureV21MainlineObservationRow(
    std::ostream& output,
    const ports::EstimatedDirectObservation& observation);

}  // namespace volleyball::adapters::capture_v21
