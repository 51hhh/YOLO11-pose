#pragma once

#include "volleyball/core/ids.hpp"
#include "volleyball/core/math_types.hpp"
#include "volleyball/core/time.hpp"

#include <optional>
#include <string>
#include <vector>

namespace volleyball::core {

enum class CameraRole { left, right };

enum class PixelFormat { bayer8, gray8, bgr8, rgb8 };

struct RegionOfInterest {
    std::uint32_t x{0};
    std::uint32_t y{0};
    std::uint32_t width{0};
    std::uint32_t height{0};
};

// 该类型只表达一帧的采集事实，不携带检测、深度或滤波结果。
struct CameraFrameRecord {
    CameraId camera_id;
    CameraRole role{CameraRole::left};
    FrameIndex frame_index{0};
    Timestamp hardware_timestamp;
    std::optional<Timestamp> exposure_start;
    Timestamp capture_timestamp;
    std::optional<Timestamp> receive_timestamp;
    std::optional<Timestamp> inference_done_timestamp;
    std::optional<double> exposure_us;
    std::optional<double> gain_db;
    RegionOfInterest roi;
    PixelFormat pixel_format{PixelFormat::bayer8};
    bool dropped{false};
    std::optional<std::string> drop_reason;
};

struct FrameReference {
    CameraId camera_id;
    CameraRole role{CameraRole::left};
    FrameIndex frame_index{0};
};

// 一个 detector 候选，无论最终是否被选中都应记录。
struct Detection2D {
    FrameReference frame;
    CandidateId candidate_id{0};
    Vec2 center_px;
    std::optional<double> radius_px;
    std::optional<BoundingBox2D> bounding_box;
    std::optional<Vec2> pixel_velocity_pxps;
    std::optional<double> score;
    std::optional<Covariance2D> covariance_px2;
    bool selected{false};
    PipelineId source_pipeline;
};

enum class AssociationDecision { accepted, rejected };

enum class AssociationRejectReason {
    none,
    class_mismatch,
    nonpositive_disparity,
    disparity_out_of_range,
    epipolar_error,
    size_ratio,
    shifted_iou,
    temporal_mismatch,
    triangulation_failure,
    quality_gate,
};

// 所有左右候选配对尝试都使用该类型，被拒绝项不会被丢弃。
struct StereoAssociationAttempt {
    PairId pair_id{0};
    FrameIndex left_frame_index{0};
    FrameIndex right_frame_index{0};
    CandidateId left_candidate_id{0};
    CandidateId right_candidate_id{0};
    Timestamp left_capture;
    Timestamp right_capture;
    std::int64_t delta_t_lr_ns{0};
    std::optional<double> disparity_px;
    std::optional<double> epipolar_error_px;
    AssociationDecision decision{AssociationDecision::rejected};
    AssociationRejectReason reject_reason{AssociationRejectReason::quality_gate};
};

// 直接三角化观测：这是 trajectory.csv 唯一允许的 3D 数据语义。
struct TriangulatedObservation {
    ObservationId observation_id{0};
    PairId source_pair_id{0};
    Timestamp capture_reference;
    CoordinateFrameId coordinate_frame;
    Vec3 position_m;
    std::optional<Covariance3D> covariance_m2;
    std::optional<double> confidence;
    std::optional<double> reprojection_error_px;
};

enum class ValidationStatus { accepted, rejected };

enum class ObservationRejectReason {
    none,
    nonfinite,
    range,
    covariance,
    reprojection,
    temporal_continuity,
    innovation,
};

// 验证结果包含原直接观测，但不修改它。
struct ValidatedObservation {
    TriangulatedObservation measurement;
    ValidationStatus status{ValidationStatus::rejected};
    std::vector<ObservationRejectReason> reasons;
};

// 派生状态与直接观测是不同类型，不能写入 trajectory.csv。
struct TrackEstimate {
    TrackId track_id{0};
    Timestamp state_time;
    CoordinateFrameId coordinate_frame;
    Vec3 position_m;
    Vec3 velocity_mps;
    Vec3 acceleration_mps2;
    std::optional<Covariance3D> position_covariance_m2;
};

struct LandingPrediction {
    TrackId source_track_id{0};
    Timestamp prediction_time;
    CoordinateFrameId coordinate_frame;
    Vec3 landing_position_m;
    double time_to_land_s{0.0};
    std::optional<Covariance3D> landing_covariance_m2;
};

enum class DomainError {
    none,
    missing_identifier,
    clock_domain_mismatch,
    timestamp_order,
    invalid_roi,
    missing_drop_reason,
    invalid_numeric_value,
    invalid_decision_reason,
};

struct DomainValidationResult {
    DomainError error{DomainError::none};
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == DomainError::none;
    }
};

[[nodiscard]] DomainValidationResult validate(const CameraFrameRecord& frame) noexcept;
[[nodiscard]] DomainValidationResult validate(const Detection2D& detection) noexcept;
[[nodiscard]] DomainValidationResult validate(
    const StereoAssociationAttempt& attempt) noexcept;
[[nodiscard]] DomainValidationResult validate(
    const TriangulatedObservation& observation) noexcept;

}  // namespace volleyball::core

