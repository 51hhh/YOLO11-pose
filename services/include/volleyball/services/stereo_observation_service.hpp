#pragma once

#include "volleyball/algorithms/stereo/association.hpp"
#include "volleyball/algorithms/stereo/triangulation.hpp"
#include "volleyball/ports/detector.hpp"
#include "volleyball/ports/observation_publisher.hpp"
#include "volleyball/ports/trial_writer.hpp"

#include <optional>
#include <string>
#include <vector>

namespace volleyball::services {

struct ObservationGateConfig {
    double min_depth_m{0.0};
    double max_depth_m{100.0};
    double max_reprojection_error_px{5.0};
};

struct StereoObservationServiceConfig {
    algorithms::stereo::AssociationConfig association;
    algorithms::stereo::TriangulationConfig triangulation;
    core::RectifiedStereoCalibration calibration;
    ObservationGateConfig gate;
};

enum class ProcessStatus {
    observation,
    no_match,
    detector_error,
    invalid_input,
    write_error,
    publish_error,
};

struct StereoProcessResult {
    ProcessStatus status{ProcessStatus::invalid_input};
    std::optional<core::TriangulatedObservation> observation;
    std::vector<algorithms::stereo::ScoredAssociation> associations;
    std::string message;

    [[nodiscard]] bool ok() const noexcept {
        return status == ProcessStatus::observation ||
               status == ProcessStatus::no_match;
    }
};

// 用例层只组合端口与纯算法。MVS/CUDA/TensorRT/ROS2 必须由外层
// adapter 注入，不得在此层出现。
class StereoObservationService {
public:
    StereoObservationService(
        StereoObservationServiceConfig config,
        ports::Detector& left_detector,
        ports::Detector& right_detector,
        ports::TrialWriter& writer,
        ports::ObservationPublisher* publisher = nullptr);

    [[nodiscard]] StereoProcessResult process(
        const ports::StereoFrame& frame);
    [[nodiscard]] ports::WriteResult finalize();
    void abort(std::string reason) noexcept;

private:
    [[nodiscard]] StereoProcessResult fail(
        ProcessStatus status,
        std::string message) noexcept;
    [[nodiscard]] bool appendDetections(
        const std::vector<core::Detection2D>& detections,
        std::string& error);
    [[nodiscard]] bool appendAssociations(
        const std::vector<algorithms::stereo::ScoredAssociation>& associations,
        std::string& error);
    [[nodiscard]] bool validateDetectionBatch(
        const ports::DetectionBatch& batch,
        const core::CameraFrameRecord& frame,
        std::string& error) const;
    [[nodiscard]] bool passesGate(
        const core::TriangulatedObservation& observation) const noexcept;

    StereoObservationServiceConfig config_;
    ports::Detector& left_detector_;
    ports::Detector& right_detector_;
    ports::TrialWriter& writer_;
    ports::ObservationPublisher* publisher_;
    core::PairId next_pair_id_{0};
    core::ObservationId next_observation_id_{0};
    bool closed_{false};
};

}  // namespace volleyball::services
