#include "volleyball/services/stereo_observation_service.hpp"

#include <cmath>
#include <utility>

namespace volleyball::services {
namespace {

bool writeAccepted(const ports::WriteResult& result, std::string& error) {
    if (result.status == ports::WriteStatus::accepted) {
        return true;
    }
    error = result.message.empty() ? "trial writer rejected a record" : result.message;
    return false;
}

const core::Detection2D* findCandidate(
    const std::vector<core::Detection2D>& candidates,
    core::CandidateId candidate_id) noexcept {
    for (const auto& candidate : candidates) {
        if (candidate.candidate_id == candidate_id) {
            return &candidate;
        }
    }
    return nullptr;
}

}  // namespace

StereoObservationService::StereoObservationService(
    StereoObservationServiceConfig config,
    ports::Detector& left_detector,
    ports::Detector& right_detector,
    ports::TrialWriter& writer,
    ports::ObservationPublisher* publisher)
    : config_(std::move(config)),
      left_detector_(left_detector),
      right_detector_(right_detector),
      writer_(writer),
      publisher_(publisher) {}

StereoProcessResult StereoObservationService::fail(
    ProcessStatus status,
    std::string message) noexcept {
    writer_.abort(message);
    closed_ = true;
    return {status, std::nullopt, {}, std::move(message)};
}

bool StereoObservationService::appendDetections(
    const std::vector<core::Detection2D>& detections,
    std::string& error) {
    for (const auto& detection : detections) {
        if (!writeAccepted(writer_.append(detection), error)) {
            return false;
        }
    }
    return true;
}

bool StereoObservationService::passesGate(
    const core::TriangulatedObservation& observation) const noexcept {
    const double depth = observation.position_m.z;
    const double reprojection = observation.reprojection_error_px.value_or(0.0);
    return std::isfinite(depth) && depth >= config_.gate.min_depth_m &&
           depth <= config_.gate.max_depth_m &&
           std::isfinite(reprojection) &&
           reprojection <= config_.gate.max_reprojection_error_px;
}

StereoProcessResult StereoObservationService::process(
    const ports::StereoFrame& frame) {
    if (closed_) {
        return {ProcessStatus::write_error, std::nullopt, {}, "service is closed"};
    }
    const auto left_valid = core::validate(frame.left.metadata);
    const auto right_valid = core::validate(frame.right.metadata);
    if (!left_valid || !right_valid || !frame.left.image || !frame.right.image ||
        frame.left.metadata.role != core::CameraRole::left ||
        frame.right.metadata.role != core::CameraRole::right) {
        return fail(ProcessStatus::invalid_input, "invalid left/right stereo frame");
    }
    if (frame.left.metadata.capture_timestamp.clock_domain !=
            frame.right.metadata.capture_timestamp.clock_domain ||
        frame.left.metadata.capture_timestamp.nanoseconds !=
            frame.right.metadata.capture_timestamp.nanoseconds) {
        return fail(
            ProcessStatus::invalid_input,
            "hardware-triggered stereo frames must share one exposure timestamp");
    }

    std::string error;
    if (!writeAccepted(writer_.append(frame.left.metadata), error) ||
        !writeAccepted(writer_.append(frame.right.metadata), error)) {
        return fail(ProcessStatus::write_error, std::move(error));
    }

    auto left_result = left_detector_.detect(frame.left);
    auto right_result = right_detector_.detect(frame.right);
    if (!left_result.success || !right_result.success) {
        const std::string message = !left_result.success
            ? left_result.message
            : right_result.message;
        return fail(
            ProcessStatus::detector_error,
            message.empty() ? "stereo detector failed" : message);
    }
    if (!appendDetections(left_result.batch.candidates, error) ||
        !appendDetections(right_result.batch.candidates, error)) {
        return fail(ProcessStatus::write_error, std::move(error));
    }

    auto associations = algorithms::stereo::associateCandidates(
        left_result.batch.candidates,
        right_result.batch.candidates,
        frame.left.metadata.capture_timestamp,
        frame.right.metadata.capture_timestamp,
        config_.association,
        next_pair_id_);
    next_pair_id_ += associations.attempts.size();
    for (const auto& scored : associations.attempts) {
        if (!writeAccepted(writer_.append(scored.attempt), error)) {
            return fail(ProcessStatus::write_error, std::move(error));
        }
    }
    if (!associations.best_accepted_index) {
        return {
            ProcessStatus::no_match,
            std::nullopt,
            std::move(associations.attempts),
            "no stereo association passed all gates",
        };
    }

    const auto& selected =
        associations.attempts[*associations.best_accepted_index].attempt;
    const auto* left_candidate = findCandidate(
        left_result.batch.candidates, selected.left_candidate_id);
    const auto* right_candidate = findCandidate(
        right_result.batch.candidates, selected.right_candidate_id);
    if (!left_candidate || !right_candidate) {
        return fail(ProcessStatus::invalid_input, "selected candidate disappeared");
    }
    auto triangulated = algorithms::stereo::triangulateRectifiedCenters(
        *left_candidate,
        *right_candidate,
        selected,
        config_.calibration,
        config_.triangulation,
        next_observation_id_++);
    if (!triangulated || !passesGate(*triangulated.observation)) {
        return {
            ProcessStatus::no_match,
            std::nullopt,
            std::move(associations.attempts),
            triangulated ? "triangulated observation failed quality gate"
                         : triangulated.message,
        };
    }
    const auto& observation = *triangulated.observation;
    if (!writeAccepted(writer_.append(observation), error)) {
        return fail(ProcessStatus::write_error, std::move(error));
    }
    if (publisher_) {
        core::ValidatedObservation validated;
        validated.measurement = observation;
        validated.status = core::ValidationStatus::accepted;
        const auto published = publisher_->publish(validated);
        if (!published.success) {
            return fail(
                ProcessStatus::publish_error,
                published.message.empty() ? "observation publish failed"
                                          : published.message);
        }
    }
    return {
        ProcessStatus::observation,
        observation,
        std::move(associations.attempts),
        {},
    };
}

ports::WriteResult StereoObservationService::finalize() {
    if (closed_) {
        return {ports::WriteStatus::failed, "service is closed"};
    }
    const auto result = writer_.finalize();
    if (result.status == ports::WriteStatus::accepted) {
        closed_ = true;
    }
    return result;
}

}  // namespace volleyball::services
