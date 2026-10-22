#include "volleyball/adapters/fake/in_memory_trial_writer.hpp"

#include <utility>

namespace volleyball::adapters::fake {

template <typename Record>
ports::WriteResult InMemoryTrialWriter::appendValidated(
    const Record& record,
    std::vector<Record>& destination) {
    if (finalized_ || !abort_reason_.empty()) {
        return {ports::WriteStatus::failed, "writer is no longer open"};
    }
    const auto result = core::validate(record);
    if (!result) {
        return {ports::WriteStatus::failed, result.message};
    }
    destination.push_back(record);
    return {ports::WriteStatus::accepted, {}};
}

ports::WriteResult InMemoryTrialWriter::append(
    const ports::StereoFrame& frame) {
    if (finalized_ || !abort_reason_.empty()) {
        return {ports::WriteStatus::failed, "writer is no longer open"};
    }
    const auto left = core::validate(frame.left.metadata);
    const auto right = core::validate(frame.right.metadata);
    if (!left || !right) {
        return {
            ports::WriteStatus::failed,
            !left ? left.message : right.message,
        };
    }
    if (frame.left.metadata.role != core::CameraRole::left ||
        frame.right.metadata.role != core::CameraRole::right ||
        frame.left.metadata.camera_id == frame.right.metadata.camera_id ||
        frame.left.metadata.frame_index != frame.right.metadata.frame_index) {
        return {ports::WriteStatus::failed, "invalid stereo frame identity"};
    }
    frames_.push_back(frame);
    return {ports::WriteStatus::accepted, {}};
}

ports::WriteResult InMemoryTrialWriter::append(
    const core::Detection2D& record) {
    return appendValidated(record, detections_);
}

ports::WriteResult InMemoryTrialWriter::append(
    const core::StereoAssociationAttempt& record) {
    return appendValidated(record, associations_);
}

ports::WriteResult InMemoryTrialWriter::append(
    const ports::EstimatedDirectObservation& record) {
    if (!ports::isValid(record.evidence)) {
        return {ports::WriteStatus::failed, "invalid observation evidence"};
    }
    const auto result = appendValidated(record.measurement, observations_);
    if (result.status == ports::WriteStatus::accepted) {
        observation_evidence_.push_back(record.evidence);
    }
    return result;
}

ports::WriteResult InMemoryTrialWriter::finalize() {
    if (finalized_ || !abort_reason_.empty()) {
        return {ports::WriteStatus::failed, "writer is no longer open"};
    }
    finalized_ = true;
    return {ports::WriteStatus::accepted, {}};
}

void InMemoryTrialWriter::abort(std::string reason) noexcept {
    if (!finalized_) {
        abort_reason_ = std::move(reason);
    }
}

}  // namespace volleyball::adapters::fake

