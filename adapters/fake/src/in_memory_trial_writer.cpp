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
    const core::CameraFrameRecord& record) {
    return appendValidated(record, frames_);
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
    const core::TriangulatedObservation& record) {
    return appendValidated(record, observations_);
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

