#pragma once

#include "volleyball/ports/trial_writer.hpp"

#include <string>
#include <vector>

namespace volleyball::adapters::fake {

// 用于单元测试和 CPU 回放的内存写入器，不是生产存储实现。
class InMemoryTrialWriter final : public ports::TrialWriter {
public:
    [[nodiscard]] ports::WriteResult append(
        const core::CameraFrameRecord& record) override;
    [[nodiscard]] ports::WriteResult append(
        const core::Detection2D& record) override;
    [[nodiscard]] ports::WriteResult append(
        const core::StereoAssociationAttempt& record) override;
    [[nodiscard]] ports::WriteResult append(
        const core::TriangulatedObservation& record) override;
    [[nodiscard]] ports::WriteResult finalize() override;
    void abort(std::string reason) noexcept override;

    [[nodiscard]] const auto& frames() const noexcept { return frames_; }
    [[nodiscard]] const auto& detections() const noexcept { return detections_; }
    [[nodiscard]] const auto& associations() const noexcept { return associations_; }
    [[nodiscard]] const auto& observations() const noexcept { return observations_; }
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }
    [[nodiscard]] const std::string& abortReason() const noexcept { return abort_reason_; }

private:
    template <typename Record>
    ports::WriteResult appendValidated(
        const Record& record,
        std::vector<Record>& destination);

    std::vector<core::CameraFrameRecord> frames_;
    std::vector<core::Detection2D> detections_;
    std::vector<core::StereoAssociationAttempt> associations_;
    std::vector<core::TriangulatedObservation> observations_;
    bool finalized_{false};
    std::string abort_reason_;
};

}  // namespace volleyball::adapters::fake

