#pragma once

#include "volleyball/ports/trial_writer.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace volleyball::adapters::capture_v21 {

struct MainlineStageTrialWriterConfig {
    // 目录由上层独占创建，并预先写入 context.json。写入器不得创建或覆盖
    // 另一个 trial 的 staging。
    std::filesystem::path stage_directory;
};

// 主线 v5 staging 写入器。它保存逐相机采集事实、全部视觉证据和原始图像，
// 但不直接发布最终 v2.1 trial；发布仍由独立 finalizer 校验后完成。
class MainlineStageTrialWriter final : public ports::TrialWriter {
public:
    explicit MainlineStageTrialWriter(MainlineStageTrialWriterConfig config);
    ~MainlineStageTrialWriter() override;

    MainlineStageTrialWriter(const MainlineStageTrialWriter&) = delete;
    MainlineStageTrialWriter& operator=(const MainlineStageTrialWriter&) = delete;

    [[nodiscard]] bool open(std::string& error);
    [[nodiscard]] bool isOpen() const noexcept;

    [[nodiscard]] ports::WriteResult append(
        const ports::StereoFrame& frame) override;
    [[nodiscard]] ports::WriteResult append(
        const core::Detection2D& record) override;
    [[nodiscard]] ports::WriteResult append(
        const core::StereoAssociationAttempt& record) override;
    [[nodiscard]] ports::WriteResult append(
        const core::TriangulatedObservation& record) override;
    [[nodiscard]] ports::WriteResult finalize() override;
    void abort(std::string reason) noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volleyball::adapters::capture_v21
