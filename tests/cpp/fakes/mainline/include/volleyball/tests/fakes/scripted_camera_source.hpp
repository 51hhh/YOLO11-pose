#pragma once

#include "volleyball/ports/camera_source.hpp"

#include <cstddef>
#include <vector>

namespace volleyball::tests::fakes {

// 用于主机单元测试和确定性回放的脚本相机源，不模拟真实采集时序。
class ScriptedCameraSource final : public ports::CameraSource {
public:
    explicit ScriptedCameraSource(std::vector<ports::SourceResult> results);

    [[nodiscard]] ports::SourceResult next() override;
    [[nodiscard]] std::size_t calls() const noexcept { return next_index_; }

private:
    std::vector<ports::SourceResult> results_;
    std::size_t next_index_{0};
};

}  // namespace volleyball::tests::fakes
