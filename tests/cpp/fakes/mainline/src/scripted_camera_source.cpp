#include "volleyball/tests/fakes/scripted_camera_source.hpp"

#include <optional>
#include <utility>

namespace volleyball::tests::fakes {

ScriptedCameraSource::ScriptedCameraSource(
    std::vector<ports::SourceResult> results)
    : results_(std::move(results)) {}

ports::SourceResult ScriptedCameraSource::next() {
    if (next_index_ >= results_.size()) {
        return {
            ports::SourceStatus::error,
            std::nullopt,
            "scripted camera source exhausted without a terminal status",
        };
    }
    return std::move(results_[next_index_++]);
}

}  // namespace volleyball::tests::fakes
