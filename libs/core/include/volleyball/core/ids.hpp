#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace volleyball::core {

template <typename Tag>
class StringId {
public:
    StringId() = default;
    explicit StringId(std::string value) : value_(std::move(value)) {}

    [[nodiscard]] const std::string& value() const noexcept { return value_; }
    [[nodiscard]] bool empty() const noexcept { return value_.empty(); }

    friend bool operator==(const StringId& lhs, const StringId& rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }
    friend bool operator!=(const StringId& lhs, const StringId& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const StringId& lhs, const StringId& rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }

private:
    std::string value_;
};

struct CameraIdTag;
struct ClockDomainIdTag;
struct CoordinateFrameIdTag;
struct PipelineIdTag;
struct TrialIdTag;
struct MappingIdTag;

using CameraId = StringId<CameraIdTag>;
using ClockDomainId = StringId<ClockDomainIdTag>;
using CoordinateFrameId = StringId<CoordinateFrameIdTag>;
using PipelineId = StringId<PipelineIdTag>;
using TrialId = StringId<TrialIdTag>;
using MappingId = StringId<MappingIdTag>;

using FrameIndex = std::uint64_t;
using CandidateId = std::uint64_t;
using PairId = std::uint64_t;
using ObservationId = std::uint64_t;
using TrackId = std::uint64_t;

}  // namespace volleyball::core

