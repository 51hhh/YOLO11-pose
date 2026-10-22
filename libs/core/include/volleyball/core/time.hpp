#pragma once

#include "volleyball/core/ids.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace volleyball::core {

struct Timestamp {
    std::int64_t nanoseconds{0};
    ClockDomainId clock_domain;
};

struct ClockMapping {
    MappingId id;
    ClockDomainId source_clock;
    ClockDomainId reference_clock;
    std::int64_t source_anchor_ns{0};
    std::int64_t reference_anchor_ns{0};
    double drift_ppm{0.0};
    double uncertainty_ns{0.0};
    std::optional<std::int64_t> valid_from_source_ns;
    std::optional<std::int64_t> valid_to_source_ns;
};

enum class ClockMapError {
    none,
    invalid_mapping,
    wrong_clock_domain,
    outside_validity_interval,
    overflow,
};

struct ClockMapResult {
    std::optional<Timestamp> timestamp;
    ClockMapError error{ClockMapError::none};
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return timestamp.has_value();
    }
};

[[nodiscard]] ClockMapResult mapTimestamp(
    const Timestamp& source,
    const ClockMapping& mapping) noexcept;

}  // namespace volleyball::core

