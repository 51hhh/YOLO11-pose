#include "volleyball/core/time.hpp"

#include <cmath>
#include <limits>

namespace volleyball::core {
namespace {

ClockMapResult failure(ClockMapError error, std::string message) noexcept {
    return ClockMapResult{std::nullopt, error, std::move(message)};
}

}  // namespace

ClockMapResult mapTimestamp(
    const Timestamp& source,
    const ClockMapping& mapping) noexcept {
    if (mapping.id.empty() || mapping.source_clock.empty() ||
        mapping.reference_clock.empty() ||
        mapping.source_clock == mapping.reference_clock ||
        !std::isfinite(mapping.drift_ppm) ||
        !std::isfinite(mapping.uncertainty_ns) ||
        mapping.uncertainty_ns < 0.0 || mapping.drift_ppm <= -1.0e6 ||
        (mapping.valid_from_source_ns && mapping.valid_to_source_ns &&
         *mapping.valid_to_source_ns < *mapping.valid_from_source_ns)) {
        return failure(ClockMapError::invalid_mapping, "invalid clock mapping");
    }
    if (source.clock_domain != mapping.source_clock) {
        return failure(
            ClockMapError::wrong_clock_domain,
            "source timestamp does not use mapping source clock");
    }
    if ((mapping.valid_from_source_ns &&
         source.nanoseconds < *mapping.valid_from_source_ns) ||
        (mapping.valid_to_source_ns &&
         source.nanoseconds > *mapping.valid_to_source_ns)) {
        return failure(
            ClockMapError::outside_validity_interval,
            "source timestamp is outside mapping validity interval");
    }

    // 只对小的相对时间使用浮点，避免大 epoch 丢失纳秒精度。
    const long double relative =
        static_cast<long double>(source.nanoseconds) -
        static_cast<long double>(mapping.source_anchor_ns);
    const long double scale =
        1.0L + static_cast<long double>(mapping.drift_ppm) * 1.0e-6L;
    const long double mapped_relative = std::round(relative * scale);
    const long double mapped =
        static_cast<long double>(mapping.reference_anchor_ns) + mapped_relative;
    constexpr auto kMin = static_cast<long double>(
        std::numeric_limits<std::int64_t>::min());
    constexpr auto kMax = static_cast<long double>(
        std::numeric_limits<std::int64_t>::max());
    if (mapped < kMin || mapped > kMax) {
        return failure(ClockMapError::overflow, "mapped timestamp overflows int64");
    }
    return ClockMapResult{
        Timestamp{static_cast<std::int64_t>(mapped), mapping.reference_clock},
        ClockMapError::none,
        {},
    };
}

}  // namespace volleyball::core

