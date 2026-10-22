#include "volleyball/adapters/fake/in_memory_trial_writer.hpp"
#include "volleyball/core/observation.hpp"
#include "volleyball/core/time.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

namespace core = volleyball::core;
namespace fake = volleyball::adapters::fake;
namespace ports = volleyball::ports;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

core::Timestamp timestamp(std::int64_t nanoseconds, const char* domain) {
    return {nanoseconds, core::ClockDomainId{domain}};
}

core::CameraFrameRecord validFrame() {
    core::CameraFrameRecord frame;
    frame.camera_id = core::CameraId{"left-serial"};
    frame.role = core::CameraRole::left;
    frame.frame_index = 42;
    frame.hardware_timestamp = timestamp(999, "camera");
    frame.exposure_start = timestamp(1000, "camera");
    frame.capture_timestamp = timestamp(1100, "camera");
    frame.receive_timestamp = timestamp(1200, "camera");
    frame.inference_done_timestamp = timestamp(1300, "camera");
    frame.exposure_us = 100.0;
    frame.gain_db = 2.0;
    frame.roi = {0, 0, 1440, 1080};
    return frame;
}

void testClockMappingPreservesEpochPrecision() {
    core::ClockMapping mapping;
    mapping.id = core::MappingId{"camera-to-system-0"};
    mapping.source_clock = core::ClockDomainId{"camera"};
    mapping.reference_clock = core::ClockDomainId{"system"};
    mapping.source_anchor_ns = 8'000'000'000'000'000'000LL;
    mapping.reference_anchor_ns = 1'700'000'000'000'000'000LL;
    mapping.drift_ppm = 12.5;
    mapping.uncertainty_ns = 100.0;
    mapping.valid_from_source_ns = mapping.source_anchor_ns;
    mapping.valid_to_source_ns = mapping.source_anchor_ns + 1'000'000;

    const auto source = timestamp(mapping.source_anchor_ns + 800'000, "camera");
    const auto result = core::mapTimestamp(source, mapping);
    check(static_cast<bool>(result), "valid clock mapping must succeed");
    check(
        result.timestamp &&
            result.timestamp->nanoseconds == mapping.reference_anchor_ns + 800'010,
        "clock mapping must round the relative nanoseconds exactly");
    check(
        result.timestamp &&
            result.timestamp->clock_domain == core::ClockDomainId{"system"},
        "clock mapping must label the reference domain");
}

void testClockMappingRejectsInvalidInputs() {
    core::ClockMapping mapping;
    mapping.id = core::MappingId{"map"};
    mapping.source_clock = core::ClockDomainId{"camera"};
    mapping.reference_clock = core::ClockDomainId{"system"};
    mapping.valid_from_source_ns = 100;
    mapping.valid_to_source_ns = 200;

    const auto wrong_domain = core::mapTimestamp(timestamp(150, "host"), mapping);
    check(
        wrong_domain.error == core::ClockMapError::wrong_clock_domain,
        "wrong source clock must be rejected");
    const auto outside = core::mapTimestamp(timestamp(99, "camera"), mapping);
    check(
        outside.error == core::ClockMapError::outside_validity_interval,
        "timestamp outside mapping interval must be rejected");
}

void testFrameValidation() {
    auto frame = validFrame();
    check(static_cast<bool>(core::validate(frame)), "valid frame must pass");

    frame.receive_timestamp = timestamp(1200, "host");
    check(
        core::validate(frame).error == core::DomainError::clock_domain_mismatch,
        "one row must not mix camera and host clocks");

    frame = validFrame();
    frame.dropped = true;
    check(
        core::validate(frame).error == core::DomainError::missing_drop_reason,
        "dropped frame must retain a reason");

    frame = validFrame();
    frame.hardware_timestamp.nanoseconds = -1;
    check(
        core::validate(frame).error == core::DomainError::invalid_numeric_value,
        "camera timestamps must not be negative");
}

void testAssociationValidation() {
    core::StereoAssociationAttempt attempt;
    attempt.left_capture = timestamp(1000, "camera");
    attempt.right_capture = timestamp(1025, "camera");
    attempt.delta_t_lr_ns = 25;
    attempt.decision = core::AssociationDecision::rejected;
    attempt.reject_reason = core::AssociationRejectReason::epipolar_error;
    check(static_cast<bool>(core::validate(attempt)), "valid rejection must pass");

    attempt.reject_reason = core::AssociationRejectReason::none;
    check(
        core::validate(attempt).error == core::DomainError::invalid_decision_reason,
        "rejected association requires a reason");

    attempt.reject_reason = core::AssociationRejectReason::epipolar_error;
    attempt.left_capture.nanoseconds = -1;
    check(
        core::validate(attempt).error == core::DomainError::invalid_numeric_value,
        "association timestamps must not be negative");
}

void testWriterSeparatesRawAndDerivedTypes() {
    fake::InMemoryTrialWriter writer;
    const auto result = writer.append(validFrame());
    check(result.status == ports::WriteStatus::accepted, "writer must accept valid frame");
    check(writer.frames().size() == 1, "writer must retain accepted frame");
    check(
        writer.finalize().status == ports::WriteStatus::accepted,
        "writer must finalize once");
    check(
        writer.append(validFrame()).status == ports::WriteStatus::failed,
        "finalized writer must reject further records");
}

void testTriangulatedObservationValidation() {
    core::TriangulatedObservation observation;
    observation.capture_reference = timestamp(10, "trajectory");
    observation.coordinate_frame = core::CoordinateFrameId{"stereo_rig"};
    observation.position_m = {1.0, 2.0, 3.0};
    observation.confidence = 0.9;
    check(
        static_cast<bool>(core::validate(observation)),
        "finite direct observation must pass");
    observation.position_m.z = std::numeric_limits<double>::quiet_NaN();
    check(
        core::validate(observation).error == core::DomainError::invalid_numeric_value,
        "non-finite direct observation must fail");
}

}  // namespace

int main() {
    testClockMappingPreservesEpochPrecision();
    testClockMappingRejectsInvalidInputs();
    testFrameValidation();
    testAssociationValidation();
    testWriterSeparatesRawAndDerivedTypes();
    testTriangulatedObservationValidation();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "core contract tests passed\n";
    return EXIT_SUCCESS;
}

