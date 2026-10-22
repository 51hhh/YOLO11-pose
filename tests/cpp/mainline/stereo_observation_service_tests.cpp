#include "volleyball/adapters/fake/in_memory_trial_writer.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include "stereo_service_fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

namespace core = volleyball::core;
namespace fake = volleyball::adapters::fake;
namespace ports = volleyball::ports;
namespace services = volleyball::services;
namespace support = volleyball::test_support;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

using support::CollectingPublisher;
using support::StaticDetector;
using support::TestImage;
using support::detection;
using support::serviceConfig;
using support::stereoFrame;
using support::time;

void testServicePreservesAllAttemptsAndPublishesDirectObservation() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
        detection(core::CameraRole::left, "left", 1, 680.0, 600.0, 0.40),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.5, 0.90),
        detection(core::CameraRole::right, "right", 1, 660.0, 300.0, 0.30),
    });
    fake::InMemoryTrialWriter writer;
    CollectingPublisher publisher;
    services::StereoObservationService service(
        serviceConfig(), left, right, writer, &publisher);

    const auto result = service.process(stereoFrame());
    check(result.ok(), "valid stereo frame must be processed");
    check(result.status == services::ProcessStatus::observation,
          "valid pair must produce an observation");
    check(result.associations.size() == 4,
          "every left/right candidate combination must be retained");
    check(writer.frames().size() == 2, "both camera facts must be written");
    check(writer.detections().size() == 4, "all detector candidates must be written");
    check(
        std::count_if(
            writer.detections().begin(), writer.detections().end(),
            [](const core::Detection2D& candidate) {
                return candidate.selected;
            }) == 2,
        "only the two candidates behind the direct observation may be selected");
    check(writer.associations().size() == 4, "all pair attempts must be written");
    check(
        std::count_if(
            writer.associations().begin(), writer.associations().end(),
            [](const core::StereoAssociationAttempt& attempt) {
                return attempt.decision == core::AssociationDecision::accepted;
            }) == 1,
        "exactly one pair may be accepted for one direct observation");
    check(writer.observations().size() == 1, "one direct observation must be written");
    check(publisher.observations.size() == 1, "accepted direct observation must publish");
    check(result.observation &&
              std::abs(result.observation->position_m.z - 16.0) < 1.0e-9,
          "rectified triangulation must use z=fx*baseline/disparity");
    check(result.observation && result.observation->covariance_m2.has_value(),
          "triangulation must propagate center uncertainty");
    check(service.finalize().status == ports::WriteStatus::accepted,
          "successful service must finalize the raw writer");
}

void testUnselectedGatePassingPairIsRecordedAsRejected() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
        detection(core::CameraRole::left, "left", 1, 695.0, 500.0, 0.80),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.90),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::observation,
          "best gate-passing pair must produce an observation");
    check(writer.associations().size() == 2,
          "both gate-passing attempts must be retained");
    check(writer.associations()[0].decision == core::AssociationDecision::accepted,
          "best pair must remain accepted");
    check(
        writer.associations()[1].decision == core::AssociationDecision::rejected &&
            writer.associations()[1].reject_reason ==
                core::AssociationRejectReason::not_selected,
        "unselected gate-passing pair must have an explicit rejection reason");
}

void testTriangulationFailureRejectsSelectedPair() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 699.5, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    auto service_config = serviceConfig();
    service_config.triangulation.min_disparity_px = 1.0;
    services::StereoObservationService service(
        service_config, left, right, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::no_match,
          "triangulation failure must not produce an observation");
    check(writer.associations().size() == 1,
          "failed selected pair must still be retained");
    check(
        writer.associations()[0].decision == core::AssociationDecision::rejected &&
            writer.associations()[0].reject_reason ==
                core::AssociationRejectReason::triangulation_failure,
        "triangulation failure must rewrite final pair semantics");
    check(writer.observations().empty(),
          "triangulation failure must not write a direct observation");
    check(
        std::none_of(
            writer.detections().begin(), writer.detections().end(),
            [](const core::Detection2D& candidate) {
                return candidate.selected;
            }),
        "failed triangulation must leave every candidate unselected");
}

void testObservationGateRejectsSelectedPair() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    auto service_config = serviceConfig();
    service_config.gate.max_depth_m = 10.0;
    services::StereoObservationService service(
        service_config, left, right, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::no_match,
          "quality gate failure must not produce an observation");
    check(
        writer.associations().size() == 1 &&
            writer.associations()[0].decision ==
                core::AssociationDecision::rejected &&
            writer.associations()[0].reject_reason ==
                core::AssociationRejectReason::quality_gate,
        "quality gate failure must be explicit in the retained pair");
}

void testDetectorFailureAbortsTrial() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    right.setFail(true);
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);
    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "detector error must be explicit");
    check(!writer.abortReason().empty(), "partial trial must be aborted");
    check(service.finalize().status == ports::WriteStatus::failed,
          "aborted service must not publish a trial");
}

void testMalformedDetectorBatchAbortsTrial() {
    auto wrong_frame =
        detection(core::CameraRole::left, "other-left", 0, 700.0, 500.0, 0.95);
    StaticDetector left({wrong_frame});
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "candidate from another frame/camera must be rejected");
    check(!writer.abortReason().empty(),
          "malformed detector output must abort the partial trial");
}

void testDuplicateCandidateIdsAbortTrial() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
        detection(core::CameraRole::left, "left", 0, 695.0, 500.0, 0.80),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "duplicate candidate IDs in one frame must be rejected");
    check(!writer.abortReason().empty(),
          "duplicate candidate IDs must abort the partial trial");
}

void testServiceOwnsSelectedFlags() {
    auto left_candidate =
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95);
    auto right_candidate =
        detection(core::CameraRole::right, "right", 0, 650.0, 700.0, 0.95);
    left_candidate.selected = true;
    right_candidate.selected = true;
    StaticDetector left({left_candidate});
    StaticDetector right({right_candidate});
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::no_match,
          "epipolar mismatch fixture must not produce an observation");
    check(
        std::none_of(
            writer.detections().begin(), writer.detections().end(),
            [](const core::Detection2D& candidate) {
                return candidate.selected;
            }),
        "detector-provided selected flags must not leak into raw evidence");
}

void testArrivalOrderNeverDefinesStereoIdentity() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);
    auto input = stereoFrame();
    check(input.left.metadata.receive_timestamp->nanoseconds >
              input.right.metadata.receive_timestamp->nanoseconds,
          "fixture must contain reversed USB arrival order");
    check(service.process(input).status == services::ProcessStatus::observation,
          "USB arrival order must not alter watermark-aligned pairing");
}

void testCaptureDeltaUsesConfiguredTemporalGate() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);
    auto within_tolerance = stereoFrame();
    within_tolerance.right.metadata.capture_timestamp = time(1050);

    const auto accepted = service.process(within_tolerance);
    check(accepted.status == services::ProcessStatus::observation,
          "bounded left/right capture offset must remain usable");
    check(
        writer.associations().size() == 1 &&
            writer.associations()[0].delta_t_lr_ns == 50,
        "measured capture delta must be retained in the pair evidence");

    StaticDetector late_left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector late_right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter late_writer;
    services::StereoObservationService late_service(
        serviceConfig(), late_left, late_right, late_writer);
    auto outside_tolerance = stereoFrame();
    outside_tolerance.right.metadata.capture_timestamp = time(1101);

    const auto rejected = late_service.process(outside_tolerance);
    check(rejected.status == services::ProcessStatus::no_match,
          "capture offset beyond tolerance must not produce 3D");
    check(
        late_writer.associations().size() == 1 &&
            late_writer.associations()[0].reject_reason ==
                core::AssociationRejectReason::temporal_mismatch,
        "temporal rejection must remain auditable");
}

bool rejectsInvalidStereoFrame(ports::StereoFrame input) {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(serviceConfig(), left, right, writer);
    const auto result = service.process(input);
    return result.status == services::ProcessStatus::invalid_input &&
           !writer.abortReason().empty();
}

void testStereoFrameIdentityAndBufferContract() {
    auto mismatched_index = stereoFrame();
    mismatched_index.right.metadata.frame_index += 1;
    check(rejectsInvalidStereoFrame(std::move(mismatched_index)),
          "left/right frame identity mismatch must abort the trial");

    auto duplicate_camera = stereoFrame();
    duplicate_camera.right.metadata.camera_id = core::CameraId{"left"};
    check(rejectsInvalidStereoFrame(std::move(duplicate_camera)),
          "left/right camera IDs must be distinct");

    auto dropped = stereoFrame();
    dropped.left.metadata.dropped = true;
    dropped.left.metadata.drop_reason = "injected loss";
    check(rejectsInvalidStereoFrame(std::move(dropped)),
          "dropped frame must never enter detection");

    auto wrong_size = stereoFrame();
    wrong_size.left.image = std::make_shared<TestImage>(1280, 1080);
    check(rejectsInvalidStereoFrame(std::move(wrong_size)),
          "image dimensions must match frame metadata");

    auto wrong_format = stereoFrame();
    wrong_format.left.image = std::make_shared<TestImage>(
        1440, 1080, core::PixelFormat::bgr8);
    check(rejectsInvalidStereoFrame(std::move(wrong_format)),
          "image pixel format must match frame metadata");

    auto short_buffer = stereoFrame();
    short_buffer.left.image = std::make_shared<TestImage>(
        1440, 1080, core::PixelFormat::gray8, 1440, 1024);
    check(rejectsInvalidStereoFrame(std::move(short_buffer)),
          "image allocation must cover stride times height");
}

}  // namespace

int main() {
    testServicePreservesAllAttemptsAndPublishesDirectObservation();
    testUnselectedGatePassingPairIsRecordedAsRejected();
    testTriangulationFailureRejectsSelectedPair();
    testObservationGateRejectsSelectedPair();
    testDetectorFailureAbortsTrial();
    testMalformedDetectorBatchAbortsTrial();
    testDuplicateCandidateIdsAbortTrial();
    testServiceOwnsSelectedFlags();
    testArrivalOrderNeverDefinesStereoIdentity();
    testCaptureDeltaUsesConfiguredTemporalGate();
    testStereoFrameIdentityAndBufferContract();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stereo observation service tests passed\n";
    return EXIT_SUCCESS;
}
