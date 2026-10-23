#include "volleyball/tests/fakes/in_memory_trial_writer.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include "stereo_service_fixtures.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace core = volleyball::core;
namespace fake = volleyball::tests::fakes;
namespace ports = volleyball::ports;
namespace services = volleyball::services;
namespace stereo_algorithms = volleyball::algorithms::stereo;
namespace support = volleyball::test_support;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

class FixedEstimator final : public ports::DirectObservationEstimator {
public:
    explicit FixedEstimator(ports::DirectObservationResult result)
        : result_(std::move(result)) {}

    [[nodiscard]] ports::DirectObservationResult estimate(
        const ports::DirectObservationRequest& request) override {
        ++calls_;
        saw_gray_inputs_ =
            request.rectified_frame.left.geometry_gray &&
            request.rectified_frame.right.geometry_gray &&
            request.rectified_frame.left.geometry_gray->pixelFormat() ==
                core::PixelFormat::gray8 &&
            request.rectified_frame.right.geometry_gray->pixelFormat() ==
                core::PixelFormat::gray8;
        return result_;
    }

    [[nodiscard]] std::size_t calls() const noexcept { return calls_; }
    [[nodiscard]] bool sawGrayInputs() const noexcept {
        return saw_gray_inputs_;
    }

private:
    ports::DirectObservationResult result_;
    std::size_t calls_{0};
    bool saw_gray_inputs_{false};
};

struct DetectorBarrierState {
    std::mutex mutex;
    std::condition_variable entered;
    int active{0};
    int maximum_active{0};
    int total_entered{0};
    bool timed_out{false};
};

class BarrierDetector final : public ports::Detector {
public:
    BarrierDetector(
        std::shared_ptr<DetectorBarrierState> state,
        double center_x)
        : state_(std::move(state)), center_x_(center_x) {}

    [[nodiscard]] ports::DetectionResult detect(
        const ports::RectifiedCameraFrame& frame) override {
        {
            std::unique_lock<std::mutex> lock(state_->mutex);
            ++state_->active;
            state_->maximum_active =
                std::max(state_->maximum_active, state_->active);
            ++state_->total_entered;
            state_->entered.notify_all();
            if (!state_->entered.wait_for(
                    lock, std::chrono::milliseconds(500), [&] {
                        return state_->total_entered >= 2;
                    })) {
                state_->timed_out = true;
            }
            --state_->active;
        }

        auto candidate = support::detection(
            frame.source.role,
            frame.source.camera_id.value().c_str(),
            0,
            center_x_,
            500.0,
            0.95);
        candidate.frame.frame_index = frame.source.frame_index;
        return {
            true,
            {frame.source,
             core::PipelineId{"fixture-detector"},
             {std::move(candidate)}},
            {},
        };
    }

private:
    std::shared_ptr<DetectorBarrierState> state_;
    double center_x_;
};

class ThrowingDetector final : public ports::Detector {
public:
    [[nodiscard]] ports::DetectionResult detect(
        const ports::RectifiedCameraFrame&) override {
        throw std::runtime_error("injected detector exception");
    }
};

using support::CollectingPublisher;
using support::StaticDetector;
using support::TestImage;
using support::detection;
using support::passingObservationEstimator;
using support::passingPreprocessor;
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
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer, &publisher);

    const auto result = service.process(stereoFrame());
    check(result.ok(), "valid stereo frame must be processed");
    check(result.status == services::ProcessStatus::observation,
          "valid pair must produce an observation");
    check(result.associations.size() == 4,
          "every left/right candidate combination must be retained");
    check(writer.frames().size() == 1,
          "both camera facts must be written as one stereo transaction");
    check(
        writer.frames()[0].left.image->pixelFormat() ==
            core::PixelFormat::bayer8 &&
            left.lastDetectorFormat() == core::PixelFormat::bgr8,
        "writer must retain raw Bayer while detector receives rectified BGR");
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
    check(publisher.publications.size() == 1,
          "accepted direct observation must publish");
    const auto& publication = publisher.publications.front();
    check(
        publication.observation.measurement.observation_id ==
                writer.observations().front().observation_id &&
            publication.association.pair_id ==
                publication.observation.measurement.source_pair_id &&
            std::any_of(
                writer.associations().begin(),
                writer.associations().end(),
                [&](const core::StereoAssociationAttempt& attempt) {
                    return attempt.pair_id == publication.association.pair_id &&
                           attempt.decision ==
                               core::AssociationDecision::accepted;
                }),
        "publication must preserve the written observation and pair identity");
    check(
        publication.left_frame.camera_id == core::CameraId{"left"} &&
            publication.right_frame.camera_id == core::CameraId{"right"} &&
            publication.left_detection.selected &&
            publication.right_detection.selected &&
            publication.estimation.method ==
                core::StereoDepthMethodId{"detection_center"} &&
            publication.estimation.disparity_px == 50.0 &&
            publication.estimation.support_points == 1,
        "publication must retain frames, detections, and estimator evidence");
    check(result.observation &&
              std::abs(result.observation->position_m.z - 16.0) < 1.0e-9,
          "rectified triangulation must use z=fx*baseline/disparity");
    check(result.observation && result.observation->covariance_m2.has_value(),
          "triangulation must propagate center uncertainty");
    check(service.finalize().status == ports::WriteStatus::accepted,
          "successful service must finalize the raw writer");
}

void testRuntimeServiceCanPublishWithoutRecordingWriter() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.5, 0.90),
    });
    CollectingPublisher publisher;
    services::StereoObservationService service(
        serviceConfig(),
        passingPreprocessor(),
        left,
        right,
        passingObservationEstimator(),
        static_cast<ports::TrialWriter*>(nullptr),
        &publisher);

    const auto result = service.process(stereoFrame());
    const auto finalized = service.finalize();
    check(
        result.status == services::ProcessStatus::observation &&
            publisher.publications.size() == 1,
        "runtime mode must publish direct observations without a fake writer");
    check(
        finalized.status == ports::WriteStatus::accepted,
        "runtime mode without recording must finalize its service lifecycle");
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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

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
    auto triangulation = support::triangulationConfig();
    triangulation.min_disparity_px = 1.0;
    stereo_algorithms::RectifiedCenterEstimator estimator(
        support::calibration(), triangulation);
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right, estimator, writer);

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
        service_config, passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

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

void testEstimatorInfrastructureFailureAbortsTrial() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    FixedEstimator estimator({
        ports::DirectObservationStatus::failed,
        std::nullopt,
        core::StereoDepthMethodId{"roi_multi_point"},
        "injected ROI backend failure",
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right, estimator, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::estimator_error,
          "estimator infrastructure failure must be distinct from no-match");
    check(estimator.calls() == 1 && estimator.sawGrayInputs(),
          "estimator must receive the validated rectified Gray products");
    check(writer.abortReason() == "injected ROI backend failure",
          "estimator infrastructure failure must abort the trial");
}

void testMalformedEstimatorResultAbortsTrial() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    FixedEstimator estimator({
        ports::DirectObservationStatus::observation,
        std::nullopt,
        {},
        {},
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right, estimator, writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::estimator_error,
          "malformed estimator output must be a protocol failure");
    check(!writer.abortReason().empty(),
          "malformed estimator output must abort the partial trial");
}

bool serviceConstructionRejects(
    services::StereoObservationServiceConfig config) {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    try {
        services::StereoObservationService service(
            std::move(config), passingPreprocessor(), left, right,
            passingObservationEstimator(), writer);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void testInvalidStaticConfigFailsDuringServiceConstruction() {
    auto config = serviceConfig();
    config.association.max_size_ratio = 0.5;
    check(serviceConstructionRejects(config),
          "invalid association config must fail before processing");

    config = serviceConfig();
    config.gate.max_depth_m = config.gate.min_depth_m;
    check(serviceConstructionRejects(config),
          "invalid observation gate must fail before processing");
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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);
    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "detector error must be explicit");
    check(!writer.abortReason().empty(), "partial trial must be aborted");
    check(service.finalize().status == ports::WriteStatus::failed,
          "aborted service must not publish a trial");
}

void testStereoDetectorsRunConcurrently() {
    auto state = std::make_shared<DetectorBarrierState>();
    BarrierDetector left(state, 700.0);
    BarrierDetector right(state, 650.0);
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::observation,
          "parallel detector results must preserve normal observation flow");
    check(state->maximum_active == 2 && !state->timed_out,
          "left/right detector calls must overlap on persistent workers");
}

void testDetectorExceptionBecomesAuditableFailure() {
    ThrowingDetector left;
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

    const auto result = service.process(stereoFrame());
    check(
        result.status == services::ProcessStatus::detector_error &&
            result.message.find("injected detector exception") !=
                std::string::npos,
        "detector exceptions must become explicit detector failures");
    check(!writer.abortReason().empty(),
          "detector exceptions must abort the partial trial");
}

void testPreprocessFailureAbortsAfterRawEvidence() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    support::StaticStereoPreprocessor preprocessor;
    preprocessor.setFail(true);
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), preprocessor, left, right,
        passingObservationEstimator(), writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::preprocess_error,
          "preprocess failure must have a distinct status");
    check(writer.frames().size() == 1,
          "raw camera evidence must be retained before preprocessing starts");
    check(left.calls() == 0 && right.calls() == 0,
          "detectors must not receive a failed preprocess product");
    check(!writer.abortReason().empty(),
          "preprocess failure must abort the partial trial");
}

void testMalformedPreprocessProductsAbortTrial() {
    auto rejects = [](support::StaticStereoPreprocessor& preprocessor) {
        StaticDetector left({
            detection(
                core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
        });
        StaticDetector right({
            detection(
                core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
        });
        fake::InMemoryTrialWriter writer;
        services::StereoObservationService service(
            serviceConfig(), preprocessor, left, right,
            passingObservationEstimator(), writer);
        const auto result = service.process(stereoFrame());
        return result.status == services::ProcessStatus::preprocess_error &&
               left.calls() == 0 && right.calls() == 0 &&
               !writer.abortReason().empty();
    };

    support::StaticStereoPreprocessor swapped;
    swapped.setSwapSources(true);
    check(rejects(swapped),
          "preprocessor must not swap or rewrite source frame identity");

    support::StaticStereoPreprocessor wrong_format;
    wrong_format.setInvalidLeftDetectorFormat(true);
    check(rejects(wrong_format),
          "preprocessor must provide the declared BGR detector product");
}

void testExplicitAbortClosesServiceExactlyOnce() {
    StaticDetector left({
        detection(core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

    service.abort("camera source failed");
    service.abort("must not replace the first reason");

    check(writer.abortReason() == "camera source failed",
          "explicit abort must preserve the first terminal reason");
    check(service.process(stereoFrame()).status ==
              services::ProcessStatus::write_error,
          "explicit abort must close the processing service");
    check(service.finalize().status == ports::WriteStatus::failed,
          "explicit abort must prevent later finalization");
}

void testMalformedDetectorBatchAbortsTrial() {
    auto wrong_frame =
        detection(core::CameraRole::left, "other-left", 0, 700.0, 500.0, 0.95);
    StaticDetector left({wrong_frame});
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "duplicate candidate IDs in one frame must be rejected");
    check(!writer.abortReason().empty(),
          "duplicate candidate IDs must abort the partial trial");
}

void testDetectorCandidateMustIntersectItsSourceImage() {
    auto outside =
        detection(core::CameraRole::left, "left", 0, -1.0, 500.0, 0.95);
    StaticDetector left({outside});
    StaticDetector right({
        detection(core::CameraRole::right, "right", 0, 650.0, 500.0, 0.95),
    });
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

    const auto result = service.process(stereoFrame());
    check(result.status == services::ProcessStatus::detector_error,
          "candidate center outside its source image must be rejected");
    check(!writer.abortReason().empty(),
          "out-of-image detector output must abort the partial trial");
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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);

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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);
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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);
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
        serviceConfig(), passingPreprocessor(), late_left, late_right,
        passingObservationEstimator(), late_writer);
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
    services::StereoObservationService service(
        serviceConfig(), passingPreprocessor(), left, right,
        passingObservationEstimator(), writer);
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
        1440, 1080, core::PixelFormat::bayer8, 1440, 1024);
    check(rejectsInvalidStereoFrame(std::move(short_buffer)),
          "image allocation must cover stride times height");
}

}  // namespace

int main() {
    testServicePreservesAllAttemptsAndPublishesDirectObservation();
    testRuntimeServiceCanPublishWithoutRecordingWriter();
    testUnselectedGatePassingPairIsRecordedAsRejected();
    testTriangulationFailureRejectsSelectedPair();
    testObservationGateRejectsSelectedPair();
    testEstimatorInfrastructureFailureAbortsTrial();
    testMalformedEstimatorResultAbortsTrial();
    testInvalidStaticConfigFailsDuringServiceConstruction();
    testPreprocessFailureAbortsAfterRawEvidence();
    testMalformedPreprocessProductsAbortTrial();
    testDetectorFailureAbortsTrial();
    testStereoDetectorsRunConcurrently();
    testDetectorExceptionBecomesAuditableFailure();
    testExplicitAbortClosesServiceExactlyOnce();
    testMalformedDetectorBatchAbortsTrial();
    testDuplicateCandidateIdsAbortTrial();
    testDetectorCandidateMustIntersectItsSourceImage();
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
