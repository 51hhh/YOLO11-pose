#include "volleyball/adapters/fake/in_memory_trial_writer.hpp"
#include "volleyball/adapters/fake/scripted_camera_source.hpp"
#include "volleyball/services/stereo_pipeline_runner.hpp"

#include "stereo_service_fixtures.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

support::StaticDetector leftDetector() {
    return support::StaticDetector({
        support::detection(
            core::CameraRole::left, "left", 0, 700.0, 500.0, 0.95),
    });
}

support::StaticDetector rightDetector() {
    return support::StaticDetector({
        support::detection(
            core::CameraRole::right, "right", 0, 650.0, 500.0, 0.90),
    });
}

ports::SourceResult frameResult() {
    return {
        ports::SourceStatus::frame,
        support::stereoFrame(),
        {},
    };
}

ports::SourceResult endResult() {
    return {ports::SourceStatus::end_of_stream, std::nullopt, {}};
}

ports::SourceResult timeoutResult() {
    return {ports::SourceStatus::timeout, std::nullopt, {}};
}

class FinalizeFailingWriter final : public ports::TrialWriter {
public:
    [[nodiscard]] ports::WriteResult append(
        const ports::StereoFrame&) override {
        return accepted();
    }
    [[nodiscard]] ports::WriteResult append(
        const core::Detection2D&) override {
        return accepted();
    }
    [[nodiscard]] ports::WriteResult append(
        const core::StereoAssociationAttempt&) override {
        return accepted();
    }
    [[nodiscard]] ports::WriteResult append(
        const ports::EstimatedDirectObservation&) override {
        return accepted();
    }
    [[nodiscard]] ports::WriteResult finalize() override {
        return {ports::WriteStatus::failed, "injected finalize failure"};
    }
    void abort(std::string reason) noexcept override {
        if (abort_reason.empty()) {
            abort_reason = std::move(reason);
        }
    }

    std::string abort_reason;

private:
    static ports::WriteResult accepted() {
        return {ports::WriteStatus::accepted, {}};
    }
};

void testCompletesAndFinalizesExactlyOnce() {
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source({frameResult(), endResult()});
    services::StereoPipelineRunner runner({}, source, service);

    const auto result = runner.run();
    check(result.ok(), "frame followed by end-of-stream must complete");
    check(result.frames_received == 1 && result.observations == 1,
          "completed summary must count the direct observation");
    check(result.no_matches == 0 && result.timeouts == 0,
          "completed summary must not invent misses or timeouts");
    check(writer.finalized() && writer.abortReason().empty(),
          "normal completion must finalize without aborting");
    check(source.calls() == 2,
          "runner must stop reading immediately after end-of-stream");

    const auto repeated = runner.run();
    check(repeated.status == services::RunStatus::already_finished,
          "runner must reject a second lifecycle invocation");
    check(source.calls() == 2,
          "second invocation must not touch the source or writer");
}

void testToleratesConfiguredTimeoutThenResumes() {
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source(
        {timeoutResult(), frameResult(), endResult()});
    services::StereoPipelineRunnerConfig config;
    config.max_consecutive_timeouts = 1;
    services::StereoPipelineRunner runner(config, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::completed,
          "one configured timeout must be recoverable");
    check(result.timeouts == 1 && result.frames_received == 1,
          "timeout and resumed frame counts must remain visible");
    check(writer.finalized(), "recovered trial must still finalize");
}

void testTimeoutLimitAbortsPartialTrial() {
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source(
        {timeoutResult(), timeoutResult(), endResult()});
    services::StereoPipelineRunnerConfig config;
    config.max_consecutive_timeouts = 1;
    services::StereoPipelineRunner runner(config, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::source_timeout,
          "timeout beyond the configured limit must terminate the trial");
    check(result.timeouts == 2 && source.calls() == 2,
          "runner must stop exactly at the timeout limit violation");
    check(!writer.finalized() && !writer.abortReason().empty(),
          "timed-out trial must abort instead of publishing partial data");
}

void testSourceErrorAndProtocolErrorAbort() {
    {
        auto left = leftDetector();
        auto right = rightDetector();
        fake::InMemoryTrialWriter writer;
        services::StereoObservationService service(
            support::serviceConfig(), support::passingPreprocessor(), left, right,
            support::passingObservationEstimator(), writer);
        fake::ScriptedCameraSource source({
            {ports::SourceStatus::error, std::nullopt, "USB transfer failed"},
        });
        services::StereoPipelineRunner runner({}, source, service);

        const auto result = runner.run();
        check(result.status == services::RunStatus::source_error,
              "camera source failure must be distinguished from processing");
        check(writer.abortReason() == "USB transfer failed",
              "source failure reason must reach the trial writer");
    }

    {
        auto left = leftDetector();
        auto right = rightDetector();
        fake::InMemoryTrialWriter writer;
        services::StereoObservationService service(
            support::serviceConfig(), support::passingPreprocessor(), left, right,
            support::passingObservationEstimator(), writer);
        fake::ScriptedCameraSource source({
            {ports::SourceStatus::frame, std::nullopt, {}},
        });
        services::StereoPipelineRunner runner({}, source, service);

        const auto result = runner.run();
        check(result.status == services::RunStatus::source_protocol_error,
              "frame status without a frame payload must be rejected");
        check(!writer.abortReason().empty(),
              "source protocol violation must abort the trial");
    }
}

void testProcessingFailureAbortsAndPreservesReason() {
    auto left = leftDetector();
    auto right = rightDetector();
    right.setFail(true);
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source({frameResult(), endResult()});
    services::StereoPipelineRunner runner({}, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::processing_error,
          "detector failure must terminate as processing error");
    check(writer.abortReason() == "injected detector failure",
          "runner must preserve the service's first failure reason");
    check(source.calls() == 1,
          "runner must not consume later input after processing failure");
}

void testNoMatchIsAValidFrameOutcome() {
    support::StaticDetector left(std::vector<core::Detection2D>{});
    support::StaticDetector right(std::vector<core::Detection2D>{});
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source({frameResult(), endResult()});
    services::StereoPipelineRunner runner({}, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::completed,
          "a frame without a stereo match must not fail the trial");
    check(result.no_matches == 1 && result.observations == 0,
          "summary must distinguish no-match from observation");
    check(writer.finalized(), "no-match trial must remain publishable");
}

void testFinalizeFailureAbortsTrial() {
    auto left = leftDetector();
    auto right = rightDetector();
    FinalizeFailingWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source({endResult()});
    services::StereoPipelineRunner runner({}, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::finalize_error,
          "writer finalization failure must be a distinct terminal state");
    check(writer.abort_reason == "injected finalize failure",
          "failed finalization must abort the writer with its reason");
}

void testRequestedRuntimeStopFinalizesWithoutReadingAnotherFrame() {
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source({frameResult(), endResult()});
    services::StereoPipelineRunnerConfig config;
    config.stop_requested = [] { return true; };
    config.finalize_on_stop = true;
    services::StereoPipelineRunner runner(config, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::stopped,
          "runtime signal stop must be an explicit terminal state");
    check(writer.finalized() && writer.abortReason().empty(),
          "runtime stop may close a non-capture writer normally");
    check(source.calls() == 0,
          "an already-requested stop must not consume another camera frame");
}

void testRequestedCaptureStopAbortsPartialTrial() {
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(), support::passingPreprocessor(), left, right,
        support::passingObservationEstimator(), writer);
    fake::ScriptedCameraSource source({frameResult(), endResult()});
    services::StereoPipelineRunnerConfig config;
    config.stop_requested = [] { return true; };
    config.finalize_on_stop = false;
    services::StereoPipelineRunner runner(config, source, service);

    const auto result = runner.run();
    check(result.status == services::RunStatus::stopped,
          "capture cancellation must be distinguishable from source failure");
    check(!writer.finalized() && !writer.abortReason().empty(),
          "capture cancellation must never publish a partial trial");
}

}  // namespace

int main() {
    testCompletesAndFinalizesExactlyOnce();
    testToleratesConfiguredTimeoutThenResumes();
    testTimeoutLimitAbortsPartialTrial();
    testSourceErrorAndProtocolErrorAbort();
    testProcessingFailureAbortsAndPreservesReason();
    testNoMatchIsAValidFrameOutcome();
    testFinalizeFailureAbortsTrial();
    testRequestedRuntimeStopFinalizesWithoutReadingAnotherFrame();
    testRequestedCaptureStopAbortsPartialTrial();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stereo pipeline runner tests passed\n";
    return EXIT_SUCCESS;
}
