#include "volleyball/tests/fakes/in_memory_trial_writer.hpp"
#include "volleyball/services/managed_stereo_trial.hpp"

#include "stereo_service_fixtures.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace core = volleyball::core;
namespace fake = volleyball::tests::fakes;
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

class ManagedSource final : public ports::ManagedCameraSource {
public:
    ManagedSource(
        std::vector<std::string>& events,
        std::vector<ports::SourceResult> results)
        : events_(events), results_(std::move(results)) {}

    [[nodiscard]] bool start(std::string& error) override {
        events_.push_back("camera.start");
        if (!start_success_) {
            error = "injected camera start failure";
            return false;
        }
        started_ = report_started_;
        return true;
    }

    void stop() noexcept override {
        events_.push_back("camera.stop");
        started_ = false;
    }

    [[nodiscard]] bool isStarted() const noexcept override {
        return started_;
    }

    [[nodiscard]] ports::SourceResult next() override {
        events_.push_back("camera.next");
        if (next_index_ >= results_.size()) {
            return {
                ports::SourceStatus::error,
                std::nullopt,
                "managed source script exhausted",
            };
        }
        return std::move(results_[next_index_++]);
    }

    void setStartSuccess(bool value) noexcept { start_success_ = value; }
    void setReportStarted(bool value) noexcept { report_started_ = value; }
    void setInitiallyStarted(bool value) noexcept { started_ = value; }

private:
    std::vector<std::string>& events_;
    std::vector<ports::SourceResult> results_;
    std::size_t next_index_{0};
    bool start_success_{true};
    bool report_started_{true};
    bool started_{false};
};

class ManagedTrigger final : public ports::TriggerController {
public:
    explicit ManagedTrigger(std::vector<std::string>& events)
        : events_(events) {}

    [[nodiscard]] bool start(std::string& error) override {
        events_.push_back("trigger.start");
        if (!start_success_) {
            error = "injected trigger start failure";
            return false;
        }
        running_ = report_running_;
        return true;
    }

    void stop() noexcept override {
        events_.push_back("trigger.stop");
        running_ = false;
    }

    [[nodiscard]] bool isRunning() const noexcept override {
        return running_;
    }

    void setStartSuccess(bool value) noexcept { start_success_ = value; }
    void setReportRunning(bool value) noexcept { report_running_ = value; }
    void setInitiallyRunning(bool value) noexcept { running_ = value; }

private:
    std::vector<std::string>& events_;
    bool start_success_{true};
    bool report_running_{true};
    bool running_{false};
};

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
    return {ports::SourceStatus::frame, support::stereoFrame(), {}};
}

ports::SourceResult endResult() {
    return {ports::SourceStatus::end_of_stream, std::nullopt, {}};
}

void testManagedTrialOrdersStartupAndShutdown() {
    std::vector<std::string> events;
    ManagedSource source(events, {frameResult(), endResult()});
    ManagedTrigger trigger(events);
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(),
        support::passingPreprocessor(),
        left,
        right,
        support::passingObservationEstimator(),
        writer);

    const auto result = services::runManagedStereoTrial(
        {}, source, trigger, service);
    check(result.status == services::RunStatus::completed,
          "managed trial must complete a valid script");
    check(
        events == std::vector<std::string>{
            "camera.start",
            "trigger.start",
            "camera.next",
            "camera.next",
            "trigger.stop",
            "camera.stop",
        },
        "camera/trigger lifecycle order must be exact");
    check(writer.finalized() && writer.abortReason().empty(),
          "successful managed trial must finalize exactly once");
}

void testCameraStartFailureNeverStartsTrigger() {
    std::vector<std::string> events;
    ManagedSource source(events, {});
    source.setStartSuccess(false);
    ManagedTrigger trigger(events);
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(),
        support::passingPreprocessor(),
        left,
        right,
        support::passingObservationEstimator(),
        writer);

    const auto result = services::runManagedStereoTrial(
        {}, source, trigger, service);
    check(result.status == services::RunStatus::camera_start_error,
          "camera startup failure must be distinct");
    check(
        events == std::vector<std::string>{
            "camera.start", "camera.stop"},
        "failed camera startup must clean up without starting GPIO");
    check(writer.abortReason() == "injected camera start failure",
          "camera startup reason must abort the trial");
}

void testTriggerStartFailureStopsTriggerBeforeCamera() {
    std::vector<std::string> events;
    ManagedSource source(events, {});
    ManagedTrigger trigger(events);
    trigger.setStartSuccess(false);
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(),
        support::passingPreprocessor(),
        left,
        right,
        support::passingObservationEstimator(),
        writer);

    const auto result = services::runManagedStereoTrial(
        {}, source, trigger, service);
    check(result.status == services::RunStatus::trigger_start_error,
          "trigger startup failure must be distinct");
    check(
        events == std::vector<std::string>{
            "camera.start",
            "trigger.start",
            "trigger.stop",
            "camera.stop",
        },
        "trigger startup failure must stop GPIO before camera");
    check(writer.abortReason() == "injected trigger start failure",
          "trigger startup reason must abort the trial");
}

void testFalseLifecycleStateIsRejected() {
    std::vector<std::string> events;
    ManagedSource source(events, {});
    source.setReportStarted(false);
    ManagedTrigger trigger(events);
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(),
        support::passingPreprocessor(),
        left,
        right,
        support::passingObservationEstimator(),
        writer);

    const auto result = services::runManagedStereoTrial(
        {}, source, trigger, service);
    check(result.status == services::RunStatus::lifecycle_error,
          "successful start without started state must be rejected");
    check(
        events == std::vector<std::string>{
            "camera.start", "camera.stop"},
        "lifecycle protocol failure must still clean up in safe order");
}

void testPrestartedHardwareIsNotTakenOver() {
    std::vector<std::string> events;
    ManagedSource source(events, {});
    source.setInitiallyStarted(true);
    ManagedTrigger trigger(events);
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(),
        support::passingPreprocessor(),
        left,
        right,
        support::passingObservationEstimator(),
        writer);

    const auto result = services::runManagedStereoTrial(
        {}, source, trigger, service);
    check(result.status == services::RunStatus::lifecycle_error,
          "prestarted hardware must be rejected");
    check(events.empty(),
          "managed trial must not start or stop hardware it does not own");
    check(source.isStarted(),
          "rejected prestarted camera must remain under caller ownership");
}

void testSourceFailureStillStopsHardwareInSafeOrder() {
    std::vector<std::string> events;
    ManagedSource source(events, {{
        ports::SourceStatus::error,
        std::nullopt,
        "injected acquisition failure",
    }});
    ManagedTrigger trigger(events);
    auto left = leftDetector();
    auto right = rightDetector();
    fake::InMemoryTrialWriter writer;
    services::StereoObservationService service(
        support::serviceConfig(),
        support::passingPreprocessor(),
        left,
        right,
        support::passingObservationEstimator(),
        writer);

    const auto result = services::runManagedStereoTrial(
        {}, source, trigger, service);
    check(result.status == services::RunStatus::source_error,
          "runner source failure must remain visible");
    check(
        events == std::vector<std::string>{
            "camera.start",
            "trigger.start",
            "camera.next",
            "trigger.stop",
            "camera.stop",
        },
        "runtime failure must stop trigger before camera");
    check(writer.abortReason() == "injected acquisition failure",
          "runtime source failure must abort the trial");
}

}  // namespace

int main() {
    testManagedTrialOrdersStartupAndShutdown();
    testCameraStartFailureNeverStartsTrigger();
    testTriggerStartFailureStopsTriggerBeforeCamera();
    testFalseLifecycleStateIsRejected();
    testPrestartedHardwareIsNotTakenOver();
    testSourceFailureStillStopsHardwareInSafeOrder();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "managed stereo trial tests passed\n";
    return EXIT_SUCCESS;
}
