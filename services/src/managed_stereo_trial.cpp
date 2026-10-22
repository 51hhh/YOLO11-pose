#include "volleyball/services/managed_stereo_trial.hpp"

#include <exception>
#include <string>
#include <utility>

namespace volleyball::services {
namespace {

std::string messageOr(std::string message, const char* fallback) {
    return message.empty() ? std::string{fallback} : std::move(message);
}

RunSummary startupFailure(
    RunStatus status,
    std::string message,
    ports::ManagedCameraSource& source,
    ports::TriggerController& trigger,
    StereoObservationService& service,
    bool camera_start_attempted,
    bool trigger_start_attempted) {
    message = messageOr(std::move(message), "managed stereo trial failed");
    if (trigger_start_attempted) {
        trigger.stop();
    }
    if (camera_start_attempted) {
        source.stop();
    }
    service.abort(message);
    return {status, 0, 0, 0, 0, std::move(message)};
}

}  // namespace

RunSummary runManagedStereoTrial(
    StereoPipelineRunnerConfig config,
    ports::ManagedCameraSource& source,
    ports::TriggerController& trigger,
    StereoObservationService& service) {
    if (source.isStarted() || trigger.isRunning()) {
        return startupFailure(
            RunStatus::lifecycle_error,
            "managed stereo trial requires stopped camera and trigger",
            source,
            trigger,
            service,
            false,
            false);
    }

    std::string error;
    try {
        if (!source.start(error)) {
            return startupFailure(
                RunStatus::camera_start_error,
                messageOr(std::move(error), "camera source failed to start"),
                source,
                trigger,
                service,
                true,
                false);
        }
    } catch (const std::exception& exception) {
        return startupFailure(
            RunStatus::camera_start_error,
            std::string{"camera source start threw an exception: "} +
                exception.what(),
            source,
            trigger,
            service,
            true,
            false);
    } catch (...) {
        return startupFailure(
            RunStatus::camera_start_error,
            "camera source start threw a non-standard exception",
            source,
            trigger,
            service,
            true,
            false);
    }
    if (!source.isStarted()) {
        return startupFailure(
            RunStatus::lifecycle_error,
            "camera source reported success without entering started state",
            source,
            trigger,
            service,
            true,
            false);
    }

    try {
        error.clear();
        if (!trigger.start(error)) {
            return startupFailure(
                RunStatus::trigger_start_error,
                messageOr(std::move(error), "external trigger failed to start"),
                source,
                trigger,
                service,
                true,
                true);
        }
    } catch (const std::exception& exception) {
        return startupFailure(
            RunStatus::trigger_start_error,
            std::string{"external trigger start threw an exception: "} +
                exception.what(),
            source,
            trigger,
            service,
            true,
            true);
    } catch (...) {
        return startupFailure(
            RunStatus::trigger_start_error,
            "external trigger start threw a non-standard exception",
            source,
            trigger,
            service,
            true,
            true);
    }
    if (!trigger.isRunning()) {
        return startupFailure(
            RunStatus::lifecycle_error,
            "external trigger reported success without entering running state",
            source,
            trigger,
            service,
            true,
            true);
    }

    StereoPipelineRunner runner(config, source, service);
    RunSummary result;
    try {
        result = runner.run();
    } catch (const std::exception& exception) {
        result = startupFailure(
            RunStatus::lifecycle_error,
            std::string{"managed trial runner threw an exception: "} +
                exception.what(),
            source,
            trigger,
            service,
            true,
            true);
        return result;
    } catch (...) {
        result = startupFailure(
            RunStatus::lifecycle_error,
            "managed trial runner threw a non-standard exception",
            source,
            trigger,
            service,
            true,
            true);
        return result;
    }
    trigger.stop();
    source.stop();
    return result;
}

}  // namespace volleyball::services
