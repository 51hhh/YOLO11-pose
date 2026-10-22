#include "volleyball/services/stereo_pipeline_runner.hpp"

#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace volleyball::services {
namespace {

std::string messageOr(std::string message, const char* fallback) {
    return message.empty() ? std::string{fallback} : std::move(message);
}

}  // namespace

StereoPipelineRunner::StereoPipelineRunner(
    StereoPipelineRunnerConfig config,
    ports::CameraSource& source,
    StereoObservationService& service)
    : config_(config), source_(source), service_(service) {}

RunSummary StereoPipelineRunner::run() {
    if (finished_) {
        return {
            RunStatus::already_finished,
            0,
            0,
            0,
            0,
            "stereo pipeline runner has already finished",
        };
    }

    RunSummary summary;
    std::size_t consecutive_timeouts = 0;
    auto fail = [&](RunStatus status, std::string message) {
        message = messageOr(std::move(message), "stereo pipeline failed");
        service_.abort(message);
        summary.status = status;
        summary.message = std::move(message);
        finished_ = true;
        return summary;
    };
    auto stopRequested = [&]() -> std::optional<RunSummary> {
        if (!config_.stop_requested) return std::nullopt;
        bool requested = false;
        try {
            requested = config_.stop_requested();
        } catch (const std::exception& error) {
            return fail(
                RunStatus::lifecycle_error,
                std::string{"stop callback threw an exception: "} +
                    error.what());
        } catch (...) {
            return fail(
                RunStatus::lifecycle_error,
                "stop callback threw a non-standard exception");
        }
        if (!requested) return std::nullopt;
        if (!config_.finalize_on_stop) {
            return fail(RunStatus::stopped, "stereo pipeline stop requested");
        }
        ports::WriteResult result;
        try {
            result = service_.finalize();
        } catch (const std::exception& error) {
            return fail(
                RunStatus::finalize_error,
                std::string{"stop finalization threw an exception: "} +
                    error.what());
        } catch (...) {
            return fail(
                RunStatus::finalize_error,
                "stop finalization threw a non-standard exception");
        }
        if (result.status != ports::WriteStatus::accepted) {
            return fail(
                RunStatus::finalize_error,
                messageOr(
                    std::move(result.message),
                    "trial writer rejected stop finalization"));
        }
        summary.status = RunStatus::stopped;
        summary.message = "stereo pipeline stop requested";
        finished_ = true;
        return summary;
    };

    while (true) {
        if (auto stopped = stopRequested()) return *stopped;
        ports::SourceResult source_result;
        try {
            source_result = source_.next();
        } catch (const std::exception& error) {
            return fail(
                RunStatus::source_error,
                std::string{"camera source threw an exception: "} + error.what());
        } catch (...) {
            return fail(
                RunStatus::source_error,
                "camera source threw a non-standard exception");
        }

        const bool has_frame = source_result.stereo_frame.has_value();
        if ((source_result.status == ports::SourceStatus::frame) != has_frame) {
            return fail(
                RunStatus::source_protocol_error,
                "camera source frame status and payload disagree");
        }
        if (auto stopped = stopRequested()) return *stopped;

        switch (source_result.status) {
        case ports::SourceStatus::frame: {
            consecutive_timeouts = 0;
            ++summary.frames_received;
            StereoProcessResult process_result;
            try {
                process_result = service_.process(*source_result.stereo_frame);
            } catch (const std::exception& error) {
                return fail(
                    RunStatus::processing_error,
                    std::string{"stereo processing threw an exception: "} +
                        error.what());
            } catch (...) {
                return fail(
                    RunStatus::processing_error,
                    "stereo processing threw a non-standard exception");
            }
            if (process_result.status == ProcessStatus::observation) {
                ++summary.observations;
                break;
            }
            if (process_result.status == ProcessStatus::no_match) {
                ++summary.no_matches;
                break;
            }
            return fail(
                RunStatus::processing_error,
                messageOr(
                    std::move(process_result.message),
                    "stereo observation service rejected a frame"));
        }
        case ports::SourceStatus::timeout:
            ++summary.timeouts;
            ++consecutive_timeouts;
            if (consecutive_timeouts > config_.max_consecutive_timeouts) {
                return fail(
                    RunStatus::source_timeout,
                    messageOr(
                        std::move(source_result.message),
                        "camera source exceeded the consecutive timeout limit"));
            }
            break;
        case ports::SourceStatus::error:
            return fail(
                RunStatus::source_error,
                messageOr(
                    std::move(source_result.message),
                    "camera source reported an error"));
        case ports::SourceStatus::end_of_stream: {
            ports::WriteResult result;
            try {
                result = service_.finalize();
            } catch (const std::exception& error) {
                return fail(
                    RunStatus::finalize_error,
                    std::string{"trial finalization threw an exception: "} +
                        error.what());
            } catch (...) {
                return fail(
                    RunStatus::finalize_error,
                    "trial finalization threw a non-standard exception");
            }
            if (result.status != ports::WriteStatus::accepted) {
                return fail(
                    RunStatus::finalize_error,
                    messageOr(
                        std::move(result.message),
                        "trial writer rejected finalization"));
            }
            summary.status = RunStatus::completed;
            summary.message = std::move(source_result.message);
            finished_ = true;
            return summary;
        }
        default:
            return fail(
                RunStatus::source_protocol_error,
                "camera source returned an unknown status");
        }
    }
}

}  // namespace volleyball::services
