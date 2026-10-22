#include "volleyball/services/stereo_pipeline_runner.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace volleyball::services {
namespace {

std::string messageOr(std::string message, const char* fallback) {
    return message.empty() ? std::string{fallback} : std::move(message);
}

bool isTerminal(ports::SourceStatus status) noexcept {
    return status == ports::SourceStatus::end_of_stream ||
           status == ports::SourceStatus::error;
}

// 相机 SDK 只由生产线程调用。队列保存带所有权的 StereoFrame，因此消费方
// 处理期间、队列中的帧以及生产方正在填充的下一帧均不会复用同一缓冲。
class BoundedSourceQueue {
public:
    struct PopResult {
        std::optional<ports::SourceResult> item;
        bool producer_finished{false};
    };

    BoundedSourceQueue(
        ports::CameraSource& source,
        std::size_t capacity,
        SourceQueueOverflowPolicy overflow)
        : source_(source), capacity_(capacity), overflow_(overflow) {}

    ~BoundedSourceQueue() { stop(); }

    BoundedSourceQueue(const BoundedSourceQueue&) = delete;
    BoundedSourceQueue& operator=(const BoundedSourceQueue&) = delete;

    void start() {
        producer_ = std::thread([this] { produce(); });
    }

    [[nodiscard]] PopResult popFor(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait_for(lock, timeout, [&] {
            return stop_requested_ || !queue_.empty() || producer_finished_;
        });
        if (queue_.empty()) {
            return {std::nullopt, producer_finished_ || stop_requested_};
        }
        ports::SourceResult item = std::move(queue_.front());
        queue_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return {std::move(item), false};
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
        if (producer_.joinable()) producer_.join();
    }

    [[nodiscard]] std::size_t framesDropped() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return frames_dropped_;
    }

    [[nodiscard]] std::size_t highWatermark() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return high_watermark_;
    }

private:
    [[nodiscard]] ports::SourceResult readOne() noexcept {
        try {
            return source_.next();
        } catch (const std::exception& error) {
            return {
                ports::SourceStatus::error,
                std::nullopt,
                std::string{"camera source threw an exception: "} +
                    error.what(),
            };
        } catch (...) {
            return {
                ports::SourceStatus::error,
                std::nullopt,
                "camera source threw a non-standard exception",
            };
        }
    }

    void produce() noexcept {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_requested_) break;
            }
            ports::SourceResult item = readOne();
            const bool terminal = isTerminal(item.status);
            std::unique_lock<std::mutex> lock(mutex_);
            if (overflow_ == SourceQueueOverflowPolicy::block) {
                not_full_.wait(lock, [&] {
                    return stop_requested_ || queue_.size() < capacity_;
                });
                if (stop_requested_) break;
            } else if (queue_.size() >= capacity_) {
                if (queue_.front().status == ports::SourceStatus::frame) {
                    ++frames_dropped_;
                }
                queue_.pop_front();
            }
            queue_.push_back(std::move(item));
            if (queue_.size() > high_watermark_) {
                high_watermark_ = queue_.size();
            }
            if (terminal) producer_finished_ = true;
            lock.unlock();
            not_empty_.notify_one();
            if (terminal) return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            producer_finished_ = true;
        }
        not_empty_.notify_all();
    }

    ports::CameraSource& source_;
    const std::size_t capacity_;
    const SourceQueueOverflowPolicy overflow_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<ports::SourceResult> queue_;
    std::thread producer_;
    bool stop_requested_{false};
    bool producer_finished_{false};
    std::size_t frames_dropped_{0};
    std::size_t high_watermark_{0};
};

}  // namespace

StereoPipelineRunner::StereoPipelineRunner(
    StereoPipelineRunnerConfig config,
    ports::CameraSource& source,
    StereoObservationService& service)
    : config_(std::move(config)), source_(source), service_(service) {}

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
    std::unique_ptr<BoundedSourceQueue> source_queue;
    auto refreshQueueMetrics = [&] {
        if (!source_queue) return;
        summary.frames_dropped = source_queue->framesDropped();
        summary.queue_high_watermark = source_queue->highWatermark();
    };
    auto stopQueue = [&] {
        if (!source_queue) return;
        source_queue->stop();
        refreshQueueMetrics();
    };
    auto fail = [&](RunStatus status, std::string message) {
        message = messageOr(std::move(message), "stereo pipeline failed");
        stopQueue();
        service_.abort(message);
        summary.status = status;
        summary.message = std::move(message);
        finished_ = true;
        return summary;
    };
    auto finalize = [&](RunStatus status, std::string message)
        -> std::optional<RunSummary> {
        stopQueue();
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
        return finalize(RunStatus::stopped, "stereo pipeline stop requested");
    };
    auto reportProgress = [&]() -> std::optional<RunSummary> {
        if (!config_.progress_callback ||
            config_.progress_interval_frames == 0 ||
            summary.frames_received == 0 ||
            summary.frames_received % config_.progress_interval_frames != 0) {
            return std::nullopt;
        }
        refreshQueueMetrics();
        try {
            config_.progress_callback(summary);
        } catch (const std::exception& error) {
            return fail(
                RunStatus::lifecycle_error,
                std::string{"progress callback threw an exception: "} +
                    error.what());
        } catch (...) {
            return fail(
                RunStatus::lifecycle_error,
                "progress callback threw a non-standard exception");
        }
        return std::nullopt;
    };

    if (config_.source_queue_capacity > 0) {
        try {
            source_queue = std::make_unique<BoundedSourceQueue>(
                source_,
                config_.source_queue_capacity,
                config_.source_queue_overflow);
            source_queue->start();
        } catch (const std::exception& error) {
            return fail(
                RunStatus::lifecycle_error,
                std::string{"cannot start camera source queue: "} +
                    error.what());
        } catch (...) {
            return fail(
                RunStatus::lifecycle_error,
                "cannot start camera source queue");
        }
    }

    while (true) {
        if (auto stopped = stopRequested()) return *stopped;
        ports::SourceResult source_result;
        if (source_queue) {
            auto popped = source_queue->popFor(std::chrono::milliseconds{10});
            if (!popped.item) {
                if (popped.producer_finished) {
                    return fail(
                        RunStatus::source_error,
                        "camera source queue ended without a terminal record");
                }
                continue;
            }
            source_result = std::move(*popped.item);
        } else {
            try {
                source_result = source_.next();
            } catch (const std::exception& error) {
                return fail(
                    RunStatus::source_error,
                    std::string{"camera source threw an exception: "} +
                        error.what());
            } catch (...) {
                return fail(
                    RunStatus::source_error,
                    "camera source threw a non-standard exception");
            }
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
            } else if (process_result.status == ProcessStatus::no_match) {
                ++summary.no_matches;
            } else {
                return fail(
                    RunStatus::processing_error,
                    messageOr(
                        std::move(process_result.message),
                        "stereo observation service rejected a frame"));
            }
            if (auto failed = reportProgress()) return *failed;
            break;
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
        case ports::SourceStatus::end_of_stream:
            return *finalize(
                RunStatus::completed, std::move(source_result.message));
        default:
            return fail(
                RunStatus::source_protocol_error,
                "camera source returned an unknown status");
        }
    }
}

}  // namespace volleyball::services
