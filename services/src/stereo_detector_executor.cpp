#include "stereo_detector_executor.hpp"

#include <exception>
#include <stdexcept>
#include <string>

namespace volleyball::services {
namespace {

ports::DetectionResult detectorFailure(
    const ports::RectifiedCameraFrame& frame,
    std::string message) {
    ports::DetectionResult result;
    result.batch.frame = frame.source;
    result.message = std::move(message);
    return result;
}

}  // namespace

StereoDetectorExecutor::StereoDetectorExecutor(
    ports::Detector& left_detector,
    ports::Detector& right_detector)
    : detectors_{&left_detector, &right_detector} {
    workers_[0] = std::thread(&StereoDetectorExecutor::worker, this, 0U);
    try {
        workers_[1] = std::thread(&StereoDetectorExecutor::worker, this, 1U);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        task_ready_.notify_all();
        workers_[0].join();
        throw;
    }
}

StereoDetectorExecutor::~StereoDetectorExecutor() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    task_ready_.notify_all();
    for (auto& worker_thread : workers_) {
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
}

std::pair<ports::DetectionResult, ports::DetectionResult>
StereoDetectorExecutor::detect(const ports::RectifiedStereoFrame& frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_) {
        throw std::logic_error("stereo detector executor is stopping");
    }
    if (task_active_) {
        throw std::logic_error(
            "stereo detector executor does not accept overlapping calls");
    }

    frames_[0] = frame.left;
    frames_[1] = frame.right;
    results_[0] = {};
    results_[1] = {};
    task_active_ = true;
    ++generation_;
    if (generation_ == 0) {
        ++generation_;
    }
    const std::uint64_t requested_generation = generation_;
    task_ready_.notify_all();

    task_done_.wait(lock, [&] {
        return completed_generation_[0] == requested_generation &&
               completed_generation_[1] == requested_generation;
    });
    task_active_ = false;
    return {std::move(results_[0]), std::move(results_[1])};
}

void StereoDetectorExecutor::worker(std::size_t side) noexcept {
    std::uint64_t observed_generation = 0;
    while (true) {
        ports::RectifiedCameraFrame frame;
        std::uint64_t task_generation = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_ready_.wait(lock, [&] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }
            task_generation = generation_;
            frame = frames_[side];
        }

        ports::DetectionResult result;
        try {
            result = detectors_[side]->detect(frame);
        } catch (const std::exception& exception) {
            result = detectorFailure(
                frame,
                std::string{"detector threw: "} + exception.what());
        } catch (...) {
            result = detectorFailure(
                frame, "detector threw a non-standard exception");
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            results_[side] = std::move(result);
            observed_generation = task_generation;
            completed_generation_[side] = task_generation;
        }
        task_done_.notify_one();
    }
}

}  // namespace volleyball::services
