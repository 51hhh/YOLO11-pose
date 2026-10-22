#pragma once

#include "volleyball/ports/detector.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>

namespace volleyball::services {

// 两个长期工作线程分别独占左右 Detector。一次双目任务先同时唤醒两侧，
// 再等待两侧完成，使独立 TensorRT stream 能真正重叠提交和执行。
class StereoDetectorExecutor final {
public:
    StereoDetectorExecutor(
        ports::Detector& left_detector,
        ports::Detector& right_detector);
    ~StereoDetectorExecutor();

    StereoDetectorExecutor(const StereoDetectorExecutor&) = delete;
    StereoDetectorExecutor& operator=(const StereoDetectorExecutor&) = delete;

    [[nodiscard]] std::pair<ports::DetectionResult, ports::DetectionResult>
    detect(const ports::RectifiedStereoFrame& frame);

private:
    void worker(std::size_t side) noexcept;

    ports::Detector* detectors_[2];
    std::thread workers_[2];
    std::mutex mutex_;
    std::condition_variable task_ready_;
    std::condition_variable task_done_;
    ports::RectifiedCameraFrame frames_[2];
    ports::DetectionResult results_[2];
    std::uint64_t generation_{0};
    std::uint64_t completed_generation_[2]{0, 0};
    bool task_active_{false};
    bool stopping_{false};
};

}  // namespace volleyball::services
