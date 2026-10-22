/**
 * @file pipeline_run_state.h
 * @brief Pipeline 运行、失败与清理责任的无依赖状态机。
 *
 * running=false 只表示不再接受/调度新工作，不能表示线程和设备已经清理。
 * cleanup_pending 独立保存一次且仅一次的 join/stream drain/camera stop 责任。
 */

#ifndef STEREO_3D_PIPELINE_PIPELINE_RUN_STATE_H_
#define STEREO_3D_PIPELINE_PIPELINE_RUN_STATE_H_

#include <atomic>

namespace stereo3d {

enum class PipelineRunPhase {
    IDLE,
    RUNNING,
    CLEANUP_PENDING,
    CLEANING,
    FAILED_CLEANUP_PENDING,
    FAILED_CLEANING,
    FAILED_CLEANED,
};

class PipelineRunState {
public:
    /** @return true 表示本次真正开始；false 表示已运行、待清理或已失败。 */
    bool begin() {
        PipelineRunPhase expected = PipelineRunPhase::IDLE;
        return phase_.compare_exchange_strong(
            expected, PipelineRunPhase::RUNNING,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    void requestStop() {
        PipelineRunPhase expected = PipelineRunPhase::RUNNING;
        phase_.compare_exchange_strong(
            expected, PipelineRunPhase::CLEANUP_PENDING,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    void markFailed() {
        PipelineRunPhase phase = phase_.load(std::memory_order_acquire);
        while (true) {
            PipelineRunPhase target = PipelineRunPhase::FAILED_CLEANED;
            switch (phase) {
                case PipelineRunPhase::RUNNING:
                case PipelineRunPhase::CLEANUP_PENDING:
                    target = PipelineRunPhase::FAILED_CLEANUP_PENDING;
                    break;
                case PipelineRunPhase::CLEANING:
                    target = PipelineRunPhase::FAILED_CLEANING;
                    break;
                case PipelineRunPhase::IDLE:
                    target = PipelineRunPhase::FAILED_CLEANED;
                    break;
                case PipelineRunPhase::FAILED_CLEANUP_PENDING:
                case PipelineRunPhase::FAILED_CLEANING:
                case PipelineRunPhase::FAILED_CLEANED:
                    return;
            }
            if (phase_.compare_exchange_weak(
                    phase, target,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return;
            }
        }
    }

    /** @return true 时，调用方取得本轮唯一的清理责任。 */
    bool claimCleanup() {
        PipelineRunPhase phase = phase_.load(std::memory_order_acquire);
        while (phase == PipelineRunPhase::CLEANUP_PENDING ||
               phase == PipelineRunPhase::FAILED_CLEANUP_PENDING) {
            const PipelineRunPhase target =
                phase == PipelineRunPhase::CLEANUP_PENDING
                    ? PipelineRunPhase::CLEANING
                    : PipelineRunPhase::FAILED_CLEANING;
            if (phase_.compare_exchange_weak(
                    phase, target,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    /** 清理 owner 在所有 join/drain/stop 完成后调用。 */
    bool finishCleanup() {
        PipelineRunPhase phase = phase_.load(std::memory_order_acquire);
        while (phase == PipelineRunPhase::CLEANING ||
               phase == PipelineRunPhase::FAILED_CLEANING) {
            const PipelineRunPhase target =
                phase == PipelineRunPhase::CLEANING
                    ? PipelineRunPhase::IDLE
                    : PipelineRunPhase::FAILED_CLEANED;
            if (phase_.compare_exchange_weak(
                    phase, target,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    bool isRunning() const {
        return phase_.load(std::memory_order_acquire) ==
               PipelineRunPhase::RUNNING;
    }

    bool isIdle() const {
        return phase_.load(std::memory_order_acquire) ==
               PipelineRunPhase::IDLE;
    }

    bool hasFailed() const {
        const PipelineRunPhase phase = phase_.load(std::memory_order_acquire);
        return phase == PipelineRunPhase::FAILED_CLEANUP_PENDING ||
               phase == PipelineRunPhase::FAILED_CLEANING ||
               phase == PipelineRunPhase::FAILED_CLEANED;
    }

    PipelineRunPhase phase() const {
        return phase_.load(std::memory_order_acquire);
    }

private:
    std::atomic<PipelineRunPhase> phase_{PipelineRunPhase::IDLE};
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PIPELINE_RUN_STATE_H_
