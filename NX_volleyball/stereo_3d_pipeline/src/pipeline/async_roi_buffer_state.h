/**
 * @file async_roi_buffer_state.h
 * @brief 异步 ROI 快照缓冲的可测试所有权状态机。
 *
 * CUDA/VPI 同步只负责证明资源已经不再被设备访问；本状态机负责决定
 * buffer 是否允许重新进入空闲池。QUARANTINED 是单向状态，只能在整个
 * Pipeline 销毁并重新初始化后恢复。
 */

#ifndef STEREO_3D_PIPELINE_ASYNC_ROI_BUFFER_STATE_H_
#define STEREO_3D_PIPELINE_ASYNC_ROI_BUFFER_STATE_H_

namespace stereo3d {

enum class AsyncRoiBufferOwnership {
    UNINITIALIZED,
    FREE,
    IN_USE,
    WAITING_DIAGNOSTIC,
    QUARANTINED,
};

enum class AsyncRoiBufferTransition {
    NONE,
    ENQUEUE_FREE,
    INVALID,
};

class AsyncRoiBufferState {
public:
    void initializeFree() {
        ownership_ = AsyncRoiBufferOwnership::FREE;
        diagnostic_refs_ = 0;
    }

    bool acquire() {
        if (ownership_ != AsyncRoiBufferOwnership::FREE ||
            diagnostic_refs_ != 0) {
            return false;
        }
        ownership_ = AsyncRoiBufferOwnership::IN_USE;
        return true;
    }

    bool addDiagnosticRef() {
        if (ownership_ != AsyncRoiBufferOwnership::IN_USE) {
            return false;
        }
        ++diagnostic_refs_;
        return true;
    }

    AsyncRoiBufferTransition release() {
        if (ownership_ == AsyncRoiBufferOwnership::QUARANTINED) {
            return AsyncRoiBufferTransition::NONE;
        }
        if (ownership_ != AsyncRoiBufferOwnership::IN_USE) {
            return AsyncRoiBufferTransition::INVALID;
        }
        if (diagnostic_refs_ > 0) {
            ownership_ = AsyncRoiBufferOwnership::WAITING_DIAGNOSTIC;
            return AsyncRoiBufferTransition::NONE;
        }
        ownership_ = AsyncRoiBufferOwnership::FREE;
        return AsyncRoiBufferTransition::ENQUEUE_FREE;
    }

    AsyncRoiBufferTransition releaseDiagnosticRef() {
        if (diagnostic_refs_ <= 0) {
            return AsyncRoiBufferTransition::INVALID;
        }
        --diagnostic_refs_;
        if (ownership_ == AsyncRoiBufferOwnership::WAITING_DIAGNOSTIC &&
            diagnostic_refs_ == 0) {
            ownership_ = AsyncRoiBufferOwnership::FREE;
            return AsyncRoiBufferTransition::ENQUEUE_FREE;
        }
        return AsyncRoiBufferTransition::NONE;
    }

    void quarantine() {
        ownership_ = AsyncRoiBufferOwnership::QUARANTINED;
    }

    AsyncRoiBufferOwnership ownership() const { return ownership_; }
    int diagnosticRefs() const { return diagnostic_refs_; }
    bool isInUse() const {
        return ownership_ == AsyncRoiBufferOwnership::IN_USE;
    }
    bool isQuarantined() const {
        return ownership_ == AsyncRoiBufferOwnership::QUARANTINED;
    }

private:
    AsyncRoiBufferOwnership ownership_ =
        AsyncRoiBufferOwnership::UNINITIALIZED;
    int diagnostic_refs_ = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ASYNC_ROI_BUFFER_STATE_H_
