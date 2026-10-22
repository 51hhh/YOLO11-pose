/**
 * @file async_grab_state.h
 * @brief Single-flight handshake between the pipeline and camera grab thread.
 *
 * stop() rejects new work but deliberately preserves an accepted request. The
 * camera thread must finish that request and publish completion before the
 * pipeline may touch the destination image.
 */

#ifndef STEREO_3D_PIPELINE_ASYNC_GRAB_STATE_H_
#define STEREO_3D_PIPELINE_ASYNC_GRAB_STATE_H_

#include <condition_variable>
#include <mutex>

namespace stereo3d {

class AsyncGrabState {
public:
    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = true;
        pending_slot_ = -1;
        active_ = false;
        done_ = false;
        result_ok_ = false;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            accepting_ = false;
        }
        request_cv_.notify_all();
        done_cv_.notify_all();
    }

    bool request(int slot_index, int slot_count) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!accepting_ || active_ || pending_slot_ >= 0 ||
                slot_index < 0 || slot_index >= slot_count) {
                return false;
            }
            pending_slot_ = slot_index;
            active_ = true;
            done_ = false;
            result_ok_ = false;
        }
        request_cv_.notify_one();
        return true;
    }

    /**
     * @return true with slot_index for accepted work; false after stop once
     *         no accepted request remains.
     */
    bool waitForRequest(int& slot_index) {
        std::unique_lock<std::mutex> lock(mutex_);
        request_cv_.wait(lock, [this] {
            return pending_slot_ >= 0 || !accepting_;
        });
        if (pending_slot_ < 0) return false;
        slot_index = pending_slot_;
        pending_slot_ = -1;
        return true;
    }

    bool complete(bool ok) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!active_ || pending_slot_ >= 0) return false;
            active_ = false;
            done_ = true;
            result_ok_ = ok;
        }
        done_cv_.notify_all();
        return true;
    }

    /**
     * Wait only for the accepted request's completion. Stopping intake is not
     * completion and therefore cannot expose a raw image still being written.
     */
    bool waitForCompletion() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] {
            return done_ || !active_;
        });
        return done_ && result_ok_;
    }

private:
    std::mutex mutex_;
    std::condition_variable request_cv_;
    std::condition_variable done_cv_;
    bool accepting_ = false;
    int pending_slot_ = -1;
    bool active_ = false;
    bool done_ = false;
    bool result_ok_ = false;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ASYNC_GRAB_STATE_H_
