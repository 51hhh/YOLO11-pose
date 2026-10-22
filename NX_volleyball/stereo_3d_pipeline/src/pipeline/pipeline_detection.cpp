#include "pipeline.h"
#include "pipeline_depth_modes.h"
#include "../utils/logger.h"

#include <algorithm>

namespace stereo3d {

namespace {

void limitDetectionsByConfidence(std::vector<Detection>& detections,
                                 int max_detections) {
    if (max_detections <= 0 ||
        detections.size() <= static_cast<size_t>(max_detections)) {
        return;
    }
    auto by_confidence = [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    };
    const auto keep_end = detections.begin() + max_detections;
    std::nth_element(detections.begin(), keep_end, detections.end(), by_confidence);
    detections.resize(static_cast<size_t>(max_detections));
    std::sort(detections.begin(), detections.end(), by_confidence);
}

}  // namespace

TRTDetector* Pipeline::getDetector(int /*frame_id*/) const {
    return detector_.get();
}

cudaStream_t Pipeline::getDLAStream(int /*frame_id*/) const {
    return streams_.cudaStreamDLA;
}

TRTDetector* Pipeline::getRightDetector() const {
    return detector_right_.get();
}

cudaStream_t Pipeline::getRightDLAStream(int /*frame_id*/) const {
    return streams_.cudaStreamDLA_R;
}

bool Pipeline::dualYoloEnabled() const {
    return config_.dual_yolo.enabled && detector_right_;
}

bool Pipeline::leftDetectorUsesBGR() const {
    return isBGRFormat(config_.detector_input_format);
}

bool Pipeline::rightDetectorUsesBGR() const {
    const std::string fmt = config_.dual_yolo.right_input_format.empty()
        ? config_.detector_input_format : config_.dual_yolo.right_input_format;
    return isBGRFormat(fmt);
}

bool Pipeline::colorPipelineEnabled() const {
    return leftDetectorUsesBGR() ||
           (config_.dual_yolo.enabled && rightDetectorUsesBGR());
}

float Pipeline::activeDisparityOffset() const {
    return config_.disparity_offset.enabled
        ? config_.disparity_offset.d0
        : 0.0f;
}

bool Pipeline::checkCudaFrame(cudaError_t status, FrameSlot& slot,
                              const char* operation) const {
    if (status == cudaSuccess) return true;
    LOG_ERROR("[Pipeline] %s failed: frame=%d err=%s",
              operation, slot.frame_id, cudaGetErrorString(status));
    slot.grab_failed = true;
    return false;
}

bool Pipeline::recordDetectDoneEvents(FrameSlot& slot) const {
    slot.detect_done_event_recorded = false;
    slot.detect_right_done_event_recorded = false;
    bool ok = true;

    // 左目 lock/enqueue 失败时也要刷新 event，否则下游可能等待到旧 slot 事件。
    const cudaError_t left_status = cudaEventRecord(
        slot.evtDetectDone,
        slot.detection_submitted
            ? getDLAStream(slot.frame_id)
            : streams_.cudaStreamGPU);
    if (checkCudaFrame(left_status, slot, "record left detect completion event")) {
        slot.detect_done_event_recorded = true;
    } else {
        ok = false;
    }

    if (dualYoloEnabled() && slot.is_detect_frame && slot.right_detection_submitted) {
        const cudaError_t right_status = cudaEventRecord(
            slot.evtDetectRightDone, getRightDLAStream(slot.frame_id));
        if (checkCudaFrame(right_status, slot,
                           "record right detect completion event")) {
            slot.detect_right_done_event_recorded = true;
        } else {
            ok = false;
        }
    }
    return ok;
}

bool Pipeline::waitDetectDone(cudaStream_t stream, FrameSlot& slot) const {
    bool ok = true;
    auto wait_or_fallback_sync = [&](bool source_may_have_work, bool recorded,
                                     cudaEvent_t event,
                                     cudaStream_t source_stream,
                                     const char* wait_operation,
                                     const char* sync_operation) {
        if (recorded) {
            if (!checkCudaFrame(cudaStreamWaitEvent(stream, event, 0),
                                slot, wait_operation)) {
                ok = false;
            } else {
                return;
            }
        }

        // event 未由本帧成功记录时绝不能等待旧 event。若已有推理提交，
        // 直接同步源 stream，确保 ring slot 的缓冲不会在 GPU 仍使用时复用。
        if (source_may_have_work &&
            !checkCudaFrame(cudaStreamSynchronize(source_stream),
                            slot, sync_operation)) {
            ok = false;
        }
        if (!recorded) ok = false;
    };

    wait_or_fallback_sync(
        slot.detection_submitted || slot.detect_done_event_recorded,
        slot.detect_done_event_recorded,
        slot.evtDetectDone,
        slot.detection_submitted
            ? getDLAStream(slot.frame_id)
            : streams_.cudaStreamGPU,
        "wait for left detect completion event",
        "fallback synchronize left detector stream");

    if (dualYoloEnabled() && slot.is_detect_frame &&
        slot.right_detection_submitted) {
        wait_or_fallback_sync(
            true, slot.detect_right_done_event_recorded,
            slot.evtDetectRightDone, getRightDLAStream(slot.frame_id),
            "wait for right detect completion event",
            "fallback synchronize right detector stream");
    }
    return ok;
}

bool Pipeline::synchronizeDetectDone(FrameSlot& slot,
                                     const char* operation) const {
    const bool wait_ok = waitDetectDone(streams_.cudaStreamFuse, slot);
    const bool sync_ok = checkCudaFrame(
        cudaStreamSynchronize(streams_.cudaStreamFuse), slot, operation);
    return wait_ok && sync_ok;
}

bool Pipeline::detectEventsReady(const FrameSlot& slot) const {
    if (!slot.is_detect_frame || !slot.detection_submitted ||
        !slot.detect_done_event_recorded) {
        return false;
    }

    cudaError_t err = cudaEventQuery(slot.evtDetectDone);
    if (err == cudaErrorNotReady) {
        return false;
    }
    if (err != cudaSuccess) {
        LOG_WARN("[Pipeline] left detect event query failed: frame=%d err=%s",
                 slot.frame_id, cudaGetErrorString(err));
        return false;
    }

    if (dualYoloEnabled()) {
        if (!slot.right_detection_submitted ||
            !slot.detect_right_done_event_recorded) {
            return false;
        }
        err = cudaEventQuery(slot.evtDetectRightDone);
        if (err == cudaErrorNotReady) {
            return false;
        }
        if (err != cudaSuccess) {
            LOG_WARN("[Pipeline] right detect event query failed: frame=%d err=%s",
                     slot.frame_id, cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

void Pipeline::collectRightDetections(FrameSlot& slot, int slot_index) {
    slot.detections_right.clear();
    if (!dualYoloEnabled() || !slot.is_detect_frame ||
        !slot.right_detection_submitted) {
        return;
    }

    slot.detections_right = detector_right_->collect(slot_index,
                                                     config_.rect_width,
                                                     config_.rect_height);
    limitDetectionsByConfidence(slot.detections_right, config_.max_detections);
}

void Pipeline::collectRoiDetections(FrameSlot& slot, int slot_index) {
    auto* det = getDetector(slot.frame_id);
    if (slot.detection_submitted) {
        slot.detections = det->collect(slot_index,
                                       config_.rect_width,
                                       config_.rect_height);
        limitDetectionsByConfidence(slot.detections, config_.max_detections);
    } else {
        slot.detections.clear();
    }
    collectRightDetections(slot, slot_index);
}

}  // namespace stereo3d
