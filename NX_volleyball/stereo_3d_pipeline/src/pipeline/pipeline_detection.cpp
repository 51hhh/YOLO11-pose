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

void Pipeline::recordDetectDoneEvents(FrameSlot& slot) const {
    // 左目 lock/enqueue 失败时也要刷新 event，否则下游可能等待到旧 slot 事件。
    cudaEventRecord(slot.evtDetectDone,
                    slot.detection_submitted
                        ? getDLAStream(slot.frame_id)
                        : streams_.cudaStreamGPU);
    if (dualYoloEnabled() && slot.is_detect_frame && slot.right_detection_submitted) {
        cudaEventRecord(slot.evtDetectRightDone, getRightDLAStream(slot.frame_id));
    }
}

void Pipeline::waitDetectDone(cudaStream_t stream, const FrameSlot& slot) const {
    cudaStreamWaitEvent(stream, slot.evtDetectDone, 0);
    if (dualYoloEnabled() && slot.is_detect_frame && slot.right_detection_submitted) {
        cudaStreamWaitEvent(stream, slot.evtDetectRightDone, 0);
    }
}

bool Pipeline::detectEventsReady(const FrameSlot& slot) const {
    if (!slot.is_detect_frame || !slot.detection_submitted) {
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
        if (!slot.right_detection_submitted) {
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
