#include "pipeline.h"
#include "pipeline_roi_match_helpers.h"
#include "../utils/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace stereo3d {

void Pipeline::tracker_handle_detect_result(FrameSlot& slot) {
    if (!tracker_) return;

    if (slot.detections.empty()) {
        tracker_lost_count_++;
        if (tracker_lost_count_ >= config_.tracker.lost_threshold) {
            if (tracker_state_ != TrackerState::IDLE) {
                tracker_state_ = TrackerState::LOST;
                tracker_->reset();
                tracker_state_ = TrackerState::IDLE;
                LOG_INFO("[Tracker] LOST -> IDLE (no YOLO det for %d frames)",
                         tracker_lost_count_);
            }
        }
        return;
    }

    const auto& best = *std::max_element(
        slot.detections.begin(), slot.detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence < b.confidence;
        });

    const uint8_t* imgPtr = static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    int imgPitch = slot.rectGray_L_gpu.pitchBytes;
    if (!imgPtr) return;

    if (tracker_state_ == TrackerState::IDLE ||
        tracker_state_ == TrackerState::LOST) {
        tracker_->setTarget(imgPtr, imgPitch,
                            config_.rect_width, config_.rect_height, best);
        tracker_state_ = TrackerState::TRACKING;
        tracker_lost_count_ = 0;
        LOG_INFO("[Tracker] setTarget: (%.0f,%.0f) %dx%d conf=%.2f",
                 best.cx, best.cy, (int)best.width, (int)best.height,
                 best.confidence);
    } else {
        tracker_->setTarget(imgPtr, imgPitch,
                            config_.rect_width, config_.rect_height, best);
        tracker_lost_count_ = 0;
    }
}

void Pipeline::tracker_infill(FrameSlot& slot) {
    if (!tracker_ || tracker_state_ != TrackerState::TRACKING) {
        slot.sot_bbox_result = SOTResult{};
        slot.bbox_source = BboxSource::NONE;
        return;
    }

    const uint8_t* imgPtr = static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    int imgPitch = slot.rectGray_L_gpu.pitchBytes;
    if (!imgPtr) {
        slot.sot_bbox_result = SOTResult{};
        slot.bbox_source = BboxSource::NONE;
        return;
    }

    SOTResult result = tracker_->track(imgPtr, imgPitch,
                                       config_.rect_width, config_.rect_height);

    if (result.valid && result.confidence >= config_.tracker.min_confidence) {
        slot.sot_bbox_result = result;
        slot.bbox_source = BboxSource::TRACKER;
        tracker_lost_count_ = 0;
    } else {
        slot.sot_bbox_result = SOTResult{};
        slot.bbox_source = BboxSource::NONE;
        tracker_lost_count_++;
        if (tracker_lost_count_ >= config_.tracker.lost_threshold) {
            tracker_state_ = TrackerState::LOST;
            tracker_->reset();
            tracker_state_ = TrackerState::IDLE;
        }
    }
}

void Pipeline::stage2_roi_fuse_tracker(FrameSlot& slot, int slot_index) {
    (void)slot_index;
    NVTX_RANGE("Stage2_ROIFuseTracker");
    std::lock_guard<std::mutex> post_lock(roi_postprocess_mutex_);
    slot.results.clear();

    if (slot.bbox_source != BboxSource::TRACKER || !slot.sot_bbox_result.valid) {
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly(fusionDtForFrame(slot));
            stampFrameMetadata(slot);
        }
        NVTX_RANGE_POP();
        return;
    }

    const auto& sot = slot.sot_bbox_result;
    Detection pseudo_det;
    pseudo_det.cx = sot.cx;
    pseudo_det.cy = sot.cy;
    pseudo_det.width = sot.width;
    pseudo_det.height = sot.height;
    pseudo_det.confidence = sot.confidence;
    pseudo_det.class_id = 0;
    slot.detections = {pseudo_det};

    if (!roi_matcher_) {
        LOG_ERROR("Tracker ROI fuse requested but ROIStereoMatcher is not initialized");
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly(fusionDtForFrame(slot));
            stampFrameMetadata(slot);
        }
        NVTX_RANGE_POP();
        return;
    }

    const uint8_t* leftPtr =
        static_cast<const uint8_t*>(slot.rectGray_L_gpu.data);
    const uint8_t* rightPtr =
        static_cast<const uint8_t*>(slot.rectGray_R_gpu.data);
    const int leftPitch = slot.rectGray_L_gpu.pitchBytes;
    const int rightPitch = slot.rectGray_R_gpu.pitchBytes;
    if (!leftPtr || !rightPtr || leftPitch <= 0 || rightPitch <= 0) {
        if (hybrid_depth_) {
            std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
            slot.results = hybrid_depth_->predictOnly(fusionDtForFrame(slot));
            stampFrameMetadata(slot);
        }
        NVTX_RANGE_POP();
        return;
    }

    std::vector<stereo3d::Object3D> roi_results;
    {
        ScopedTimer troi("Stage2_ROIMatchTracker");
        roi_results = roi_matcher_->match(
            leftPtr, leftPitch, rightPtr, rightPitch,
            config_.rect_width, config_.rect_height,
            slot.detections, streams_.cudaStreamFuse);
        globalPerf().record("Stage2_ROIMatchTracker", troi.elapsedMs());
    }

    if (hybrid_depth_) {
        std::lock_guard<std::mutex> hd_lock(hybrid_depth_mutex_);
        const double dt = fusionDtForFrame(slot);
        slot.results = hybrid_depth_->estimate(slot.detections, roi_results, dt);
        stampFrameMetadata(slot);
    } else {
        slot.results = std::move(roi_results);
        stampFrameMetadata(slot);
    }

    NVTX_RANGE_POP();
}

}  // namespace stereo3d
