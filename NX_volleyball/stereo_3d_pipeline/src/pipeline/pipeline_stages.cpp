#include "pipeline.h"
#include "pipeline_roi_match_helpers.h"
#include "../utils/logger.h"

#include <cstdint>

namespace stereo3d {

void Pipeline::stage1_detect(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage1_Detect");
    slot.detection_submitted = false;
    slot.right_detection_submitted = false;

    auto* det = getDetector(slot.frame_id);
    auto stream = getDLAStream(slot.frame_id);
    cudaStreamWaitEvent(stream, slot.evtRectDone, 0);

    // BGR mode uses rectified BGR; gray mode uses rectified gray.
    const auto& leftDetectGpu = leftDetectorUsesBGR()
        ? slot.rectBGR_L_gpu
        : slot.rectGray_L_gpu;
    if (!leftDetectGpu.data || leftDetectGpu.pitchBytes <= 0) {
        LOG_WARN("stage1_detect: invalid left detect CUDA pointer");
        NVTX_RANGE_POP();
        return;
    }

    const bool submitted = det->enqueue(slot_index, leftDetectGpu.data,
                                        leftDetectGpu.pitchBytes,
                                        config_.rect_width, config_.rect_height,
                                        stream);

    if (!submitted) {
        LOG_WARN("stage1_detect: left TRT enqueue failed");
        NVTX_RANGE_POP();
        return;
    }
    slot.detection_submitted = true;

    if (dualYoloEnabled() && slot.is_detect_frame) {
        auto* detR = getRightDetector();
        auto streamR = getRightDLAStream(slot.frame_id);
        cudaStreamWaitEvent(streamR, slot.evtRectDone, 0);

        const auto& rightDetectGpu = rightDetectorUsesBGR()
            ? slot.rectBGR_R_gpu
            : slot.rectGray_R_gpu;
        if (!rightDetectGpu.data || rightDetectGpu.pitchBytes <= 0) {
            LOG_WARN("stage1_detect: invalid right detect CUDA pointer");
            NVTX_RANGE_POP();
            return;
        }

        const bool submittedR = detR->enqueue(slot_index, rightDetectGpu.data,
                                              rightDetectGpu.pitchBytes,
                                              config_.rect_width,
                                              config_.rect_height,
                                              streamR);

        if (!submittedR) {
            LOG_WARN("stage1_detect: right TRT enqueue failed");
            NVTX_RANGE_POP();
            return;
        }
        slot.right_detection_submitted = true;
    }
    NVTX_RANGE_POP();
}

void Pipeline::stage2_stereo(FrameSlot& slot) {
    NVTX_RANGE("Stage2_Stereo");

    switch (config_.disparity_strategy) {
        case DisparityStrategy::FULL_FRAME:
            stereo_->compute(streams_.vpiStreamGPU,
                             slot.rectGray_vpiL, slot.rectGray_vpiR,
                             slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::HALF_RESOLUTION:
            stereo_->computeHalfRes(streams_.vpiStreamGPU,
                                    slot.rectGray_vpiL, slot.rectGray_vpiR,
                                    slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::ROI_ONLY:
            // ROI mode uses separate ROI matcher in pipelineLoopROI stage2.
            LOG_ERROR("stage2_stereo called in ROI_ONLY mode - this is a bug");
            return;
    }

    // Downstream Stage3 syncs vpiStreamGPU before fusion.
    NVTX_RANGE_POP();
}

void Pipeline::stage3_fuse(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage3_Fuse");

    slot.results.clear();

    collectRoiDetections(slot, slot_index);

    VPIImageData dispData;
    VPIStatus st = vpiImageLockData(slot.disparityMap, VPI_LOCK_READ,
                     VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dispData);
    if (st != VPI_SUCCESS) {
        LOG_WARN("stage3_fuse: vpiImageLockData failed (%d)", (int)st);
        NVTX_RANGE_POP();
        return;
    }

    const int16_t* disp_ptr = static_cast<const int16_t*>(
        dispData.buffer.pitch.planes[0].data);
    int disp_pitch = dispData.buffer.pitch.planes[0].pitchBytes;

    float dispScale = (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION)
                      ? 2.0f : 1.0f;
    slot.results = fusion_->computeBatch(slot.detections, disp_ptr, disp_pitch,
                                         config_.rect_width, config_.rect_height,
                                         streams_.cudaStreamFuse, dispScale);
    stampFrameMetadata(slot);

    vpiImageUnlock(slot.disparityMap);
    NVTX_RANGE_POP();
}

}  // namespace stereo3d
