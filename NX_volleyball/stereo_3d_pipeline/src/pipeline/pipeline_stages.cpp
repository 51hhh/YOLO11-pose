#include "pipeline.h"
#include "pipeline_roi_match_helpers.h"
#include "../utils/logger.h"

#include <vpi/algo/ConvertImageFormat.h>

#include <cstdint>
#include <chrono>

// Custom CUDA Bayer -> BGR8 kernel in detect_preprocess.cu.
extern "C" void launchBayerToBGR8(const unsigned char* bayer, unsigned char* bgr,
                                   int width, int height,
                                   int bayer_pitch, int bgr_pitch,
                                   cudaStream_t stream);

namespace stereo3d {

void Pipeline::stage0_grab_and_rectify(FrameSlot& slot, bool grab_preloaded) {
    NVTX_RANGE("Stage0_GrabRect");

#ifdef HIK_CAMERA_ENABLED
    if (camera_ && !grab_preloaded) {
        VPIImageData imgDataL, imgDataR;
        VPIStatus stL = vpiImageLockData(
            slot.rawL, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataL);
        VPIStatus stR = vpiImageLockData(
            slot.rawR, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgDataR);

        GrabResult resL, resR;
        bool grab_ok = false;
        if (stL == VPI_SUCCESS && stR == VPI_SUCCESS) {
            grab_ok = camera_->grabFramePair(
                static_cast<uint8_t*>(imgDataL.buffer.pitch.planes[0].data),
                static_cast<uint8_t*>(imgDataR.buffer.pitch.planes[0].data),
                imgDataL.buffer.pitch.planes[0].pitchBytes,
                imgDataR.buffer.pitch.planes[0].pitchBytes,
                1000, resL, resR);
            const uint64_t fallback_capture_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            slot.host_capture_timestamp_ns = chooseCaptureTimestampNs(
                resL.host_timestamp, resR.host_timestamp, fallback_capture_ns);
            slot.left_timestamp_us = resL.timestamp_us;
            slot.right_timestamp_us = resR.timestamp_us;
            slot.left_frame_number = resL.frame_number;
            slot.right_frame_number = resR.frame_number;
            slot.left_frame_counter = resL.frame_counter;
            slot.right_frame_counter = resR.frame_counter;
            slot.left_trigger_index = resL.trigger_index;
            slot.right_trigger_index = resR.trigger_index;
            slot.stereo_timestamp_residual_ns =
                resL.stereo_timestamp_residual_ns;
        } else {
            LOG_WARN("[Pipeline] stage0 raw lock failed: L=%d R=%d",
                     (int)stL, (int)stR);
        }

        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rawL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rawR);

        if (!grab_ok) {
            slot.grab_failed = true;
            LOG_WARN("[Pipeline] Frame %d grab failed, skipping", slot.frame_id);
        }
    }
#endif

    vpiStreamSync(streams_.vpiStreamPVA);

    if (config_.tnr_enabled) {
        ScopedTimer tt("TNR");

        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rawL, tnrNV12L_, nullptr);
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rawR, tnrNV12R_, nullptr);

        VPITNRParams tnrParams;
        vpiInitTemporalNoiseReductionParams(&tnrParams);
        tnrParams.preset = config_.tnr_preset;
        tnrParams.strength = config_.tnr_strength;

        VPIImage prevL = tnrFirstFrame_ ? nullptr : tnrOutNV12L_;
        VPIImage prevR = tnrFirstFrame_ ? nullptr : tnrOutNV12R_;

        vpiSubmitTemporalNoiseReduction(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                         tnrPayloadL_, prevL,
                                         tnrNV12L_, tnrOutNV12L_, &tnrParams);
        vpiSubmitTemporalNoiseReduction(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                         tnrPayloadR_, prevR,
                                         tnrNV12R_, tnrOutNV12R_, &tnrParams);

        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    tnrOutNV12L_, slot.rawL, nullptr);
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    tnrOutNV12R_, slot.rawR, nullptr);

        tnrFirstFrame_ = false;
        globalPerf().record("TNR", tt.elapsedMs());
    }

    if (colorPipelineEnabled()) {
        const int rw = config_.camera.width;
        const int rh = config_.camera.height;

        launchBayerToBGR8(
            static_cast<const unsigned char*>(slot.rawL_gpu.data),
            static_cast<unsigned char*>(slot.tempBGR_L_gpu.data),
            rw, rh,
            slot.rawL_gpu.pitchBytes,
            slot.tempBGR_L_gpu.pitchBytes,
            streams_.cudaStreamBGR);

        launchBayerToBGR8(
            static_cast<const unsigned char*>(slot.rawR_gpu.data),
            static_cast<unsigned char*>(slot.tempBGR_R_gpu.data),
            rw, rh,
            slot.rawR_gpu.pitchBytes,
            slot.tempBGR_R_gpu.pitchBytes,
            streams_.cudaStreamBGR);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("BayerToBGR kernel launch failed: %s",
                      cudaGetErrorString(err));
            slot.grab_failed = true;
            NVTX_RANGE_POP();
            return;
        }
        err = cudaStreamSynchronize(streams_.cudaStreamBGR);
        if (err != cudaSuccess) {
            LOG_ERROR("BayerToBGR stream sync failed: %s",
                      cudaGetErrorString(err));
            slot.grab_failed = true;
            NVTX_RANGE_POP();
            return;
        }

        rectifier_->submitBGR(streams_.vpiStreamPVA,
                              slot.tempBGR_L, slot.tempBGR_R,
                              slot.rectBGR_vpiL, slot.rectBGR_vpiR);

        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rectBGR_vpiL, slot.rectGray_vpiL, nullptr);
        vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                                    slot.rectBGR_vpiR, slot.rectGray_vpiR, nullptr);
    } else {
        rectifier_->submit(streams_.vpiStreamPVA,
                           slot.rawL, slot.rawR,
                           slot.rectGray_vpiL, slot.rectGray_vpiR);
    }

    NVTX_RANGE_POP();
}

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
