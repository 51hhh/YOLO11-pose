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

    auto checkVpi = [&](VPIStatus status, const char* operation) {
        if (status == VPI_SUCCESS) return true;
        LOG_ERROR("[Pipeline] %s failed: frame=%d status=%d",
                  operation, slot.frame_id, (int)status);
        slot.grab_failed = true;
        return false;
    };

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
            slot.left_receive_monotonic_ns = resL.receive_monotonic_ns;
            slot.right_receive_monotonic_ns = resR.receive_monotonic_ns;
            slot.left_receive_realtime_ns = resL.receive_realtime_ns;
            slot.right_receive_realtime_ns = resR.receive_realtime_ns;
            slot.left_sdk_host_timestamp_raw = resL.host_timestamp;
            slot.right_sdk_host_timestamp_raw = resR.host_timestamp;
            slot.left_timestamp_us = resL.timestamp_us;
            slot.right_timestamp_us = resR.timestamp_us;
            slot.left_frame_number = resL.frame_number;
            slot.right_frame_number = resR.frame_number;
            slot.left_frame_counter = resL.frame_counter;
            slot.right_frame_counter = resR.frame_counter;
            slot.left_trigger_index = resL.trigger_index;
            slot.right_trigger_index = resR.trigger_index;
            attachPwmTriggerMetadata(slot, resL.frame_counter);
            slot.stereo_timestamp_residual_ns =
                resL.stereo_timestamp_residual_ns;
        } else {
            LOG_WARN("[Pipeline] stage0 raw lock failed: L=%d R=%d",
                     (int)stL, (int)stR);
        }

        if (stL == VPI_SUCCESS &&
            !checkVpi(vpiImageUnlock(slot.rawL), "unlock left raw image")) {
            grab_ok = false;
        }
        if (stR == VPI_SUCCESS &&
            !checkVpi(vpiImageUnlock(slot.rawR), "unlock right raw image")) {
            grab_ok = false;
        }

        if (!grab_ok) {
            slot.grab_failed = true;
            LOG_WARN("[Pipeline] Frame %d grab failed, skipping", slot.frame_id);
        }
    }
#endif

    // 异步抓取失败时 raw buffer 可能仍保存上一轮像素，绝不能继续生成一帧
    // 看似有效的校正图。VPI 提交失败也统一沿 grab_failed 进入采集失败契约。
    if (slot.grab_failed) {
        NVTX_RANGE_POP();
        return;
    }

    if (!checkVpi(vpiStreamSync(streams_.vpiStreamPVA),
                  "sync rectification stream before reuse")) {
        NVTX_RANGE_POP();
        return;
    }

    if (config_.tnr_enabled) {
        ScopedTimer tt("TNR");

        if (!checkVpi(vpiSubmitConvertImageFormat(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          slot.rawL, tnrNV12L_, nullptr),
                      "submit left raw-to-NV12") ||
            !checkVpi(vpiSubmitConvertImageFormat(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          slot.rawR, tnrNV12R_, nullptr),
                      "submit right raw-to-NV12")) {
            NVTX_RANGE_POP();
            return;
        }

        VPITNRParams tnrParams;
        if (!checkVpi(vpiInitTemporalNoiseReductionParams(&tnrParams),
                      "initialize TNR parameters")) {
            NVTX_RANGE_POP();
            return;
        }
        tnrParams.preset = config_.tnr_preset;
        tnrParams.strength = config_.tnr_strength;

        VPIImage prevL = tnrFirstFrame_ ? nullptr : tnrOutNV12L_;
        VPIImage prevR = tnrFirstFrame_ ? nullptr : tnrOutNV12R_;

        if (!checkVpi(vpiSubmitTemporalNoiseReduction(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          tnrPayloadL_, prevL, tnrNV12L_, tnrOutNV12L_,
                          &tnrParams),
                      "submit left TNR") ||
            !checkVpi(vpiSubmitTemporalNoiseReduction(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          tnrPayloadR_, prevR, tnrNV12R_, tnrOutNV12R_,
                          &tnrParams),
                      "submit right TNR") ||
            !checkVpi(vpiSubmitConvertImageFormat(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          tnrOutNV12L_, slot.rawL, nullptr),
                      "submit left NV12-to-raw") ||
            !checkVpi(vpiSubmitConvertImageFormat(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          tnrOutNV12R_, slot.rawR, nullptr),
                      "submit right NV12-to-raw")) {
            NVTX_RANGE_POP();
            return;
        }

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

        if (!rectifier_->submitBGR(streams_.vpiStreamPVA,
                                   slot.tempBGR_L, slot.tempBGR_R,
                                   slot.rectBGR_vpiL, slot.rectBGR_vpiR)) {
            slot.grab_failed = true;
            NVTX_RANGE_POP();
            return;
        }

        if (!checkVpi(vpiSubmitConvertImageFormat(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          slot.rectBGR_vpiL, slot.rectGray_vpiL, nullptr),
                      "submit left BGR-to-gray") ||
            !checkVpi(vpiSubmitConvertImageFormat(
                          streams_.vpiStreamPVA, VPI_BACKEND_CUDA,
                          slot.rectBGR_vpiR, slot.rectGray_vpiR, nullptr),
                      "submit right BGR-to-gray")) {
            NVTX_RANGE_POP();
            return;
        }
    } else {
        if (!rectifier_->submit(streams_.vpiStreamPVA,
                                slot.rawL, slot.rawR,
                                slot.rectGray_vpiL, slot.rectGray_vpiR)) {
            slot.grab_failed = true;
            NVTX_RANGE_POP();
            return;
        }
    }

    NVTX_RANGE_POP();
}

bool Pipeline::stage1_detect(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage1_Detect");
    slot.detection_submitted = false;
    slot.right_detection_submitted = false;
    slot.detect_done_event_recorded = false;
    slot.detect_right_done_event_recorded = false;

    auto fail = [&](const char* message) {
        LOG_ERROR("stage1_detect: %s (frame=%d slot=%d)",
                  message, slot.frame_id, slot_index);
        slot.grab_failed = true;
        NVTX_RANGE_POP();
        return false;
    };

    auto* det = getDetector(slot.frame_id);
    auto stream = getDLAStream(slot.frame_id);
    if (!det || !stream) {
        return fail("left detector or CUDA stream is null");
    }
    if (!checkCudaFrame(cudaStreamWaitEvent(stream, slot.evtRectDone, 0),
                        slot, "left detector wait for rectification")) {
        NVTX_RANGE_POP();
        return false;
    }

    // BGR mode uses rectified BGR; gray mode uses rectified gray.
    const auto& leftDetectGpu = leftDetectorUsesBGR()
        ? slot.rectBGR_L_gpu
        : slot.rectGray_L_gpu;
    if (!leftDetectGpu.data || leftDetectGpu.pitchBytes <= 0) {
        return fail("invalid left detect CUDA pointer");
    }

    const bool submitted = det->enqueue(slot_index, leftDetectGpu.data,
                                        leftDetectGpu.pitchBytes,
                                        config_.rect_width, config_.rect_height,
                                        stream);

    if (!submitted) {
        // enqueue() 可能在推理已入队、后续 D2H 提交失败后返回 false。
        // 此时不能立刻复用该 slot 的 TensorRT 缓冲。
        checkCudaFrame(cudaStreamSynchronize(stream), slot,
                       "drain left detector after enqueue failure");
        return fail("left TRT enqueue failed");
    }
    slot.detection_submitted = true;

    if (dualYoloEnabled() && slot.is_detect_frame) {
        auto* detR = getRightDetector();
        auto streamR = getRightDLAStream(slot.frame_id);
        if (!detR || !streamR) {
            return fail("right detector or CUDA stream is null");
        }
        if (!checkCudaFrame(cudaStreamWaitEvent(streamR, slot.evtRectDone, 0),
                            slot, "right detector wait for rectification")) {
            NVTX_RANGE_POP();
            return false;
        }

        const auto& rightDetectGpu = rightDetectorUsesBGR()
            ? slot.rectBGR_R_gpu
            : slot.rectGray_R_gpu;
        if (!rightDetectGpu.data || rightDetectGpu.pitchBytes <= 0) {
            return fail("invalid right detect CUDA pointer");
        }

        const bool submittedR = detR->enqueue(slot_index, rightDetectGpu.data,
                                              rightDetectGpu.pitchBytes,
                                              config_.rect_width,
                                              config_.rect_height,
                                              streamR);

        if (!submittedR) {
            checkCudaFrame(cudaStreamSynchronize(streamR), slot,
                           "drain right detector after enqueue failure");
            return fail("right TRT enqueue failed");
        }
        slot.right_detection_submitted = true;
    }
    NVTX_RANGE_POP();
    return true;
}

bool Pipeline::stage2_stereo(FrameSlot& slot) {
    NVTX_RANGE("Stage2_Stereo");

    bool submitted = false;
    switch (config_.disparity_strategy) {
        case DisparityStrategy::FULL_FRAME:
            submitted = stereo_->compute(
                streams_.vpiStreamGPU,
                slot.rectGray_vpiL, slot.rectGray_vpiR,
                slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::HALF_RESOLUTION:
            submitted = stereo_->computeHalfRes(
                streams_.vpiStreamGPU,
                slot.rectGray_vpiL, slot.rectGray_vpiR,
                slot.disparityMap, slot.confidenceMap);
            break;

        case DisparityStrategy::ROI_ONLY:
            // ROI mode uses separate ROI matcher in pipelineLoopROI stage2.
            LOG_ERROR("stage2_stereo called in ROI_ONLY mode - this is a bug");
            slot.grab_failed = true;
            NVTX_RANGE_POP();
            return false;
    }

    // Downstream Stage3 syncs vpiStreamGPU before fusion.
    if (!submitted) {
        LOG_ERROR("stage2_stereo: VPI submit failed (frame=%d)", slot.frame_id);
        slot.grab_failed = true;
        // HALF_RESOLUTION 可能在前几个 submit 成功、后续 submit 失败。
        // 失败路径也必须排空 stream，避免 slot 图像仍被 VPI 使用时复用。
        const VPIStatus drain_status = vpiStreamSync(streams_.vpiStreamGPU);
        if (drain_status != VPI_SUCCESS) {
            LOG_ERROR("stage2_stereo: failed to drain VPI stream "
                      "after submit error (frame=%d status=%d)",
                      slot.frame_id, (int)drain_status);
        }
    }
    NVTX_RANGE_POP();
    return submitted;
}

bool Pipeline::stage3_fuse(FrameSlot& slot, int slot_index) {
    NVTX_RANGE("Stage3_Fuse");

    slot.results.clear();

    collectRoiDetections(slot, slot_index);

    if (!fusion_) {
        LOG_ERROR("stage3_fuse: Coordinate3D fusion is not initialized "
                  "(frame=%d)", slot.frame_id);
        slot.grab_failed = true;
        NVTX_RANGE_POP();
        return false;
    }

    VPIImageData dispData;
    VPIStatus st = vpiImageLockData(slot.disparityMap, VPI_LOCK_READ,
                     VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dispData);
    if (st != VPI_SUCCESS) {
        LOG_ERROR("stage3_fuse: disparity image lock failed "
                  "(frame=%d status=%d)", slot.frame_id, (int)st);
        slot.grab_failed = true;
        NVTX_RANGE_POP();
        return false;
    }

    const int16_t* disp_ptr = static_cast<const int16_t*>(
        dispData.buffer.pitch.planes[0].data);
    int disp_pitch = dispData.buffer.pitch.planes[0].pitchBytes;
    if (!disp_ptr || disp_pitch <= 0) {
        LOG_ERROR("stage3_fuse: invalid disparity CUDA view "
                  "(frame=%d ptr=%p pitch=%d)",
                  slot.frame_id, static_cast<const void*>(disp_ptr), disp_pitch);
        slot.grab_failed = true;
        const VPIStatus unlock_status = vpiImageUnlock(slot.disparityMap);
        if (unlock_status != VPI_SUCCESS) {
            LOG_ERROR("stage3_fuse: disparity cleanup unlock failed "
                      "(frame=%d status=%d)",
                      slot.frame_id, (int)unlock_status);
        }
        NVTX_RANGE_POP();
        return false;
    }

    float dispScale = (config_.disparity_strategy == DisparityStrategy::HALF_RESOLUTION)
                      ? 2.0f : 1.0f;
    const bool compute_ok = fusion_->tryComputeBatch(
        slot.detections, disp_ptr, disp_pitch,
        config_.rect_width, config_.rect_height,
        streams_.cudaStreamFuse, slot.results, dispScale);
    if (!compute_ok) {
        LOG_ERROR("stage3_fuse: Coordinate3D CUDA chain failed (frame=%d)",
                  slot.frame_id);
        slot.grab_failed = true;
        slot.results.clear();
    }

    const VPIStatus unlock_status = vpiImageUnlock(slot.disparityMap);
    if (unlock_status != VPI_SUCCESS) {
        LOG_ERROR("stage3_fuse: disparity image unlock failed "
                  "(frame=%d status=%d)",
                  slot.frame_id, (int)unlock_status);
        slot.grab_failed = true;
        slot.results.clear();
    }
    if (!slot.grab_failed) stampFrameMetadata(slot);
    NVTX_RANGE_POP();
    return !slot.grab_failed;
}

}  // namespace stereo3d
