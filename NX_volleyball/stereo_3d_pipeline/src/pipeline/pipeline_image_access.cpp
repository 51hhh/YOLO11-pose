/**
 * @file pipeline_image_access.cpp
 * @brief VPI-owned images exported as short-lived CUDA views.
 *
 * A view is valid only between vpiImageLockData and the matching unlock.
 * This file is the single owner of those boundaries for pipeline hot paths;
 * consumers must also match locked_frame_id before dereferencing a view.
 */

#include "pipeline.h"
#include "../utils/logger.h"

namespace stereo3d {
namespace {

bool lockCudaView(FrameSlot& slot,
                  VPIImage image,
                  VPILockMode mode,
                  FrameSlot::CudaImageView& view,
                  const char* name) {
    if (!image) {
        LOG_ERROR("[Pipeline] cannot lock null VPI image: frame=%d image=%s",
                  slot.frame_id, name);
        slot.grab_failed = true;
        return false;
    }
    if (view.locked) {
        if (view.locked_frame_id == slot.frame_id && view.data &&
            view.pitchBytes > 0) {
            return true;
        }
        LOG_ERROR("[Pipeline] stale VPI CUDA lock: frame=%d image=%s "
                  "locked_frame=%d",
                  slot.frame_id, name, view.locked_frame_id);
        slot.grab_failed = true;
        return false;
    }

    VPIImageData data{};
    const VPIStatus status = vpiImageLockData(
        image, mode, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data);
    if (status != VPI_SUCCESS) {
        LOG_ERROR("[Pipeline] VPI CUDA lock failed: frame=%d image=%s status=%d",
                  slot.frame_id, name, (int)status);
        slot.grab_failed = true;
        return false;
    }

    view.data = data.buffer.pitch.planes[0].data;
    view.pitchBytes = data.buffer.pitch.planes[0].pitchBytes;
    view.locked = true;
    view.locked_frame_id = slot.frame_id;
    if (view.data && view.pitchBytes > 0) return true;

    LOG_ERROR("[Pipeline] VPI CUDA lock returned invalid view: "
              "frame=%d image=%s ptr=%p pitch=%d",
              slot.frame_id, name, view.data, view.pitchBytes);
    const VPIStatus unlock_status = vpiImageUnlock(image);
    if (unlock_status != VPI_SUCCESS) {
        LOG_ERROR("[Pipeline] invalid-view cleanup unlock failed: "
                  "frame=%d image=%s status=%d",
                  slot.frame_id, name, (int)unlock_status);
    }
    view = FrameSlot::CudaImageView{};
    slot.grab_failed = true;
    return false;
}

bool unlockCudaView(FrameSlot& slot,
                    VPIImage image,
                    FrameSlot::CudaImageView& view,
                    const char* name) {
    if (!view.locked) {
        view = FrameSlot::CudaImageView{};
        return true;
    }
    if (!image) {
        LOG_ERROR("[Pipeline] cannot unlock null VPI image: "
                  "frame=%d image=%s", slot.frame_id, name);
        view = FrameSlot::CudaImageView{};
        slot.grab_failed = true;
        return false;
    }
    const VPIStatus status = vpiImageUnlock(image);
    view = FrameSlot::CudaImageView{};
    if (status == VPI_SUCCESS) return true;

    LOG_ERROR("[Pipeline] VPI CUDA unlock failed: frame=%d image=%s status=%d",
              slot.frame_id, name, (int)status);
    slot.grab_failed = true;
    return false;
}

}  // namespace

bool Pipeline::lockStage0CudaViews(FrameSlot& slot) {
    if (!colorPipelineEnabled()) return true;
    ScopedTimer timer("Stage0_VpiCudaLock");
    const bool ok =
        lockCudaView(slot, slot.rawL, VPI_LOCK_READ,
                     slot.rawL_gpu, "rawL") &&
        lockCudaView(slot, slot.rawR, VPI_LOCK_READ,
                     slot.rawR_gpu, "rawR") &&
        lockCudaView(slot, slot.tempBGR_L, VPI_LOCK_WRITE,
                     slot.tempBGR_L_gpu, "tempBGR_L") &&
        lockCudaView(slot, slot.tempBGR_R, VPI_LOCK_WRITE,
                     slot.tempBGR_R_gpu, "tempBGR_R");
    if (!ok) unlockStage0CudaViews(slot);
    globalPerf().record("Stage0_VpiCudaLock", timer.elapsedMs());
    return ok;
}

bool Pipeline::unlockStage0CudaViews(FrameSlot& slot) {
    if (!slot.rawL_gpu.locked && !slot.rawR_gpu.locked &&
        !slot.tempBGR_L_gpu.locked && !slot.tempBGR_R_gpu.locked) {
        return true;
    }
    ScopedTimer timer("Stage0_VpiCudaUnlock");
    bool ok = true;
    ok = unlockCudaView(slot, slot.tempBGR_R,
                        slot.tempBGR_R_gpu, "tempBGR_R") && ok;
    ok = unlockCudaView(slot, slot.tempBGR_L,
                        slot.tempBGR_L_gpu, "tempBGR_L") && ok;
    ok = unlockCudaView(slot, slot.rawR, slot.rawR_gpu, "rawR") && ok;
    ok = unlockCudaView(slot, slot.rawL, slot.rawL_gpu, "rawL") && ok;
    globalPerf().record("Stage0_VpiCudaUnlock", timer.elapsedMs());
    return ok;
}

bool Pipeline::lockRectifiedCudaViews(FrameSlot& slot) {
    ScopedTimer timer("Rectified_VpiCudaLock");
    const bool need_gray =
        config_.disparity_strategy == DisparityStrategy::ROI_ONLY ||
        !leftDetectorUsesBGR() ||
        (dualYoloEnabled() && !rightDetectorUsesBGR());
    const bool need_bgr = colorPipelineEnabled();

    bool ok = true;
    if (need_bgr) {
        ok = lockCudaView(slot, slot.rectBGR_vpiL, VPI_LOCK_READ,
                          slot.rectBGR_L_gpu, "rectBGR_L") && ok;
        ok = lockCudaView(slot, slot.rectBGR_vpiR, VPI_LOCK_READ,
                          slot.rectBGR_R_gpu, "rectBGR_R") && ok;
    }
    if (need_gray) {
        ok = lockCudaView(slot, slot.rectGray_vpiL, VPI_LOCK_READ,
                          slot.rectGray_L_gpu, "rectGray_L") && ok;
        ok = lockCudaView(slot, slot.rectGray_vpiR, VPI_LOCK_READ,
                          slot.rectGray_R_gpu, "rectGray_R") && ok;
    }
    if (!ok) unlockRectifiedCudaViews(slot);
    globalPerf().record("Rectified_VpiCudaLock", timer.elapsedMs());
    return ok;
}

bool Pipeline::unlockRectifiedCudaViews(FrameSlot& slot) {
    if (!slot.rectGray_L_gpu.locked && !slot.rectGray_R_gpu.locked &&
        !slot.rectBGR_L_gpu.locked && !slot.rectBGR_R_gpu.locked) {
        return true;
    }
    ScopedTimer timer("Rectified_VpiCudaUnlock");
    bool ok = true;
    ok = unlockCudaView(slot, slot.rectGray_vpiR,
                        slot.rectGray_R_gpu, "rectGray_R") && ok;
    ok = unlockCudaView(slot, slot.rectGray_vpiL,
                        slot.rectGray_L_gpu, "rectGray_L") && ok;
    ok = unlockCudaView(slot, slot.rectBGR_vpiR,
                        slot.rectBGR_R_gpu, "rectBGR_R") && ok;
    ok = unlockCudaView(slot, slot.rectBGR_vpiL,
                        slot.rectBGR_L_gpu, "rectBGR_L") && ok;
    globalPerf().record("Rectified_VpiCudaUnlock", timer.elapsedMs());
    return ok;
}

bool Pipeline::rectifiedGrayCudaViewsLocked(const FrameSlot& slot) const {
    return slot.rectGray_L_gpu.locked || slot.rectGray_R_gpu.locked;
}

}  // namespace stereo3d
