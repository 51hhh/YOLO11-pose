# Color Pipeline & P0 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BayerRG8→BGR color pipeline via VPI (Approach A) with CUDA fallback (Approach B), fix P0 bugs (Kalman dt, camera reconnect, process_accel), and P1 improvements (logger, ROI CUDA).

**Architecture:** Triple-buffered 4-stage async pipeline. New debayer+BGR remap chain inserted into Stage 0 before existing remap. Detection switches to BGR input; stereo matching stays grayscale via conversion.

**Tech Stack:** VPI 3.2+ (debayer/remap), OpenCV CUDA (fallback debayer), TensorRT 10.3, CUDA 12.x, C++17

**Existing Infrastructure:** `frame_slot.h` already declares `cv::cuda::GpuMat bgrRawL/R, rectBGR_L/R, rectGray_L/R` — unused but ready for Approach B.

---

## Task 1: VPI Debayer Integration (Approach A - Core)

**Files:**
- Modify: `src/pipeline/frame_slot.h:61-83` (add VPI BGR images)
- Modify: `src/pipeline/pipeline.cpp:104-137` (allocate new VPI images)
- Modify: `src/pipeline/pipeline.cpp:548-601` (Stage 0 debayer chain)

- [ ] **Step 1.1: Add VPI BGR/Gray images to FrameSlot**

In `src/pipeline/frame_slot.h`, after line 64 (`VPIImage rectR`), add new VPI images:

```cpp
    // --- Color pipeline (VPI) ---
    VPIImage tempBGR_L = nullptr;   ///< 左 debayer 输出 BGR (raw res)
    VPIImage tempBGR_R = nullptr;   ///< 右 debayer 输出 BGR (raw res)
    VPIImage rectBGR_vpiL = nullptr; ///< 左校正 BGR (rect res, 检测用)
    VPIImage rectBGR_vpiR = nullptr; ///< 右校正 BGR (rect res)
    VPIImage rectGray_vpiL = nullptr; ///< 左校正灰度 (rect res, 立体匹配用)
    VPIImage rectGray_vpiR = nullptr; ///< 右校正灰度 (rect res)
```

In the destructor/cleanup section, add `vpiImageDestroy` for each new image (checking nullptr first).

- [ ] **Step 1.2: Change rawL/rawR format to BAYER8_RGGB**

In `src/pipeline/pipeline.cpp`, around line 113, change:
```cpp
// Before:
vpiImageCreate(config_.raw_width, config_.raw_height, VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rawL);
// After:
vpiImageCreate(config_.raw_width, config_.raw_height, VPI_IMAGE_FORMAT_BAYER8_RGGB, flags, &slots_[i].rawL);
```
Same for rawR at line 117.

**Note:** The raw bytes from BayerRG8 camera are identical — only the VPI metadata changes.

- [ ] **Step 1.3: Allocate new VPI images in init()**

In `src/pipeline/pipeline.cpp`, after rectR allocation (~line 126), add:
```cpp
// Color pipeline images
vpiImageCreate(config_.raw_width, config_.raw_height,
               VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].tempBGR_L);
vpiImageCreate(config_.raw_width, config_.raw_height,
               VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].tempBGR_R);
vpiImageCreate(config_.rect_width, config_.rect_height,
               VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].rectBGR_vpiL);
vpiImageCreate(config_.rect_width, config_.rect_height,
               VPI_IMAGE_FORMAT_BGR8, flags, &slots_[i].rectBGR_vpiR);
vpiImageCreate(config_.rect_width, config_.rect_height,
               VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rectGray_vpiL);
vpiImageCreate(config_.rect_width, config_.rect_height,
               VPI_IMAGE_FORMAT_U8, flags, &slots_[i].rectGray_vpiR);
```

- [ ] **Step 1.4: Implement color Stage 0 chain**

In `src/pipeline/pipeline.cpp`, in `stage0_grab_and_rectify()`, after the `vpiImageUnlock` calls for rawL/rawR (around line 569), replace the old remap submission with:

```cpp
// --- Color pipeline: Debayer → BGR Remap → Gray Convert ---
// 1. Debayer: BAYER8_RGGB → BGR8 (VIC hardware ISP)
vpiSubmitDebayer(streams_.vpiStreamPVA, VPI_BACKEND_VIC,
                 slot.rawL, slot.tempBGR_L,
                 VPI_INTERPOLATION_LINEAR, 2/*bayer border*/,
                 nullptr/*params*/);
vpiSubmitDebayer(streams_.vpiStreamPVA, VPI_BACKEND_VIC,
                 slot.rawR, slot.tempBGR_R,
                 VPI_INTERPOLATION_LINEAR, 2, nullptr);

// 2. Remap: BGR8 校正 (CUDA — 三通道带宽大, CUDA更快)
rectifier_->submitBGR(streams_.vpiStreamPVA,
                      slot.tempBGR_L, slot.tempBGR_R,
                      slot.rectBGR_vpiL, slot.rectBGR_vpiR);

// 3. BGR → Gray: 给立体匹配 (VIC)
vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_VIC,
                            slot.rectBGR_vpiL, slot.rectGray_vpiL, nullptr);
vpiSubmitConvertImageFormat(streams_.vpiStreamPVA, VPI_BACKEND_VIC,
                            slot.rectBGR_vpiR, slot.rectGray_vpiR, nullptr);
```

- [ ] **Step 1.5: Build on NX and verify VPI debayer compiles**

```bash
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline && mkdir -p build && cd build && cmake .. && make -j4
```

Expected: Compiles without VPI API errors. If `vpiSubmitDebayer` is not available, log the error for Approach B fallback.

---

## Task 2: VPI Rectifier BGR Support

**Files:**
- Modify: `src/rectify/vpi_rectifier.h:41-43` (add submitBGR method)
- Modify: `src/rectify/vpi_rectifier.cpp:93-100` (implement BGR remap)

- [ ] **Step 2.1: Add submitBGR() method to VPIRectifier**

In `src/rectify/vpi_rectifier.h`, after the existing `submit()` declaration, add:
```cpp
    /// BGR remap — same LUTs, different image format
    void submitBGR(VPIStream stream,
                   VPIImage bgrL, VPIImage bgrR,
                   VPIImage rectBGR_L, VPIImage rectBGR_R);
```

- [ ] **Step 2.2: Implement submitBGR()**

In `src/rectify/vpi_rectifier.cpp`, after the existing `submit()` function, add:
```cpp
void VPIRectifier::submitBGR(VPIStream stream,
                              VPIImage bgrL, VPIImage bgrR,
                              VPIImage rectBGR_L, VPIImage rectBGR_R) {
    // VPI remap works on any image format — same LUT, different input/output format
    // Use CUDA backend for BGR (3x bandwidth, CUDA handles better)
    vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remapL_, bgrL, rectBGR_L,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remapR_, bgrR, rectBGR_R,
                   VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
}
```

**Note:** The existing remap LUTs (spatial coordinates) are format-agnostic. VPI applies the same pixel mapping regardless of channel count. Using CUDA backend instead of VIC for the 3-channel case.

- [ ] **Step 2.3: Build and verify no link errors**

```bash
make -j4
```

Expected: Clean build. `submitBGR` symbol resolves correctly.

---

## Task 3: Detection Path Switch to BGR

**Files:**
- Modify: `src/pipeline/pipeline.cpp:638-654` (stage1_detect use rectBGR)
- Modify: `config/pipeline_yolo26_mixed.yaml` (input_format: "bgr")

- [ ] **Step 3.1: Change stage1_detect() to use rectBGR**

In `src/pipeline/pipeline.cpp`, in `stage1_detect()` (around line 638-654), change:
```cpp
// Before:
vpiImageLockData(slot.rectL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);

// After (color pipeline):
vpiImageLockData(slot.rectBGR_vpiL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData);
```

And the corresponding unlock:
```cpp
// Before:
vpiImageUnlock(slot.rectL);
// After:
vpiImageUnlock(slot.rectBGR_vpiL);
```

**Note:** The detector's `enqueue()` receives a raw GPU pointer + pitch. The `bgrToRGBLetterboxKernel` in `detect_preprocess.cu` already handles 3-channel input with interleaved BGR format. The pitch will be `width * 3` instead of `width * 1`.

- [ ] **Step 3.2: Change config input_format to "bgr"**

In `config/pipeline_yolo26_mixed.yaml`, change:
```yaml
# Before:
input_format: "gray"
# After:
input_format: "bgr"
```

- [ ] **Step 3.3: Verify trt_detector handles BGR pitch correctly**

Read `src/detect/trt_detector.cpp` around the `enqueue` function to confirm it passes the pitch from `imgData` directly to the CUDA preprocessing kernel. The BGR pitch (width×3) should propagate correctly. If the detector assumes pitch = width (U8), this needs fixing.

**Check:** In `detect_preprocess.cu`, the `bgrToRGBLetterboxKernel` signature — does it take a `pitch` parameter? Ensure it's not assuming `pitch = width * 3` but uses the actual pitch from VPI lock data.

---

## Task 4: Stereo Matching Path Switch to Gray

**Files:**
- Modify: `src/pipeline/pipeline.cpp:723-750` (stage2 uses rectGray)

- [ ] **Step 4.1: Change stage2_roi_match_fuse() to use rectGray**

In `src/pipeline/pipeline.cpp`, in `stage2_roi_match_fuse()` (around line 723-750), change:
```cpp
// Before:
vpiImageLockData(slot.rectL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
vpiImageLockData(slot.rectR, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);
// ...
vpiImageUnlock(slot.rectL);
vpiImageUnlock(slot.rectR);

// After:
vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);
// ...
vpiImageUnlock(slot.rectGray_vpiL);
vpiImageUnlock(slot.rectGray_vpiR);
```

**Note:** The ROI stereo matcher expects `const uint8_t*` single-channel data. The rectGray VPI images are U8 format, so the pointer and pitch are identical in interface.

- [ ] **Step 4.2: Build and run dry test**

```bash
make -j4
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline
timeout 8 ./build/stereo_pipeline -c config/pipeline_yolo26_mixed.yaml 2>&1
```

Expected: Pipeline initializes, debayer chain submits without error. Camera grab works. TRT detector loads with BGR format. If VPI debayer returns error code, note it for Task 7 (Approach B).

---

## Task 5: P0 Bug Fix — Kalman dt Actual Interval

**Files:**
- Modify: `src/fusion/hybrid_depth.h:97-101` (add dt parameter)
- Modify: `src/fusion/hybrid_depth.cpp` (use actual dt)
- Modify: `src/pipeline/pipeline.cpp` (compute and pass dt)

- [ ] **Step 5.1: Add actual_dt parameter to estimate()**

In `src/fusion/hybrid_depth.h`, change the estimate signature:
```cpp
// Before:
std::vector<Object3D> estimate(
    const std::vector<Detection>& detections,
    const std::vector<Object3D>& roi_results);

// After:
std::vector<Object3D> estimate(
    const std::vector<Detection>& detections,
    const std::vector<Object3D>& roi_results,
    double actual_dt = 0.0);  // 0 = use config dt (backward compatible)
```

- [ ] **Step 5.2: Use actual_dt in Kalman predict**

In `src/fusion/hybrid_depth.cpp`, at the beginning of `estimate()`, add:
```cpp
const double dt = (actual_dt > 0.001) ? actual_dt : config_.dt;
```

Then replace all `config_.dt` references within estimate() with `dt`.

- [ ] **Step 5.3: Compute and pass real dt in pipeline**

In `src/pipeline/pipeline.cpp`, add a member to Pipeline class (in pipeline.h):
```cpp
std::chrono::steady_clock::time_point last_fuse_time_{};
```

In the fuse call site (stage2_roi_match_fuse or equivalent), before calling estimate():
```cpp
auto now = std::chrono::steady_clock::now();
double dt = 0.01; // default
if (last_fuse_time_.time_since_epoch().count() > 0) {
    dt = std::chrono::duration<double>(now - last_fuse_time_).count();
    dt = std::clamp(dt, 0.002, 0.1);
}
last_fuse_time_ = now;
// Pass dt to estimate()
auto output = hybrid_depth_->estimate(dets, roi_results, dt);
```

- [ ] **Step 5.4: Build and verify**

```bash
make -j4
```

Expected: Clean build. The dt parameter defaults to 0.0 (backward compatible).

---

## Task 6: P0 Bug Fix — Camera Reconnect + process_accel

**Files:**
- Modify: `src/capture/hikvision_camera.h:127-135` (add reconnect members)
- Modify: `src/capture/hikvision_camera.cpp` (implement reconnect logic)
- Modify: `config/pipeline_yolo26_mixed.yaml` (process_accel: 50)

- [ ] **Step 6.1: Add reconnect capability to HikvisionCamera**

In `src/capture/hikvision_camera.h`, add private members:
```cpp
    static constexpr int MAX_CONSECUTIVE_FAILURES = 10;
    static constexpr int MAX_RECONNECT_RETRIES = 3;
    int consecutive_failures_ = 0;
    bool reconnect();
```

- [ ] **Step 6.2: Implement failure counting in grabFramePair**

In `src/capture/hikvision_camera.cpp`, in `grabFramePair()`, after the existing sync jump return false:

At the success path (before `return true`):
```cpp
consecutive_failures_ = 0;
```

At each failure path (before `return false`):
```cpp
consecutive_failures_++;
if (consecutive_failures_ >= MAX_CONSECUTIVE_FAILURES) {
    LOG_WARN("[HikCam] %d consecutive grab failures, attempting reconnect...",
             consecutive_failures_);
    if (reconnect()) {
        LOG_INFO("[HikCam] Reconnect successful");
        consecutive_failures_ = 0;
    } else {
        LOG_ERROR("[HikCam] Reconnect failed after %d retries", MAX_RECONNECT_RETRIES);
    }
}
```

- [ ] **Step 6.3: Implement reconnect()**

In `src/capture/hikvision_camera.cpp`, add:
```cpp
bool HikvisionCamera::reconnect() {
    for (int attempt = 0; attempt < MAX_RECONNECT_RETRIES; ++attempt) {
        LOG_INFO("[HikCam] Reconnect attempt %d/%d", attempt + 1, MAX_RECONNECT_RETRIES);
        stopGrabbing();
        close();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (open(config_) && startGrabbing()) {
            return true;
        }
    }
    return false;
}
```

- [ ] **Step 6.4: Update process_accel in config**

In `config/pipeline_yolo26_mixed.yaml`, change:
```yaml
process_accel: 50.0
```

- [ ] **Step 6.5: Build and verify**

```bash
make -j4
```

Expected: Clean build. Reconnect method compiles. No behavioral change unless grab actually fails 10 times.

---

## Task 7: Approach B Fallback (OpenCV CUDA Debayer)

**Trigger:** Only implement if Task 1 Step 1.5 reveals VPI debayer issues (missing VIC backend, API errors, or framerate below 100fps).

**Files:**
- Modify: `src/pipeline/pipeline.cpp` (Stage 0 alternative path)
- Uses existing: `frame_slot.h:71-83` (cv::cuda::GpuMat fields already declared)

- [ ] **Step 7.1: Implement OpenCV CUDA debayer path**

In `src/pipeline/pipeline.cpp`, `stage0_grab_and_rectify()`, as alternative to VPI debayer:

```cpp
// --- Approach B: OpenCV CUDA debayer + remap ---
// 1. Upload raw Bayer to GPU
slot.rawBayerL.upload(slot.hostBayerL, cudaStream);

// 2. Debayer: BayerRG8 → BGR (CUDA)
cv::cuda::cvtColor(slot.rawBayerL, slot.bgrRawL, cv::COLOR_BayerRG2BGR, 0, cudaStream);

// 3. Remap: BGR → rectified BGR (CUDA)
cv::cuda::remap(slot.bgrRawL, slot.rectBGR_L, mapXL_gpu, mapYL_gpu,
                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), cudaStream);

// 4. Gray: BGR → Grayscale (CUDA)
cv::cuda::cvtColor(slot.rectBGR_L, slot.rectGray_L, cv::COLOR_BGR2GRAY, 0, cudaStream);

// Same for R camera...
```

- [ ] **Step 7.2: A/B compare framerate**

Run both pipelines with 100fps target. Log per-stage timing. Compare:
- Approach A (VPI): debayer(VIC) + remap(CUDA) + gray(VIC)
- Approach B (OCV): debayer(CUDA) + remap(CUDA) + gray(CUDA)

---

## Task 8: P1 — Logger Timestamps + Thread ID

**Files:**
- Modify: `src/utils/logger.h:19-31`

- [ ] **Step 8.1: Rewrite logMsg with timestamp**

Replace the entire `logMsg` function in `src/utils/logger.h`:

```cpp
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

inline void logMsg(LogLevel level, const char* fmt, ...) {
    static const char* prefixes[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    char buf[2048];
    int off = snprintf(buf, sizeof(buf), "[%ld.%03ld][T%ld][%s] ",
                       ts.tv_sec % 10000, ts.tv_nsec / 1000000,
                       (long)syscall(SYS_gettid),
                       prefixes[static_cast<int>(level)]);
    va_list args;
    va_start(args, fmt);
    off += vsnprintf(buf + off, sizeof(buf) - off, fmt, args);
    va_end(args);
    if (off < (int)sizeof(buf) - 1) { buf[off++] = '\n'; buf[off] = '\0'; }
    fputs(buf, stderr);
}
```

**Key improvement:** Single `fputs` call = atomic on most Linux systems. Includes monotonic timestamp and thread ID.

- [ ] **Step 8.2: Build and verify log output format**

```bash
make -j4
timeout 5 ./build/stereo_pipeline -c config/pipeline_yolo26_mixed.yaml 2>&1 | head -20
```

Expected: `[1234.567][T12345][INFO ] Pipeline initialized`

---

## Task 9: P1 — ROI CUDA Thread Optimization

**Files:**
- Modify: `src/stereo/roi_stereo_match.cu:54-175`

- [ ] **Step 9.1: Redesign kernel thread mapping**

Replace the current kernel with cooperative disparity search:

```cuda
// New thread mapping:
// 25 sample points × 5 threads/point = 125 threads (of 128)
// Each thread group of 5 searches a disjoint disparity sub-range
const int pointIdx = threadIdx.x / THREADS_PER_POINT;  // 0..24
const int subRange = threadIdx.x % THREADS_PER_POINT;  // 0..4
const int dStart = subRange * (maxDisparity / THREADS_PER_POINT);
const int dEnd   = (subRange + 1) * (maxDisparity / THREADS_PER_POINT);

// Each thread finds local bestSAD in its sub-range
int localBestSAD = INT_MAX, localBestD = -1;
for (int d = dStart; d < dEnd && d <= maxSearch; d++) {
    int sad = computeSAD(leftImg, rightImg, px, py, d, patchRadius, leftPitch, rightPitch);
    if (sad < localBestSAD) { localBestSAD = sad; localBestD = d; }
}

// Warp-level reduction across 5 threads to find global minimum
// Using __shfl_down_sync for threads within same point group
```

- [ ] **Step 9.2: Implement warp-level reduction**

After each thread has its local best, reduce across the 5 threads:
```cuda
__shared__ int s_sad[128], s_disp[128];
s_sad[threadIdx.x] = localBestSAD;
s_disp[threadIdx.x] = localBestD;
__syncthreads();

// Thread 0 of each point-group (stride THREADS_PER_POINT) finds global minimum
if (subRange == 0) {
    int bestSAD = s_sad[threadIdx.x];
    int bestD = s_disp[threadIdx.x];
    for (int t = 1; t < THREADS_PER_POINT; t++) {
        int idx = threadIdx.x + t;
        if (s_sad[idx] < bestSAD) {
            bestSAD = s_sad[idx];
            bestD = s_disp[idx];
        }
    }
    // Continue with sub-pixel refinement using bestD...
}
```

- [ ] **Step 9.3: Preserve sub-pixel refinement and uniqueness check**

The sub-pixel parabola fitting and uniqueness validation logic (lines 157-175) should remain unchanged, just executed by the leader thread of each point group.

- [ ] **Step 9.4: Build and benchmark**

```bash
make -j4
# Run with timing output
timeout 10 ./build/stereo_pipeline -c config/pipeline_yolo26_mixed.yaml 2>&1 | grep -i "stereo\|match\|latency"
```

Expected: ROI matching time reduced from ~0.5ms to ~0.1-0.15ms per detection.

---

## Task 10: Sync to NX + Full Integration Test

**Files:** All modified files from Tasks 1-9

- [ ] **Step 10.1: Sync all modified files to NX**

```powershell
# From Windows
$files = @(
    "src/pipeline/frame_slot.h",
    "src/pipeline/pipeline.cpp",
    "src/pipeline/pipeline.h",
    "src/rectify/vpi_rectifier.h",
    "src/rectify/vpi_rectifier.cpp",
    "src/capture/hikvision_camera.h",
    "src/capture/hikvision_camera.cpp",
    "src/fusion/hybrid_depth.h",
    "src/fusion/hybrid_depth.cpp",
    "src/utils/logger.h",
    "src/stereo/roi_stereo_match.cu",
    "config/pipeline_yolo26_mixed.yaml"
)
foreach ($f in $files) {
    pscp -pw nvidia "NX_volleyball\stereo_3d_pipeline\$f" nvidia@192.168.31.56:/home/nvidia/NX_volleyball/stereo_3d_pipeline/$f
}
```

- [ ] **Step 10.2: Fix line endings and build**

```bash
find src config -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.yaml" | xargs sed -i 's/\r$//'
cd build && make -j4
```

- [ ] **Step 10.3: Run full pipeline test**

```bash
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline
./build/stereo_pipeline -c config/pipeline_yolo26_mixed.yaml 2>&1 | tee /tmp/color_pipeline_test.log
```

**Verify:**
1. Log shows `input_format: BGR` in detector init
2. Debayer submits without error (VPI or OpenCV CUDA)
3. Detection finds volleyball (color should improve detection rate)
4. Stereo matching produces valid 3D coordinates
5. FPS maintains 100Hz
6. Logger shows timestamps: `[sec.ms][Ttid][LEVEL]`
7. No sync jump errors (camera reconnect not triggered in normal operation)

---

## Execution Order

| Phase | Tasks | Dependencies |
|-------|-------|-------------|
| **Phase 1** | Task 1 + Task 2 | None (core color pipeline) |
| **Phase 2** | Task 3 + Task 4 | Depends on Phase 1 |
| **Phase 3** | Task 5 + Task 6 | Independent of Phase 1-2 |
| **Phase 4** | Task 7 (if needed) | Only if VPI debayer fails |
| **Phase 5** | Task 8 + Task 9 | Independent |
| **Phase 6** | Task 10 | All above complete |
