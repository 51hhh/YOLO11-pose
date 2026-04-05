# Color Pipeline & P0 Fixes Design Spec

> **Goal:** Add BayerRG8 → BGR color pipeline for improved YOLO detection accuracy, fix Kalman dt/process_accel/camera reconnect bugs, improve logger and ROI CUDA utilization.

> **Target:** Lab prototype validation. Detection accuracy and 3D accuracy are priorities.

## Hard Constraints

- Resolution: 1440x1080 (non-negotiable)
- Framerate: 100fps PWM (non-negotiable)
- Camera config: BayerRG8, exposure 9867us, gain 11.99dB, USB trigger (tested optimal)
- Platform: Jetson Orin NX Super 16GB, VPI 3.2+, TensorRT 10.3

---

## Part 1: VPI Color Pipeline (Approach A)

### Current Pixel Path (gray)

```
BayerRG8(raw) → memcpy → VPI U8 → VPI Remap(U8→U8) → rectU8 → grayToRGBLetterbox → TRT
```

Problem: VPI Remap destroys Bayer mosaic structure. `input_format: "bayer"` kernel produces artifacts on remapped data. Model receives identical R=G=B channels, losing all color contrast.

### New Pixel Path (color)

```
BayerRG8(raw) → memcpy → VPI BAYER8_RGGB → vpiSubmitDebayer(VIC) → tempBGR(1440x1080)
                                                                        |
                                             vpiSubmitRemap(CUDA, BGR→BGR) → rectBGR(1280x720)
                                                                        |
                                    +-----------+-----------+
                                    |                       |
                            bgrToRGBLetterbox       vpiConvertImage(BGR→U8)
                                    |                       |
                              TRT detect              rectGray(1280x720)
                                                            |
                                                     ROI stereo match
```

### Design Decisions

1. **Debayer BEFORE remap**: Must recover color before spatial interpolation, otherwise remap mixes different color channels (producing pseudo-color artifacts).

2. **VIC for debayer**: Hardware ISP, does not consume GPU SM resources. Runs in parallel with GPU detect from previous frame.

3. **CUDA for BGR remap**: Three-channel remap has 3x bandwidth; CUDA handles this better than VIC. Runs async on same VPI stream.

4. **Gray conversion for stereo**: `vpiSubmitConvertImageFormat(BGR→U8)` on VIC. ~0.2ms. ROI SAD stereo matching stays on single-channel for now. Future: test color SAD (3-channel accumulation) and compare.

5. **Existing `bgrToRGBLetterboxKernel`**: Already implemented in `detect_preprocess.cu`. Only config change: `input_format: "bgr"`.

### Memory Budget

| Image | Format | Size | Per Slot | ×3 Slots | ×2 Cameras |
|-------|--------|------|----------|----------|------------|
| tempBGR | BGR8 1440×1080 | 4.4MB | 4.4MB | 13.3MB | 26.6MB |
| rectBGR | BGR8 1280×720 | 2.8MB | 2.8MB | 8.3MB | 16.6MB |
| rectGray | U8 1280×720 | 0.9MB | 0.9MB | 2.8MB | 5.5MB |
| **Total new** | | | **8.1MB** | **24.4MB** | **48.7MB** |

48.7MB additional on 16GB system — acceptable.

### 100fps Timing Verification

Triple-buffer async pipeline (ROI mode):

| Stage | Operations | Hardware | Wall Time | Parallel With |
|-------|-----------|----------|-----------|---------------|
| Stage 0 | Grab (USB) | CPU 2 threads | 6.4ms | — |
| Stage 0 | Debayer (async) | VIC | 0.5ms submit→VIC | Stage 1 GPU |
| Stage 0 | BGR Remap (async) | CUDA | 2.0ms async | Stage 1 GPU |
| Stage 0 | Gray Convert (async) | VIC | 0.2ms async | Stage 1 GPU |
| Stage 1 | WaitRect (sync) | — | ~3ms wait | — |
| Stage 1 | Detect | GPU | 2.93ms | Stage 0 Grab |
| Stage 2 | ROI Match + Fuse | GPU | <1ms | Stage 0+1 |

Critical path = max(Stage0, Stage1, Stage2) = max(6.5ms, 5.93ms, 1ms) = **6.5ms < 10ms** ✓

### Files to Modify

- `src/pipeline/frame_slot.h`: Add tempBGR, rectBGR, rectGray VPI images
- `src/pipeline/pipeline.cpp`: Stage 0 new VPI chain (debayer+remap+gray)
- `src/rectify/vpi_rectifier.cpp`: Support BGR remap initialization (new LUT for BGR)
- `src/capture/hikvision_camera.cpp`: Change raw image format to BAYER8_RGGB
- `config/pipeline_yolo26_mixed.yaml`: `input_format: "bgr"`

### Approach B (Fallback): CUDA Fused Debayer-Letterbox

If VPI debayer on VIC is not available or too slow:

```
BayerRG8(raw) → VPI Remap(U8→U8, current) → rectU8(gray for stereo)
rawL(BAYER) → CUDA bayerDemosaicLetterboxKernel(raw→RGB, using remap coords) → TRT detect
```

Problem: Requires raw→rect coordinate mapping in kernel. Complex, error-prone. Only implement if Approach A fails 100fps constraint.

---

## Part 2: P0 Bug Fixes

### 2.1 Kalman dt: Use Actual Frame Interval

**Problem:** `dt` is hardcoded from config (0.01s). Frame drops cause prediction drift.

**Solution:**
- Add `std::chrono::steady_clock::time_point last_fuse_time_` to pipeline
- Compute actual dt between fuse calls: `dt = now - last_fuse_time_`
- Clamp to safe range: `std::clamp(dt, 0.002, 0.1)` (2ms to 100ms)
- Pass real dt to `HybridDepth::estimate()` as new parameter

**Files:**
- `src/fusion/hybrid_depth.h`: Add `double actual_dt` parameter to `estimate()`
- `src/fusion/hybrid_depth.cpp`: Use `actual_dt` in Kalman predict instead of `config_.dt`
- `src/pipeline/pipeline.cpp`: Compute and pass actual dt

### 2.2 process_accel: Increase to 50

**Problem:** Current value 30 m/s² is below volleyball spike acceleration (40-60 m/s²).

**Solution:** Change `process_accel: 50.0` in config YAML.

**Files:**
- `config/pipeline_yolo26_mixed.yaml`: `process_accel: 50.0`

### 2.3 Continuous Skip Protection + Camera Auto-Reconnect

**Problem:** `grab_failed` has no upper limit. Camera disconnect = permanent failure.

**Solution:**
- Add `consecutive_failures_` counter in `HikvisionCamera`
- After `MAX_CONSECUTIVE_FAILURES` (10) sequential fails: trigger `reconnect()`
- `reconnect()`: close() → sleep(500ms) → open(original_config) → max 3 retries
- Log reconnection attempts and results

**Files:**
- `src/capture/hikvision_camera.h`: Add counter, reconnect() method
- `src/capture/hikvision_camera.cpp`: Implement failure counting and reconnect logic

---

## Part 3: P1 Improvements

### 3.1 Logger: Add Timestamp + Thread ID

**Problem:** No timestamps, no thread identification. Dual fprintf not atomic.

**Solution:**
- Single fprintf with prefix: `[sec.ms][Ttid][LEVEL]`
- Use `clock_gettime(CLOCK_MONOTONIC)` for monotonic timestamps
- Use `syscall(SYS_gettid)` for thread ID
- Single fprintf call = atomic write on most systems

**Format:** `[1234.567][T12345][INFO ] Pipeline initialized`

**Files:**
- `src/utils/logger.h`: Rewrite logMsg() with prefix

### 3.2 ROI CUDA Thread Utilization Optimization

**Problem:** 128 threads/block, only 25 active (5×5 grid). Each does serial d=0..255 search. 19.5% utilization.

**Solution:** Cooperative disparity search.
- 25 sample points × 5 threads/point = 125 active threads
- Each thread searches a disparity sub-range (d/5 values)
- Warp-level reduction to find global minimum SAD
- Expected: matching time from ~0.5ms to ~0.1ms per detection

**Files:**
- `src/stereo/roi_stereo_match.cu`: Rewrite kernel with cooperative search

---

## Scope Summary

| Category | Item | Priority | Est. Effort |
|----------|------|----------|-------------|
| Color Pipeline | VPI debayer + BGR remap chain | P0 | 2 days |
| Color Pipeline | Approach B fallback (if A fails) | P0-fallback | 1 day |
| Bug Fix | Kalman dt actual interval | P0 | 0.5 day |
| Bug Fix | process_accel to 50 | P0 | 5 min |
| Bug Fix | Camera reconnect | P0 | 0.5 day |
| Improvement | Logger timestamps | P1 | 0.5 day |
| Improvement | ROI CUDA optimization | P1 | 1 day |

**Total: ~5 days for all items**

---

## Success Criteria

1. **Detection accuracy**: Measurable improvement on test set (target: >96% at conf 0.3)
2. **Framerate**: Maintain 100fps with color pipeline
3. **Kalman tracking**: No drift after frame drops (verify with controlled drop test)
4. **Camera resilience**: Auto-recover from simulated USB disconnect
5. **Stereo A/B test**: Compare gray SAD vs color SAD matching quality
