# SOT Tracker 补帧实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 YOLO 检测间隙帧插入 SOT (Single Object Tracking) 补帧，使检测层从 ~33Hz 恢复到 100Hz 全帧 bbox 输出，对下游完全透明。

**Architecture:** 固定间隔 `detect_interval=N` 控制 YOLO 推理频率（30~40Hz），其余帧由 NanoTrack/MixFormerV2 在 GPU 上同步推理补帧。YOLO 在 DLA 上异步 enqueue/collect（~7ms），tracker 在 cudaStreamGPU 上同步执行（~0.5-1ms）。LOST 状态下临时设置 `detect_interval=1` 提前触发 YOLO 重检测。

**Tech Stack:** C++17, CUDA, TensorRT, VPI, yaml-cpp

---

## 文件结构

### 新增文件
| 文件 | 职责 |
|------|------|
| `src/track/sot_tracker.h` | SOT 抽象接口 (setTarget/track/reset/isTracking) |
| `src/track/nanotrack_trt.h` | NanoTrack TRT 实现声明 |
| `src/track/nanotrack_trt.cpp` | NanoTrack TRT 实现: backbone + correlation head |
| `src/track/mixformer_trt.h` | MixFormerV2-small TRT 实现声明 |
| `src/track/mixformer_trt.cpp` | MixFormerV2-small TRT 实现: unified backbone |
| `src/track/crop_resize.cu` | CUDA kernel: 从全帧裁剪+缩放 ROI patch |
| `config/pipeline_yolo11m_960_nanotrack.yaml` | YOLO11-M@960 + NanoTrack 配置 |
| `config/pipeline_yolo11m_960_mixformer.yaml` | YOLO11-M@960 + MixFormerV2 配置 |

### 修改文件
| 文件 | 改动 |
|------|------|
| `src/pipeline/pipeline.h` | +TrackerConfig, +SOTTracker 成员, +detect_interval 参数, +is_detect_frame() |
| `src/pipeline/pipeline.cpp` | pipelineLoopROI() 主循环中插入 tracker 逻辑; init() 加载 tracker |
| `src/pipeline/frame_slot.h` | +sot_bbox (Detection), +is_detect_frame, +bbox_source 枚举 |
| `CMakeLists.txt` | +track 源文件和 CUDA 文件 |

### 不改
| 文件 | 原因 |
|------|------|
| `src/fusion/hybrid_depth.*` | 下游透明 |
| `src/stereo/roi_stereo_matcher.*` | 下游透明 |
| `src/detect/trt_detector.*` | YOLO 检测器复用 |

---

## 核心设计约束

### 异步时序（detect_interval=3 示例，YOLO ~7ms on DLA）

```
Frame 0: [YOLO enqueue F0]  tracker: 无模板 → 无输出    output: none (warmup)
Frame 1: [DLA running...]   tracker: 无模板 → 无输出    output: none (warmup)
Frame 2: [YOLO collect F0]  → det bbox → output: YOLO权威
         [YOLO enqueue F3]  tracker.setTarget(F0_image_cached, det_bbox)
Frame 3: [DLA running...]   tracker.track(F3) → sot_bbox  output: tracker补帧
Frame 4: [DLA running...]   tracker.track(F4) → sot_bbox  output: tracker补帧
Frame 5: [YOLO collect F3]  → det bbox → output: YOLO权威
         [YOLO enqueue F6]  tracker.setTarget(F3_image_cached, det_bbox)
...
```

**关键**: YOLO collect 到的是 N-2 帧结果（流水线延迟），tracker 模板刷新时使用 **collect 对应帧的图像**（缓存在 FrameSlot 中），而非当前帧。

### 分帧决策
```
is_detect_frame = (frame_id % detect_interval == 0)
                  || (tracker_state == LOST)  // 丢失时强制每帧YOLO
```

### 状态机
```
IDLE     → YOLO检测到目标 → TRACKING
TRACKING → YOLO帧: YOLO权威 + 刷新模板
         → 补帧: tracker输出
         → tracker连续 lost_threshold 帧无输出 → LOST
LOST     → detect_interval 临时=1 (每帧YOLO)
         → YOLO检到 → TRACKING
         → YOLO 连续 N 帧未检到 → IDLE
```

---

## Task 1: SOT 抽象接口

**Files:**
- Create: `src/track/sot_tracker.h`

- [ ] **Step 1: 创建 SOT 抽象接口头文件**

```cpp
// src/track/sot_tracker.h
#ifndef STEREO_3D_PIPELINE_SOT_TRACKER_H_
#define STEREO_3D_PIPELINE_SOT_TRACKER_H_

#include <cuda_runtime.h>
#include "../pipeline/frame_slot.h"

namespace stereo3d {

struct SOTResult {
    float cx, cy, width, height;
    float confidence;
    bool valid;

    SOTResult() : cx(0), cy(0), width(0), height(0), confidence(0), valid(false) {}
};

enum class TrackerState {
    IDLE,       // 无目标
    TRACKING,   // 正常跟踪
    LOST        // 目标丢失，等待 YOLO 重检测
};

class SOTTracker {
public:
    virtual ~SOTTracker() = default;

    // 初始化 TRT 引擎
    virtual bool init(const std::string& engine_path, cudaStream_t stream) = 0;

    // 设置/刷新跟踪模板 (YOLO 检测到目标时调用)
    // gpu_image: 校正后灰度图 GPU 指针, pitch: 行字节跨度
    virtual void setTarget(const void* gpu_image, int pitch,
                           int img_width, int img_height,
                           const Detection& det) = 0;

    // 在当前帧执行跟踪 (同步, 在 stream 上执行)
    virtual SOTResult track(const void* gpu_image, int pitch,
                            int img_width, int img_height) = 0;

    // 重置跟踪器 (清除模板)
    virtual void reset() = 0;

    // 是否有活跃模板
    virtual bool hasTarget() const = 0;

    // 获取模型输入尺寸 (用于 crop)
    virtual int getTemplateSize() const = 0;
    virtual int getSearchSize() const = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_SOT_TRACKER_H_
```

- [ ] **Step 2: Commit**

```bash
git add src/track/sot_tracker.h
git commit -m "feat(track): add SOT tracker abstract interface"
```

---

## Task 2: CUDA 裁剪缩放 kernel

**Files:**
- Create: `src/track/crop_resize.cu`

- [ ] **Step 1: 实现 CUDA crop+resize kernel**

kernel 功能: 从全帧灰度图中根据 bbox 裁剪带 context_factor 的区域，双线性缩放到目标尺寸。

```cuda
// src/track/crop_resize.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace stereo3d {

__global__ void cropResizeBilinearKernel(
    const uint8_t* __restrict__ src, int src_pitch, int src_w, int src_h,
    float* __restrict__ dst, int dst_size,
    float roi_x, float roi_y, float roi_w, float roi_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_size || dy >= dst_size) return;

    // 目标像素 → 源 ROI 坐标 (双线性)
    float sx = roi_x + (dx + 0.5f) * roi_w / dst_size - 0.5f;
    float sy = roi_y + (dy + 0.5f) * roi_h / dst_size - 0.5f;

    // 边界 clamp
    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1)));

    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float fx = sx - x0, fy = sy - y0;

    float v00 = src[y0 * src_pitch + x0];
    float v10 = src[y0 * src_pitch + x1];
    float v01 = src[y1 * src_pitch + x0];
    float v11 = src[y1 * src_pitch + x1];

    float val = v00 * (1-fx)*(1-fy) + v10 * fx*(1-fy)
              + v01 * (1-fx)*fy     + v11 * fx*fy;

    // 归一化到 [0,1]
    dst[dy * dst_size + dx] = val / 255.0f;
}

void cropResizeGPU(
    const uint8_t* src, int src_pitch, int src_w, int src_h,
    float* dst, int dst_size,
    float cx, float cy, float w, float h,
    float context_factor,
    cudaStream_t stream)
{
    // 扩展 ROI (加 context)
    float roi_w = w * context_factor;
    float roi_h = h * context_factor;
    float roi_x = cx - roi_w * 0.5f;
    float roi_y = cy - roi_h * 0.5f;

    dim3 block(16, 16);
    dim3 grid((dst_size + 15) / 16, (dst_size + 15) / 16);
    cropResizeBilinearKernel<<<grid, block, 0, stream>>>(
        src, src_pitch, src_w, src_h,
        dst, dst_size,
        roi_x, roi_y, roi_w, roi_h);
}

}  // namespace stereo3d
```

- [ ] **Step 2: Commit**

```bash
git add src/track/crop_resize.cu
git commit -m "feat(track): add CUDA crop+resize kernel for SOT"
```

---

## Task 3: NanoTrack TRT 实现

**Files:**
- Create: `src/track/nanotrack_trt.h`
- Create: `src/track/nanotrack_trt.cpp`

NanoTrack 架构: 两个 TRT 引擎 — backbone (共享) 提取 template/search 特征，head (cross-correlation) 输出 score map + bbox offset。

- [ ] **Step 1: 创建 NanoTrack 头文件**

```cpp
// src/track/nanotrack_trt.h
#ifndef STEREO_3D_PIPELINE_NANOTRACK_TRT_H_
#define STEREO_3D_PIPELINE_NANOTRACK_TRT_H_

#include "sot_tracker.h"
#include <NvInfer.h>
#include <string>
#include <memory>

namespace stereo3d {

// 前向声明 crop kernel
void cropResizeGPU(const uint8_t* src, int src_pitch, int src_w, int src_h,
                   float* dst, int dst_size,
                   float cx, float cy, float w, float h,
                   float context_factor, cudaStream_t stream);

class NanoTrackTRT : public SOTTracker {
public:
    NanoTrackTRT();
    ~NanoTrackTRT() override;

    bool init(const std::string& engine_path, cudaStream_t stream) override;
    void setTarget(const void* gpu_image, int pitch,
                   int img_width, int img_height,
                   const Detection& det) override;
    SOTResult track(const void* gpu_image, int pitch,
                    int img_width, int img_height) override;
    void reset() override;
    bool hasTarget() const override;
    int getTemplateSize() const override { return 127; }
    int getSearchSize() const override { return 255; }

private:
    bool loadEngines(const std::string& backbone_path, const std::string& head_path);
    void extractFeature(const float* patch, int size, float* feature);
    SOTResult decodeOutput(int search_size, int img_width, int img_height);

    // TRT 组件 — backbone
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* backboneEngine_ = nullptr;
    nvinfer1::IExecutionContext* backboneCtx_ = nullptr;

    // TRT 组件 — head (cross-correlation)
    nvinfer1::ICudaEngine* headEngine_ = nullptr;
    nvinfer1::IExecutionContext* headCtx_ = nullptr;

    // GPU 缓冲
    float* d_template_patch_ = nullptr;   // [1, 1, 127, 127]
    float* d_search_patch_   = nullptr;   // [1, 1, 255, 255]
    float* d_template_feat_  = nullptr;   // backbone 输出
    float* d_search_feat_    = nullptr;   // backbone 输出
    float* d_head_output_    = nullptr;   // head 输出 (score_map + bbox)
    float* h_head_output_    = nullptr;   // pinned host

    // 状态
    cudaStream_t stream_ = nullptr;
    bool has_target_ = false;
    Detection last_det_;                  // 上一次目标位置 (用于 search 区域中心)

    // head 输出尺寸
    int score_map_size_ = 0;
    int head_output_elements_ = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NANOTRACK_TRT_H_
```

- [ ] **Step 2: 创建 NanoTrack 实现文件**

实现 `init()`, `setTarget()`, `track()`, `reset()`, 其中：
- `init()`: 加载 backbone + head 两个 TRT 引擎, 分配 GPU 缓冲
- `setTarget()`: 裁剪 template patch (127x127) → backbone 提取特征 → 缓存
- `track()`: 裁剪 search patch (255x255, 以 last_det_ 为中心) → backbone → head → decode score map → 返回 bbox
- `decodeOutput()`: 取 score map argmax → 转换回原图坐标

```cpp
// src/track/nanotrack_trt.cpp — 完整实现
// (内容较长，详见 Task 3 Step 2 代码块)
```

核心 track() 逻辑:
```cpp
SOTResult NanoTrackTRT::track(const void* gpu_image, int pitch,
                               int img_width, int img_height) {
    SOTResult result;
    if (!has_target_) return result;

    // 1. 裁剪 search patch (以 last_det_ 为中心, context_factor=2.0)
    cropResizeGPU(static_cast<const uint8_t*>(gpu_image), pitch, img_width, img_height,
                  d_search_patch_, 255,
                  last_det_.cx, last_det_.cy, last_det_.width, last_det_.height,
                  2.0f, stream_);

    // 2. 提取 search 特征
    extractFeature(d_search_patch_, 255, d_search_feat_);

    // 3. Head: cross-correlation (template_feat × search_feat → score+bbox)
    void* headBindings[] = { d_template_feat_, d_search_feat_, d_head_output_ };
    headCtx_->executeV2(headBindings);
    cudaStreamSynchronize(stream_);

    // 4. D2H + decode
    cudaMemcpyAsync(h_head_output_, d_head_output_,
                    head_output_elements_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    result = decodeOutput(255, img_width, img_height);
    if (result.valid) {
        last_det_.cx = result.cx;
        last_det_.cy = result.cy;
        last_det_.width = result.width;
        last_det_.height = result.height;
    }
    return result;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/track/nanotrack_trt.h src/track/nanotrack_trt.cpp
git commit -m "feat(track): NanoTrack TRT implementation"
```

---

## Task 4: MixFormerV2-small TRT 实现

**Files:**
- Create: `src/track/mixformer_trt.h`
- Create: `src/track/mixformer_trt.cpp`

MixFormerV2 架构: 单一 TRT 引擎, 输入 template_patch + search_patch (拼接), 输出 score + bbox_offset。

- [ ] **Step 1: 创建 MixFormerV2 头文件**

```cpp
// src/track/mixformer_trt.h
#ifndef STEREO_3D_PIPELINE_MIXFORMER_TRT_H_
#define STEREO_3D_PIPELINE_MIXFORMER_TRT_H_

#include "sot_tracker.h"
#include <NvInfer.h>
#include <string>

namespace stereo3d {

void cropResizeGPU(const uint8_t* src, int src_pitch, int src_w, int src_h,
                   float* dst, int dst_size,
                   float cx, float cy, float w, float h,
                   float context_factor, cudaStream_t stream);

class MixFormerTRT : public SOTTracker {
public:
    MixFormerTRT();
    ~MixFormerTRT() override;

    bool init(const std::string& engine_path, cudaStream_t stream) override;
    void setTarget(const void* gpu_image, int pitch,
                   int img_width, int img_height,
                   const Detection& det) override;
    SOTResult track(const void* gpu_image, int pitch,
                    int img_width, int img_height) override;
    void reset() override;
    bool hasTarget() const override;
    int getTemplateSize() const override { return 128; }
    int getSearchSize() const override { return 256; }

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    float* d_template_patch_ = nullptr;  // [1, 1, 128, 128]
    float* d_search_patch_   = nullptr;  // [1, 1, 256, 256]
    float* d_output_         = nullptr;  // [score, cx, cy, w, h]
    float* h_output_         = nullptr;  // pinned host

    cudaStream_t stream_ = nullptr;
    bool has_target_ = false;
    Detection last_det_;
    int output_elements_ = 0;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_MIXFORMER_TRT_H_
```

- [ ] **Step 2: 创建 MixFormerV2 实现文件**

类似 NanoTrack 但更简单（单引擎）:
- `init()`: 加载单个 TRT 引擎
- `setTarget()`: 裁剪 template patch → 缓存在 GPU（不提取特征，每次 track 都传入）
- `track()`: 裁剪 search patch → engine(template, search) → decode → bbox

- [ ] **Step 3: Commit**

```bash
git add src/track/mixformer_trt.h src/track/mixformer_trt.cpp
git commit -m "feat(track): MixFormerV2-small TRT implementation"
```

---

## Task 5: FrameSlot 扩展

**Files:**
- Modify: `src/pipeline/frame_slot.h`

- [ ] **Step 1: 添加 SOT 相关字段到 FrameSlot**

在 `frame_slot.h` 中添加:

```cpp
// 在 Detection 结构体之后添加
enum class BboxSource {
    NONE,       // 无检测
    YOLO,       // YOLO 检测
    TRACKER     // SOT 补帧
};

// 在 FrameSlot 结构体中添加 (Stage 1 检测结果区域之后):
    // =========== SOT Tracker 补帧 ===========
    Detection sot_bbox;                   ///< SOT tracker 输出 (补帧用)
    BboxSource bbox_source = BboxSource::NONE; ///< 最终 bbox 来源
    bool is_detect_frame = true;          ///< 是否为 YOLO 检测帧
```

在 `reset()` 中添加清理:
```cpp
    sot_bbox = Detection();
    bbox_source = BboxSource::NONE;
    is_detect_frame = true;
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/frame_slot.h
git commit -m "feat(track): extend FrameSlot with SOT fields"
```

---

## Task 6: PipelineConfig + Pipeline 成员扩展

**Files:**
- Modify: `src/pipeline/pipeline.h`

- [ ] **Step 1: 添加 TrackerConfig 和 tracker 成员**

在 `pipeline.h` 的 `PipelineConfig` 中添加 tracker 配置:

```cpp
    // SOT Tracker 补帧 (YOLO 间隔帧)
    struct TrackerConfig {
        bool enabled = false;              ///< 是否启用 SOT 补帧
        std::string type = "nanotrack";    ///< "nanotrack" | "mixformer"
        std::string engine_path;           ///< TRT 引擎路径 (NanoTrack: backbone路径前缀)
        std::string head_engine_path;      ///< NanoTrack head 引擎路径 (MixFormer 不用)
        int detect_interval = 3;           ///< YOLO 检测间隔 (每N帧一次, 其余补帧)
        int lost_threshold = 5;            ///< tracker 连续无输出帧数 → LOST
        float min_confidence = 0.3f;       ///< tracker 最低置信度
    } tracker;
```

在 `Pipeline` 私有成员中添加:

```cpp
    #include "../track/sot_tracker.h"
    // (在 include 区域)

    // (在私有成员区域, hybrid_depth_ 之后)
    std::unique_ptr<SOTTracker> tracker_;           ///< SOT 补帧跟踪器
    TrackerState tracker_state_ = TrackerState::IDLE;
    int tracker_lost_count_ = 0;                    ///< 连续丢失帧数
    int effective_detect_interval_ = 3;             ///< 运行时检测间隔 (LOST 时=1)
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/pipeline.h
git commit -m "feat(track): add TrackerConfig and tracker members to Pipeline"
```

---

## Task 7: Pipeline 初始化 — tracker 加载

**Files:**
- Modify: `src/pipeline/pipeline.cpp`

- [ ] **Step 1: 在 init() 中加载 tracker**

在 `pipeline.cpp` 的 `init()` 方法末尾（hybrid_depth_ 初始化之后）添加 tracker 初始化:

```cpp
    // SOT Tracker 初始化
    if (config_.tracker.enabled) {
        if (config_.tracker.type == "nanotrack") {
            tracker_ = std::make_unique<NanoTrackTRT>();
        } else if (config_.tracker.type == "mixformer") {
            tracker_ = std::make_unique<MixFormerTRT>();
        } else {
            LOG_ERROR("Unknown tracker type: %s", config_.tracker.type.c_str());
            return false;
        }

        if (!tracker_->init(config_.tracker.engine_path, streams_.cudaStreamGPU)) {
            LOG_ERROR("Failed to init SOT tracker");
            return false;
        }
        effective_detect_interval_ = config_.tracker.detect_interval;
        LOG_INFO("SOT tracker (%s) initialized, detect_interval=%d",
                 config_.tracker.type.c_str(), effective_detect_interval_);
    }
```

添加对应 include:
```cpp
#include "../track/nanotrack_trt.h"
#include "../track/mixformer_trt.h"
```

- [ ] **Step 2: 在 loadConfig (YAML 解析) 中添加 tracker 配置解析**

在 YAML 解析部分添加:
```cpp
    if (yaml["tracker"]) {
        auto t = yaml["tracker"];
        config.tracker.enabled = t["enabled"].as<bool>(false);
        config.tracker.type = t["type"].as<std::string>("nanotrack");
        config.tracker.engine_path = t["engine_path"].as<std::string>("");
        config.tracker.head_engine_path = t["head_engine_path"].as<std::string>("");
        config.tracker.detect_interval = t["detect_interval"].as<int>(3);
        config.tracker.lost_threshold = t["lost_threshold"].as<int>(5);
        config.tracker.min_confidence = t["min_confidence"].as<float>(0.3f);
    }
```

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/pipeline.cpp
git commit -m "feat(track): init SOT tracker in Pipeline::init()"
```

---

## Task 8: pipelineLoopROI() — 核心补帧逻辑

**Files:**
- Modify: `src/pipeline/pipeline.cpp`

这是最关键的改动。在现有的 `pipelineLoopROI()` 四阶段循环中嵌入 tracker 逻辑。

### 核心修改策略

现有流程:
```
Phase A: requestGrab(N+1)
Phase B: stage1_detect(N)     ← 每帧都 enqueue YOLO
Phase C: stage2 fuse(N-1)     ← 每帧都 collect+fuse
Phase D: waitGrab + rectify
```

修改后:
```
Phase A: requestGrab(N+1)
Phase B: if is_detect_frame → stage1_detect(N)
         tracker.track(N)   ← 每帧都跑 tracker (有模板时)
Phase C: if is_fuse_detect_frame → collect YOLO + 刷新 tracker 模板
         else                    → 使用 tracker bbox
         stage2_roi (用最终 bbox)
Phase D: waitGrab + rectify
```

- [ ] **Step 1: 修改 Phase B — 条件性 YOLO enqueue + tracker track**

在 `pipelineLoopROI()` 的 Phase B 中:

```cpp
        // Phase B: Stage1 — 条件性检测 + tracker 跟踪
        bool current_is_detect_frame = false;
        if (next_detect_frame < next_grab_frame) {
            int slot_idx = next_detect_frame % RING_BUFFER_SIZE;
            auto& slot = slots_[slot_idx];

            if (slot.grab_failed) {
                vpiStreamSync(streams_.vpiStreamPVA);
                next_detect_frame++;
                next_fuse_frame = next_detect_frame - 1;
            } else {
                // 判断是否为 YOLO 检测帧
                current_is_detect_frame =
                    !tracker_ ||
                    (slot.frame_id % effective_detect_interval_ == 0) ||
                    (tracker_state_ == TrackerState::LOST);
                slot.is_detect_frame = current_is_detect_frame;

                {
                    ScopedTimer tw("Stage1_WaitRect");
                    cudaEventRecord(slot.evtRectDone, streams_.cudaStreamGPU);
                    globalPerf().record("Stage1_WaitRect", tw.elapsedMs());
                }

                if (current_is_detect_frame) {
                    // YOLO 帧: 正常 enqueue
                    auto dlaStream = getDLAStream(slot.frame_id);
                    cudaStreamWaitEvent(dlaStream, slot.evtRectDone, 0);

                    {
                        ScopedTimer t1("Stage1_DetectSubmit");
                        stage1_detect(slot, slot_idx);
                        globalPerf().record("Stage1_DetectSubmit", t1.elapsedMs());
                    }
                    cudaEventRecord(slot.evtDetectDone, dlaStream);
                }

                // Tracker: 每帧都跑 (有模板时)
                if (tracker_ && tracker_->hasTarget()) {
                    // 获取当前帧 GPU 图像指针
                    VPIImage trackImg = (config_.detector_input_format == "bgr")
                                        ? slot.rectBGR_vpiL : slot.rectGray_vpiL;
                    VPIImageData imgData;
                    if (vpiImageLockData(trackImg, VPI_LOCK_READ,
                                         VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData) == VPI_SUCCESS) {
                        ScopedTimer tt("SOT_Track");
                        slot.sot_bbox_result = tracker_->track(
                            imgData.buffer.pitch.planes[0].data,
                            imgData.buffer.pitch.planes[0].pitchBytes,
                            config_.rect_width, config_.rect_height);
                        globalPerf().record("SOT_Track", tt.elapsedMs());
                        vpiImageUnlock(trackImg);
                    }
                }

                next_detect_frame++;
            }
        }
```

- [ ] **Step 2: 修改 Phase C — 条件性 collect/fuse + tracker 模板刷新**

Phase C 中根据 `is_detect_frame` 分流:

```cpp
        // Phase C: 融合
        int fuse_slot_idx = -1;
        if (next_fuse_frame < next_detect_frame - 1) {
            fuse_slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
            auto& fuse_slot = slots_[fuse_slot_idx];

            if (fuse_slot.is_detect_frame) {
                // YOLO 帧: collect 检测结果
                {
                    ScopedTimer tw("Stage2_WaitDetect");
                    cudaStreamWaitEvent(streams_.cudaStreamFuse, fuse_slot.evtDetectDone, 0);
                    cudaStreamSynchronize(streams_.cudaStreamFuse);
                    globalPerf().record("Stage2_WaitDetect", tw.elapsedMs());
                }

                auto* det = getDetector(fuse_slot.frame_id);
                int fuse_slot_ring = fuse_slot_idx % RING_BUFFER_SIZE;
                fuse_slot.detections = det->collect(fuse_slot_ring,
                                                     config_.rect_width, config_.rect_height);

                if (!fuse_slot.detections.empty()) {
                    fuse_slot.bbox_source = BboxSource::YOLO;
                    // 刷新 tracker 模板 (用 fuse_slot 对应帧的图像)
                    if (tracker_) {
                        VPIImage trackImg = (config_.detector_input_format == "bgr")
                                            ? fuse_slot.rectBGR_vpiL : fuse_slot.rectGray_vpiL;
                        VPIImageData imgData;
                        if (vpiImageLockData(trackImg, VPI_LOCK_READ,
                                             VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData) == VPI_SUCCESS) {
                            tracker_->setTarget(
                                imgData.buffer.pitch.planes[0].data,
                                imgData.buffer.pitch.planes[0].pitchBytes,
                                config_.rect_width, config_.rect_height,
                                fuse_slot.detections[0]);
                            vpiImageUnlock(trackImg);
                        }
                        tracker_state_ = TrackerState::TRACKING;
                        tracker_lost_count_ = 0;
                        effective_detect_interval_ = config_.tracker.detect_interval;
                    }
                } else {
                    // YOLO 检测到空 = 无目标
                    fuse_slot.bbox_source = BboxSource::NONE;
                    if (tracker_) {
                        tracker_lost_count_++;
                        if (tracker_lost_count_ >= config_.tracker.lost_threshold) {
                            tracker_state_ = TrackerState::LOST;
                            effective_detect_interval_ = 1; // 每帧 YOLO
                        }
                    }
                }
            } else {
                // 补帧: 使用 tracker bbox
                if (fuse_slot.sot_bbox_result.valid &&
                    fuse_slot.sot_bbox_result.confidence >= config_.tracker.min_confidence) {
                    // 将 SOTResult 转换为 Detection 放入 detections
                    Detection sot_det;
                    sot_det.cx = fuse_slot.sot_bbox_result.cx;
                    sot_det.cy = fuse_slot.sot_bbox_result.cy;
                    sot_det.width = fuse_slot.sot_bbox_result.width;
                    sot_det.height = fuse_slot.sot_bbox_result.height;
                    sot_det.confidence = fuse_slot.sot_bbox_result.confidence;
                    sot_det.class_id = 0;  // volleyball
                    fuse_slot.detections = { sot_det };
                    fuse_slot.bbox_source = BboxSource::TRACKER;
                    tracker_lost_count_ = 0;
                } else {
                    fuse_slot.detections.clear();
                    fuse_slot.bbox_source = BboxSource::NONE;
                    tracker_lost_count_++;
                    if (tracker_lost_count_ >= config_.tracker.lost_threshold) {
                        tracker_state_ = TrackerState::LOST;
                        effective_detect_interval_ = 1;
                    }
                }
            }

            // ROI 匹配 + 深度融合 (与原逻辑相同, detections 已填充)
            {
                ScopedTimer t2("Stage2_ROIMatchFuse");
                stage2_roi_match_fuse_from_detections(fuse_slot, fuse_slot_idx);
                globalPerf().record("Stage2_ROIMatchFuse", t2.elapsedMs());
            }
        }
```

注意: 需要新增 `stage2_roi_match_fuse_from_detections()` 方法，它与现有 `stage2_roi_match_fuse()` 类似但跳过 collect 步骤（detections 已由上层填充）。

- [ ] **Step 3: 新增 stage2_roi_match_fuse_from_detections()**

```cpp
// 补帧模式的 ROI 匹配融合 — detections 已由 tracker 或 YOLO 填充
void Pipeline::stage2_roi_match_fuse_from_detections(FrameSlot& slot, int slot_index) {
    slot.results.clear();

    if (slot.detections.empty()) {
        if (hybrid_depth_) {
            slot.results = hybrid_depth_->predictOnly();
        }
        return;
    }

    // 获取校正后灰度左右图 GPU 指针
    VPIImageData imgDataL, imgDataR;
    auto stL = vpiImageLockData(slot.rectGray_vpiL, VPI_LOCK_READ,
                                VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataL);
    auto stR = vpiImageLockData(slot.rectGray_vpiR, VPI_LOCK_READ,
                                VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgDataR);
    if (stL != VPI_SUCCESS || stR != VPI_SUCCESS) {
        if (stL == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiL);
        if (stR == VPI_SUCCESS) vpiImageUnlock(slot.rectGray_vpiR);
        if (hybrid_depth_) slot.results = hybrid_depth_->predictOnly();
        return;
    }

    const uint8_t* leftPtr  = static_cast<const uint8_t*>(imgDataL.buffer.pitch.planes[0].data);
    int leftPitch  = imgDataL.buffer.pitch.planes[0].pitchBytes;
    const uint8_t* rightPtr = static_cast<const uint8_t*>(imgDataR.buffer.pitch.planes[0].data);
    int rightPitch = imgDataR.buffer.pitch.planes[0].pitchBytes;

    auto roi_results = roi_matcher_->match(
        leftPtr, leftPitch, rightPtr, rightPitch,
        config_.rect_width, config_.rect_height,
        slot.detections, streams_.cudaStreamFuse);

    vpiImageUnlock(slot.rectGray_vpiL);
    vpiImageUnlock(slot.rectGray_vpiR);

    if (hybrid_depth_) {
        auto now = std::chrono::steady_clock::now();
        double dt = 0.01;
        if (last_fuse_time_.time_since_epoch().count() > 0) {
            dt = std::chrono::duration<double>(now - last_fuse_time_).count();
            dt = std::clamp(dt, 0.002, 0.1);
        }
        last_fuse_time_ = now;
        slot.results = hybrid_depth_->estimate(slot.detections, roi_results, dt);
    } else {
        slot.results = std::move(roi_results);
    }
}
```

- [ ] **Step 4: 在 pipeline.h 中声明新方法**

```cpp
    void stage2_roi_match_fuse_from_detections(FrameSlot& slot, int slot_index);
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/pipeline.h src/pipeline/pipeline.cpp
git commit -m "feat(track): integrate SOT tracker into pipelineLoopROI()"
```

---

## Task 9: FrameSlot 添加 SOTResult 字段

**Files:**
- Modify: `src/pipeline/frame_slot.h`

- [ ] **Step 1: 在 FrameSlot 中添加 SOTResult 存储**

需要 forward-include `sot_tracker.h` 中的 `SOTResult`，或将 `SOTResult` 移到 `frame_slot.h`（避免循环依赖）。

最佳方案: 将 `SOTResult` 定义移入 `frame_slot.h`（它是纯数据结构），`sot_tracker.h` include `frame_slot.h`。

```cpp
// 在 frame_slot.h 中添加 (Detection 之后)
struct SOTResult {
    float cx, cy, width, height;
    float confidence;
    bool valid;
    SOTResult() : cx(0), cy(0), width(0), height(0), confidence(0), valid(false) {}
};
```

在 FrameSlot 中添加:
```cpp
    SOTResult sot_bbox_result;            ///< SOT tracker 跟踪结果
```

在 `reset()` 中添加:
```cpp
    sot_bbox_result = SOTResult();
```

- [ ] **Step 2: 更新 sot_tracker.h 移除 SOTResult 定义，改为 include frame_slot.h**

`sot_tracker.h` 已经 include `frame_slot.h`，所以只需移除其中的 `SOTResult` 定义。

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/frame_slot.h src/track/sot_tracker.h
git commit -m "refactor(track): move SOTResult to frame_slot.h, avoid circular dep"
```

---

## Task 10: CMakeLists.txt 更新

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: 添加 track 源文件**

在 `PIPELINE_SOURCES` 中添加:
```cmake
  src/track/nanotrack_trt.cpp
  src/track/mixformer_trt.cpp
```

在 `CUDA_SOURCES` 中添加:
```cmake
  src/track/crop_resize.cu
```

- [ ] **Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add track module sources to CMakeLists"
```

---

## Task 11: 配置文件

**Files:**
- Create: `config/pipeline_yolo11m_960_nanotrack.yaml`
- Create: `config/pipeline_yolo11m_960_mixformer.yaml`

- [ ] **Step 1: 创建 NanoTrack 配置**

基于 `pipeline_roi.yaml` 模板，修改检测器和添加 tracker 节：

```yaml
# YOLO11-M@960 DLA + NanoTrack GPU 补帧
detector:
  engine_path: "/home/nvidia/NX_volleyball/model/yolo11m_dla0_int8_960.engine"
  input_size: 960
  use_dla: true
  dla_core: 0
  allow_gpu_fallback: true
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 10
  input_format: "gray"

tracker:
  enabled: true
  type: "nanotrack"
  engine_path: "/home/nvidia/NX_volleyball/model/nanotrack_backbone.engine"
  head_engine_path: "/home/nvidia/NX_volleyball/model/nanotrack_head.engine"
  detect_interval: 3          # 100Hz / 3 ≈ 33Hz YOLO
  lost_threshold: 5
  min_confidence: 0.3
```

- [ ] **Step 2: 创建 MixFormerV2 配置**

```yaml
tracker:
  enabled: true
  type: "mixformer"
  engine_path: "/home/nvidia/NX_volleyball/model/mixformerv2_small.engine"
  detect_interval: 3
  lost_threshold: 5
  min_confidence: 0.3
```

- [ ] **Step 3: Commit**

```bash
git add config/pipeline_yolo11m_960_nanotrack.yaml config/pipeline_yolo11m_960_mixformer.yaml
git commit -m "config: add YOLO11-M@960 + NanoTrack/MixFormer configs"
```

---

## Task 12: 编译验证 (NX 远程)

- [ ] **Step 1: 同步代码到 NX**

```powershell
# Windows → NX rsync
scp -r NX_volleyball/stereo_3d_pipeline/ nvidia@192.168.31.56:/home/nvidia/NX_volleyball/stereo_3d_pipeline/
```

- [ ] **Step 2: SSH 编译**

```bash
ssh nvidia@192.168.31.56 "cd /home/nvidia/NX_volleyball/stereo_3d_pipeline/build && cmake .. && make -j4"
```

- [ ] **Step 3: 修复编译错误 (如果有)**

- [ ] **Step 4: Commit 修复**

---

## Task 13: 模型准备 (NanoTrack ONNX → TRT)

- [ ] **Step 1: 确认 NanoTrack ONNX 模型获取方式**

NanoTrack 模型来源:
- PyTracking 或 nanotrack 官方仓库导出 ONNX
- 需要 backbone 和 head 两个 ONNX

- [ ] **Step 2: 在 NX 上转换 TRT 引擎**

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=nanotrack_backbone.onnx \
  --saveEngine=nanotrack_backbone.engine \
  --fp16 --workspace=1024
```

- [ ] **Step 3: 类似转换 MixFormerV2**

---

## Task 14: 集成测试

- [ ] **Step 1: 运行 NanoTrack 配置**

```bash
./stereo_pipeline --config config/pipeline_yolo11m_960_nanotrack.yaml
```

观察:
- FPS 是否接近 100Hz
- YOLO/tracker bbox 交替输出是否正常
- LOST 状态恢复是否正常

- [ ] **Step 2: 运行 MixFormerV2 配置**

- [ ] **Step 3: 性能对比和调参**

---

## 执行顺序与依赖

```
Task 1 (接口) ──┐
Task 2 (CUDA)  ──┼── Task 3 (NanoTrack) ──┐
                 │                          │
                 └── Task 4 (MixFormer) ───┤
                                            │
Task 5 (FrameSlot) ────────────────────────┤
Task 9 (SOTResult移动) ───────────────────┤
Task 6 (PipelineConfig) ──────────────────┤
                                            │
                                            ├── Task 7 (init) ── Task 8 (主循环) ── Task 10 (CMake)
                                            │
Task 11 (配置文件) ─── Task 12 (编译) ── Task 13 (模型) ── Task 14 (测试)
```

Tasks 1-6, 9-11 可大量并行。Task 7-8 是关键路径（依赖所有前置）。Task 12-14 顺序执行。
