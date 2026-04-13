# 检测层增强：YOLO-Observer + Tracker 补帧

## 定位

这是**第一层检测层**的增强方案。对下游完全透明 — 输出与 YOLO26@640 每帧检测相同格式的 bbox，后续双目深度、Kalman、3D 轨迹等不做任何改变。

## 问题

YOLO11-M@960 推理 ~7ms，100Hz 相机下只能 ~40FPS 检测。剩余 ~60 帧无 bbox 输出。

## 方案

用轻量跟踪模型 (NanoTrack / MixFormerV2-small) 在 YOLO 间隙帧产生 bbox，使检测层恢复 100FPS 全帧输出。

```
        100Hz 相机帧流
             │
     ┌───────┴───────┐
     │               │
  YOLO 帧            补帧
  (~33Hz)           (~67Hz)
     │               │
  DLA 异步         GPU 同步
  enqueue→collect  tracker.track()
     │               │
  bbox (权威)      bbox (跟踪)
     │               │
     └───────┬───────┘
             │
        统一 bbox 输出
             │
        ┌────┴────┐
        │  下游   │  (双目深度/Kalman/3D — 不改)
        └─────────┘
```

## 异步时序

**关键约束**: YOLO ~7ms > 帧间隔 10ms 时无法每帧推理。使用现有 enqueue/collect 异步流水线。

```
时序 (detect_interval=3, YOLO ~7ms on DLA):

Frame 0:  YOLO.enqueue(F0)      tracker.track(F0) → (首帧无模板, 无输出)
Frame 1:  (DLA 推理中...)       tracker.track(F1) → (无模板, 无输出)
Frame 2:  YOLO.collect(F0)→det  tracker.track(F2) → tracker bbox
          ↳ 用 det 输出         ↳ 刷新 tracker 模板
          YOLO.enqueue(F3)
Frame 3:  (DLA 推理中...)       tracker.track(F3) → bbox ← 输出
Frame 4:  (DLA 推理中...)       tracker.track(F4) → bbox ← 输出
Frame 5:  YOLO.collect(F3)→det  tracker.track(F5) → tracker bbox
          ↳ 用 det 输出         ↳ 刷新模板
          YOLO.enqueue(F6)
...

输出帧率 = 100Hz (每帧都有 bbox, F0/F1 除外等 YOLO 初检)
```

## 分帧逻辑

```cpp
// Phase B: 检测层
if (tracker.hasTarget()) {
    sot_bbox = tracker.track(frame);       // 每帧, GPU ~0.5-1ms
}

if (is_yolo_frame) {
    yolo.enqueue(frame);                   // 每N帧, DLA 异步
}

// Phase C: 融合
if (fuse_slot.is_detect_frame) {
    yolo_det = yolo.collect(fuse_slot);    // YOLO 结果到达
    if (yolo_det.valid) {
        output_bbox = yolo_det.bbox;       // YOLO 权威
        tracker.setTarget(frame, yolo_det.bbox);  // 刷新模板
    } else {
        output_bbox = none;                // YOLO 没检到 = 无输出
    }
} else {
    // 补帧: tracker 输出
    if (sot_bbox.valid) {
        output_bbox = sot_bbox;            // tracker 补帧
    } else {
        output_bbox = none;                // tracker 也丢了 = 无输出
    }
}

// output_bbox → 传给下游 (等同于 YOLO26@640 的每帧检测结果)
```

## 状态管理

```
IDLE        → YOLO 首次检测到 → TRACKING
TRACKING    → YOLO 帧: YOLO 权威输出 + 刷新模板
            → 补帧: tracker 正常跟踪
            → tracker 连续 K 帧无输出 → LOST
LOST        → YOLO 重新检测到 → TRACKING
            → 可选: 临时提高 YOLO 频率 (detect_interval=1)
```

3 个状态，逻辑简单。不做 Kalman predict、不做搜索恢复 — YOLO 是唯一的重新捕获手段。

## 硬件分配

| 组件 | 硬件 | 时间 | 模式 |
|------|------|------|------|
| YOLO11-M@960 | DLA0 | ~7ms | 异步 enqueue/collect |
| NanoTrack | GPU (cudaStreamGPU) | ~0.5ms | 同步 |
| MixFormerV2-small | GPU (cudaStreamGPU) | ~1ms | 同步 |
| 畸变校正 | VIC/PVA | ~1ms | 同步 |

DLA 和 GPU 并行不冲突。tracker 使用现有 cudaStreamGPU 即可。

## 文件规划

### 新增
| 文件 | 说明 |
|------|------|
| `src/track/sot_tracker.h` | 抽象接口 (setTarget/track/reset) |
| `src/track/nanotrack_trt.h/cpp` | NanoTrack TRT 推理 |
| `src/track/mixformer_trt.h/cpp` | MixFormerV2-small TRT 推理 |
| `src/track/crop_patch.cu` | CUDA 裁剪缩放 kernel |
| `config/pipeline_yolo11m_960_nanotrack.yaml` | NanoTrack 配置 |
| `config/pipeline_yolo11m_960_mixformer.yaml` | MixFormerV2 配置 |

### 修改
| 文件 | 改动 |
|------|------|
| `pipeline.h` | +TrackerConfig, +tracker 成员 |
| `pipeline.cpp` | pipelineLoopROI() 插入 tracker 逻辑 |
| `frame_slot.h` | +sot_result, +is_detect_frame |
| `CMakeLists.txt` | +track 源文件 |

### 不改
| 文件 | 原因 |
|------|------|
| `hybrid_depth.h/cpp` | 下游不变 |
| `roi_matcher.*` | 下游不变 |
| `trt_detector.*` | YOLO 检测器复用 |

## 待回答

- [ ] YOLO11-M@960 模型文件从哪里来？需要训练/导出？
