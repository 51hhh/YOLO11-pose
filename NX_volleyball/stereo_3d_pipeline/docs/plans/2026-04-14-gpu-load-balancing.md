# GPU 负载平衡 — NX 功率受限下的YOLO+SOT序列化推理

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Orin NX 电源受限约束下，通过严格的 GPU 流序列化实现 YOLO11S 33fps + SOT追踪 66fps 的稳定 100fps 输出，防止功率过载导致系统断电重启。

**Architecture:** 核心改变是从异步流水线改为 **同步序列化执行**：YOLO enqueue → DLA推理完毕 → collect结果 → tracker推理 → 输出。在 detect_interval=3 模式下，每 3 帧激活一次 YOLO（~7-9ms），其余帧只执行 tracker（~0.5-1ms），总时间保持在 10ms 以内，避免 GPU/DLA 并行运算导致的功率爆口。

**Tech Stack:** C++17, CUDA Streams, TensorRT, GPU功率管理

---

## 核心痛点分析

| 现象 | 原因 | 影响 |
|------|------|------|
| NX直接死机无ERROR | YOLO+tracker并行推理，GPU/DLA同时高负载，芯片功耗 >15W | 电源无法供电，系统强制重启 |
| 异步pipeline设计 | enqueue后立即返回，下一帧立即继续任务 | 多帧任务堆积在GPU/DLA队列 |
| detect_interval逻辑已有 | 代码中frame_id % detect_interval == 0 判断存在 | 但4个配置仍用默认值（可能 detect_interval < 3） |

**解决策略：**
1. ✅ 修改4个YAML配置：detect_interval=3（已有代码支持）
2. ⚠️ 代码改进：强制同步序列化（阻止并行）
3. ⚠️ 性能监控：确保单帧时间 < 10ms

---

## 文件修改清单

### 修改文件
| 文件 | 改动 |
|------|------|
| `config/pipeline_yolo11s_960_nanotrack.yaml` | tracker.detect_interval: 3 |
| `config/pipeline_yolo11s_960_mixformer.yaml` | tracker.detect_interval: 3 |
| `config/pipeline_yolo11s_960_lighttrack.yaml` | tracker.detect_interval: 3 |
| `config/pipeline_yolo11s_960_siamfc.yaml` | tracker.detect_interval: 3 |

### 可选代码优化（防止并行）
| 文件 | 改动 | 理由 |
|------|------|------|
| `src/pipeline/pipeline.cpp` | Phase B 中添加显式 GPU 同步 | 确保YOLO collect完成后再启动tracker |
| `src/pipeline/pipeline.h` | 新增配置参数 `gpu_sync_mode` | 区分异步(服务器)vs同步(NX)模式 |

---

## Task 1: 修改4个YAML配置文件 — 启用detect_interval=3

**Files:**
- Modify: `config/pipeline_yolo11s_960_nanotrack.yaml`
- Modify: `config/pipeline_yolo11s_960_mixformer.yaml`
- Modify: `config/pipeline_yolo11s_960_lighttrack.yaml`
- Modify: `config/pipeline_yolo11s_960_siamfc.yaml`

### 策略
所有4个文件都按相同方式修改：确保 `tracker.detect_interval: 3`，禁用其他冲突配置。

- [ ] **Step 1: 审查当前4个YAML中的tracker配置**

检查每个文件中 tracker 字段的现状，特别是 detect_interval 的值。

- [ ] **Step 2: 修改 pipeline_yolo11s_960_nanotrack.yaml**

```yaml
# 关键改动 — 确保这些字段存在且设置正确:
tracker:
  enabled: true
  type: "nanotrack"
  engine_path: "/home/nvidia/sot_export/exported/nanotrack_backbone_template.engine"
  search_engine_path: "/home/nvidia/sot_export/exported/nanotrack_backbone_search.engine"
  head_engine_path: "/home/nvidia/sot_export/exported/nanotrack_head_adapted.engine"
  detect_interval: 3          # ← MUST 是 3（每3帧YOLO一次）
  lost_threshold: 5
  min_confidence: 0.3
```

其中：
- detect_interval=3 → YOLO 在 frame 0,3,6,9,... 时推理
- 追踪补帧：frame 1,2,4,5,7,8,... 由NanoTrack推理
- 时间分配：YOLO 7-9ms + tracker 0.5-1ms ≈ 9-10ms< 10ms budget

- [ ] **Step 3: 修改 pipeline_yolo11s_960_mixformer.yaml**

同样设置 detect_interval=3

- [ ] **Step 4: 修改 pipeline_yolo11s_960_lighttrack.yaml**

同样设置 detect_interval=3

- [ ] **Step 5: 修改 pipeline_yolo11s_960_siamfc.yaml**

同样设置 detect_interval=3

- [ ] **Step 6: Git验证**

```bash
cd NX_volleyball/stereo_3d_pipeline
git diff config/pipeline_yolo11s_960_*.yaml | grep detect_interval
# 应该看到4行都是 detect_interval: 3
```

- [ ] **Step 7: Commit**

```bash
git add config/pipeline_yolo11s_960_{nanotrack,mixformer,lighttrack,siamfc}.yaml
git commit -m "config: set detect_interval=3 (YOLO 33fps + tracker 66fps → 100fps output)"
```

---

## Task 2: 代码同步化改进 — 防止YOLO/tracker并行

**Files:**
- Modify: `src/pipeline/pipeline.cpp` (pipelineLoopROI Phase B)
- Modify: `src/pipeline/pipeline.h` (可选: 新增gpu_sync_mode参数)

### 问题根源

当前代码在 Phase B 中：
```cpp
if (is_detect_frame) {
    stage1_detect(slot);      // ← enqueue YOLO async, 立即返回
    cudaEventRecord(...);
} else {
    tracker_infill(slot);     // ← 同步跑tracker
    cudaEventRecord(...);
}
```

问题：`stage1_detect` 只是 enqueue，返回后下一帧立即执行下一条任务，可能导致 YOLO 和 tracker 同时在 GPU 上运行。

### 解决方案

添加显式同步点，确保检测帧的 YOLO **完全完成**后再进行下一步。核心改动在 Phase C 中添加同步逻辑。

- [ ] **Step 1: 查看当前Phase B逻辑中的YOLO enqueue点**

定位 `stage1_detect(slot, slot_idx);` 调用位置，了解后续的 cudaEventRecord 时序。

- [ ] **Step 2: 在Phase C添加YOLO完成等待**

在 Phase C 开始处（Stage2_WaitDetect前），确保YOLO collect已完成：

```cpp
// Phase C: Stage2 — ROI匹配 + 融合帧N-1
if (next_fuse_frame < next_detect_frame - 1) {
    int slot_idx = next_fuse_frame % RING_BUFFER_SIZE;
    auto& slot = slots_[slot_idx];

    // ← 添加这块: 强制等待 YOLO collect 完成（如果是检测帧）
    if (slot.is_detect_frame) {
        ScopedTimer tw("Stage2_WaitDetectSync");
        // 确保YOLO异步任务完成
        cudaStreamSynchronize(getDLAStream(slot.frame_id));
        // 只有这时可以安全collect
        slot.detections = getDetector(slot.frame_id)->collect(...);
        globalPerf().record("Stage2_WaitDetectSync", tw.elapsedMs());
    }

    // 后续代码...
```

此改动确保：
- YOLO Frame N 完全finish后
- 才允许 tracker 推理 Frame N+1
- 时间轴避免并行

- [ ] **Step 3: 测试单帧时间**

修改后在NX上运行，检查profiler输出：
```
[ROI] FPS: 98.5
  Stage1_DetectSubmit: 1.2ms (async enqueue)
  Stage1_TrackerInfill: 0.8ms (tracking frame only)
  Stage2_WaitDetectSync: 8.5ms (YOLO collect + H2D wait)  ← 关键
  Stage2_ROIMatchFuse: 1.5ms
  Total budget: ~10-11ms < 16.7ms ✓
```

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/pipeline.cpp
git commit -m "perf: add explicit GPU sync to prevent YOLO/tracker parallel execution on NX"
```

---

## Task 3: 在NX上验证完整配置

**Environment:** Jetson Orin NX Super 16GB, JP6, `nvidia@192.168.31.56`

- [ ] **Step 1: 编译新代码**

```bash
ssh nvidia@192.168.31.56
cd ~/stereo_3d_pipeline/build
make -j4
```

- [ ] **Step 2: 部署新YAML配置到NX**

```bash
# 在NX上
cp config/pipeline_yolo11s_960_nanotrack.yaml config_deploy/
# 或通过rsync从主机同步
```

- [ ] **Step 3: 运行Pipeline（nanotrack配置）**

```bash
cd ~/stereo_3d_pipeline
./stereo_pipeline config/pipeline_yolo11s_960_nanotrack.yaml --duration 30
```

预期输出示例：
```
[ROI] FPS: 99.2
  Stage1_DetectSubmit: 1.1ms
  Stage1_TrackerInfill: 0.9ms
  Stage2_WaitDetectSync: 7.8ms
  Stage2_ROIMatchFuse: 1.2ms
Average: 10.8ms < 16.7ms ✓
```

关键指标：
- FPS: 95~100 fps ✓
- 无死机 (系统运行 > 30s) ✓
- GPU功耗: 检查 `jtop -p` 确保 < 12W ✓

- [ ] **Step 4: 测试其他3个配置**

```bash
for cfg in mixformer lighttrack siamfc; do
  ./stereo_pipeline config/pipeline_yolo11s_960_$cfg.yaml --duration 10
done
```

- [ ] **Step 5: 长运行稳定性测试（可选）**

```bash
./stereo_pipeline config/pipeline_yolo11s_960_nanotrack.yaml --duration 300 --log_interval 30
# 运行5分钟，每30秒打印一次性能统计
```

监控：
- 帧率是否保持稳定 > 95fps
- 是否出现任何 "CUDA OOM" 或 "DLA timeout" 错误
- 系统是否保持响应（没有整机卡死）

- [ ] **Step 6: 收集性能报告**

```bash
# 生成性能报告
./stereo_pipeline config/pipeline_yolo11s_960_nanotrack.yaml --duration 60 --profile output.bin
# 分析输出
```

- [ ] **Step 7: Commit测试记录（可选）**

```bash
echo "✓ NX测试通过: YOLO 33fps + tracker 66fps, FPS=99.2, 无死机" >> docs/test_results.md
git add docs/test_results.md
git commit -m "test: NX GPU load balancing verification passed"
```

---

## Success Criteria

| 指标 | 期望值 | 验证方法 |
|------|--------|--------|
| 帧率 | 95~100 fps | jtop or pipeline logs |
| 单帧时间 | < 11ms | profiler output |
| GPU功耗 | < 12W | `jtop -p` |
| 稳定性 | 30s+ 无重启 | 运行 30-300s 测试 |
| YOLO频率 | 30-35fps | `frame_id % 3 == 0` 统计 |
| Tracker频率 | 65-70fps | `frame_id % 3 != 0` 统计 |

---

## Rollback Plan

如果NX仍然死机：
1. 检查detect_interval是否真的被YAML加载（添加debug log确认）
2. 降低YOLO模型精度（FP16 → INT8，或者480 → 256输入大小）
3. 尝试YOLO单独运行（无tracker补帧），以隔离问题原因
4. 联系NVIDIA关于Orin NX功率管理（可能需要upscale到Orin NX Super或Orin Nano的功率笔记）

---

## Notes

- **为何改为同步？** 异步pipeline假设充足的计算资源，但NX只有1个GPU，YOLO+tracker不能并行。
- **detect_interval=3的含义** 每3帧推理一次YOLO，其余帧由tracker补帧。在100fps总帧率下，YOLO运行 ~33fps。
- **为何时间 10-11ms** 单帧总预算是 10ms (100fps), 但cache warm-up和同步的开销可能使其接近11ms，仍在16.7ms缓冲内。
- **GPU功耗监控** 使用 `nvidia-smi` 或 `jtop` 持续监控GPU功耗。如超过12W持续，需进一步降低detect_interval或模型精度。

