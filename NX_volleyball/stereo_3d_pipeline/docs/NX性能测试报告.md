# Jetson Orin NX 16GB — stereo_3d_pipeline 性能测试报告

> 测试日期: 2026-04-01  
> 硬件: Jetson Orin NX 16GB Super (SM 8.7, 8 SMs)  
> 软件: JetPack 6.1, CUDA 12.6, TensorRT 10.3.0, VPI 3.2.4, OpenCV 4.10  
> 模型: YOLOv11-26n (yolo26_fp16.engine, FP16, 640×640)  
> 相机: 海康 GigE 双目 (1440×1080, 自由采集 FreeRun)

---

## 1. TRT 单模型推理 Benchmark

使用 `trtexec` 官方工具, 200 iterations + 10s duration:

| 指标 | Mean | Median | P90 | P95 | P99 | Min | Max |
|---|---|---|---|---|---|---|---|
| **GPU Compute** | 3.71ms | 3.64ms | 3.69ms | 4.29ms | 5.29ms | 3.58ms | 6.83ms |
| **Host Latency** | 4.06ms | 3.98ms | 4.06ms | 4.63ms | 5.64ms | 3.83ms | 7.16ms |
| **H2D Transfer** | 0.33ms | 0.32ms | 0.35ms | 0.35ms | 0.41ms | — | — |
| **D2H Transfer** | 0.02ms | 0.02ms | 0.03ms | 0.03ms | 0.03ms | — | — |
| **Enqueue** | 2.34ms | 2.40ms | 2.64ms | 2.74ms | 3.17ms | 1.84ms | 4.98ms |

**理论推理吞吐**: 269 QPS (FPS)  
**GPU 计算效率**: 3.64ms median → 实际用时仅为设计预算(6-15ms)的 24-61%

---

## 2. 完整四级流水线 Benchmark

15 秒实际运行 (真实相机画面流入):

| Stage | 描述 | Avg (ms) | Min (ms) | Max (ms) | 帧数 |
|---|---|---|---|---|---|
| **Stage 0** | Grab + VPI Remap (CUDA) | **3.64** | 2.27 | 16.16 | 295 |
| **Stage 1** | TRT Detect (GPU FP16) | **5.48** | 3.84 | 94.74* | 294 |
| **Stage 2** | VPI Stereo Disparity | **0.02** | 0.01 | 0.10 | 294 |
| **Stage 3** | 3D Fusion + Output | **0.04** | 0.02 | 0.34 | 293 |

> *Stage 1 max 94.74ms 为首帧 warmup, 稳态下 ≤ 7ms

**实测吞吐**: ~19.7 FPS (295 帧 / 15 秒)

### 注意事项
- Stage 2 的 0.02ms 为 VPI 异步提交代价, 非实际 GPU 计算时间 (同步在下游 event wait 中)
- 实际 FPS 受限于 Stage 1 (TRT Detect) 为瓶颈环节 — 串行模式下约 ~5.5ms/帧
- 当前 pipeline 帧间流水线尚在单线程顺序执行, 并行 overlap 未完全发挥

---

## 3. 硬件利用率 (tegrastats)

113 个采样点 (200ms 间隔, ~22.6 秒):

| 资源 | 平均 | 最小 | 最大 |
|---|---|---|---|
| **GPU (GR3D)** | 60.9% | 0% | 99% |
| **CPU (8 核)** | 24.8% | 16% | 40% |
| **RAM** | 5918 MB | 5049 MB | 6540 MB |
| **总功耗** | 16.1 W | 8.2 W | 22.8 W |
| **EMC (内存带宽)** | 7-8% @ 2133 MHz | — | — |

### 硬件加速器状态
| 加速器 | 状态 | 说明 |
|---|---|---|
| GPU (Ampere) | **活跃 61%** | TRT 推理 + VPI Stereo + CUDA Remap |
| NVDLA0 | **OFF** | 当前 engine 未启用 DLA |
| NVDLA1 | **OFF** | 未使用 |
| PVA0 | **0%** | Remap 已改为 CUDA (PVA 不支持 Remap) |
| VIC | **OFF** | 未使用 |
| NVJPG | **OFF** | 不需要 |

### 温度
- CPU: ~63-65°C
- GPU: ~59-62°C
- TJ (Junction): ~65°C (安全范围, 热降频阈值 ~97°C)

---

## 4. 瓶颈分析

### 4.1 当前瓶颈: Stage 1 (TRT 推理) — 5.48ms

| 分析维度 | 详情 |
|---|---|
| **瓶颈 Stage** | Stage 1: TRT Detect (GPU FP16) |
| **平均耗时** | 5.48 ms/帧 |
| **理论上限** | 1000 / 5.48 ≈ **182 FPS** (仅推理) |
| **trtexec 裸推理** | 3.64 ms → 275 FPS |
| **差异原因** | Pipeline 中包含 preprocess (GPU resize+normalize) + postprocess (CPU NMS) |
| **GPU 利用率** | 61% — 有提升空间 (异步提交/DLA offload 可释放 GPU) |

### 4.2 Stage 0 (Grab + Remap) — 3.64ms

| 分析维度 | 详情 |
|---|---|
| **相机 Grab** | ~2-3ms (GigE 传输 → Host → VPI Host Lock memcpy) |
| **VPI Remap** | ~1ms (CUDA backend, 1280×720 双线性插值) |
| **优化方向** | 使用 CUDA Pinned Memory + cudaMemcpy2DAsync 替代 Host Lock |

### 4.3 实测 FPS vs 理论 FPS

| 模式 | FPS | 说明 |
|---|---|---|
| **trtexec 裸推理** | 269 | 无 preprocess, 无 postprocess, 无相机 |
| **Pipeline (无相机 dry-run)** | ~29.7 | 445帧/15s, 黑帧无目标 |
| **Pipeline (真实相机)** | ~19.7 | 295帧/15s, GigE bandwidth + memcpy |
| **设计目标** | 60-100 | 需要帧间流水线并行 overlap |

### 4.4 关键发现

1. **GPU 仅用 61%** — 说明 pipeline 序列化运行, GPU 空闲时在等 CPU/相机
2. **DLA/PVA 完全未使用** — pipeline 所有计算在 GPU 上, 硬件加速器闲置
3. **CPU 利用率 25%** — 相机 Grab/NMS 在 CPU 线程, 利用率低
4. **内存带宽 8%** — 远未饱和 (2133 MHz DDR5)
5. **功耗 16W** — NX 15W 模式下可能有功耗降频

---

## 5. 优化建议

### 优先级 P0: 启用帧间并行流水线
当前 pipeline loop 虽然设计了 3 级 overlap, 但 Stage 0→1→2→3 实际串行执行。
- **预期**: max(Stage_i) ≈ 5.5ms → **~180 FPS** (理论)
- **实现**: 确保 CUDA Events 异步同步正确, Stage 0/1/2 真正并行

### 优先级 P1: DLA Offload 推理
将 TRT 推理从 GPU 移到 DLA:
```bash
trtexec --onnx=yolo26.onnx --fp16 --useDLACore=0 --allowGPUFallback \
        --memPoolSize=workspace:4294967296 --saveEngine=yolo26_dla_fp16.engine
```
- DLA 处理推理 → GPU 同时做 VPI Stereo → 真正并行
- 预期 GPU 利用率降至 30-40%, 推理 offload 到 DLA

---

## 5.1 本轮已落地改造（2026-04-02）

### ✅ P0: 流水线并行化（代码已完成）

核心改造点：

1. `pipelineLoop` 调度重排为：
        - Stage3（上一帧融合）
        - Stage1/2（当前帧异步提交）
        - Stage0（下一帧抓取）
2. 移除 Stage1 内部同步阻塞：
        - `TRTDetector` 新增 `enqueue(slot)` + `collect(slot)` 两阶段接口
        - Stage1 仅提交推理与 D2H，不阻塞
        - Stage3 在等待 `evtDetectDone` 后回收检测结果
3. Stage2 同步下沉：
        - Stage2 只做异步提交
        - Stage3 统一等待 `vpiStreamGPU` 完成，避免在提交路径串行化
4. 新增性能项：
        - `Stage1_DetectSubmit`
        - `Stage2_StereoSubmit`
        - `Stage3_WaitDetect`
        - `Stage3_WaitStereo`

### ✅ P1: DLA 引擎构建（脚本已完成）

新增与更新：

- `scripts/build_engine.sh`
  - 支持 `gpu|dla` 模式
  - 统一 FP16
  - 使用 `--memPoolSize=workspace:4294967296`（规避 MiB 解析问题）
- `scripts/build_dla_engine.sh`
  - 一键构建 `yolo26_dla_fp16.engine`
- `config/pipeline_dla.yaml`
  - DLA 运行配置模板（`use_dla: true`）
- `scripts/pipeline_perf_compare.sh`
  - GPU vs DLA 端到端对比脚本

### 🔁 复测命令（NX）

1. 构建 DLA Engine
        - `./scripts/build_dla_engine.sh`
2. 对比测试
        - `./scripts/pipeline_perf_compare.sh`
3. 查看报告
        - `benchmark_results/pipeline_compare_*.md`

> 说明：由于本地工作区不直接连接 NX 硬件，本报告中的“已落地改造”指代码/脚本已完成；
> 最新 FPS、GPU/DLA 占用需在 NX 端按上述命令复测后回填。

### 优先级 P2: 降分辨率策略
- 当前: 640×640 推理 + 1280×720 视差 (全帧)
- 方案A: 320×320 推理 → ~1ms (7× faster)
- 方案B: 半分辨率视差 (640×360) → 节省 75% 计算量

### 优先级 P3: 零拷贝 Camera → GPU
- 使用 `cudaHostAlloc` pinned buffer 避免 Host Lock overhead
- 或使用 MMapped DMA buffer 直接从 GigE NIC → GPU

### 优先级 P4: 提升 GPU/EMC 频率
```bash
sudo jetson_clocks      # 锁定最高频率
sudo nvpmodel -m 0      # MAXN模式 (15W → 25W)
```
- 当前 GPU: 305 MHz (可达 ~900 MHz)
- 当前 EMC: 2133 MHz (7%)

---

## 6. 总结

| 指标 | 当前值 | 目标值 | 差距 |
|---|---|---|---|
| **吞吐量** | 19.7 FPS | 60-100 FPS | 3-5× |
| **GPU 利用率** | 61% | 80-90% | 需要并行 |
| **推理延迟** | 5.48 ms | ≤ 10 ms | ✅ 已达标 |
| **端到端延迟** | ~50 ms | ≤ 20 ms | 需要流水线 |
| **DLA 使用** | 0% | 50%+ | 需要 DLA engine |
| **PVA 使用** | 0% | 20%+ | VPI Remap 不支持 PVA |

**结论**: 单阶段性能优异 (推理 3.64ms), 但帧间流水线并行度不足导致整体吞吐量仅 ~20 FPS。
优先启用真正的流水线并行 + DLA offload 即可达到 60+ FPS 目标。
