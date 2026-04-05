# DLA 综合分析报告

## Jetson Orin NX Super — YOLO 排球检测 DLA 部署

**平台**: NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super (16GB)  
**JetPack**: R36.4.7 (JetPack 6)  
**TensorRT**: 10.3.0.30  
**CUDA**: 12.6.68  
**DLA**: 2× NVDLA v2.0  
**功耗模式**: MAXN_SUPER (Mode 0)  
**日期**: 2026-04-04  

---

## 1. 模型分析

### 1.1 可用 ONNX 模型

| 模型 | 文件 | 输入形状 | 输出形状 | 后处理 | TopK 节点 |
|------|------|---------|---------|--------|----------|
| YOLO26n (Nano) | yolo26n.onnx (9.2MB) | [1,3,320,320] | [1,300,6] | Post-NMS | 2× TopK |
| YOLO26 (Full) | yolo26.onnx (9.3MB) | [1,3,640,640] | [1,5,8400] | Pre-NMS | 无 |

> **关键发现**: yolo26n 的 TopK 节点不完全兼容 DLA。DLA 环境下 TopK 需要 GPU 回退 (fallback)。yolo26 无 TopK，理论上更适合纯 DLA 部署，但 640×640 分辨率导致 DLA 性能极差。

### 1.2 DLA 精度支持

| 精度模式 | 是否支持 | 说明 |
|---------|---------|------|
| **FP16** | ✅ 支持 | NVDLA v2.0 原生支持 FP16 |
| **INT8** | ✅ 支持 | 需要校准数据 (或 `--best` 模式时 TRT 自动使用 FP16 回退) |
| FP32 | ❌ 不支持 | DLA 不支持 FP32，需全部转为 FP16/INT8 |

> **结论**: DLA **不限 INT8**，可同时使用 FP16 和 INT8。

---

## 2. 引擎构建结果

### 2.1 320×320 引擎 (yolo26n.onnx)

| 引擎 | 精度 | 后端 | 大小 | 状态 | 备注 |
|------|------|------|------|------|------|
| yolo26n_gpu_fp16_320 | FP16 | GPU | 7.0MB | ✅ | |
| yolo26n_gpu_int8_320 | INT8 | GPU | 4.3MB | ✅ | |
| yolo26n_dla0_int8_320 | INT8 | DLA0 | 3.7MB | ✅ | |
| yolo26n_dla1_int8_320 | INT8 | DLA1 | 3.7MB | ✅ | |
| yolo26n_dla0_fp16_320 | FP16 | DLA0 | 6.7MB | ✅ | **需 8GB workspace** |
| yolo26n_dla1_fp16_320 | FP16 | DLA1 | 6.7MB | ✅ | **需 8GB workspace** |

> DLA FP16 320 构建需要 8GB 工作空间 (`--memPoolSize=workspace:8589934592` bytes 格式)，因 TopK 注意力层的 GPU 回退需要大量临时内存。4GB 不足会导致构建失败。

### 2.2 640×640 引擎 (yolo26.onnx)

| 引擎 | 精度 | 后端 | 大小 | 状态 | 备注 |
|------|------|------|------|------|------|
| yolo26_gpu_fp16_640 | FP16 | GPU | 7.9MB | ✅ | |
| yolo26_gpu_int8_640 | INT8 | GPU | 4.5MB | ✅ | 构建耗时 ~11min |
| yolo26_dla0_fp16_640 | FP16 | DLA0 | 6.8MB | ✅ | |
| yolo26_dla1_fp16_640 | FP16 | DLA1 | 6.8MB | ✅ | |
| yolo26_dla0_int8_640 | INT8 | DLA0 | 3.8MB | ✅ | |
| yolo26_dla1_int8_640 | INT8 | DLA1 | 3.7MB | ✅ | |
| yolo26_dla0_standalone_fp16 | FP16 | DLA0 | — | ❌ | I/O 格式限制 |
| yolo26_dla0_standalone_int8 | INT8 | DLA0 | — | ❌ | I/O 格式限制 |

---

## 3. trtexec 独立延迟基准

### 3.1 320×320 引擎

| 引擎 | p99 延迟 | 吞吐量 | vs GPU FP16 |
|------|---------|--------|------------|
| **GPU FP16** | 1.82ms | 549 qps | 1.00× |
| **GPU INT8** | 1.61ms | 625 qps | 1.14× ↑ |
| **DLA0 INT8** | 2.61ms | 368 qps | 0.67× |
| DLA1 INT8 | 3.25ms | 309 qps | 0.56× |
| **DLA0 FP16** | 6.84ms | 145 qps | 0.26× |
| DLA1 FP16 | 6.58ms | 149 qps | 0.27× |
| DLA0 INT8 (旧) | 3.50ms | 278 qps | 0.51× |
| DLA1 INT8 (旧) | 3.44ms | 281 qps | 0.51× |

### 3.2 640×640 引擎

| 引擎 | p99 延迟 | 吞吐量 | vs GPU FP16 |
|------|---------|--------|------------|
| **GPU FP16** | 3.80ms | 276 qps | 1.00× |
| **GPU INT8** | 3.09ms | 339 qps | 1.23× ↑ |
| **DLA0 FP16** | 28.20ms | 35 qps | 0.13× |
| DLA1 FP16 | 28.69ms | 35 qps | 0.13× |
| **DLA0 INT8** | 16.27ms | 62 qps | 0.23× |
| DLA1 INT8 | 16.36ms | 62 qps | 0.23× |

### 3.3 DLA vs GPU 性能比

| 模型 | 精度 | DLA p99 | GPU p99 | DLA/GPU 比 |
|------|------|---------|---------|-----------|
| 320×320 | INT8 | 2.61ms | 1.61ms | 1.62× (DLA 慢 62%) |
| 320×320 | FP16 | 6.84ms | 1.82ms | 3.76× (DLA 慢 276%) |
| 640×640 | INT8 | 16.27ms | 3.09ms | 5.26× (DLA 慢 426%) |
| 640×640 | FP16 | 28.20ms | 3.80ms | 7.42× (DLA 慢 642%) |

> **DLA 延迟随分辨率和精度显著恶化。** 320 INT8 是 DLA 的最佳工作点 (仅比 GPU 慢 62%)。640 FP16 时 DLA 慢 7.4 倍，完全不可用。

---

## 4. 端到端流水线性能 (实测)

### 4.1 测试条件
- 双目海康相机 1440×1080 BayerRG8 @ 100Hz PWM 触发
- VPI TNR 降噪 → VPI PVA 校正 1280×720 → YOLO 检测 → ROI 多点立体匹配
- 3 级流水线: Grab+Rectify(Stage0) | Detect(Stage1) | ROI Match+3D(Stage2)
- 15 秒运行取平均 (含引擎加载 ~2-3s)

### 4.2 完整性能对比

| 配置 | FPS | WaitDetect (ms) | DetectSubmit (ms) | WaitRect (ms) | 帧数/时长 |
|------|-----|----------------|-------------------|-------------|----------|
| **GPU INT8 640** | **88** | 0.18 | 3.42 | 5.37 | 1321/15s |
| **GPU FP16 640** | **83** | 0.34 | 3.55 | 5.33 | 828/10s |
| **GPU FP16 320** | **83** | 0.08 | 3.24 | 5.81 | 827/10s |
| **DLA0 INT8 320** | **83** | 0.25 | 2.93 | 6.19 | 830/10s |
| Dual DLA INT8 320 | 80 | 0.31 | 2.76 | 6.65 | 797/10s |
| DLA0 FP16 320 | 53 | 3.25 | 3.29 | 9.88 | 797/15s |
| DLA0 INT8 640 | 34 | 13.05 | 2.59 | 10.62 | 504/15s |
| Dual DLA FP16 640 | 22 | 25.60 | 2.89 | 12.20 | 324/15s |
| DLA0 FP16 640 | 20 | 26.45 | 2.57 | 13.03 | 204/10s |

### 4.3 瓶颈分析

```
100Hz 相机帧率限制: 10ms/帧

GPU INT8 640:  检测 ~3.6ms  ← 远低于 10ms → 相机瓶颈 → 88 FPS ✅ 最佳
GPU FP16 640:  检测 ~3.9ms  ← 远低于 10ms → 相机瓶颈 → 83 FPS ✅
DLA INT8 320:  检测 ~3.2ms  ← 远低于 10ms → 相机瓶颈 → 83 FPS ✅
DLA FP16 320:  检测 ~6.5ms  ← 低于 10ms 但GPU回退争抢 → GPU瓶颈 → 53 FPS ⚠️
DLA INT8 640:  检测 ~16ms   ← 超过 10ms  → 检测瓶颈 → 34 FPS ❌
DLA FP16 640:  检测 ~29ms   ← 远超 10ms  → 检测瓶颈 → 20 FPS ❌
```

### 4.4 WaitRect 异常分析

DLA 配置下 WaitRect 显著高于 GPU 配置:
- GPU 方案: WaitRect ≈ 5.3-5.8ms
- DLA 方案: WaitRect ≈ 6.2-13.0ms

原因: DLA 的 GPU fallback 层 (MatMul/TopK) 与 VPI 校正/TNR 争抢 GPU 资源，导致 PVA 校正流水被拖慢。

---

## 5. 双 DLA 并行分析

### 5.1 实现架构

```
Frame 0 → DLA0 (cudaStreamDLA)   → evtDetectDone → ROI Match
Frame 1 → DLA1 (cudaStreamDLA1)  → evtDetectDone → ROI Match
Frame 2 → DLA0 (cudaStreamDLA)   → evtDetectDone → ROI Match
...
```

每帧根据 `frame_id % 2` 选择 DLA0 或 DLA1，使用独立 CUDA Stream。

### 5.2 trtexec 双 DLA 并行测试 (理想场景)

| 模式 | 单 DLA qps | 双 DLA 合计 qps | 提升 |
|------|-----------|----------------|------|
| INT8 640 | 62 | 77.7 | +25% |
| INT8 320 | 368 | 541 | +47% |
| FP16 640 | 35 | 39.6 | +13% |

> trtexec 双进程完全独立运行时，INT8 320 接近 1.47× 线性加速。但 FP16 640 因 GPU fallback 竞争仅 +13%。

### 5.3 流水线双 DLA 实测

| 模式 | 单 DLA FPS | 双 DLA FPS | 提升 |
|------|-----------|-----------|------|
| INT8 320 | 83 | 80 | **-3.6%** ↓ |
| FP16 640 | 20 | 22 | **+10%** ↑ |

### 5.4 双 DLA 效果低于预期的原因

**流水线结构限制**: 当前 3 级流水线中，Stage 2 (collect) 在 Stage 1 (submit) 之前执行：

```
每次循环:
  1. Stage 2: 等待上一帧 DLA 完成 → collect() 阻塞
  2. Stage 1: 等待校正完成 → submit() 提交到 DLA
  3. Stage 0: 采集下一帧
```

Frame N+1 的 DLA 推理只在 Frame N-1 的结果 collect 完成后才提交。这导致 **DLA0 和 DLA1 无法真正并行**，两者交替执行而非重叠。

**INT8 320 (-3.6%)**: 检测 ~3ms 远低于 10ms 相机间隔，本就不是瓶颈。双 DLA 只增加上下文切换开销。

**FP16 640 (+10%)**: 检测 ~29ms 是严重瓶颈。双 DLA 小幅降低 GPU fallback 竞争，但两个 DLA 的 GPU fallback 层仍然竞争同一 GPU。

### 5.5 优化方向

要实现真正双 DLA 并行，需要调整流水线循环顺序为 **submit → collect → grab**:
```
每次循环:
  1. Stage 1: submit() 提交当前帧到 DLA  ← 先提交
  2. Stage 2: collect() 获取前一帧结果    ← 此时新老帧的 DLA 并行
  3. Stage 0: 采集下一帧
```

预计 DLA 640 场景可提升至 ~35-40 FPS (接近 trtexec 双并行的 39.6 qps)。

---

## 6. DLA 层分布与兼容性

### 6.1 DLA 子图分析

通过 `--dumpLayerInfo` 分析引擎层执行设备:

**yolo26n (320×320) DLA INT8 — 5 个 DLA 子图 + GPU 回退:**

| DLA 子图 | 覆盖层范围 |
|----------|-----------|
| ForeignNode[1] | `/model.0/conv/Conv` → `/model.10/m/m.0/attn/Transpose` (主干网络) |
| ForeignNode[2] | `/model.10/m/m.0/attn/Split_18` → `+Transpose_1` (注意力分支) |
| ForeignNode[3] | `/model.10/Split` → `/model.22/m.0/m.0.1/attn/Transpose` (中间层) |
| ForeignNode[4] | `/model.22/m.0/m.0.1/attn/Split_44` → `+Transpose_1` (注意力分支) |
| ForeignNode[5] | `/model.23/one2one_cv2.0` → `one2one_cv2.2.2/Conv` (检测头+NMS) |

**GPU 回退层:**
- `MatMul` × 2 (注意力矩阵乘法) — 每个注意力模块 2 次
- `Reformatting CopyNode` × 多次 (DLA↔GPU 格式转换)

**yolo26 (640×640) DLA INT8 — 5 个 DLA 子图 + GPU 回退:**

| DLA 子图 | 覆盖层范围 |
|----------|-----------|
| ForeignNode[1] | `/model.0/conv/Conv` → `/model.10/m/m.0/attn/Transpose` |
| ForeignNode[2] | `/model.10/m/m.0/attn/Split_18` → `+Transpose_1` |
| ForeignNode[3] | `/model.10/Split` → `/model.22/m.0/m.0.1/attn/Transpose` |
| ForeignNode[4] | `/model.22/m.0/m.0.1/attn/Split_40` → `+Transpose_1` |
| ForeignNode[5] | `/model.23/cv2.0` → `/model.23/Concat_2` (检测头) |

**GPU 回退层 (640 额外):**
- `MatMul` × 4 (注意力矩阵乘法，640 特征图更大 → 耗时更久)
- `Reshape` × 3 (输出维度变换)
- `Sigmoid`, `Mul` (后处理激活)
- `Reformatting CopyNode` × ~20 (大量 DLA↔GPU 数据搬运)

> **关键洞察**: 两个模型都有相似的注意力 (Attention) 模块导致 DLA→GPU→DLA "乒乓"执行。但 640 模型的特征图为 320 的 4 倍面积，MatMul 的 GPU fallback 成本呈二次增长，加上更多的 Reformatting CopyNode 数据搬运，这解释了 640 DLA 性能急剧下降。

### 6.2 DLA Standalone Loadable 测试

| 模型 | 精度 | 状态 | 失败原因 |
|------|------|------|---------|
| yolo26.onnx | FP16 | ❌ 失败 | I/O 格式限制 |
| yolo26.onnx | INT8 | ❌ 失败 | I/O 格式限制 |

**错误信息**: `I/O formats for safe DLA capability are restricted to fp16/int8:dla_linear, fp16/int8:hwc4, fp16:chw16 or int8:chw32`

**分析**: DLA Standalone 要求所有层都运行在 DLA 上，不允许 GPU fallback。由于注意力模块的 MatMul 运算不被 DLA 原生支持，且模型 I/O 格式不符合 DLA 安全格式要求，Standalone 构建失败。

**要使 Standalone 成功**，需要:
1. 移除或替换模型中的 Attention/MatMul 层
2. 设置输入输出为 DLA 兼容格式: `--inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16`
3. 对于此 YOLO 模型，这意味着需要修改模型架构

---

## 7. Letterbox 预处理与坐标还原

### 7.1 GPU CUDA Letterbox 实现

已实现等比缩放 + 灰色填充的 GPU letterbox 预处理内核:

```
输入: 1280×720 灰度 (校正后)
输出: 320×320 或 640×640 RGB (归一化到 [0,1])
填充色: 114/255 ≈ 0.447 (YOLO 标准灰色)

缩放比: scale = min(modelW/rectW, modelH/rectH)
        = min(320/1280, 320/720) = 0.25 (320模型)
        = min(640/1280, 640/720) = 0.5  (640模型)
```

### 7.2 检测框到原始图像坐标还原

```cpp
// TRTDetector::collect() 中自动还原
float invScale = 1.0f / scale;
float origX = (detectX - padX) * invScale;  // 还原到 1280×720 坐标
float origY = (detectY - padY) * invScale;
```

对于 640 模型 (scale=0.5):
- 检测框精度: 0.5 像素 (在 1280×720 空间中 = 1 原始像素)
- 亚像素还原: 除以 scale=0.5 等价于 ×2，保持小数精度

对于 320 模型 (scale=0.25):
- 检测框精度: 0.25 像素 → 还原后 1 原始像素
- 640 模型的空间分辨率是 320 的 2×

---

## 8. 最优配置建议

### 8.1 推荐配置矩阵

| 场景 | 推荐配置 | FPS | 理由 |
|------|---------|-----|------|
| **高精度优先** | GPU INT8 640 | 88 | 最高分辨率 + 最高帧率 |
| **均衡首选** | GPU FP16 640 | 83 | 高分辨率，无 INT8 精度风险 |
| **DLA 卸载 GPU** | DLA0 INT8 320 | 83 | 释放 GPU 给立体视觉 |
| **最低功耗** | DLA0 INT8 320 | 83 | DLA 功耗远低于 GPU |
| ❌ 不推荐 | DLA FP16 任何 | ≤53 | GPU fallback 争抢导致低效 |
| ❌ 不推荐 | DLA × 640 | ≤34 | 大特征图 DLA/GPU 乒乓严重 |
| ❌ 不推荐 | 当前双 DLA | ≤80 | 流水线结构限制并行度 |

### 8.2 生产部署建议

**推荐方案: DLA0 INT8 320 + GPU 立体视觉**
- 检测 (DLA): ~3ms，远低于 10ms 相机间隔
- GPU 完全释放给: VPI TNR、PVA 校正、立体匹配
- 功耗优化: DLA 运算不计入 GPU 功耗预算
- 配置文件: `config/pipeline_roi.yaml`

**替代方案: GPU INT8 640** (需要高精检测时)
- 配置文件: `config/pipeline_roi_gpu_int8_640.yaml`

### 8.3 未来优化路线

| 优化 | 预期效果 | 复杂度 |
|------|---------|--------|
| 调整流水线顺序实现真双DLA并行 | DLA 640 INT8: 34→50+ FPS | 中 |
| 替换 Attention 为 DLA 友好操作 | DLA 延迟 -30~50% | 高 (需重训模型) |
| DLA INT8 校准 (用真实排球数据) | 检测精度提升 | 低 |
| 200Hz+ 相机升级 | 单 DLA INT8 可到 ~310 FPS | 硬件 |

---

## 9. 详细数据附录

### 9.1 完整 trtexec Benchmark v3 输出

```
=================================================================
 DLA Comprehensive Benchmark v3 - Sat Apr  4 05:59:43 AM UTC 2026
 Platform: NVIDIA Jetson Orin NX Super (16GB)
 JetPack: R36.4.7, TensorRT 10.3.0, CUDA 12.6
 Mode: MAXN_SUPER
=================================================================

===== PHASE 1: BUILD ENGINES =====
-- 320x320 --
GPU_FP16_320 (7.0M)  OK
GPU_INT8_320 (4.3M)  OK
DLA0_INT8_320 (3.7M) OK
DLA1_INT8_320 (3.7M) OK
DLA0_FP16_320 (6.7M) OK (8GB workspace)
DLA1_FP16_320 (6.7M) OK (8GB workspace)
-- 640x640 --
GPU_FP16_640 (7.9M)  OK
GPU_INT8_640 (4.5M)  OK (~11min build)
DLA0_FP16_640 (6.8M) OK
DLA1_FP16_640 (6.8M) OK
DLA0_INT8_640 (3.8M) OK
DLA1_INT8_640 (3.7M) OK

===== PHASE 2: BENCHMARK (300 iterations) =====
-- 320x320 --
[GPU_FP16_320]      p99=1.82ms  549 qps
[GPU_INT8_320]      p99=1.61ms  625 qps
[DLA0_INT8_320]     p99=2.61ms  368 qps
[DLA1_INT8_320]     p99=3.25ms  309 qps
[DLA0_FP16_320]     p99=6.84ms  145 qps
[DLA1_FP16_320]     p99=6.58ms  149 qps
[DLA0_INT8_320_old] p99=3.50ms  278 qps
[DLA1_INT8_320_old] p99=3.44ms  281 qps
-- 640x640 --
[GPU_FP16_640]      p99=3.80ms  276 qps
[GPU_INT8_640]      p99=3.09ms  339 qps
[DLA0_FP16_640]     p99=28.20ms  35 qps
[DLA1_FP16_640]     p99=28.69ms  35 qps
[DLA0_INT8_640]     p99=16.27ms  62 qps
[DLA1_INT8_640]     p99=16.36ms  62 qps

===== PHASE 4: DLA STANDALONE =====
yolo26.onnx FP16: FAILED (I/O format restriction)
yolo26.onnx INT8: FAILED (I/O format restriction)

===== PHASE 5: DUAL DLA PARALLEL (trtexec) =====
INT8 640: DLA0 38.9qps + DLA1 38.8qps = 77.7qps combined
INT8 320: DLA0 270qps + DLA1 271qps = 541qps combined
FP16 640: DLA0 19.8qps + DLA1 19.8qps = 39.6qps combined
```

### 9.2 引擎文件清单

```
/home/nvidia/NX_volleyball/model/
├── yolo26n.onnx                      (9.2MB) 320×320 Post-NMS
├── yolo26.onnx                       (9.3MB) 640×640 Pre-NMS
├── yolo26n_gpu_fp16_320.engine       (7.0MB)
├── yolo26n_gpu_int8_320.engine       (4.3MB)
├── yolo26n_dla0_int8.engine          (3.7MB) 原版
├── yolo26n_dla1_int8.engine          (3.7MB) 原版
├── yolo26n_dla0_int8_320.engine      (3.7MB) benchmark 版
├── yolo26n_dla1_int8_320.engine      (3.7MB) benchmark 版
├── yolo26n_dla0_fp16_320.engine      (6.7MB)
├── yolo26n_dla1_fp16_320.engine      (6.7MB)
├── yolo26_gpu_fp16_640.engine        (7.9MB)
├── yolo26_gpu_int8_640.engine        (4.5MB)
├── yolo26_dla0_fp16_640.engine       (6.8MB)
├── yolo26_dla1_fp16_640.engine       (6.8MB)
├── yolo26_dla0_int8_640.engine       (3.8MB)
└── yolo26_dla1_int8_640.engine       (3.7MB)
```

### 9.3 流水线配置文件清单

```
config/
├── pipeline_roi.yaml                 DLA0 INT8 320 (生产推荐)
├── pipeline_roi_dual_dla.yaml        双DLA INT8 320
├── pipeline_roi_dual_dla_640.yaml    双DLA FP16 640
├── pipeline_roi_640.yaml             DLA0 FP16 640
├── pipeline_roi_gpu640.yaml          GPU FP16 640
├── pipeline_roi_gpu320.yaml          GPU FP16 320
├── pipeline_roi_gpu_int8_640.yaml    GPU INT8 640
├── pipeline_roi_dla_fp16_320.yaml    DLA0 FP16 320
└── pipeline_roi_dla_int8_640.yaml    DLA0 INT8 640
```

---

## 10. 待完成项

- [ ] INT8 精度验证 (需放置排球目标)
- [ ] DLA INT8 校准数据集制作
- [ ] 流水线 submit→collect 顺序优化 (真双 DLA 并行)
- [ ] 模型 Attention 层 DLA 适配研究

---

*报告基于 Jetson Orin NX Super 实测数据，2026-04-04。*

### 5.2 未来扩展 (更高帧率相机)

若升级到 200Hz+ 相机，检测将成为瓶颈：
- **GPU FP16 640**: ~3.9ms → 理论极限 ~256 FPS → 足够
- **DLA INT8 320**: ~3.2ms → 理论极限 ~312 FPS → 足够
- **双 DLA INT8 320**: ~1.6ms → 理论极限 ~625 FPS → 超额

此时 DLA 的价值在于**卸载 GPU**，让 GPU 处理立体视觉或其他计算密集任务。

---

## 6. DLA 层兼容性

### 6.1 yolo26n.onnx (320×320)
- 包含 **TopK** (2×) 和 **Gather** (1×) 节点
- TopK 不被 DLA 原生支持，需 GPU 回退
- INT8 模式：大部分层在 DLA 上执行，TopK/NMS 回退到 GPU
- 不可构建 DLA Standalone Loadable (需手动去除 NMS 后处理)

### 6.2 yolo26.onnx (640×640)
- **无 TopK** 节点 (Pre-NMS 输出)
- 理论上可构建 DLA Standalone Loadable
- 但 640×640 下 DLA 吞吐量极低 (20 FPS)，实用价值有限
- (待 benchmark 完成后补充 Standalone 构建结果)

---

## 7. Letterbox 预处理

### 7.1 实现
已实现 GPU CUDA letterbox 预处理内核 (`grayToRGBLetterboxKernel`):
- 等比缩放 + 灰色 (114/255) 填充
- 支持 1280×720 灰度输入 → 320×320 或 640×640 RGB 输出
- 坐标还原: `(x - padX) * invScale`, `(y - padY) * invScale`

### 7.2 坐标还原公式
```
scale = min(modelW/rectW, modelH/rectH)
newW = round(rectW * scale), newH = round(rectH * scale)
padX = (modelW - newW) / 2, padY = (modelH - newH) / 2

还原: 原始 x = (检测 x - padX) / scale
      原始 y = (检测 y - padY) / scale
```

---

## 8. 待完成项

- [ ] 640 DLA INT8 引擎构建 + 测试 (benchmark 进行中)
- [ ] 双 DLA 640 FP16 测试 (等 DLA1 FP16 640 引擎)
- [ ] DLA Standalone Loadable 适用性测试 (yolo26.onnx)
- [ ] trtexec 独立延迟基准 (benchmark Phase 2)
- [ ] DLA 层分布详细报告 (benchmark Phase 3)
- [ ] INT8 精度验证 (需放置排球目标)

---

*注: 本报告基于实际 NX 设备测试数据。640 相关结果将在 benchmark 完成后更新。*
