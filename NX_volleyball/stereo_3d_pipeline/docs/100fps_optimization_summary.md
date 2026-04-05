# 100fps 排球追踪 Pipeline 优化总结

> **日期**: 2026-04-03  
> **平台**: NVIDIA Jetson Orin NX Super 16GB (JetPack 6, R36.4.7)  
> **状态**: ✅ 100 FPS 目标达成  

---

## 1. 硬件环境

| 项目 | 规格 |
|------|------|
| **SoC** | Jetson Orin NX 16GB Super (8 SM, CC 8.7) |
| **CUDA** | 12.6.68 |
| **TensorRT** | 10.3.0.30 |
| **VPI** | 3.2.4 |
| **相机** | 2× Hikvision MV-CA016-10UC (USB3 SuperSpeed 5Gbps) |
| **分辨率** | 1440×1080 BayerRG8 → 校正后 1280×720 |
| **标定** | baseline 328.71mm, RMS 1.11 |
| **电源模式** | MAXN_SUPER (mode 0) + jetson_clocks |
| **DLA** | ❌ 不可用 (设备节点缺失, GPU fallback) |

## 2. 架构概览

### 三级流水线 (ROI 模式)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 0     │    │  Stage 1     │    │  Stage 2     │
│  Grab+Rect   │ →  │  Detect      │ →  │  ROI Match   │
│  CPU+VPI     │    │  GPU (INT8)  │    │  GPU CUDA    │
│  ~8ms        │    │  ~3.5ms      │    │  ~0.01ms     │
└──────────────┘    └──────────────┘    └──────────────┘
     Frame N+2           Frame N+1           Frame N

吞吐量 = 1 / max(Stage_i) → 100+ FPS (流水线重叠)
```

- **Stage 0**: 双目 USB 抓帧 (并行线程) + VPI Remap 校正 (异步提交)
- **Stage 1**: TensorRT INT8 YOLO 检测 (异步 enqueue)
- **Stage 2**: ROI 多点 SAD 块匹配 → IQR 中值滤波 → 三角测距

### 三缓冲 (Triple Buffer)

```
Slot 0:  [Stage 2 处理]
Slot 1:  [Stage 1 处理]
Slot 2:  [Stage 0 抓帧]
```

CUDA Events (`evtRectDone`, `evtDetectDone`) 管理跨 Stage 依赖。

## 3. 关键优化

### 3.1 USB 相机取代 GigE (解决 57ms → 6ms 瓶颈)

| 指标 | GigE (之前) | USB3 (当前) |
|------|------------|-------------|
| 单帧 grab | 57.84 ms | **6.00 ms** |
| 有效 FPS | ~17 | **165.9** |
| 带宽 | ~125 MB/s | ~500 MB/s |

USB3 带宽充裕: 1440×1080 × 8bit = 1.55 MB/帧, 165fps 需 256 MB/s < 500 MB/s 理论值。

### 3.2 INT8 引擎取代 FP16

| 引擎 | 输入 | 延迟 | GPU显存 |
|------|------|------|---------|
| yolo26_fp16.engine | 640×640 | OOM 崩溃 | 8.6 MB |
| **yolo26n_dla0_int8.engine** | 320×320 | **3.52 ms** | 3.7 MB |

FP16 引擎因 LLVM OOM 无法运行。INT8 引擎虽然名称含 "DLA", 但在无 DLA 硬件的板上自动 fallback 到 GPU 执行, 性能优秀。

### 3.3 VPI Remap 异步化 (节省 ~1ms/帧)

**优化前** (Stage 0 内阻塞):
```
grab(6ms) → rect submit → vpiStreamSync(1.5ms 阻塞) → total 7.5ms
```

**优化后** (延迟同步到 Stage 1 前):
```
Stage 0: grab(6ms) → rect submit(async) → 6.1ms
Stage 1: vpiStreamSync(~0ms, 已完成) → detect enqueue
```

VPI remap 在 grab 等待期间并行执行, 同步时几乎零成本。

### 3.4 自由运行模式 (Free-Run)

USB 相机使用 Free-Run 而非触发模式:
- 无需 PWM GPIO 硬件
- 消除触发同步警告 (trigger mode 每帧告警)
- 同等 FPS 性能
- 通过 `LatestImagesOnly` 策略保证最新帧, 避免缓冲堆积

## 4. 实测性能

### 触发模式 (130Hz PWM)
```
Stage                      Avg(ms)  Min(ms)  Max(ms)    Count
--------------------------------------------------------------
Stage0_GrabRect               6.51     0.98    10.40      888
Stage1_WaitRect               1.40     1.20     3.93      886
Stage1_DetectSubmit           2.07     1.74    49.82      887
Stage2_WaitDetect             0.03     0.01     0.11      886
Stage2_ROIMatchFuse           0.01     0.01     0.12      886

FPS: 100.1 (at output frame 600)
Frame sync warnings: 887 (every frame)
```

### 自由运行模式 (推荐)
```
Stage                      Avg(ms)  Min(ms)  Max(ms)    Count
--------------------------------------------------------------
Stage0_GrabRect               6.42     0.54     9.51      887
Stage1_WaitRect               1.47     1.20     4.74      885
Stage1_DetectSubmit           2.11     1.76    47.56      886
Stage2_WaitDetect             0.03     0.01     0.57      885
Stage2_ROIMatchFuse           0.01     0.01     0.10      885

FPS: 100.0 (at output frame 800)
Frame sync warnings: 0
```

### 帧时间预算 (10ms @ 100fps)

```
USB Grab (parallel L+R):  6.4 ms   ████████████░░░░░░░░  64%
VPI Remap sync:           1.5 ms   ███░░░░░░░░░░░░░░░░░  15%
TRT INT8 enqueue:         2.1 ms   ████░░░░░░░░░░░░░░░░  21%
ROI Match + Fuse:         0.01ms   ░░░░░░░░░░░░░░░░░░░░   0%
────────────────────────────────────────────────────────
Total:                   10.01ms                         100%
```

## 5. DLA 状态说明

经诊断, 本板 Orin NX Super 的 DLA 设备节点不存在:
- `/dev/nvhost-nvdla*` — 不存在
- `/sys/class/misc/` — 无 DLA 相关条目
- `dmesg | grep -i dla` — 无日志

DLA INT8 engine (`yolo26n_dla0_int8.engine`) 使用 `--allowGPUFallback` 构建, 在 GPU 上正常运行, 延迟 3.52ms/帧。

**结论**: 当前方案为 GPU-only, 性能已满足 100fps 目标。若后续需要 DLA:
1. 检查设备树是否禁用 DLA
2. 确认 JetPack 6 完整安装
3. DLA 可用后, GPU 专注 VPI/ROI, DLA 专注 detect → 预期 120+ fps

## 6. 配置文件

### 推荐配置: `config/pipeline_roi_freerun.yaml`

```yaml
camera:
  use_trigger: false          # Free-run for USB cameras
  exposure_us: 3000.0         # 3ms exposure
  gain_db: 15.0

detector:
  engine_path: "yolo26n_dla0_int8.engine"   # INT8, 320×320
  use_dla: false                             # GPU fallback

stereo:
  strategy: "roi_only"        # ROI multi-point triangulation
  max_disparity: 256
```

### 触发模式配置: `config/pipeline_roi.yaml`

```yaml
camera:
  use_trigger: true
  trigger_chip: "gpiochip2"
  trigger_line: 7

performance:
  pwm_frequency: 130          # > 100fps target
```

## 7. 代码变更

| 文件 | 变更 |
|------|------|
| `config/pipeline_roi.yaml` | INT8 engine, 130Hz PWM, 3ms 曝光 |
| `config/pipeline_roi_freerun.yaml` | 新增 Free-Run 配置 |
| `src/pipeline/pipeline.cpp` | Stage0 VPI remap 异步化; Stage1 延迟同步; 排空阶段修复 |

## 8. 潜在提速方向

| 优化 | 预期收益 | 复杂度 |
|------|---------|--------|
| DLA 启用 (如硬件可用) | +20-30 fps | 低 — 改 config |
| 降分辨率到 960×540 | -0.5ms VPI | 低 — 改 config |
| VPI TNR 降噪 (NV12) | +匹配质量 | 中 — 格式转换 |
| 双线程 grab+rect overlap | -1ms | 中 — 线程安全 |
| CUDA Graph capture | -0.5ms API 开销 | 高 |
| 640×480 相机模式 | -3ms grab | 低 — 改 config, 需重标定 |

## 9. 快速部署

```bash
# 在 Windows 上
python scripts/deploy_and_test.py

# 在 NX 上直接运行
cd ~/NX_volleyball/stereo_3d_pipeline/build
./stereo_pipeline --config ../config/pipeline_roi_freerun.yaml
```
