# stereo_3d_pipeline — 实时双目排球追踪

Jetson Orin NX Super 16GB 双目深度管线。**实测 98 FPS** (100Hz PWM 触发)。

> **当前状态**: 生产就绪 — ROI 模式 + DLA INT8 + VPI TNR，2026-04-03 验证通过。

## 架构

ROI 模式 3 阶段流水线，3 帧重叠并行：

| Stage | 硬件    | 功能                          | 实测耗时 |
|-------|---------|-------------------------------|----------|
| 0     | CUDA/VIC | 海康 USB 抓帧 + VPI TNR 降噪 + VPI Remap 校正 | 0.69 ms (抓帧) + 0.04 ms (TNR) |
| 1     | CUDA+DLA | VPI Remap 等待 + DLA/GPU YOLO 检测提交 | 7.63 ms (Remap 等待) + 2.90 ms (检测提交) |
| 2     | CUDA    | DLA 检测等待 + ROI 多点匹配 + 3D 测距 | 0.12 ms + 0.01 ms |

DLA 检测与 VPI Remap 完美并行 — 检测等待仅 0.12ms。

### 关键技术

- **VPI TNR (时域降噪)**: NV12 格式 CUDA 后端, 0.04ms/帧对, 几乎零开销
- **VPI Remap (畸变校正/LDC)**: OpenCV 生成 undistort+rectify 映射表 → VPI WarpMap → CUDA/VIC Remap（可配置）
- **DLA INT8 检测**: yolo26n 640×640, 3.37ms/帧, 313 qps, 释放 GPU
- **ROI 多点匹配**: 检测后仅在目标区域做 CUDA 立体匹配, 0.01ms (vs 全帧 70ms)
- **PWM 硬件触发**: libgpiod 100Hz 精确同步双目

## TensorRT 引擎对比

| 引擎 | 精度 | 设备 | 延迟 | 吞吐 | 用途 |
|------|------|------|------|------|------|
| yolo26n_dla0_int8_640 | INT8 | **DLA0** | **3.37ms** | 313 qps | ✅ 生产配置 |
| yolo26_fp16 | FP16 | GPU | 4.10ms | 265 qps | 备选 (无需 DLA) |
| yolo26_dla_fp16 | FP16 | DLA0 | 27ms | 37 qps | ❌ 太慢 |

## 目录结构

```
stereo_3d_pipeline/
├── CMakeLists.txt
├── config/
│   ├── pipeline.yaml           # 默认全帧模式配置
│   └── pipeline_roi.yaml       # ✅ ROI 模式配置 (98 FPS 生产配置)
├── scripts/
│   ├── build_engine.sh         # TensorRT FP16 引擎构建脚本 (GPU/DLA)
│   ├── build_dla_engine.sh     # DLA 引擎一键构建脚本
│   └── pipeline_perf_compare.sh # GPU vs DLA 流水线对比脚本
└── src/
    ├── main.cpp                # 入口: 加载配置 → 运行 Pipeline
    ├── calibration/
    │   ├── stereo_calibration.h/cpp  # 标定参数加载 (OpenCV YAML)
    │   ├── capture_chessboard.cpp    # 棋盘格图像采集工具 (C++)
    │   ├── stereo_calibrate.cpp      # 双目标定计算工具 (C++)
    │   └── pwm_trigger.h            # libgpiod PWM 触发封装
    ├── capture/
    │   └── hikvision_camera.h/cpp    # 海康 MVS SDK 零拷贝双目采集
    ├── detect/
    │   ├── trt_detector.h/cpp        # TensorRT+NVDLA 检测器
    │   └── detect_preprocess.cu      # CUDA 预处理核 (BayerRG → CHW)
    ├── fusion/
    │   ├── coordinate_3d.h/cpp       # 视差→3D 坐标转换
    │   └── depth_extract.cu          # 直方图峰值深度提取 (CUDA)
    ├── pipeline/
    │   ├── pipeline.h/cpp            # 核心流水线 (全帧4阶段 + ROI 3阶段)
    │   ├── frame_slot.h              # 三缓冲帧槽
    │   └── sync.h                    # CUDA/VPI Stream 同步管理
    ├── rectify/
    │   └── vpi_rectifier.h/cpp       # VPI CUDA Remap 校正 (LDC)
    ├── stereo/
    │   ├── vpi_stereo.h/cpp          # VPI CUDA 视差计算 (全帧/半分辨率)
    │   ├── roi_stereo_matcher.h/cpp  # ✅ ROI 多点 CUDA 匹配器
    │   ├── roi_stereo_match.cu       # ✅ ROI 匹配 CUDA kernel
    │   └── onnx_stereo.h/cpp         # ONNX Runtime DL推理 (CREStereo/HITNet)
    └── utils/
        ├── logger.h                  # printf 格式日志
        ├── profiler.h                # 性能统计 (NVTX + 均值聚合)
        └── zero_copy_alloc.h         # cudaHostAllocMapped 零拷贝池
```

## 依赖

| 依赖          | 版本要求                | 说明                        |
|---------------|-------------------------|-----------------------------|
| JetPack       | 6.x (R36.4+)           | CUDA, TensorRT, VPI, cuDNN  |
| CUDA Toolkit  | 12.6+                  | Orin NX Super               |
| VPI           | 3.2+                   | Remap(VIC/CUDA) + TNR(CUDA) |
| TensorRT      | 10.3+                  | INT8 + DLA 推理              |
| OpenCV        | 4.x                    | 标定、图像处理               |
| yaml-cpp      | 0.6+                   | 配置文件解析                 |
| 海康 MVS SDK  | 3.x+                   | `/opt/MVS` 安装              |
| libgpiod      | 1.x                    | PWM 触发                     |

## 编译

```bash
# 在 Jetson 上
cd stereo_3d_pipeline
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译产物：
- `stereo_pipeline` — 主程序
- `stereo_depth_viewer` — **13算法深度对比 Viewer**（支持 headless 基准测试）
- `capture_chessboard` — 标定图像采集工具（需要海康 SDK）
- `stereo_calibrate` — 双目标定计算工具（纯 OpenCV）

可选依赖（自动检测）：
- **ONNX Runtime** → CREStereo/HITNet DL推理 (`HAS_ONNXRUNTIME`)
- **OpenCV ximgproc** → WLS视差滤波 (`HAS_XIMGPROC`)

## 深度算法 Viewer

`stereo_depth_viewer` 集成 13 种算法，支持实时预览和 headless 基准测试：

| 模式 | 算法 | 后端 |
|---|---|---|
| 0 | 原始立体对 | - |
| 1-3 | VPI CUDA SGM (Full/Half/Bilateral) | VPI |
| 4-8 | OpenCV CUDA SGM/BM/BP/CSBP, SGBM CPU | OpenCV |
| 9 | SGBM + WLS 后处理 | ximgproc |
| 10 | SGBM + Census 变换 | 自研 |
| 11-12 | CREStereo / HITNet | ONNX Runtime |

### Headless 基准测试
```bash
./stereo_depth_viewer --headless \
    --crestereo dl_models/crestereo_init_iter10_480x640.onnx \
    --hitnet dl_models/hitnet_eth3d_480x640.onnx
```
输出：`diagnose_output/benchmark_report.json` + `comparison_grid.png`

详细对比结果见 `docs/深度算法对比报告.md`。

自定义 CUDA 架构：
```bash
cmake .. -DCUDA_ARCH="87"   # 仅 Orin NX
```

## 标定流程

实操版检查清单与交接模板见：`docs/相机标定实施手册.md`。

### 1. 采集棋盘格图像

使用 C++ 采集工具（与主程序使用相同的海康 SDK 和 PWM 触发）：

```bash
# PWM 触发模式（默认）
./capture_chessboard -o calibration_images -e 3000

# 自由运行模式（无需外部触发信号）
./capture_chessboard --free-run -o calibration_images

# 自定义参数
./capture_chessboard --board-w 9 --board-h 6 --left 0 --right 1 -e 2000
```

操作方式：
- **空格** — 采集当前帧（需左右均检测到棋盘格）
- **q / ESC** — 退出
- **c** — 清空已采集图像

建议采集 20-30 对图像，覆盖不同角度和位置。

### 2. 执行标定

```bash
# 方格边长 30mm
./stereo_calibrate -s 30.0 -d calibration_images -o calibration/stereo_calib.yaml

# 自定义棋盘格尺寸
./stereo_calibrate -s 25.0 --board-w 11 --board-h 8 -d calibration_images

# 跳过可视化
./stereo_calibrate -s 30.0 --no-vis
```

标定工具自动完成：
1. 多级角点检测（严格→宽松降级策略）
2. 亚像素精化（`cornerSubPix`）
3. 单目标定 + 焦距合理性检查
4. 2σ 异常图像剔除
5. 立体标定（`CALIB_USE_INTRINSIC_GUESS`）
6. 立体校正（`stereoRectify`, `alpha=0`）
7. 深度精度报告
8. 校正效果可视化预览

输出文件 `stereo_calib.yaml` 与 `StereoCalibration::load()` 完全兼容。

### 3. 配置使用

将标定文件放入 `calibration/` 目录，并在 `config/pipeline.yaml` 中指定：

```yaml
calibration:
  file: "calibration/stereo_calib.yaml"
```

## TensorRT 引擎构建

```bash
# GPU FP16
./scripts/build_engine.sh /home/nvidia/NX_volleyball/model/yolo26.onnx /home/nvidia/NX_volleyball/model/yolo26_fp16.engine gpu

# DLA FP16 (Core 0 + GPU Fallback)
./scripts/build_dla_engine.sh /home/nvidia/NX_volleyball/model/yolo26.onnx /home/nvidia/NX_volleyball/model/yolo26_dla_fp16.engine
```

## 并行化性能对比

```bash
./scripts/pipeline_perf_compare.sh
```

脚本会输出 `benchmark_results/pipeline_compare_*.md`，包含：
- GPU / DLA 场景 FPS 对比
- Stage0 / Stage1 提交耗时
- Stage3 等待 Detect/Stereo 耗时
- tegrastats 的 GPU 平均利用率

## 运行

```bash
# ROI 模式 (98 FPS, 推荐)
./stereo_pipeline --config config/pipeline_roi.yaml

# 全帧模式 (用于调试/对比)
./stereo_pipeline --config config/pipeline.yaml

# GPU Mixed 模式 (yolo26_mixed, 低延迟)
./stereo_pipeline --config config/pipeline_yolo26_mixed.yaml

# 纯 GPU INT8 模式 (yolo26_gpu)
./stereo_pipeline --config config/pipeline_yolo26_gpu.yaml

# 可视化窗口 (显示检测框 + 测距 + 3D坐标)
./stereo_pipeline --config config/pipeline_yolo26_mixed.yaml --visualize
```

Ctrl+C 安全退出。

> 若通过 SSH 无桌面会话运行，OpenCV 可能无法初始化 GTK 窗口并自动回退 headless；
> 管线与检测仍会继续运行，可通过日志中的 `Ball:` 行确认检测与测距。

### 推荐生产配置（GPU Mixed, 目标 100 FPS）

- 配置文件：`config/pipeline_yolo26_mixed.yaml`
- 关键参数：
  - `rectify.backend: "CUDA"`（当前实机测试吞吐更高；也支持 `VIC`）
  - `stereo.strategy: "roi_only"`
  - `detector.engine_path: yolo26_mixed_attn_fp16.engine`
  - `detector.use_dla: false`（纯 GPU 推理）
  - `detector.input_format: "bayer"`（相机 BayerRG8 输入）
- 可视化模式建议仅用于联调；追求极限 FPS 时关闭 `--visualize`。

## 配置说明

生产配置 `config/pipeline_roi.yaml`：

```yaml
camera:
  exposure_us: 9867.0        # 用户标定曝光 (不可修改)
  gain_db: 11.9906           # 用户标定增益 (不可修改)
  use_trigger: true          # PWM 硬件触发
  width: 1440
  height: 1080

tnr:
  enabled: true              # VPI TNR 时域降噪 (0.04ms)
  preset: "outdoor_medium"
  strength: 0.6

detector:
  engine_path: "/home/nvidia/NX_volleyball/model/yolo26n_dla0_int8_640.engine"
  input_size: 640            # 640x640 检测分辨率
  use_dla: true              # 使用 DLA0 (释放 GPU)

rectify:
  backend: "CUDA"           # 当前实机推荐 CUDA；可选 "VIC"

stereo:
  strategy: "roi_only"       # ROI 多点匹配 (0.01ms)

performance:
  pwm_frequency: 100.0       # PWM 触发频率 Hz
```

## 技术要点

- **零拷贝内存**: `cudaHostAllocMapped` 在 SoC 统一内存架构上实现 CPU/GPU 共享，避免显式传输
- **VPI TNR 时域降噪**: NV12_ER 格式 CUDA 后端, V3 算法, 0.04ms/帧对, 显著降低 Bayer 噪声
- **VPI Remap 校正 (LDC)**: OpenCV 标定映射表 → VPI WarpMap → CUDA 异步 Remap
- **TensorRT INT8 + DLA**: yolo26n 在 DLA0 运行, 3.37ms@640×640, 释放 GPU 给校正/匹配
- **ROI 多点匹配**: 检测后仅在目标 bbox 区域执行 CUDA 立体匹配 + 亚像素拟合 + IQR 离群值过滤
- **三缓冲帧级交错**: 3 帧同时处于不同阶段, DLA/GPU/VPI 并行执行
- **PWM 硬件同步**: libgpiod 100Hz PWM, 双目 USB 相机同步触发

## 性能实测 (2026-04-03)

| 指标 | 值 |
|------|-----|
| 实测 FPS | **98** |
| PWM 触发率 | 100 Hz |
| 采集+提交校正 | 0.69 ms |
| VPI TNR 降噪 | 0.04 ms |
| VPI Remap 等待 | 7.63 ms (当前瓶颈) |
| DLA INT8 检测提交 | 2.90 ms |
| DLA 检测等待 | 0.12 ms (与 Remap 并行) |
| ROI 匹配+测距 | 0.01 ms |
| 平台 | Orin NX Super 16GB, JetPack 6, MAXN_SUPER |
