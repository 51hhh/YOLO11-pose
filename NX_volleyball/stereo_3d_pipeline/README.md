# stereo_3d_pipeline

Jetson Xavier NX / Orin NX 高帧率双目 3D 排球追踪流水线。

## 架构

4 阶段 Pipeline，3 帧重叠并行，目标帧率 60-100 FPS：

| Stage | 硬件    | 功能                          | 耗时估算 |
|-------|---------|-------------------------------|----------|
| 0     | CPU+CUDA | 海康相机抓帧 + VPI Remap 校正 | ~3 ms    |
| 1     | NVDLA   | TensorRT INT8 YOLOv8/v11 检测 | ~12-15 ms|
| 2     | GPU     | VPI Stereo Disparity 视差计算 | ~10-12 ms|
| 3     | GPU/CPU | 3D 坐标融合 + 输出            | ~1 ms    |

Stage 1 与 Stage 2 对同一帧异步提交，Stage 3 在下一次迭代等待并融合上一帧，形成真实帧间重叠。三帧交错调度：
- 帧 N   → Stage 3（融合输出）
- 帧 N+1 → Stage 1+2（检测+视差，异步提交）
- 帧 N+2 → Stage 0（抓帧+校正）

## 目录结构

```
stereo_3d_pipeline/
├── CMakeLists.txt
├── config/
│   └── pipeline.yaml           # 运行时配置
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
    │   ├── pipeline.h/cpp            # 核心四阶段流水线
    │   ├── frame_slot.h              # 三缓冲帧槽
    │   └── sync.h                    # CUDA/VPI Stream 同步管理
    ├── rectify/
    │   └── vpi_rectifier.h/cpp       # VPI CUDA Remap 校正
    ├── stereo/
    │   └── vpi_stereo.h/cpp          # VPI CUDA 视差计算
    └── utils/
        ├── logger.h                  # printf 格式日志
        ├── profiler.h                # 性能统计 (NVTX + 均值聚合)
        └── zero_copy_alloc.h         # cudaHostAllocMapped 零拷贝池
```

## 依赖

| 依赖          | 版本要求                | 说明                        |
|---------------|-------------------------|-----------------------------|
| JetPack       | 5.x+                   | CUDA, TensorRT, VPI, cuDNN  |
| CUDA Toolkit  | 11.4+                  | Xavier NX / Orin NX          |
| VPI           | 3.x+                   | Remap(CUDA/VIC) + Stereo(CUDA) |
| TensorRT      | 8.5+ / 10.x            | INT8 + DLA 推理              |
| OpenCV        | 4.x                    | 标定、图像处理               |
| yaml-cpp      | 0.6+                   | 配置文件解析                 |
| 海康 MVS SDK  | 3.x+                   | `/opt/MVS` 安装              |
| libgpiod      | 1.x                    | PWM 触发 (可选)              |

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
- `capture_chessboard` — 标定图像采集工具（需要海康 SDK）
- `stereo_calibrate` — 双目标定计算工具（纯 OpenCV）

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
./stereo_pipeline --config config/pipeline.yaml
```

Ctrl+C 安全退出。

## 配置说明

`config/pipeline.yaml` 主要配置项：

```yaml
camera:
  left_index: 0              # 左相机索引
  right_index: 1             # 右相机索引
  exposure_us: 3000.0        # 曝光时间 us
  use_trigger: true          # 外触发模式
  width: 1440                # 原始分辨率
  height: 1080

rectify:
  output_width: 1280         # 校正后分辨率
  output_height: 720

detector:
  engine_path: "models/yolov8n_int8.engine"
  use_dla: true
  input_size: 320            # 模型输入（320 更快，640 更准）

stereo:
  max_disparity: 128
  use_half_resolution: false # 开启后视差计算速度翻倍

fusion:
  min_depth: 0.3             # 最小有效深度 m
  max_depth: 15.0            # 最大有效深度 m

performance:
  pwm_frequency: 100.0       # PWM 触发频率 Hz
```

## 技术要点

- **零拷贝内存**: `cudaHostAllocMapped` 在 SoC 统一内存架构上实现 CPU/GPU 共享，避免显式传输
- **VPI PVA 校正**: 在 PVA 硬件上异步执行 Remap，不占用 GPU/CPU 时间
- **TensorRT INT8 + NVDLA**: 在专用 DLA 加速器上推理，释放 GPU 给视差计算
- **VPI 视差 Q8.8 定点**: S16 格式输出，直方图峰值提取避免浮点开销
- **半分辨率策略**: 可选 `use_half_resolution`，视差在半分辨率计算后通过 `disparityScale` 补偿
- **帧级交错**: 3 帧同时处于不同阶段，流水线效率最大化
