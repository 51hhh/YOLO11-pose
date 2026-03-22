# volleyball_tracking：YOLO11 Pose 排球关键点训练与单目追踪链路

`volleyball_tracking` 是这个仓库里偏算法与训练的一条实现路线，重点是把排球检测建模成 **1 类目标 + 5 个关键点** 的 pose 任务，并在检测结果之上完成几何拟合、追踪和平滑。

与同仓库下的 `../NX_volleyball` 相比，这个目录更适合展示：
- 数据定义与标注规范
- YOLOv11-Pose 训练配置
- 关键点到圆心/半径的几何建模
- ByteTrack + Kalman 的单目追踪模块化实现
- `.pt` 到 TensorRT `.engine` 的导出链路

## 核心流程

```text
输入图像
  ↓
YOLOv11n-Pose
  ↓
5关键点 (Center / Top / Bottom / Left / Right)
  ↓
加权最小二乘拟合圆
  ↓
ByteTrack 多目标追踪
  ↓
卡尔曼滤波平滑
  ↓
输出 (cx, cy, r, vx, vy, track_id)
```

## 关键点定义

```text
     Top
      ↑
Left ← Center → Right
      ↓
    Bottom
```

当前数据集定义采用 5 个关键点：
- `Center`：球心点，拟合时额外加权
- `Top / Bottom / Left / Right`：用于恢复球的几何边界
- 左右点在水平翻转增强时需要交换，对应关系已经写在 `data/dataset.yaml`

## 目录结构

```text
volleyball_tracking/
├── README.md
├── QUICKSTART.md
├── PROJECT_SUMMARY.md
├── data/
│   └── dataset.yaml               # YOLO-Pose 数据集配置
├── train/
│   ├── train.py                   # 训练脚本
│   ├── config.yaml                # 训练超参数配置
│   └── requirements.txt
├── deploy/
│   ├── export_tensorrt.py         # TensorRT 导出
│   ├── geometry.py                # 圆拟合算法
│   ├── tracker.py                 # ByteTrack + Kalman
│   ├── visualizer.py              # 可视化组件
│   └── requirements_nx.txt
├── docs/                          # 数据、训练、导出、部署文档
├── demo/                          # 预留目录，当前仅有 README
├── tools/                         # 预留目录，当前仅有 README
└── models/                        # 训练模型与导出引擎存放位置
```

## 当前已实现内容

| 模块 | 说明 |
|------|------|
| `train/train.py` | 基于 Ultralytics YOLO 的训练入口，支持配置文件、恢复训练、权重覆盖 |
| `train/config.yaml` | 包含模型、图像尺寸、训练轮数、优化器、loss 权重和增强参数 |
| `data/dataset.yaml` | 定义类别、关键点形状与翻转映射 |
| `deploy/geometry.py` | 提供 `weighted_lsq`、`ransac`、`algebraic` 三种圆拟合方法 |
| `deploy/tracker.py` | 实现高低分检测分离的 ByteTrack 两阶段匹配和 8 维状态卡尔曼滤波 |
| `deploy/visualizer.py` | 结果绘制与调试可视化组件 |
| `deploy/export_tensorrt.py` | 从 `.pt` 导出 TensorRT engine，并复制回 `models/` |
| `docs/` | 对数据准备、训练、导出和部署过程做了补充说明 |

## 目前仍是预留/占位的部分

为了保证 README 与现状一致，需要特别说明：
- `tools/` 目前还是预留目录，仓库中只有说明文档，没有实际脚本实现
- `demo/` 目前也是预留目录，还没有完整的相机/视频演示入口
- 这个子项目已经具备训练、导出、拟合和追踪模块，但还不是一个“开箱即用”的完整单目录应用

这并不是缺点，反而更适合作为作品集展示“训练链路 + 算法模块化设计”的部分。

## 快速开始

### 1. 安装训练依赖

```bash
cd volleyball_tracking/train
pip install -r requirements.txt
```

### 2. 准备预训练模型

将 Ultralytics 的预训练权重放到 `models/` 目录，例如：

```bash
cd volleyball_tracking/models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov11n-pose.pt
```

### 3. 准备数据集

当前仓库只保留了 `dataset.yaml`，图像与标签目录需要自行准备。常见结构如下：

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

标签格式为 YOLO Pose；每个目标包含 bbox 和 5 个关键点坐标/可见性。

### 4. 开始训练

```bash
cd volleyball_tracking/train
python train.py --config config.yaml
```

默认配置要点：
- 模型：`yolov11n-pose.pt`
- 输入尺寸：`640`
- 训练轮数：`100`
- 优化器：`AdamW`
- pose loss 权重：`12.0`

训练完成后，脚本会把最佳模型复制到：

```text
volleyball_tracking/models/volleyball_best.pt
```

### 5. 导出 TensorRT 引擎

```bash
cd volleyball_tracking/deploy
python export_tensorrt.py \
    --weights ../models/volleyball_best.pt \
    --imgsz 640 \
    --fp16
```

导出成功后，engine 文件会复制回 `../models/`。

## 核心模块说明

### 1. 圆拟合：`deploy/geometry.py`

`CircleFitter` 提供三种方法：
- `weighted_lsq`：默认方法，对 `Center` 点额外加权
- `ransac`：适合异常点或遮挡场景
- `algebraic`：代数拟合，速度快但精度相对保守

这部分体现了本项目的一个关键特点：**不是直接使用 bbox 中心，而是用关键点恢复球的几何圆心和半径**。

### 2. 追踪：`deploy/tracker.py`

`VolleyballTracker` 的实现重点包括：
- 高分/低分检测分离的 ByteTrack 两阶段匹配
- 圆 IoU 作为匹配度量
- 8 维状态卡尔曼滤波：`[cx, cy, r, vx, vy, vr, ax, ay]`

这让它不仅能输出目标位置，还能输出更稳定的速度和轨迹状态。

### 3. 可视化：`deploy/visualizer.py`

该模块负责把关键点、拟合圆、速度向量和轨迹结果画到图像上，适合在算法调试阶段配合自定义推理脚本使用。

## 适合作品集展示的点

- 把排球检测问题改写成 **5 关键点 pose 任务**，而不是只做 bbox 检测
- 在检测结果之后加入 **几何拟合**，得到更稳定的圆心和半径估计
- 把追踪、几何拟合、导出和可视化拆成独立模块，结构清晰，便于验证和复用
- 训练与部署链路可自然衔接到同仓库下的 `NX_volleyball` 双目 3D 系统

## 使用边界

- 当前仓库不包含实际数据集与训练好的 engine 文件
- `demo/` 与 `tools/` 仍是预留状态，公开展示时应如实说明
- 如果需要看完整的 Jetson + ROS2 双目部署链路，请继续阅读 `../NX_volleyball/README.md`

## 相关文档

- `../README.md`：仓库总览
- `QUICKSTART.md`：更偏操作步骤的快速上手说明
- `PROJECT_SUMMARY.md`：项目阶段性总结
- `docs/01_data_preparation.md`：数据与标注说明
- `docs/02_training.md`：训练过程说明
- `docs/03_tensorrt_export.md`：TensorRT 导出说明
- `docs/04_deployment.md`：部署说明
