# 🚀 快速开始指南

## 项目结构

```
volleyball_tracking/
├── README.md                    # 主文档
├── PROJECT_SUMMARY.md           # 项目总结
├── .gitignore                   # Git 忽略配置
│
├── docs/                        # 详细文档
│   ├── 01_data_preparation.md   # 数据准备
│   ├── 02_training.md           # 模型训练
│   ├── 03_tensorrt_export.md    # TensorRT 转换
│   └── 04_deployment.md         # Orin NX 部署
│
├── data/                        # 数据集
│   ├── dataset.yaml             # 数据集配置
│   ├── images/                  # 图像 (需创建)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/                  # 标注 (需创建)
│       ├── train/
│       ├── val/
│       └── test/
│
├── train/                       # 训练模块
│   ├── train.py                 # 训练脚本
│   ├── config.yaml              # 训练配置
│   └── requirements.txt         # 依赖列表
│
├── deploy/                      # 部署模块
│   ├── geometry.py              # 几何拟合
│   ├── tracker.py               # 追踪模块
│   ├── visualizer.py            # 可视化
│   ├── export_tensorrt.py       # TensorRT 导出
│   └── requirements_nx.txt      # Orin NX 依赖
│
├── tools/                       # 工具脚本 (待实现)
│   └── README.md
│
├── demo/                        # 演示程序 (待实现)
│   └── README.md
│
└── models/                      # 模型文件
    └── README.md
```

---

## 🎯 完整工作流程

### 阶段 1: 环境准备

#### 训练环境 (PC/服务器)

```bash
# 克隆项目
cd /home/rick/desktop/yolo/yoloProject/volleyball_tracking

# 创建虚拟环境
conda create -n volleyball python=3.10
conda activate volleyball

# 安装训练依赖
cd train
pip install -r requirements.txt

# 下载预训练模型
cd ../models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov11n-pose.pt
```

#### 部署环境 (Orin NX)

```bash
# 设置性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 安装依赖
cd deploy
pip3 install -r requirements_nx.txt
```

---

### 阶段 2: 数据准备

```bash
# 1. 创建数据目录
mkdir -p data/images/{train,val,test}
mkdir -p data/labels/{train,val,test}

# 2. 收集排球图像/视频
# 将图像放入 data/images/train/

# 3. 标注数据 (5个关键点)
# 使用标注工具标注:
# - Point 0: Center (球心)
# - Point 1: Top (顶部)
# - Point 2: Bottom (底部)
# - Point 3: Left (左侧)
# - Point 4: Right (右侧)

# 4. 转换为 YOLO-Pose 格式
# 每个图像对应一个 .txt 文件
# 格式: class x_center y_center width height kpt1_x kpt1_y kpt1_v ...
```

**标注示例** (`data/labels/train/image001.txt`):
```
0 0.512 0.384 0.156 0.208 0.512 0.384 2 0.512 0.300 2 0.512 0.468 2 0.456 0.384 2 0.568 0.384 2
```

---

### 阶段 3: 模型训练

```bash
cd train

# 检查配置
cat config.yaml

# 开始训练
python train.py --config config.yaml

# 监控训练 (新终端)
tensorboard --logdir runs/train --port 6006
# 浏览器打开: http://localhost:6006
```

**预期结果**:
- 训练 100 轮约需 2-4 小时 (RTX 4090)
- mAP@0.5 应达到 >0.95
- 最佳模型保存在 `runs/train/volleyball/weights/best.pt`

---

### 阶段 4: TensorRT 转换

```bash
cd deploy

# 导出 TensorRT 引擎 (FP16)
python export_tensorrt.py \
    --weights ../models/volleyball_best.pt \
    --imgsz 640 \
    --fp16

# 输出: ../models/volleyball_best.engine
```

**在 Orin NX 上转换** (推荐):
```bash
# 将 .pt 模型传输到 Orin NX
scp volleyball_best.pt orin@192.168.1.100:~/models/

# 在 Orin NX 上转换
ssh orin@192.168.1.100
cd volleyball_tracking/deploy
python3 export_tensorrt.py \
    --weights ~/models/volleyball_best.pt \
    --imgsz 640 \
    --fp16
```

---

### 阶段 5: 部署测试

#### 方法 1: Python 脚本测试

```python
# test_inference.py
import sys
sys.path.append('deploy')

from geometry import CircleFitter
from tracker import VolleyballTracker, Detection
from visualizer import Visualizer
import numpy as np

# 初始化
fitter = CircleFitter(method='weighted_lsq')
tracker = VolleyballTracker(frame_rate=150)
visualizer = Visualizer()

# 模拟检测结果
keypoints = np.array([
    [320, 240],  # Center
    [320, 220],  # Top
    [320, 260],  # Bottom
    [300, 240],  # Left
    [340, 240],  # Right
])
confidences = np.array([0.95, 0.90, 0.92, 0.88, 0.91])

# 拟合圆
cx, cy, r, quality = fitter.fit(keypoints, confidences)
print(f"圆心: ({cx:.2f}, {cy:.2f}), 半径: {r:.2f}, 质量: {quality:.2f}")

# 创建检测
det = Detection(cx, cy, r, 0.95, keypoints, confidences)

# 追踪
tracks = tracker.update([det])
print(f"追踪到 {len(tracks)} 个目标")
```

#### 方法 2: 实时相机 (需实现 demo_camera.py)

```bash
cd demo

# 实时推理
python demo_camera.py \
    --engine ../models/volleyball.engine \
    --camera 0 \
    --show
```

---

## 📊 性能验证

### 基准测试

```python
# benchmark.py
import time
import numpy as np
from geometry import CircleFitter

fitter = CircleFitter()
keypoints = np.random.rand(5, 2) * 640
confidences = np.random.rand(5)

# 预热
for _ in range(100):
    fitter.fit(keypoints, confidences)

# 测试
times = []
for _ in range(1000):
    start = time.perf_counter()
    fitter.fit(keypoints, confidences)
    times.append((time.perf_counter() - start) * 1000)

print(f"平均耗时: {np.mean(times):.3f} ms")
print(f"中位数: {np.median(times):.3f} ms")
```

**预期结果**:
- 几何拟合: ~0.05 ms
- TensorRT 推理: ~6 ms
- 总延迟: ~6.5 ms
- 帧率: ~150 FPS

---

## ✅ 检查清单

### 数据准备
- [ ] 收集 1000+ 张排球图像
- [ ] 标注 5 个关键点
- [ ] 验证标注质量
- [ ] 划分训练/验证/测试集

### 模型训练
- [ ] 安装训练依赖
- [ ] 配置训练参数
- [ ] 开始训练
- [ ] 验证 mAP >0.95

### TensorRT 转换
- [ ] 导出 ONNX
- [ ] 转换 TensorRT
- [ ] 验证推理速度

### Orin NX 部署
- [ ] 设置性能模式
- [ ] 安装依赖
- [ ] 测试推理
- [ ] 验证帧率 >150 FPS

---

## 🐛 故障排除

### 问题 1: 训练损失不收敛

**解决**:
```yaml
# config.yaml
lr0: 0.0005  # 降低学习率
pose: 15.0   # 增加关键点权重
```

### 问题 2: TensorRT 转换失败

**解决**:
```bash
# 使用兼容的 ONNX opset
python export_tensorrt.py --simplify --opset 11
```

### 问题 3: Orin NX 帧率低

**解决**:
```bash
# 最大化性能
sudo nvpmodel -m 0
sudo jetson_clocks

# 关闭 GUI
sudo systemctl stop gdm3
```

---

## 📚 下一步

1. **阅读详细文档**: `docs/01_data_preparation.md`
2. **准备数据集**: 标注 1000+ 张图像
3. **开始训练**: `python train/train.py`
4. **部署测试**: 在 Orin NX 上验证性能

---

**需要帮助?** 查看 `PROJECT_SUMMARY.md` 或详细文档。
