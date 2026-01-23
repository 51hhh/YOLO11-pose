# 🎓 模型训练指南

## 训练环境准备

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) |
| CPU | 8 核 | 16 核+ |
| 内存 | 16GB | 32GB+ |
| 存储 | 50GB SSD | 100GB NVMe |

### 软件依赖

```bash
# 创建虚拟环境
conda create -n volleyball python=3.10
conda activate volleyball

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics
pip install ultralytics==8.1.0

# 其他依赖
pip install -r requirements.txt
```

**requirements.txt**:
```
ultralytics==8.1.0
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.11.3
matplotlib==3.8.0
tensorboard==2.15.1
albumentations==1.3.1
```

---

## 训练配置

### 配置文件 `train/config.yaml`

```yaml
# 模型配置
model: yolov11n-pose.pt  # 预训练模型
task: pose               # 任务类型

# 数据配置
data: ../data/dataset.yaml
imgsz: 640               # 输入图像尺寸

# 训练超参数
epochs: 100              # 训练轮数
batch: 32                # batch size (根据 GPU 内存调整)
workers: 8               # 数据加载线程数
device: 0                # GPU 设备 ID

# 优化器
optimizer: AdamW         # 优化器类型
lr0: 0.001               # 初始学习率
lrf: 0.01                # 最终学习率 (lr0 * lrf)
momentum: 0.937          # SGD momentum
weight_decay: 0.0005     # 权重衰减

# 学习率调度
cos_lr: true             # 使用余弦退火
warmup_epochs: 3         # warmup 轮数
warmup_momentum: 0.8     # warmup momentum
warmup_bias_lr: 0.1      # warmup bias 学习率

# 损失函数权重
box: 7.5                 # bbox 损失权重
cls: 0.5                 # 分类损失权重
pose: 12.0               # 关键点损失权重
kobj: 1.0                # 关键点目标损失权重

# 数据增强
augment: true
degrees: 10.0            # 旋转角度
translate: 0.1           # 平移比例
scale: 0.5               # 缩放范围
shear: 0.0               # 剪切 (不使用)
perspective: 0.0         # 透视变换 (不使用)
flipud: 0.0              # 上下翻转 (不使用)
fliplr: 0.5              # 左右翻转概率
mosaic: 1.0              # Mosaic 增强
mixup: 0.1               # MixUp 增强
copy_paste: 0.0          # Copy-Paste (不使用)

# 颜色增强
hsv_h: 0.015             # 色调
hsv_s: 0.7               # 饱和度
hsv_v: 0.4               # 亮度

# 验证配置
val: true                # 每轮验证
save: true               # 保存检查点
save_period: 10          # 每 N 轮保存一次
plots: true              # 生成可视化图表

# 其他
patience: 50             # 早停轮数
verbose: true            # 详细输出
seed: 42                 # 随机种子
deterministic: true      # 确定性训练
```

---

## 训练脚本

### 基础训练脚本 `train/train.py`

```python
#!/usr/bin/env python3
"""
排球检测模型训练脚本
"""
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(cfg['seed'])
    
    # 加载预训练模型
    model = YOLO(cfg['model'])
    
    # 打印模型信息
    print(f"模型: {cfg['model']}")
    print(f"参数量: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")
    print(f"设备: {cfg['device']}")
    
    # 开始训练
    results = model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        device=cfg['device'],
        workers=cfg['workers'],
        
        # 优化器
        optimizer=cfg['optimizer'],
        lr0=cfg['lr0'],
        lrf=cfg['lrf'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
        
        # 学习率调度
        cos_lr=cfg['cos_lr'],
        warmup_epochs=cfg['warmup_epochs'],
        warmup_momentum=cfg['warmup_momentum'],
        warmup_bias_lr=cfg['warmup_bias_lr'],
        
        # 损失权重
        box=cfg['box'],
        cls=cfg['cls'],
        pose=cfg['pose'],
        kobj=cfg['kobj'],
        
        # 数据增强
        augment=cfg['augment'],
        degrees=cfg['degrees'],
        translate=cfg['translate'],
        scale=cfg['scale'],
        shear=cfg['shear'],
        perspective=cfg['perspective'],
        flipud=cfg['flipud'],
        fliplr=cfg['fliplr'],
        mosaic=cfg['mosaic'],
        mixup=cfg['mixup'],
        copy_paste=cfg['copy_paste'],
        hsv_h=cfg['hsv_h'],
        hsv_s=cfg['hsv_s'],
        hsv_v=cfg['hsv_v'],
        
        # 验证和保存
        val=cfg['val'],
        save=cfg['save'],
        save_period=cfg['save_period'],
        plots=cfg['plots'],
        
        # 其他
        patience=cfg['patience'],
        verbose=cfg['verbose'],
        seed=cfg['seed'],
        deterministic=cfg['deterministic'],
        
        # 项目配置
        project='runs/train',
        name='volleyball',
        exist_ok=False,
    )
    
    # 打印训练结果
    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)
    print(f"最佳模型: {results.save_dir / 'weights/best.pt'}")
    print(f"最终模型: {results.save_dir / 'weights/last.pt'}")
    print(f"mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"关键点 OKS: {results.results_dict['metrics/mAP50(P)']:.4f}")

if __name__ == '__main__':
    main()
```

### 运行训练

```bash
cd train
python train.py
```

---

## 训练监控

### TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir runs/train --port 6006

# 在浏览器打开
# http://localhost:6006
```

**监控指标**:
- `train/box_loss`: Bbox 回归损失
- `train/cls_loss`: 分类损失
- `train/pose_loss`: 关键点损失
- `val/mAP50(B)`: Bbox mAP@0.5
- `val/mAP50(P)`: 关键点 mAP@0.5
- `lr/pg0`: 学习率

### 实时监控脚本

```python
import time
from pathlib import Path

def monitor_training(log_dir):
    """实时监控训练日志"""
    results_file = Path(log_dir) / 'results.csv'
    
    print("等待训练开始...")
    while not results_file.exists():
        time.sleep(1)
    
    print("训练已开始，监控中...")
    last_line = ""
    
    while True:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                current_line = lines[-1]
                if current_line != last_line:
                    # 解析并打印关键指标
                    values = current_line.strip().split(',')
                    epoch = values[0]
                    box_loss = float(values[2])
                    pose_loss = float(values[4])
                    mAP50 = float(values[7])
                    
                    print(f"Epoch {epoch}: box_loss={box_loss:.4f}, "
                          f"pose_loss={pose_loss:.4f}, mAP50={mAP50:.4f}")
                    
                    last_line = current_line
        
        time.sleep(5)

# 使用
monitor_training('runs/train/volleyball')
```

---

## 训练技巧

### 1. 学习率调优

**学习率查找器**:
```python
from ultralytics import YOLO

model = YOLO('yolov11n-pose.pt')

# 自动查找最佳学习率
model.tune(
    data='../data/dataset.yaml',
    epochs=30,
    iterations=300,
    optimizer='AdamW',
    plots=True,
    save=False,
)
```

### 2. 渐进式训练

**阶段 1: 冻结骨干网络**
```yaml
# config_stage1.yaml
epochs: 30
freeze: 10  # 冻结前 10 层
lr0: 0.01   # 较高学习率
```

**阶段 2: 全网络微调**
```yaml
# config_stage2.yaml
model: runs/train/volleyball/weights/best.pt  # 加载阶段1模型
epochs: 70
freeze: 0   # 不冻结
lr0: 0.001  # 较低学习率
```

### 3. 多尺度训练

```yaml
# 启用多尺度
multi_scale: true
rect: false  # 不使用矩形训练
```

### 4. 混合精度训练

```yaml
# 自动混合精度 (AMP)
amp: true  # 默认启用
```

---

## 模型评估

### 验证脚本

```bash
python -m ultralytics.yolo.v8.detect.val \
    model=runs/train/volleyball/weights/best.pt \
    data=../data/dataset.yaml \
    imgsz=640 \
    batch=32 \
    device=0
```

### 自定义评估指标

```python
from ultralytics import YOLO
import numpy as np

def evaluate_circle_accuracy(model, test_data):
    """评估圆形拟合精度"""
    results = model.val(data=test_data)
    
    circle_errors = []
    
    for result in results:
        # 提取关键点
        keypoints = result.keypoints.xy.cpu().numpy()
        
        # 拟合圆
        for kpts in keypoints:
            cx, cy, r = fit_circle_weighted(kpts, np.ones(5))
            
            # 计算误差
            gt_circle = result.gt_circle  # 假设有真值
            error = np.sqrt((cx - gt_circle[0])**2 + (cy - gt_circle[1])**2)
            circle_errors.append(error)
    
    print(f"圆心平均误差: {np.mean(circle_errors):.2f} px")
    print(f"圆心中位数误差: {np.median(circle_errors):.2f} px")
    print(f"圆心 95% 误差: {np.percentile(circle_errors, 95):.2f} px")
```

---

## 模型导出

### 导出 ONNX

```python
from ultralytics import YOLO

model = YOLO('runs/train/volleyball/weights/best.pt')

# 导出 ONNX
model.export(
    format='onnx',
    imgsz=640,
    opset=11,
    simplify=True,
    dynamic=False,
)
```

### 导出 TorchScript

```python
model.export(
    format='torchscript',
    imgsz=640,
    optimize=True,
)
```

---

## 常见问题

### Q1: 训练时 GPU 内存溢出？

**解决方案**:
```yaml
# 减小 batch size
batch: 16  # 或 8

# 启用梯度累积
accumulate: 2  # 等效 batch=32
```

### Q2: 关键点损失不收敛？

**检查**:
1. 标注质量 (使用 `tools/visualize.py`)
2. 降低学习率 (`lr0: 0.0005`)
3. 增加 `pose` 损失权重 (`pose: 15.0`)

### Q3: 验证 mAP 波动大？

**解决**:
```yaml
# 增加验证集大小
# 使用余弦退火
cos_lr: true

# 增加 warmup
warmup_epochs: 5
```

### Q4: 如何恢复训练？

```python
model = YOLO('runs/train/volleyball/weights/last.pt')
model.train(resume=True)
```

---

## 训练检查清单

训练前确认:
- [ ] 数据集路径正确
- [ ] 标注格式正确 (YOLO-Pose)
- [ ] GPU 驱动和 CUDA 版本匹配
- [ ] 虚拟环境已激活
- [ ] 配置文件参数合理

训练中监控:
- [ ] 损失是否下降
- [ ] mAP 是否上升
- [ ] 学习率是否正常调度
- [ ] GPU 利用率 >80%

训练后验证:
- [ ] 在测试集上评估
- [ ] 可视化预测结果
- [ ] 检查关键点精度
- [ ] 测试推理速度

---

## 下一步

模型训练完成后，进入 [TensorRT 转换](03_tensorrt_export.md) 阶段。
