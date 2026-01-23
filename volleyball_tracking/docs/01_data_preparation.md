# 📊 数据准备指南

## 目标

准备高质量的排球检测数据集，包含 5 个关键点标注。

---

## 数据收集

### 推荐数据源

1. **自采集视频** (推荐)
   - 使用全局快门相机
   - 分辨率: 1920x1080 或更高
   - 帧率: 60-120 FPS
   - 场景多样性: 室内/室外、不同光照

2. **公开数据集**
   - [VolleyballDataset](https://github.com/mostafa-saad/deep-activity-rec)
   - [Sports-1M](https://cs.stanford.edu/people/karpathy/deepvideo/)

3. **数据量建议**
   - 训练集: 800-1000 张
   - 验证集: 200 张
   - 测试集: 200 张

---

## 标注规范

### 关键点定义

```
关键点索引:
0: Center  - 球心 (视觉中心)
1: Top     - 图像坐标系 y_min 处
2: Bottom  - 图像坐标系 y_max 处
3: Left    - 图像坐标系 x_min 处
4: Right   - 图像坐标系 x_max 处
```

### 标注原则

1. **Center (索引 0)**
   - 标注在球的视觉中心
   - 即使有运动模糊，也标在清晰部分的中心
   - 这是最重要的点，必须精确

2. **极值点 (索引 1-4)**
   - 标在球的投影边界上
   - 如果有运动模糊形成椭圆，标在模糊边缘
   - 必须是真实可见的边缘点

3. **遮挡处理**
   - 如果某个极值点被遮挡，标注 visibility=0
   - Center 点即使部分遮挡也应尽量标注

4. **多球场景**
   - 每个球独立标注
   - 确保 track_id 一致性（如果是视频序列）

---

## 半自动标注工具

### 工具功能

```bash
python tools/annotate.py --input data/images --output data/labels
```

**功能**:
1. 自动检测球的大致位置 (使用预训练 YOLO)
2. 提取轮廓并计算极值点
3. 人工微调关键点位置
4. 导出 YOLO-Pose 格式

### 使用流程

```python
# 1. 加载图像
image = cv2.imread("volleyball.jpg")

# 2. 自动检测球
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(image)

# 3. 提取轮廓
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. 计算极值点
for contour in contours:
    # 找到最左、最右、最上、最下的点
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    bottommost = tuple(contour[contour[:,:,1].argmax()][0])
    
    # 计算中心
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # 保存关键点
    keypoints = [
        [cx, cy, 2],           # Center (visible)
        [topmost[0], topmost[1], 2],
        [bottommost[0], bottommost[1], 2],
        [leftmost[0], leftmost[1], 2],
        [rightmost[0], rightmost[1], 2]
    ]
```

### 手动微调界面

**快捷键**:
- `鼠标左键`: 拖动关键点
- `数字键 0-4`: 选择关键点
- `V`: 切换 visibility (0/1/2)
- `S`: 保存当前标注
- `N`: 下一张图像
- `P`: 上一张图像
- `Q`: 退出

---

## 数据格式

### YOLO-Pose 标注格式

每个图像对应一个 `.txt` 文件，格式：

```
<class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_v> <kpt2_x> <kpt2_y> <kpt2_v> ...
```

**示例**:
```
0 0.512 0.384 0.156 0.208 0.512 0.384 2 0.512 0.300 2 0.512 0.468 2 0.456 0.384 2 0.568 0.384 2
```

**字段说明**:
- `class_id`: 类别 (0=排球)
- `x_center, y_center, width, height`: 归一化 bbox (0-1)
- `kptN_x, kptN_y`: 归一化关键点坐标 (0-1)
- `kptN_v`: visibility (0=未标注, 1=遮挡, 2=可见)

### 数据集配置文件

`data/dataset.yaml`:

```yaml
# 数据集路径
path: /home/rick/desktop/yolo/yoloProject/volleyball_tracking/data
train: images/train
val: images/val
test: images/test

# 类别
names:
  0: volleyball

# 关键点配置
kpt_shape: [5, 3]  # 5个关键点，每个3维 (x, y, visibility)

# 关键点连接 (用于可视化骨架)
flip_idx: [0, 2, 1, 4, 3]  # 水平翻转时的索引映射
```

---

## 数据增强策略

### 训练时增强

```yaml
# config.yaml 中的增强参数
augment: true

# 几何变换
degrees: 10.0        # 旋转 ±10°
translate: 0.1       # 平移 ±10%
scale: 0.5           # 缩放 0.5-1.5x
shear: 0.0           # 不使用剪切 (会扭曲圆形)
perspective: 0.0     # 不使用透视 (会扭曲圆形)
flipud: 0.0          # 不上下翻转
fliplr: 0.5          # 50% 左右翻转

# 颜色增强
hsv_h: 0.015         # 色调 ±1.5%
hsv_s: 0.7           # 饱和度 ±70%
hsv_v: 0.4           # 亮度 ±40%

# 高级增强
mosaic: 1.0          # 100% 使用 Mosaic
mixup: 0.1           # 10% 使用 MixUp
copy_paste: 0.0      # 不使用 Copy-Paste (单目标场景)

# 运动模糊 (模拟高速运动)
motion_blur: 0.3     # 30% 概率添加运动模糊
blur_kernel: [3, 7]  # 模糊核大小范围
```

### 离线增强脚本

```python
import albumentations as A

transform = A.Compose([
    # 运动模糊
    A.MotionBlur(blur_limit=7, p=0.3),
    
    # 高斯噪声
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    
    # 亮度对比度
    A.RandomBrightnessContrast(p=0.5),
    
    # 模糊
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
    ], p=0.2),
    
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# 应用增强
augmented = transform(image=image, keypoints=keypoints)
```

---

## 数据质量检查

### 自动验证脚本

```bash
python tools/validate.py --data data/labels --check-all
```

**检查项**:
1. ✅ 关键点是否在图像范围内
2. ✅ Center 点是否在 bbox 内
3. ✅ 极值点是否在 bbox 边界附近
4. ✅ 关键点是否形成合理的几何形状
5. ✅ visibility 标注是否正确

### 可视化检查

```bash
python tools/visualize.py --data data --num 50
```

**输出**:
- 在图像上绘制关键点和 bbox
- 显示拟合的圆形
- 标注关键点索引和 visibility

---

## 数据集划分

### 推荐划分比例

```python
# 70% 训练, 15% 验证, 15% 测试
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
```

### 划分脚本

```bash
python tools/split_dataset.py \
    --input data/images \
    --output data \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --seed 42
```

---

## 数据统计

### 统计脚本

```bash
python tools/stats.py --data data/dataset.yaml
```

**输出示例**:
```
数据集统计:
├── 训练集: 800 张
├── 验证集: 200 张
└── 测试集: 200 张

关键点可见性分布:
├── Center:  100.0% 可见
├── Top:     98.5% 可见
├── Bottom:  98.2% 可见
├── Left:    97.8% 可见
└── Right:   97.9% 可见

球的尺寸分布:
├── 平均半径: 24.3 px
├── 最小半径: 12.1 px
└── 最大半径: 45.7 px
```

---

## 常见问题

### Q1: 如何标注运动模糊的球？

**答**: 
- Center 点标在清晰部分的中心
- 极值点标在模糊边缘的极值位置
- 如果模糊严重到无法识别，跳过该帧

### Q2: 多个球重叠怎么办？

**答**:
- 分别标注每个球
- 如果某个球被完全遮挡，不标注
- 如果部分遮挡，标注可见的关键点，遮挡的点设 visibility=1

### Q3: 球在图像边缘被裁剪？

**答**:
- 仍然标注可见的关键点
- 被裁剪的极值点设 visibility=0
- Center 点如果可见必须标注

---

## 下一步

完成数据准备后，进入 [模型训练](02_training.md) 阶段。
