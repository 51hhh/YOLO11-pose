# 双目相机标定文件

## 说明
请将标定后的 `stereo_calib.yaml` 文件放在此目录下。

## ⚠️ 重要：当前使用占位符参数

当前的 `stereo_calib.yaml` 包含**占位符参数**，不是实际标定结果。

**影响**：
- ✅ YOLO检测正常工作
- ✅ PWM和相机同步正常
- ⚠️ 3D位置测量**不准确**（深度估计会有较大误差）

**建议**：
- 如果只需要2D检测，可以继续使用占位符
- 如果需要准确的3D位置，请进行实际标定

---

## 标定参数格式（代码期望）

```yaml
%YAML:1.0
---
# 左相机内参矩阵 (3x3)
K1: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0., cx,
           0., fy, cy,
           0., 0., 1. ]

# 左相机畸变系数 (1x5: k1, k2, p1, p2, k3)
D1: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ k1, k2, p1, p2, k3 ]

# 右相机内参矩阵 (3x3)
K2: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0., cx,
           0., fy, cy,
           0., 0., 1. ]

# 右相机畸变系数 (1x5)
D2: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ k1, k2, p1, p2, k3 ]

# 左相机投影矩阵 (3x4)
P1: !!opencv-matrix
   rows: 3
   cols: 4
   dt: d
   data: [ fx, 0., cx, 0.,
           0., fy, cy, 0.,
           0., 0., 1., 0. ]

# 右相机投影矩阵 (3x4)
# P2[0,3] = -fx * baseline
P2: !!opencv-matrix
   rows: 3
   cols: 4
   dt: d
   data: [ fx, 0., cx, -fx*baseline,
           0., fy, cy, 0.,
           0., 0., 1., 0. ]

# 基线距离 (米)
baseline: 0.25

# 可选：旋转矩阵 (3x3)
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ r11, r12, r13,
           r21, r22, r23,
           r31, r32, r33 ]

# 可选：平移向量 (3x1)
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ tx, ty, tz ]
```

---

## 快速开始（使用默认占位符）

如果还没有标定，可以先使用默认参数测试系统：

```bash
# 运行标定脚本
cd ~/NX_volleyball/calibration
python3 capture_chessboard.py
```
