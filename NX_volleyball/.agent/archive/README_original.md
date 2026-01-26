好的，我们现在**聚焦于纯海康双目方案**。

鉴于你使用 **5mm 广角镜头** 配合 **25cm 短基线**，还要覆盖 **0-15m** 的范围，这个方案的核心难点在于：**如何在远距离（15m）像素极少的情况下，依然保持高精度的检测和深度解算。**

传统的“缩放全图检测”在这里会失效（球会变成噪点）。我们必须采用 **“Coarse-to-Fine（由粗到精）” + “稀疏立体视觉”** 的架构。

以下是基于 **Orin NX 16GB** 的完整实现方案。

---

# 🏐 海康双目高速排球追踪系统 (Pure Stereo Edition)

## 1. 硬件架构与电气连接

### 1.1 物理参数
*   **相机**: 2x Hikvision MV-CA016-10UC (IMX273, 1440x1080, Global Shutter).
*   **镜头**: 2x 5mm (水平 FOV ~52°).
*   **基线**: 25cm (固定刚性连接).
*   **安装高度**: 建议 2m - 3m (避免球员遮挡).

### 1.2 同步触发 (至关重要)
双目视觉的前提是“同一时刻的图像”。软触发（软件发指令）会有几毫秒的误差，导致高速球深度计算错误。**必须使用硬触发。**

*   **连线**:
    *   Orin NX GPIO (输出 3.3V 脉冲) -> 电平转换(如需要) -> 相机1 Line0 & 相机2 Line0 (并联).
    *   或者：相机1 (Strobe输出) -> 相机2 (Trigger输入).
*   **MVS 设置**:
    *   `Trigger Mode`: On
    *   `Trigger Source`: Line0
    *   `Exposure Time`: < 1000µs (1ms) 以防止运动模糊.
    *   `Gain`: 自动或根据环境调高.

---

## 2. 软件算法流水线 (Pipeline)

为了在 Orin NX 上跑满 150+ FPS 且看清 15m 外的球，我们采用 **状态机模式**。

### 2.1 状态机设计
系统在两种模式间切换：
1.  **全图搜索模式 (Global Search)**: 当不知道球在哪时（初始或跟丢）。
2.  **ROI 锁定模式 (ROI Tracking)**: 当知道球大概在哪时（主要工作模式）。

### 2.2 详细流程图

```mermaid
graph TD
    Start[硬触发采集 L/R 图像] --> CheckState{当前状态?}
    
    %% 模式 1: 全图搜索
    CheckState -- 丢失/初始 --> Global[全图 Downsample (640x640)]
    Global --> DetGlobal[YOLO Detect]
    DetGlobal -- 无目标 --> OutputNone[输出: 无]
    DetGlobal -- 有目标 --> InitKF[初始化卡尔曼 & 切换到 ROI模式]
    
    %% 模式 2: ROI 锁定
    CheckState -- 锁定中 --> Predict[卡尔曼预测下一帧位置 (x,y)]
    Predict --> Crop[在原图 1440x1080 裁切 320x320 ROI]
    Crop --> DetROI[YOLO Detect (Batch=2)]
    
    %% 后处理
    DetROI --> MapBack[坐标还原回原图系]
    InitKF --> MapBack
    MapBack --> Undistort[关键点去畸变]
    Undistort --> Stereo[三角测量 (Triangulation)]
    Stereo --> UpdateKF[卡尔曼更新 (3D位置+速度)]
    UpdateKF --> Output[输出 3D 坐标]
```

---

## 3. 核心代码实现逻辑

### 3.1 坐标还原与去畸变 (Sparse Undistortion)
不要对整张图做去畸变（太慢）。只对检测到的 5 个关键点做数学去畸变。

```python
import cv2
import numpy as np

class StereoProcessor:
    def __init__(self, calib_file):
        # 加载标定参数 (K: 内参, D: 畸变, R/T: 外参)
        data = np.load(calib_file)
        self.K1, self.D1 = data['K1'], data['D1']
        self.K2, self.D2 = data['K2'], data['D2']
        self.P1, self.P2 = data['P1'], data['P2'] # 投影矩阵

    def process_points(self, pts_l_raw, pts_r_raw):
        """
        输入: 原始图像上的像素坐标 (N, 2)
        输出: 世界坐标系下的 3D 坐标 (X, Y, Z)
        """
        # 1. 关键点去畸变 (输入必须是 float32)
        # 这一步将“鱼眼”视角的点拉直，对应到理想针孔模型
        pts_l_undist = cv2.undistortPoints(pts_l_raw, self.K1, self.D1, P=self.P1)
        pts_r_undist = cv2.undistortPoints(pts_r_raw, self.K2, self.D2, P=self.P2)

        # 2. 三角测量
        # 输出 4D 齐次坐标 (x, y, z, w)
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pts_l_undist, pts_r_undist)
        
        # 3. 归一化
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T # 返回 [X, Y, Z]
```

### 3.2 ROI 裁切策略 (核心优化)

这是 5mm 镜头能看清 15m 目标的秘密武器。

```python
def get_roi_images(full_img_l, full_img_r, predicted_center, crop_size=320):
    """
    从 1440x1080 原图中裁切 320x320 的小图
    """
    cx, cy = int(predicted_center[0]), int(predicted_center[1])
    h, w = full_img_l.shape[:2]
    
    # 计算裁切边界，防止越界
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    
    # 修正 x1, y1 以保证尺寸一致 (边界情况)
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)

    # 裁切
    roi_l = full_img_l[y1:y2, x1:x2]
    roi_r = full_img_r[y1:y2, x1:x2]
    
    offset = (x1, y1) # 记录偏移量，用于还原坐标
    return roi_l, roi_r, offset

# 在推理后还原坐标
def map_coords_back(roi_keypoints, offset):
    """
    roi_keypoints: [x, y] in 320x320
    offset: (off_x, off_y)
    """
    original_keypoints = roi_keypoints + np.array(offset)
    return original_keypoints
```

---

## 4. 部署步骤与 Orin NX 优化

### 4.1 模型准备
你需要训练两个 YOLO 模型（或者同一个模型支持不同尺度）：
1.  **Global Model**: 输入 640x640，检测小目标（远处的球）。
2.  **ROI Model**: 输入 320x320，检测大目标（裁切后的球）。
*   *建议*: 直接训练一个强力的 640x640 模型，ROI 推理时虽然输入是 320，但可以 Resize 到 640 喂给模型，或者直接导出支持动态尺寸的 TensorRT 引擎。

**TensorRT 导出命令 (支持动态 Batch 和 动态尺寸)**:
```bash
trtexec --onnx=yolo_pose.onnx \
    --saveEngine=yolo_pose.engine \
    --fp16 \
    --minShapes=images:1x3x320x320 \
    --optShapes=images:2x3x640x640 \
    --maxShapes=images:2x3x640x640
```

### 4.2 性能压榨 (Zero-Copy)
在 Python 中处理 1440x1080 的图像拷贝非常耗时。
1.  **MVS SDK**: 配置为直接输出 Bayer 格式。
2.  **CUDA 处理**: 使用 PyCUDA 或 Cupy，直接在 GPU 上做 `Bayer -> RGB` 和 `Crop`。
3.  **推理**: TensorRT 输入直接指向 GPU 上的 Crop 区域。

---

## 5. 精度与误差分析 (25cm 基线的物理极限)

你必须接受物理定律带来的限制。

| 距离 (Z) | 视差 (d) | 理论深度误差 (ΔZ) | 实际表现 |
| :--- | :--- | :--- | :--- |
| **3m** | ~120 px | ~1.2 cm | **极高精度**，可判断触网。 |
| **9m** (网前) | ~40 px | ~11 cm | **高精度**，轨迹平滑。 |
| **15m** (底线) | ~24 px | ~31 cm | **中等精度**，Z轴会有抖动。 |

**优化策略**:
在卡尔曼滤波中，根据距离动态调整观测噪声协方差矩阵 $R$。
```python
# 伪代码
if z_measured < 5.0:
    R = diag([0.01, 0.01, 0.01]) # 近处，非常信任视觉测量
elif z_measured > 12.0:
    R = diag([0.1, 0.1, 0.5])    # 远处，Z轴误差大，更信任预测值
```

---

## 6. 总结：这套方案能成吗？

**能成，但对软件工程要求很高。**

1.  **光学**: 5mm 镜头解决了“跟丢”的问题，是正确的选择。
2.  **算力**: Orin NX 跑 ROI 模式（320x320）非常轻松，甚至可以达到 200+ FPS。
3.  **精度**: 25cm 基线在 15m 处确实有 30cm 左右的物理误差。但这对于记录“球的轨迹”和“速度”已经足够。如果是为了判罚“压线球”，这个精度是不够的（需要多目或更长基线），但作为训练分析系统完全合格。

**下一步行动**:
1.  固定相机，连接触发线。
2.  采集棋盘格图片，跑通 OpenCV 的 `stereoCalibrate`。
3.  先写一个简单的 Python 脚本，不带 ROI，测试近距离（3m）的测距精度。
4.  加入 ROI 逻辑，测试远距离追踪。