# 双目深度测量方案报告

> 平台：Jetson Orin NX Super 16GB | JetPack 6.2 | 双目海康 MV-CA016-10UC  
> 基线：328.7mm | 焦距：1485.5px (校正后) | 分辨率：1440×1080 → 校正后 1280×720  
> 目标范围：0-15m | 排球直径：23cm | 帧率：~90Hz  
> 相机序列号: LEFT=00219471413 (检测), RIGHT=00D39342665 (辅助)  
> 最后更新：2025-07

## 0. 距离-视差-单目分析

### 0.1 视差 vs 距离计算表

```
公式: disparity = focal × baseline / Z = 1485.5 × 0.3287 / Z

距离(m)  |  视差(px)  |  排球像素宽  |  当前方法
---------|-----------|-------------|----------
  1.0    |   488.4   |   341.9     |  单目 (视差超限)
  1.5    |   325.6   |   227.9     |  单目
  2.0    |   244.2   |   170.9     |  单目
  3.0    |   162.8   |   113.9     |  单目 (z_pred < stereo_min_z)
  4.0    |   122.1   |    85.4     |  融合过渡带 [3,5]m
  5.0    |    97.7   |    68.3     |  双目 (z_pred > mono_max_z)
  7.0    |    69.8   |    48.8     |  双目
 10.0    |    48.8   |    34.2     |  双目
 15.0    |    32.6   |    22.8     |  双目 (接近检测极限)

排球像素宽 = focal × 0.23m / Z
max_disparity = 512 (覆盖 Z ≥ 0.95m)
```

### 0.2 关键发现

- **max_disparity=512** 理论覆盖 Z ≥ 0.95m，但近距离排球充满画面导致圆拟合困难
- **实际双目有效范围**: ~1.5m 到 15m（circle-fit 成功率 > 80%需 Z ≥ 2.5m）
- **0-3m**: 使用单目估计（排球像素大，BBox 精度高）
- **3-5m 过渡带**: 线性融合单目+双目
- **5-15m**: 纯双目（circle-fit 成功率 ~100%）

## 1. 系统架构

### 1.1 当前生产配置 (pipeline_yolo26_mixed.yaml)

```yaml
stereo:
  strategy: roi_only           # 仅在检测到的 BBox 内算深度
  max_targets: 32              # 最多同时测 32 个目标
  max_disparity: 512           # 搜索范围 512px
  object_diameter: 0.200       # 排球等效直径 (m) — 3轮标定优化值
  use_circle_fit: true         # 使用 Kasa 圆拟合 (替代 SAD)
  min_confidence: 0.1          # 最低置信度阈值
```

### 1.2 混合测距架构

```
输入: YOLO 检测结果 (BBox) + 校正后左右图
          |
    +---- Kalman 预测 z_pred ----+
    |                             |
    v                             v
 z_pred < 3m?                  z_pred >= 3m?
    |                             |
    v                             v
 [单目测距]                   [双目 Circle-Fit]
 Z = f * 0.23 /              Sobel 边缘 + Kasa 圆拟合
 (w * 0.95)                   Z = f * B / disparity
    |                             |
    +--------+--------------------+
             v
       3-5m 过渡带?
        YES -> alpha = (5 - z_pred) / 2
               z = alpha*z_mono + (1-alpha)*z_stereo
             v
       [9D Kalman Filter]
       状态: [x, y, z, vx, vy, vz, ax, ay, az]
       观测: [x, y, z] (像素反投影)
       R_z: 单目=0.25, 双目=0.01, 过渡=线性插值
       R_xy = Rz * z^2 / f^2 + 0.001
             v
       输出: 3D坐标 + 速度 + 加速度, depth_method
```

## 2. 双目 Circle-Fit 算法 (roi_circle_match.cu)

### 2.1 为什么不用 SAD

排球表面白色/彩色条纹，大面积无纹理区域导致 SAD 匹配完全失效:
- 5×5 网格采样，仅 1/25 点通过唯一性检验
- 视差分布双峰 (1.42m ~45%, 1.63m ~35%, -1.00 ~20%)
- **结论**: 对无纹理球体，基于局部纹理的 SAD 不可用

### 2.2 Circle-Fit 算法流程

```
输入: 校正后左右图 + BBox (一个排球 = 一个 CUDA Block)
      |
Phase 1: Sobel 梯度计算
  - 左右图 BBox 区域: |Gx| + |Gy| → gradient magnitude
  - 自适应阈值: max(gradient) * 0.3
      |
Phase 2: 左图 Kasa 加权最小二乘圆拟合
  - 对梯度 > 阈值的边缘像素, 权重 = gradient
  - Kasa 方程: minimize sum_w[(xi-cx)^2 + (yi-cy)^2 - r^2]^2
  - 输出: (cx_L, cy_L, r_L), 残差 rms
      |
Phase 3: 右图窄带搜索
  - 搜索范围: d in [0, max_disparity]
  - 右图 BBox 偏移 d 后执行相同边缘提取
  - 最小化: |r_R - r_L| + |cy_R - cy_L| (半径+y轴一致性)
      |
Phase 4: 右图半径约束圆拟合
  - 在最佳偏移处精确拟合
  - 约束: |r_R - r_L| < r_L * 0.2 (半径一致性)
      |
Phase 5: 视差与三角测距
  - disparity = cx_L - cx_R
  - Z = focal * baseline / disparity
  - X = (cx_L - cx0) * Z / focal
  - Y = (cy_L - cy0) * Z / focal
  - confidence = 基于残差和半径一致性
      |
输出: Object3D { x, y, z, confidence }
```

### 2.3 性能实测

| 指标 | 数值 | 说明 |
|------|------|------|
| 单目标延迟 | ~0.99ms | Sobel + 两次 Kasa 拟合 + 搜索 |
| GPU 占用率 | <2% | 仅 1-2 个 Block |
| 总 pipeline | ~90fps | 含 YOLO + Remap + Circle-fit + Kalman |

### 2.4 精度与匹配率 (60秒多距离实测, 5559帧)

| 距离段 | 帧数 | 匹配成功率 | z_stereo 稳定性 |
|--------|------|-----------|----------------|
| < 2.0m | ~300 | ~60% | 偏移较大 |
| 2.0-2.5m | ~800 | ~85% | 中等 |
| 2.5-4.0m | ~2000 | ~98% | 稳定 ±0.05m |
| 4.0-6.0m | ~2400 | ~100% | 稳定 ±0.03m |
| 总体 | 5559 | **~90%** | — |

**关键发现: zs/zm 比例恒定 ≈ 0.946 (5.4%系统偏移)**

## 3. 系统偏差分析

### 3.1 偏差测量结果

在 1.5m - 5.5m 全距离范围, 双目测距系统性偏低 5.4%:

```
zs / zm ≈ 0.946  (恒定, 不随距离变化)
```

可能原因:
1. 标定参数微小误差 (焦距/基线 ~2-3% 偏差可导致此结果)
2. 校正后图像仍有轻微残余畸变
3. 圆拟合对边缘提取的系统倾向 (Sobel 梯度方向偏差)

### 3.2 误差模型推导

两种测距方法的方差均与 z^4 成正比:

**单目误差** (BBox 像素量化):
$$\sigma^2_{mono} = \frac{z^4}{f^2 \cdot D^2} \cdot \sigma^2_w$$
其中 $\sigma_w \approx 3px$ (BBox宽度抖动), $f=1485.5$, $D=0.23m$

$$\sigma^2_{mono} \approx \frac{z^4}{1485.5^2 \times 0.23^2} \times 9 = \frac{z^4}{12978}$$

**双目误差** (视差量化):
$$\sigma^2_{stereo} = \frac{z^4}{f^2 \cdot B^2} \cdot \sigma^2_d$$
其中 $\sigma_d \approx 1px$ (圆心定位抖动), $B=0.3287m$

$$\sigma^2_{stereo} \approx \frac{z^4}{1485.5^2 \times 0.3287^2} \times 1 = \frac{z^4}{238548}$$

**方差比**:
$$\frac{\sigma^2_{mono}}{\sigma^2_{stereo}} \approx \frac{238548}{12978} \approx 18.4$$

> 结论: 双目精度约为单目的 4.3 倍 ($\sqrt{18.4}$)。如果 $\sigma_w = 7px$，比值可达 ~90。

### 3.3 当前融合参数

| 参数 | 值 | 含义 |
|------|-----|------|
| R_mono | 0.003 | 单目观测噪声基值，R(z)=0.003×z² |
| R_stereo | 0.020 | 双目观测噪声基值，R(z)=0.020×z² |
| ivw_R_mono | 0.004 | IVW单目噪声方差 |
| ivw_R_stereo | 0.025 | IVW双目噪声方差（方差比≈6:1） |
| stereo_min_z | 3.0m | 双目最近有效距离 |
| mono_max_z | 5.0m | 单目最远有效距离 |
| process_accel | 50.0 m/s^2 | 过程噪声 sigma_a |
| bbox_scale | 0.95 | BBox vs 实际球体比例 |

## 4. 9D Kalman 滤波器

### 4.1 状态模型 (恒加速)

状态向量 $\mathbf{x} = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T$ (9维)

预测:
$$\mathbf{x}_{k+1} = F \mathbf{x}_k, \quad F = \begin{bmatrix} I_3 & dt \cdot I_3 & \frac{dt^2}{2} I_3 \\ 0 & I_3 & dt \cdot I_3 \\ 0 & 0 & I_3 \end{bmatrix}$$

观测: $\mathbf{z} = H \mathbf{x} = [x, y, z]^T$, $H = [I_3 \; 0 \; 0]$

### 4.2 噪声设计

过程噪声: $Q = \sigma_a^2 \cdot G G^T$, $G = [\frac{dt^2}{2} I_3;\; dt \cdot I_3;\; I_3]$

观测噪声: $R = \text{diag}(R_{xy}, R_{xy}, R_z)$
- $R_z$ 由 `getObsNoise(z, method)` 动态计算
- $R_{xy} = R_z \cdot z^2 / f^2 + 0.001$ (误差传播)

### 4.3 丢帧处理

| 丢帧数 | 处理 |
|--------|------|
| 0 | Kalman update (正常) |
| 1-5 | 纯 predict, 置信度递减 |
| 5-20 | 降级为"预测"状态 |
| > 40 | 删除 track |

## 5. 已知问题与待优化

### 5.1 已确认已完成

- [x] 双目标定完成 (焦距 1485.5px, 基线 328.7mm)
- [x] CUDA Bayer→BGR + Remap 校正
- [x] Circle-Fit CUDA kernel 完成并验证 (替代 SAD)
- [x] 90% 匹配率, 100% at 2.5m+
- [x] 9D Kalman 滤波实现并验证
- [x] 单目+双目混合测距 pipeline
- [x] CSV 轨迹录制 (z_mono, z_stereo, depth_method)
- [x] ~90 FPS 实时追踪

### 5.2 待解决

| 问题 | 严重程度 | 状态 |
|------|---------|------|
| 5.4% 系统偏差 (zs/zm=0.946) | 中 | 需偏差校正 |
| 融合策略优化 | 中 | 研究中 (6方案已评估) |
| hybrid_depth.h 注释过期 | 低 | "Z<4m"应为"Z<3m" |

### 5.3 融合策略候选方案

| 方案 | 核心思想 | 复杂度 | 风险 |
|------|---------|--------|------|
| D: 偏差校正 + IVW | z_stereo *= 1.057, 逆方差加权 | ~20行 | 极低 |
| A: Sage-Husa 自适应 R | 用 innovation 在线估计 R | ~80行 | 低 |
| E: 在线交叉校准 | EMA 跟踪 zs/zm 比例 | ~40行 | 低 |
| B: KalmanNet | RNN 增强 Kalman gain | ~500行+训练 | 中 |
| C: LSTM 端到端融合 | 序列输入→深度输出 | ~1000行+训练 | 高 |
| F: Covariance Intersection | 保守融合, 无需知道相关性 | ~60行 | 低 |

## 6. Pipeline 10ms 预算分配

```
+-------------------------+----------+--------------------------------+
| Stage                   | Time(ms) | Hardware                       |
+-------------------------+----------+--------------------------------+
| Camera Grab             | ~0.50    | USB3 DMA                       |
| CUDA Bayer->BGR+Remap   | ~2.84    | GPU CUDA                       |
| TRT Preprocess          | ~1.00    | GPU                            |
| TRT Inference (FP16)    | ~2.93    | GPU                            |
| TRT Postprocess         | ~0.30    | GPU                            |
| Circle-Fit Stereo       | ~0.99    | GPU CUDA                       |
| Mono+Kalman Fusion      | ~0.05    | CPU                            |
+-------------------------+----------+--------------------------------+
| Total                   | ~8.6     | < 11ms -> ~90fps               |
+-------------------------+----------+--------------------------------+
```
