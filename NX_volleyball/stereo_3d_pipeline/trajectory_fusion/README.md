# 轨迹可靠性融合实验

这个目录用于验证“多种深度测量 + 历史序列 + 物理约束”的后处理/半在线融合方案。它不替代当前实时双目链路，第一阶段只做离线实验，确认收益后再把轻量部分接回 `HybridDepthEstimator`。

## 结论

最终模型不应该是一个直接生成真实轨迹的黑盒生成器，也不应该只输出一个“深度是否可信”的二分类结果。推荐结构是：

```text
多路观测/质量特征/历史序列
        |
MeasurementReliabilityNet
        |-- 每种测距方法的 log_sigma
        |-- 每种测距方法的 bias
        |-- 每种测距方法的 outlier_prob
        |-- 可选 common_jitter
        v
鲁棒 EKF/UKF/RTS 物理滤波器
        |
最终 x/y/z/vx/vy/vz 轨迹和预测
```

神经网络负责学“什么时候信谁”，滤波器负责输出物理一致的轨迹。这样比 GAN 或端到端轨迹生成更适合当前场景：没有真实轨迹标签，且在线系统必须可解释、可限幅、可回退。

## 为什么不以 GAN 为主

- 没有真实轨迹标签时，GAN 的判别器很容易只学习“看起来平滑”，不能保证绝对深度正确。
- 对 100 FPS 在线闭环来说，GAN/扩散类生成器的延迟、稳定性和失效解释都不好控制。
- 当前噪声主要来自 bbox 抖动、视差误匹配、远距视差量化、左右漏检和同步异常，更适合建模成观测噪声、偏差和离群概率。

GAN/扩散可以作为离线数据增强或论文对照组，不建议作为主路线。

## 参考方向

已检索并纳入设计判断的方向：

- KalmanNet, `arXiv:2107.10043`: 用 RNN 辅助 Kalman，在部分模型已知时学习滤波增益。可作为后续强 baseline，但第一版不直接替代 EKF。
- Unsupervised Learned Kalman Filtering, `arXiv:2110.09005`: 用观测预测误差做无监督训练，适合没有真值轨迹的条件。
- DANSE, `arXiv:2306.03897`: 无监督数据驱动非线性状态估计，表达能力强，但工程复杂度高。
- Deep Kalman Filters, `arXiv:1511.05121` 和 DVBF, `arXiv:1605.06432`: 变分状态空间模型，适合研究对照，不适合作为第一版实时方案。
- Multi-sensor Student's t Filter, `arXiv:2204.11098`: 多传感器重尾噪声/离群鲁棒融合，直接支持当前多测距值融合。
- OOSTraj, `arXiv:2404.02227`: 无监督 noisy trajectory denoising，说明“无真值轨迹 + 轨迹去噪”方向可行，但场景和约束不同。
- `jonasTorz/physical_spline`: 物理样条轨迹拟合思路，可作为离线平滑对照。
- `Lukas-Kozel/KalmanNet-for-state-estimation`, `DanieleGammelli/DeepKalmanFilter`, `gregorsemmler/pytorch-dvbf`: 可复现项目参考。

## 损失设计

没有真实轨迹标签时，训练目标要避免“平滑但错深度”的退化。推荐组合：

1. 观测似然损失
   对每个有效测距方法 `m`，网络输出 `sigma_m`、`bias_m`、`outlier_m`。使用 Student-t 或 Huber NLL：

   ```text
   r_m = z_m - bias_m - z_state
   loss_obs = StudentT_NLL(r_m, sigma_m) * (1 - outlier_prob_m)
   ```

2. 留一法一致性
   随机遮住一种测距方法，用其他方法和历史预测当前状态，再检查被遮住观测的创新分布是否合理。这个比简单平均更能学到互补关系。

3. 物理约束
   对状态序列施加排球飞行约束：

   ```text
   x'' ~= 0
   z'' ~= 0
   y'' ~= g_camera
   jerk 不应长期过大
   ```

   如果相机坐标未严格对齐重力，先把 `g_camera` 作为可配置参数或小范围可学习偏置。

4. 不确定度校准
   归一化创新 `r/sigma` 应接近单位尺度，避免网络把所有 `sigma` 放大逃避误差。

5. 离群稀疏正则
   `outlier_prob` 允许处理错匹配和 YOLO 抖动，但不能每帧都判离群。

## 评估方式

必须同时评估准确性、稳定性、物理一致性和实时可用性：

- 静止球：`z` 标准差、峰峰值、短窗漂移、不同测距方法之间的偏差。
- 动态球：速度/加速度/jerk RMS，落点预测稳定性，深度跳变率。
- 留一法：遮住 mono/stereo/subpixel/circle 后的重建误差。
- 创新校准：`abs(residual) / sigma` 的分布是否接近预期。
- 异常场景：单目漏检、右目漏检、fallback、同步 delta 异常、远距离小视差。
- 在线成本：特征构建 + 网络推理 + EKF update 必须小于单帧预算，目标 < 0.3 ms。

## 文件

- `schema.md`: 未来采集多路观测时需要记录的字段。
- `dataset.py`: 读取旧 CSV 和未来扩展 CSV 的基础工具。
- `robust_smoother.py`: 旧 CSV 可运行的鲁棒物理平滑 baseline。
- `models.py`: 可靠性网络骨架，只输出噪声/偏差/离群概率，不直接输出最终轨迹。
- `losses.py`: 自监督观测似然、物理约束和正则项。
- `train_reliability.py`: 自监督训练脚手架。
- `evaluate_fusion.py`: 对原始/平滑输出做稳定性和物理一致性评估。

## 当前限制

现有 `TrajectoryRecorder` 只记录 `x,y,z,z_mono,z_stereo,depth_method,confidence`。它缺少训练可靠性模型最关键的质量特征，例如 bbox、左右圆心、视差、ROI 支撑点、亚像素标准差、fallback 标志、同步差值和水印帧号。因此：

- 第一版只能训练/评估很弱的可靠性模型。
- 真正要让模型学会“哪种深度在当前帧可信”，必须按 `schema.md` 扩展在线记录器。
- 在未完成新基线标定前，所有绝对深度指标只能做趋势判断，不能当最终精度结论。

## 已验证的旧 CSV baseline 结果

使用当前仓库里未跟踪的 `raw_observation_data.csv` 做脚本连通性验证：

```text
track=0 raw:    z_std=0.2307, dz_rms=0.0038, ddz_rms=0.0030, jerk_rms=0.0047
track=0 smooth: z_std=0.2312, dz_rms=0.0027, ddz_rms=0.0009, jerk_rms=0.0011
```

结论：旧 CSV 已经是在线 Kalman 输出，不是完整原始多路观测，所以离线 smoother 主要降低高阶抖动，不能显著降低整体 `z_std`。这不是模型路线失败，而是数据字段不足；后续必须记录 `schema.md` 里的原始测距和质量特征。
