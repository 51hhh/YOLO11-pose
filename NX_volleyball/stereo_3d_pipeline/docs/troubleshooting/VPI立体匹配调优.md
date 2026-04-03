# 问题速查：VPI 立体匹配调优

> 环境: VPI 3.2.4 · Orin NX · CUDA backend

## VPI CUDA SGM 推荐参数

```yaml
stereo:
  backend: cuda
  max_disparity: 128
  p1: 3          # 小梯度惩罚
  p2: 48         # 大梯度惩罚
  quality: 6     # SGM passes (5-8)
  confidence: 32768  # 0=关闭
```

## 常见问题

### 有效率仅 30% (OpenCV CUDA SGM)
OpenCV CUDA SGM 与 VPI SGM 参数体系不同：
- `num_disparities` 必须是 16 的倍数
- `block_size` 对 CUDA 版影响不同
- **建议用 VPI SGM 替代** （99% 有效率 vs 30%）

### VPI OFA 深度异常偏低
OFA (Optical Flow Accelerator) 4× 降采样后深度精度下降：
- 中位深度 1734mm（预期 ~6000mm）
- 仅适合近距离粗略深度估计
- **不推荐用于排球追踪**

### Stage 2 显示 0.02ms
VPI 异步提交仅记录 enqueue 时间，实际 GPU 计算在后续 sync 中完成。
真实 VPI SGM 耗时约 60-65ms（从 benchmark 数据获取）。

## 后处理增强

### WLS 滤波（需 ximgproc）
```cpp
auto wls = cv::ximgproc::createDisparityWLSFilter(left_matcher);
wls->setLambda(8000.0);
wls->setSigmaColor(1.5);
wls->filter(left_disp, left_img, filtered, right_disp);
```
效果：边缘更锐利，噪点大幅减少，有效率提升 ~15-20%。

### Census 变换预处理
```cpp
// 5×5 Census: 对中心像素比较24个邻居，输出24bit特征
// 对光照变化鲁棒（排球场强灯光）
```

## 性能参考

| 算法 | 延迟(ms) | FPS | 有效率 | 推荐场景 |
|---|---|---|---|---|
| VPI CUDA SGM | 64 | 15.6 | 99% | 主力 |
| OpenCV CUDA BM | 20 | 50 | 66% | 低延迟 |
| VPI OFA ds4 | 39 | 25 | 100% | 近距离 |
