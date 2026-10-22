# 灰度 ROI 深度候选

## 图像访问失败不是算法无匹配

需要 CPU Gray 的候选会成对锁定左右校正 VPI image。只有两侧 lock 成功，且地址、宽高、pitch 均覆盖整张校正图时，才把 host pointer 交给候选算法；任一 lock、view 校验、算法异常或后续 unlock 失败都会先释放已取得的 lock，再使整帧进入失败契约。它不会被编码成某个候选 `valid=false`，也不会让 HybridDepth 把基础设施故障当成一次正常漏测继续预测。纯算法意义上的“纹理不足、支持点不足、视差门限拒绝”仍保留为候选级无效结果，两类状态必须分开统计。

## 主线输入

所有当前主线 ROI 深度算法使用已经双目标定校正的左右 Gray8 图像。校正后理想对应点应主要沿 x 方向变化，视差通过标定焦距和基线转换为深度；检测框只提供初始对应和有限搜索区，不应被当作亚像素测量本身。

## 当前有效算法层级

最终投影会启用多种候选，但它们的作用不同：

| 候选 | 计算 | 输入/实现 | 能否成为当前首选或旧回退 |
|---|---|---|---|
| bbox center/edges | 是 | 左右框几何 | 旧回退可选 |
| circle center | 是 | 灰度 ROI Sobel + 加权圆拟合 | 旧回退可选 |
| edge centroid | 是 | Sobel 边缘加权质心 | 旧回退可选 |
| radial center | 是 | ROI 径向/圆几何代理 | 旧回退可选 |
| edge-pair center | 是 | 左右边缘中点 | 旧回退可选 |
| center patch | 是 | 中心小块 ZNCC 搜索 | 记录候选，不在当前旧回退集合中 |
| multi-point subpixel | 是 | 多点 ZNCC + 稳健聚合 | 当前首选 |
| CUDA template/NCC | 是 | 自定义 CCOEFF_NORMED CUDA | 记录/训练候选，不在当前最终选择集合中 |
| epipolar fallback | 是 | 单侧检出时的极线搜索/圆代理 | 旧回退可选 |
| ORB/BRISK/AKAZE/SIFT 等 | 否 | 稀疏关键点 | 当前关闭 |
| XFeat/SuperPoint/ALIKED | 否 | TensorRT 神经特征 | 当前关闭 |
| CUDA BM/SGM、VPI disparity | 否 | 稠密或半稠密视差 | 当前关闭 |
| 两个 color-IoU 候选 | 否 | BGR ROI 颜色/边缘 | 当前关闭 |

“计算”一列指当前最终旧 YAML 会启动相应代码路径；是否最终发布还要看候选选择规则。

## GPU 几何批处理

对通过 gate 的 pair，GPU batch 以每对框一个 CUDA block 的方式计算多个便宜候选：

1. 在框内计算 Sobel 梯度；
2. 以 `max(10, 0.25 * max_gradient)` 形成自适应边缘阈值；
3. 计算梯度加权边缘质心；
4. 计算最左/最右有效边缘中点；
5. 进行加权圆拟合；
6. 产生 bbox、circle、edge centroid、radial、edge-pair 等视差代理。

这些方法不提取可描述、可跨帧跟踪的 SIFT/ORB 特征点；其“点”是几何中心或边缘统计量。

## center patch 与 multi-point

中心 patch 使用半径 5 的左图小块，在右图初始视差附近 ±8 px 范围做 ZNCC 搜索。

多点亚像素会在球体内部和不同半径位置选取最多 12 个 patch，至少需要 4 个可靠点。每个点经过：

- y 残差约束；
- 反向匹配检查；
- 椭圆覆盖约束；
- 由球半径得到的物理范围约束；
- 与初始视差的绝对/相对差 gate；
- MAD/RANSAC 稳健剔除；
- 最终标准差、支持数和置信度 gate。

当前阈值包括 `min_confidence=0.25`、`max_stddev=1 px` 和 `time_budget=1.5 ms`。多点输出是对未来 3D 最有价值的等效视差，不要求识别球面接缝或真实旋转。

## 自定义 CUDA template/NCC

该分支在左框中心取 11×11 patch，在右图预计位置附近搜索，计算归一化相关并在 GPU 上归约峰值。正常模式只把很小的峰值结果传回 host；调试模式才可能保存 patch。

环境变量可切换到 OpenCV CUDA baseline 做对照：

```bash
STEREO_CUDA_TEMPLATE_BACKEND=opencv
```

当前它虽启用计算，但 `DepthCandidateMethod::ROI_CUDA_TEMPLATE_MATCH` 不属于最终旧回退候选集合，因此主要用于候选记录、离线比较和后续训练，不应把它的计数误报成最终发布深度来源。

## 最终候选选择

当前 `depth_solver=roi_subpixel_match` 会把 `ROI_MULTI_POINT` 设为 preferred：

1. 多点候选有效时，作为直接输出；
2. 多点无效时，调用旧候选选择器；
3. 旧选择器按候选向量顺序取第一个有效且被列为 legacy output 的候选；向量前部还保留 `fallback_feature_points` 和 `fallback_template` 两个历史入口，但主线投影会把它们明确关闭，因此当前可达顺序才是 epipolar fallback/circle center，之后是 radial、edge-pair、edge-centroid 和 bbox 系列；
4. center patch、CUDA template 和大部分稀疏/神经特征候选即使有效，也不会由这条旧选择器发布为最终 `z`。

根配置把三件事分开声明：`candidate_methods` 决定实际计算集合，`primary_method` 决定首选，`fallback_methods` 决定首选失败后的允许顺序。适配器会先关闭基础模板中的全部 `depth_modes`，再显式开启候选集合，因此算法开关不会从模板隐式继承。

旧 C++ 当前仍只有两个首选选择器，并具有固定的 legacy fallback 相对顺序。配置加载器会拒绝无法忠实投影的首选、漏列候选或乱序回退；这比接受配置后悄悄执行另一套算法更安全。

## “黑白 IoU 图像”到底是什么

当前主线没有这种数据产品。容易混淆的是一个已关闭的 CPU 大 ROI 归一化辅助过程：

1. 取覆盖检测框和搜索范围的共享矩形；
2. 缩放到目标直径；
3. 依据检测框绘制椭圆 label：球内 255、球外 0；
4. 用球内平均灰度填充 label 外区域；
5. 可再计算 Sobel magnitude edge map。

这里的 0/255 图是椭圆 mask，和 shifted-IoU 无关。当前关闭的 corner/texture/binary/ORB/BRISK/AKAZE/SIFT 等 CPU 分支才可能使用此类归一化图。若未来启用，应在接口中命名为 `roi_mask`、`normalized_gray`、`edge_map`，不要叫 `iou_image`。

## 推荐的候选评估方式

每种方法都应分别统计：

- attempted / valid / selected 比例；
- disparity 与多点/标定真值的差、标准差和长尾；
- 支持点数、反向匹配误差和极线残差；
- 按距离、框直径、曝光拖影、遮挡和左右置信度分桶的误差；
- GPU 时间、D2H 字节和超时比例；
- 是否真正改善最终 3D，而不只是候选自身看起来平滑。

