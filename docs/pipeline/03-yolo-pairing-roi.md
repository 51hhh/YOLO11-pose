# 双路 YOLO、左右配对和 ROI

## 双路检测

当前左右相机各有一个 `TRTDetector`，共享同一 TensorRT engine，但各自使用独立 CUDA stream。每个 detector 又按三缓冲槽创建独立 execution context、输入、输出和 pinned host 缓冲，防止异步帧互相覆盖。

当前参数：

| 参数 | 值 |
|---|---:|
| 输入 | 左右校正 BGR8 |
| 网络尺寸 | 640×640 letterbox |
| 置信度阈值 | 0.30 |
| NMS IoU | 0.40 |
| 每侧最大检测数 | 10 |
| 后端 | TensorRT GPU |
| C++ SOT 补帧 | 关闭，因此每帧双路 YOLO |

推理后处理支持两类 engine：

- 多尺度 6-output DFL：GPU 解码每个尺度，只拷回小型候选表，CPU 做 NMS；
- 单输出：异步 D2H 后由 CPU 解析/NMS；若 engine 已内置 NMS，则直接读取结果。

启动时不会只凭文件扩展名假定 engine 可用。反序列化后必须满足当前实现的 I/O 契约：恰好一个静态 float32 NCHW `[1,3,S,S]` 正方形输入；输出必须是受后处理支持的 `[1,C,N]` / `[1,N,C]` pre-NMS、`[1,N,6]` post-NMS，或三尺度成对的 6 个 NHWC DFL tensor。动态 shape、矩形输入、多输入、额外未绑定输出、非 float32 以及不完整 DFL 尺度会直接阻止启动。左右 engine 的实际 `S` 还必须分别等于 profile 的 `detector.input_resolution`，因此配置、模型清单和运行 engine 不会静默分叉。

在 engine 文件被读取之前，配置加载器还会核对模型清单中的 artifact 格式、输入尺寸/颜色格式、目标平台和 accelerator。当前清单明确是 Orin NX CUDA FP16 engine；把 profile 改成 DLA 不会只改一个布尔值后继续使用这份 GPU engine，而会在配置阶段被拒绝。

`evtDetectDone` 在预处理、推理和所需 D2H 操作都排入同一 stream 后记录。Stage 2 等待 event 和 stream 完成后才读取 host 输出。

DFL 输出路径把 count 清零、每个尺度的 decode kernel、count D2H 和检测数组 D2H 都纳入 `enqueue()` 的布尔成功契约；任一 CUDA 调用失败都返回 false，由帧级错误路径排空 stream 并拒绝 collect。ring slot id 必须严格位于 `[0, 2]`，不再用取模把越界调用静默映射到另一帧缓冲。

## 从检测框到左右候选

先保存左右全部 YOLO 候选，再枚举所有 `L × R` 组合。每个组合按固定顺序审计：

1. `class_id` 相同；
2. 两框宽高均有效；
3. 初始视差 `d = cx_left - cx_right > 0`；
4. `d <= 2048 px`；
5. 极线误差通过；
6. 宽高比例均不超过 2.0；
7. shifted-IoU 不低于 0.05。

极线阈值不是恒定 12 px，而是：

```text
y_tolerance = max(12 px, 0.35 * max(left.height, right.height))
```

通过门控的 pair 排序代价为：

```text
score = |cy_left - cy_right| / y_tolerance
      + |log(width_ratio)| + |log(height_ratio)|
      - 0.25 * right_confidence
```

后续 GPU pair 准备还可加入 bbox 视差一致性惩罚。所有拒绝组合都会携带原因，采集时不能只保存最终成功 pair。

## shifted-IoU 的准确含义

普通左右框不会在原图坐标直接重合，因为水平视差正是深度来源。代码先把右框沿 x 方向平移 `d`，使左右框中心 x 对齐，再计算 IoU：

```text
right_shifted = shift_x(right_bbox, cx_left - cx_right)
shifted_iou   = IoU(left_bbox, right_shifted)
```

因此它主要检验框的宽高和垂直覆盖是否相容。它：

- 不修改原始左右图；
- 不裁剪像素；
- 不输出黑白掩膜；
- 不代表两幅 ROI 的纹理相似度；
- 只是一项配对 gate/质量字段。

YOLO 自身 NMS 的 IoU 和双目 shifted-IoU 也是两个不同概念：前者在单相机内抑制重复检测，后者在左右相机间验证几何相容性。

## ROI 在当前代码中的含义

“ROI”主要是一个检测框及其搜索范围的坐标约束。当前 GPU 算法直接传入：

- 整帧校正 Gray8 CUDA 指针和 pitch；
- 左右检测框；
- 初始视差、patch/search radius 和几何 gate。

kernel 只访问框或搜索带内部像素，但通常不先把它们复制成独立小图。只有异步 Stage 2 为摆脱三缓冲槽生命周期，才会将整帧 Gray8 D2D 快照到自己的缓冲；这仍不是 IoU crop。

## 关联与像素匹配的边界

YOLO pair gate 只回答“哪一对框可能来自同一个球”。像素级候选才回答“框内哪个观测量能给出更稳定视差”。正确顺序是：

```text
检测候选
  -> 语义与双目几何关联
  -> 有界 ROI 像素计算
  -> 候选质量 gate
  -> 深度候选选择
  -> 3D
```

不能使用最终平滑 3D 反推或覆盖原始候选，也不能让某个 ROI matcher 的成功结果掩盖被拒绝的左右框组合。v2.1 采集分别保存 detections、pair attempts 和直接 3D，正是为了维持这条边界。

