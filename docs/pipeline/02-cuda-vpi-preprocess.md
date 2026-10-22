# CUDA/VPI 前处理

## 当前输入与输出

当前相机输出 1440×1080 `BayerRG8`，90 Hz。Stage 0 同时产生两类校正图像：

- `rectBGR_vpiL/R`：双路 YOLO 输入，也可用于显示和少数彩色候选；
- `rectGray_vpiL/R`：ROI 几何、NCC、多点匹配和采集保存的主输入。

当前校正 backend 是 VPI CUDA。变量名 `vpiStreamPVA` 是历史命名，不能据此断言当前校正在 PVA 上执行。

## 实际处理顺序

```text
海康 BayerRG8 payload
  -> SDK 解析 FrameSpecInfo
  -> 清除图像首部的 2 行水印像素
  -> 写入 rawL/rawR 的 host-pitch VPI buffer
  -> CUDA Bayer RGGB -> BGR8（全分辨率、双目）
  -> cudaStreamSynchronize(cudaStreamBGR)
  -> VPI CUDA 双目标定 remap，得到 rectified BGR8
  -> VPI CUDA BGR8 -> U8，得到 rectified Gray8
  -> vpiStreamSync
  -> 记录 evtRectDone，允许 TensorRT stream 继续
```

水印解析和清除发生在相机层，必须早于去 Bayer、YOLO 和图像保存。`FrameCounter` 等解析结果保存在元数据里，清除像素不会丢失身份信息。

## 图像资源与内存域

| 资源 | 格式 | 生产者 | 主要消费者 | 当前启用 |
|---|---|---|---|---|
| `rawL/rawR` | U8 Bayer | 海康取流，host-pitch lock 写入 | CUDA 去 Bayer | 是 |
| `tempBGR_L/R` | BGR8 | 自定义 CUDA kernel | VPI remap | 是 |
| `rectBGR_vpiL/R` | BGR8 | VPI remap | TensorRT YOLO | 是 |
| `rectGray_vpiL/R` | U8 | VPI format conversion | ROI 深度、采集 | 是 |
| `disparityMap/confidenceMap` | S16/置信度 | VPI full-frame stereo | 全帧融合 | `FrameSlot` 有字段；当前 ROI_ONLY 不分配图像、不计算 |
| TNR NV12 输入/输出 | NV12 | VPI convert/TNR | 后续 raw | 仅配置开启时分配；当前关闭 |

VPI 图像以 CPU、CUDA、PVA、VIC 可访问标志创建，初始化时通过一次 lock/unlock 缓存 CUDA 指针。这样每帧不用为同一图像反复 lock 八次左右；缓存只在 VPIImage 整个生命周期内有效。

## Stream 和同步边界

Stage 0 涉及：

- `cudaStreamBGR`：自定义 Bayer→BGR kernel；
- `vpiStreamPVA`：remap 和 BGR→Gray；当前 payload/backend 选择为 CUDA；
- `cudaStreamGPU`：在 VPI stream 已同步后记录 `evtRectDone`；
- 左右 TensorRT stream 等待 `evtRectDone`。

当前存在两个明确硬屏障：

1. Bayer→BGR 后显式 `cudaStreamSynchronize(cudaStreamBGR)`，保证 VPI remap 读取完整 BGR；
2. YOLO 提交前 `vpiStreamSync(vpiStreamPVA)`，之后才记录 CUDA event。

所以 `evtRectDone` 是“VPI 已由 CPU 同步确认完成后的门闩”，不是 VPI 工作直接记录到 CUDA event 的零阻塞互操作。文档或性能分析不能把这一段画成完全异步重叠。

## TensorRT 输入前处理

左右校正 BGR 直接留在 GPU。每个 detector 在自己的 CUDA stream 上执行融合预处理：

1. 1440×1080 等比缩放到 640×480；
2. 上下 letterbox 到 640×640；
3. BGR→RGB；
4. U8→float32，归一化并写入 CHW TensorRT 输入。

检测结束后坐标用同一 scale/padding 逆变换回 1440×1080 校正图坐标，后续极线和视差运算不在 640×640 网络坐标中进行。

## 当前关闭或替代路径

| 路径 | 状态 | 说明 |
|---|---|---|
| VPI TNR | 关闭 | 代码会走 U8↔NV12 和时序降噪；对 Bayer 语义和高速球拖影必须先在 NX 实测，当前不能视为已验证能力 |
| 纯 Gray 检测 | 关闭 | 当前模型输入为 BGR；代码保留 Gray/Bayer 输入预处理能力 |
| VPI 全帧 Stereo | 关闭 | `stereo.strategy=roi_only`，不提交 full/half-resolution disparity |
| VIC/PVA remap | 未选 | stream 支持相应 backend，但最终 profile 指定 CUDA |
| 彩色 ROI 特征 | 关闭 | 两个历史 color-IoU 候选均不启用，因此异步 Stage 2 通常不快照 BGR |

## 可测量的优化候选

以下是待实测候选，不是当前已实现结论：

- 用可证明正确的 VPI/CUDA 互操作消除 Bayer 后和 remap 后的 CPU 硬同步；
- 按 profile 条件分配 full-frame disparity、TNR 和关闭算法的资源；
- 用 Nsight Systems 检查左右 YOLO 是否与下一帧 USB 等待/Stage 0 重叠；
- 比较 CUDA、VIC 和 PVA remap 的延迟、功耗及像素一致性；
- 若修改色彩路径，必须重新验证 YOLO 精度、标定 remap 顺序和 Bayer pattern，不能只以帧率决定。

