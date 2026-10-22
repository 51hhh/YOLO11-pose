# CUDA/VPI 前处理

## 主线边界状态

平台无关主线已经用 `StereoPreprocessor` 明确表达本阶段：输入是带完整采集事实的原始 `StereoFrame`，输出是只引用来源帧身份的 `RectifiedStereoFrame`。服务会检查左右身份没有交换、图像尺寸与原始 ROI 一致，并要求同时得到 BGR8 detector 输入与 Gray8 几何输入；预处理失败不会退化为普通漏检。

`adapters/cuda_preprocess` 现已实现该端口。初始化时从标定生成紧密的左右 x/y 浮点映射并一次上传；每个有界槽位持有 pinned Bayer staging、CUDA raw、临时 BGR、校正 BGR 和校正 Gray。左右图在两条独立非阻塞 CUDA stream 上提交，输出通过共享槽位保持设备内存有效。当前端口没有 readiness event，因此 `prepare()` 必须在返回前同步两条 stream，之后 TensorRT 的独立 stream 才可安全读取。

该新适配器已在普通 GPU 上通过像素、身份、缓冲池耗尽和跨适配器生命周期测试，也会被 NX 目标编译；但原生主线入口尚未实例化它。下文其余内容描述当前实机仍在运行的 `adapters/legacy_nx` CUDA/VPI 路径。“新适配器测试通过”与“新主线已在 Orin NX 运行并达到 90 Hz”仍是两个不同状态。

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
  -> lock raw READ + tempBGR WRITE 的 CUDA view
  -> CUDA Bayer RGGB -> BGR8（全分辨率、双目）
  -> cudaStreamSynchronize(cudaStreamBGR)
  -> unlock raw/tempBGR CUDA view
  -> VPI CUDA 双目标定 remap，得到 rectified BGR8
  -> VPI CUDA BGR8 -> U8，得到 rectified Gray8
  -> vpiStreamSync
  -> 记录 evtRectDone，允许 TensorRT stream 继续
```

水印解析和清除发生在相机层，必须早于去 Bayer、YOLO 和图像保存。`FrameCounter` 等解析结果保存在元数据里，清除像素不会丢失身份信息。

Stage 0 对每个 VPI lock/unlock、submit 和 stream sync 检查返回状态；`VPIRectifier` 也把左右 remap 的提交结果返回给调用方。任一步失败时不会记录 `evtRectDone`，更不会把旧槽位像素送进 TensorRT。初始化日志只能证明 payload 创建成功，逐帧返回状态和后续同步共同证明该帧执行成功。

保留的 full/half-resolution VPI Stereo 路径也遵守同一规则：创建参数初始化、提交参数初始化、左右 rescale、disparity submit 和上采样逐项检查 `VPIStatus`。半分辨率链中途失败时会同步排空 `vpiStreamGPU` 后再允许槽位复用。当前 `nx-rdk/free-flight` 走 `ROI_ONLY`，所以这项是关闭支路的正确性保护，不是其已在实机启用的证据。

## 图像资源与内存域

| 资源 | 格式 | 生产者 | 主要消费者 | 当前启用 |
|---|---|---|---|---|
| `rawL/rawR` | U8 Bayer | 海康取流，host-pitch lock 写入 | CUDA 去 Bayer | 是 |
| `tempBGR_L/R` | BGR8 | 自定义 CUDA kernel | VPI remap | 是 |
| `rectBGR_vpiL/R` | BGR8 | VPI remap | TensorRT YOLO | 是 |
| `rectGray_vpiL/R` | U8 | VPI format conversion | ROI 深度、采集 | 是 |
| `disparityMap/confidenceMap` | S16/置信度 | VPI full-frame stereo | 全帧融合 | `FrameSlot` 有字段；当前 ROI_ONLY 不分配图像、不计算 |

VPI 图像以 CPU、CUDA、PVA、VIC 可访问标志创建。CUDA 地址按使用区间通过 `vpiImageLockData(...CUDA_PITCH_LINEAR...)` 获取，匹配的 `vpiImageUnlock()` 返回后立即清空，不跨帧缓存。

VPI 2/3 的 pitch-linear 平面地址字段是 `data`，VPI 4 改为 `pBase + offsetBytes`。所有 raw、rectified、disparity、实验候选和 viewer 地址访问统一经过 `vpi_plane_access.h`；仓库审计拒绝在其他源文件中直接访问这些字段。该适配已用 VPI 3/4 结构头桩分别做编译和偏移语义测试，真实 SDK 的完整构建仍属于目标 NX 门禁。

## VPI/CUDA 图像锁契约

NVIDIA 的 Image API 明确规定：`vpiImageLockData()` 返回的 buffer 只在图像保持 lock 时有效；图像可以有多个相容的 read lock，但每次 lock 都必须有一次 unlock。图像被外部 lock 时，VPI stream 不能对它执行不相容访问，否则会得到 `VPI_ERROR_BUFFER_LOCKED`。当前边界是：

| 图像 | lock 模式 | 外部消费者 | 释放条件 |
|---|---|---|---|
| `rawL/rawR` | READ | Bayer CUDA kernel | `cudaStreamBGR` 已排空 |
| `tempBGR_L/R` | WRITE | Bayer CUDA kernel | `cudaStreamBGR` 已排空，之后才允许 VPI remap 读取 |
| `rectBGR_vpiL/R` | READ | BGR YOLO、可选彩色 ROI | 对应 TensorRT/ROI/快照工作完成 |
| `rectGray_vpiL/R` | READ | Gray YOLO、ROI/特征点、异步快照 | 对应 TensorRT/ROI/快照工作完成 |

生产者与消费者的顺序如下：

1. VPI 生产 rectified 图后先 `vpiStreamSync`；
2. pipeline 为本帧取得所需 BGR/Gray CUDA view，并记录 `locked_frame_id`；
3. YOLO、ROI 或异步快照只接受“已锁且帧号一致”的 view；
4. 同步实时 ROI 在候选计算完成后释放；异步 ROI 必须等整帧 D2D/D2H 快照 event 完成后释放；
5. FrameCallback 在 view 释放后运行，回调可重新以 host read lock 深拷贝图像；
6. 槽位只有在 view 全部释放后才能 `reset()` 和重新写入。

全帧 Stereo 有一个额外分支：BGR YOLO 只锁 BGR，VPI Stereo 可同时读取独立的 Gray 图；Gray YOLO 则与 Stereo 读取同一 VPI image，必须先等 detector 完成并 unlock，再提交 Stereo。这个串行化是内存正确性的要求，性能影响需要在 Orin NX 上测量。

相容的 CUDA READ 与 HOST READ 可嵌套，用于少数同时需要 GPU Gray 和 CPU Gray 的候选；两次 lock 分别 unlock。任何 lock/unlock、快照 event 或 fallback stream 同步失败都会使该帧失败，不退化成普通“无匹配”。

## Stream 和同步边界

Stage 0 涉及：

- `cudaStreamBGR`：自定义 Bayer→BGR kernel；
- `vpiStreamPVA`：remap 和 BGR→Gray；当前 payload/backend 选择为 CUDA；
- `cudaStreamGPU`：在 VPI stream 已同步后记录 `evtRectDone`；
- 左右 TensorRT stream 等待 `evtRectDone`。

完整 stream 清单不能只看 `PipelineStreams`，因为异步 ROI 与 P2 诊断会按配置
另外创建低优先级 stream：

| stream | 生产的工作 | `nx-rdk` 每帧状态 | 主要完成证明 |
|---|---|---|---|
| `vpiStreamPVA` | remap、BGR→Gray；名称为历史遗留 | 开，算子 backend 为 CUDA | Stage 1 前 `vpiStreamSync` |
| `vpiStreamGPU` | full/half-frame VPI disparity | 关，不创建 stream、不提交 disparity | 若启用则 Stage 3/最终清理同步 |
| `cudaStreamBGR` | 左右 Bayer→BGR kernel | 开 | 每帧显式 synchronize |
| `cudaStreamDLA` | 左 YOLO preprocess/infer/decode/D2H | 开；名称不表示实际用了 DLA | 左 detect event/源 stream fallback |
| `cudaStreamDLA_R` | 右 YOLO preprocess/infer/decode/D2H | 开；当前 engine 为 CUDA GPU | 右 detect event/源 stream fallback |
| `cudaStreamGPU` | rect-done event 与全帧协调 | 开但当前主线负载很轻 | event 或最终 synchronize |
| `cudaStreamFuse` | 同步 ROI Stage 2 与融合支路 | 开 | Stage 2 结束同步/最终清理 |
| `cudaStreamP2Ncc` | inline CUDA template/NCC | 开且当前候选启用 | P2 worker join + 最终 synchronize |
| `cudaStreamP2XFeat/SuperPoint/Aliked` | inline 神经特征 | 候选关闭，不创建 stream 或 worker | 若启用则 worker join + 最终 synchronize |
| `async_roi_copy_stream_` | FrameSlot→异步缓冲的 Gray D2D，按需 Gray D2H/BGR D2D | 实时开，采集关 | slot/buffer copy event + 停止时同步 |
| `async_roi_stream_` | 异步 ROI 几何、NCC、多点候选 | 实时开，采集关 | worker 完成 + 停止时同步 |
| `p2_feature_diag_copy_stream_` | 独立诊断图像副本 | 当前诊断 lane 关，不创建 | 若启用则 copy event + 停止时同步 |
| `p2_feature_diag_stream_` | 低优先级 P2 诊断候选 | 当前诊断 lane 关，不创建 | 若启用则 worker join + 停止时同步 |

停止顺序是：停止新调度并 join pipeline，关闭并 join 异步 ROI/P2 worker，显式
同步四条辅助 stream，再同步 `PipelineStreams` 中的全部 VPI/CUDA stream。只有
所有 drain 都成功才释放导出的 VPI CUDA view；任何一条失败都会把本次运行标记
为 failure，并保留无法证明安全的锁或辅助缓冲到资源销毁阶段。

每个 detect event 另有本帧有效位。正常路径是“TensorRT preprocess/infer/decode/D2H 入队 → 在同一 stream 记录 event → fuse stream 等 event → 同步 fuse stream → CPU collect”。如果 enqueue 或 event API 失败，本帧立即失效；代码不会从上一次 ring-slot 复用留下的 host 输出构造空检测或旧检测。

当前存在两个明确硬屏障：

1. Bayer→BGR 后显式 `cudaStreamSynchronize(cudaStreamBGR)`，保证 VPI remap 读取完整 BGR；
2. YOLO 提交前 `vpiStreamSync(vpiStreamPVA)`，之后才记录 CUDA event。

所以 `evtRectDone` 是“VPI 已由 CPU 同步确认完成后的门闩”，不是 VPI 工作直接记录到 CUDA event 的零阻塞互操作。文档或性能分析不能把这一段画成完全异步重叠。

新增的 `Stage0_VpiCudaLock/Unlock` 与 `Rectified_VpiCudaLock/Unlock` 性能项专门测量 view 获取和释放成本。若该成本成为瓶颈，可评估外部分配 CUDA memory 再包装为 VPIImage，但 VPI 官方同时说明跨 backend 不保证原 allocation 零拷贝，必须以目标机 timeline 和像素回归为准，不能恢复“解锁后缓存地址”的做法。

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
| 纯 Gray 检测 | 关闭 | 当前模型输入为 BGR；代码保留 Gray/Bayer 输入预处理能力 |
| VPI 全帧 Stereo | 关闭 | `stereo.strategy=roi_only`，不提交 full/half-resolution disparity |
| VIC/PVA remap | 未选 | stream 支持相应 backend，但最终 profile 指定 CUDA |
| 彩色 ROI 特征 | 关闭 | 两个历史 color-IoU 候选均不启用，因此异步 Stage 2 通常不快照 BGR |

## 可测量的优化候选

以下是待实测候选，不是当前已实现结论：

- 用可证明正确的 VPI/CUDA 互操作消除 Bayer 后和 remap 后的 CPU 硬同步；
- full-frame disparity 与 P2 候选的 stream/worker 已按最终配置创建；
- 用 Nsight Systems 检查左右 YOLO 是否与下一帧 USB 等待/Stage 0 重叠；
- 比较 CUDA、VIC 和 PVA remap 的延迟、功耗及像素一致性；
- 若修改色彩路径，必须重新验证 YOLO 精度、标定 remap 顺序和 Bayer pattern，不能只以帧率决定。

