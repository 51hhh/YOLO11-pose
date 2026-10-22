# NX 图像处理流水线

本目录描述一帧双目图像从 GPIO 触发、海康取流、CUDA/VPI 前处理、双路 YOLO、左右候选关联、灰度 ROI 深度候选到 3D 发布或采集落盘的完整生命周期。内容按当前代码和最终投影配置反向核对，不把历史实验配置当作主线事实。

## 阅读顺序

1. [完整帧生命周期](01-frame-lifecycle.md)：一帧何时创建、由谁拥有、何时可复用。
2. [CUDA/VPI 前处理](02-cuda-vpi-preprocess.md)：Bayer、BGR、校正灰度图及同步边界。
3. [双路 YOLO、配对和 ROI](03-yolo-pairing-roi.md)：检测、NMS、shifted-IoU 与候选关联。
4. [灰度 ROI 深度候选](04-gray-roi-depth.md)：当前启用的几何、NCC、多点亚像素及可选特征分支。
5. [并发、数据流和启用矩阵](05-concurrency-profiles.md)：实时、采集、回放三种 profile 的实际差异。
6. [观测、性能与验收](06-observability-acceptance.md)：如何证明流水线正确、实时且没有隐性丢帧。
7. [运行依赖与实际启用判定](07-runtime-enablement.md)：区分配置、SDK、动态链接与逐帧执行证据。

## 当前主线结论

当前 `nx-rdk` 与真实采集共用同一套图像和双目算法，核心路径是：

```text
GPIO 上升沿
  -> 左右相机同时开始曝光
  -> 两路 USB 独立到达，按 FrameCounter 配对
  -> 清除 FrameSpecInfo 水印行
  -> BayerRG8 --CUDA--> BGR8
  -> BGR8 --VPI CUDA remap--> 校正 BGR8
  -> 校正 BGR8 --VPI CUDA--> 校正 Gray8
  -> 左右 TensorRT YOLO
  -> 全部 LxR 框组合依次通过语义/视差/极线/尺寸/shifted-IoU 门控
  -> 校正 Gray8 上的 ROI 几何、NCC 和多点亚像素候选
  -> 选择直接深度 -> 三角化/融合 -> 回调、ROS2 或 v2.1 采集
```

三点必须明确：

- `shifted-IoU` 是左右检测框在水平对齐后的形状与垂直重叠门控，不是像素裁剪，也不会产生“IoU 图像”。
- 主线没有先生成一张黑白 ROI 图再做全部深度。几何、NCC 和多点方法直接读取整帧校正灰度 GPU 缓冲区内的有限坐标区域。黑白椭圆图只属于当前关闭的部分 CPU 特征归一化分支。
- 配置中的“启用”必须区分资源分配、算法计算、最终候选选择、对外发布四层。某候选被计算和记录，不代表它能成为最终 `Object3D.z`。
- `stereo.triangulation.candidate_methods` 是计算集合的主线真相；适配器会清零旧模板全部深度 mode 后逐项投影，避免隐藏算法开关。

## 配置事实的优先级

判断某功能是否真正启用时按以下顺序检查：

1. 根工程 profile 合并结果；
2. `python/volleyball_vision/legacy_config.py` 投影后的旧 NX YAML；
3. C++ 初始化时的依赖和自动降级条件；
4. 运行时资源创建、日志和性能计数；
5. 目标 NX 上的实际执行轨迹。

适配器基础模板是 `adapters/legacy_nx/config/base_pipeline.yaml`。它只提供旧 C++ 参数形状，所有生产开关必须由主线 profile 投影覆盖，不能直接把模板当成运行配置。

可随时重新核对最终配置：

```bash
PYTHONPATH=python python3 -m volleyball_vision config render --profile nx-rdk
PYTHONPATH=python python3 -m volleyball_vision config render-legacy --profile nx-rdk
PYTHONPATH=python python3 -m volleyball_vision config pipeline-report --profile nx-rdk
PYTHONPATH=python python3 -m volleyball_vision config diff nx-rdk free-flight
```

`pipeline-report` 把上述两层配置汇总为可测试的阶段、候选角色和 stream 启用表，
并明确标记为 L1 配置投影证据。它不能替代目标机上的 SDK/链接门禁、启动日志、
逐帧计数或 Nsight 时间线。

`free-flight` 的同步、无丢帧执行策略由 mode 在普通 `render-legacy` 阶段就固定；
真正执行 `volleyball capture` 时只再注入 staging 路径、帧数、图像产品和采集身份。

## 代码入口

| 责任 | 当前入口 |
|---|---|
| 相机初始化、FrameSpecInfo 和配对 | `adapters/legacy_nx/src/capture/` |
| 三缓冲帧容器 | `src/pipeline/frame_slot.h` |
| CUDA/VPI stream | `src/pipeline/sync.h` |
| VPI 图像的短期 CUDA view 锁边界 | `src/pipeline/pipeline_image_access.cpp` |
| VPI 图像同步复制到 CPU 的统一锁边界 | `src/utils/vpi_image_copy.*` |
| VPI 2/3 与 VPI 4 平面地址适配 | `src/utils/vpi_plane_access.h` |
| Stage 0/1/全帧 Stage 2 | `src/pipeline/pipeline_stages.cpp` |
| ROI 模式调度 | `src/pipeline/pipeline_loops.cpp` |
| 异步 ROI 快照与 worker | `src/pipeline/pipeline_async_roi.cpp` |
| TensorRT 检测 | `src/detect/trt_detector.*` |
| CUDA 图像预处理 | `src/detect/detect_preprocess.cu` |
| 左右配对契约 | `src/stereo/depth_match_contract.*` |
| 双路 YOLO 深度候选 | `src/pipeline/pipeline_dual_yolo_match.cpp` |
| 最终候选选择 | `src/stereo/depth_candidate_*.cpp` |
| 3D 融合与回调 | `src/pipeline/pipeline_roi_stage2.cpp` |
| v2.1 采集队列、图像复制与收尾 | `src/capture/capture_v21_stage_recorder.*` |
| v2.1 staging CSV 序列化合同 | `adapters/capture_v21/` |
| 可选实时调试落盘 | `src/diagnostics/realtime_debug_dump*` |
| 本地可视化叠加 | `src/ui/display_*` |

