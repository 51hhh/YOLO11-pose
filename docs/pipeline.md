# 原生 NX 感知流水线

本文描述唯一部署入口 `apps/nx_mainline`。

## 一帧的完整路径

```text
GPIO 上升沿
  -> 左右全局快门曝光
  -> 两路 MVS payload 独立到达
  -> 解析 FrameSpecInfo，按 FrameCounter 对齐并清除水印行
  -> 有界源队列
  -> 保存原始 Bayer 帧（仅 capture）
  -> CUDA Bayer→BGR、双目标定 remap、Gray8
  -> 左右两个 TensorRT detector 并发执行
  -> 枚举并记录全部左右候选组合
  -> 时差、视差、极线、尺寸和 shifted-IoU 门控
  -> CUDA ROI 深度候选与显式顺序选择
  -> 直接 3D、协方差和方法证据
  -> ROS2 直接观测（runtime）或 v6 staging（capture）
  -> Python 严格校验并原子发布 capture v2.1
```

主线没有 VPI，也不产生所谓“IoU 黑白图”。`shifted-IoU` 只是把右框按候选视差
水平平移后计算形状重叠的关联门控。深度估计读取校正后的 CUDA Gray8 图像，并在
有限 ROI 内运行几何和多点方法。

## 三条不可混用的时间与身份链

| 事实 | 来源 | 用途 |
|---|---|---|
| 本次曝光对应哪次触发 | GPIO 上升沿序号和夹逼时间 | 曝光开始、曝光中心、轨迹时间轴 |
| payload 属于哪一帧 | FrameSpecInfo `FrameCounter` | 左右配对和漏帧检查 |
| payload 何时到达 NX | 每侧 `GetImageBuffer` 返回时钟 | USB/SDK 排队和长尾延迟 |

USB 到达先后不参与配对。当前曝光开始为 GPIO 上升沿映射值加
`exposure_delay_ns`；曝光中心为曝光开始加半个实际曝光时间。延迟尚未测量时只能
声明 `assumed_zero_unmeasured`，不能把 0 ns 假设解释成零误差。

## 所有权和并发

`HikStereoCameraSource` 是唯一 MVS 桥。相机先进入抓取状态，GPIO 后启动；退出时
GPIO 先停，相机后停。每个 payload 在归还 SDK buffer 前复制到有界的 pinned host
缓冲，队列、消费者和正在写入的下一帧不会复用同一缓冲。

`StereoPipelineRunner` 使用一个相机生产线程和一个消费编排线程：

- `nx-rdk` 使用有界 `drop_oldest`，优先限制观测年龄；每个丢弃都计数；
- `free-flight` 使用 `block`，任何缺帧、写入失败或协议错误终止整个 trial；
- 队列容量和高水位进入运行统计，不能只报告平均 FPS。

`CudaStereoPreprocessor` 拥有固定槽位池。左右图分别在非阻塞 CUDA stream 上完成
Bayer→BGR 与融合 remap/Gray；当前端口没有向后续暴露 readiness event，因此在
返回前同步两侧。输出缓冲通过共享槽位所有权保持有效，消费者释放后才可复用。

左右 detector 是两个独立 TensorRT 实例，由两个常驻工作线程并发调用；每个实例
始终只被自己的线程串行访问。此结构允许 GPU 重叠，但是否在 Orin NX 上真正并发、
是否达到 90 Hz，仍必须用目标机日志和 Nsight 证明。

## 检测、关联与直接深度

检测器只接收校正 BGR8，并必须返回同一相机、角色、帧号和唯一候选 ID。服务会
清除 detector 提供的临时 `selected`，只有最终生成直接 3D 的左右候选才标为选中。

所有左右组合都会形成可审计的关联尝试。门控顺序由平台无关算法实现；至少保存：

- 左右曝光中心及 `delta_t_lr_ns`；
- 检测中心视差和极线误差；
- 尺寸与球直径粗先验；
- shifted-IoU；
- 接受或拒绝原因。

检测中心视差只用于关联。CUDA 深度估计器产生的最终视差是独立测量，不能覆盖
原始门控事实。配置的 `primary_method + fallback_methods` 构成唯一选择顺序；正式
主线只允许圆/径向/边缘几何和 bbox 等球心代理方法。多点与中心 patch 纹理匹配仍可
作为研究算子，但在加入球体恢复并完成实测前不能进入选择或发布链。每条成功观测同时
保存实际方法、测量视差、视差标准差和支持点数，并由同一证据传播 3D 协方差。只有
当前双侧已关联端口能够实际生成、且在冻结顺序中的方法可以作为候选。

合法“没有可用深度”是一次 rejected 测量；CUDA、图像、契约或写入错误是 fatal
failure。后者不能伪装成漏检后继续发布。

## 三种运行模式

| profile | 硬件 | 队列 | 输出 |
|---|---|---|---|
| `nx-rdk` | MVS + GPIO + CUDA + TensorRT | `drop_oldest` | ROS2 直接观测，不写 trial |
| `free-flight` | 同一视觉链 | `block` | 原始 Bayer、全部候选/配对/3D，原子发布 v2.1 |
| `replay` | 不启动硬件 | `block` | 从冻结 v2.1 生成可重算派生结果 |

采集 writer 在任何视觉处理前保存 host 可读的原始 Bayer 帧，并分别记录帧、检测、
关联和直接观测四张 v6 staging 表。只有表、原图和完成状态全部关闭成功，Python
finalizer 才会校验时钟映射、引用闭包、哈希和配置指纹并发布最终 v2.1 trial。

实时 ROS2 只发布带完整来源证据的直接观测。滤波速度、轨迹预测、落点和控制目标
属于 RDK 或离线派生层，不能冒充 NX 测量。

## 失败与停止不变量

- 只有正常源结束或实时主动停止可以 finalize；超时越限、源错误和处理错误都 abort。
- 同一个 runner 和 writer 只能运行一次；失败后必须重建资源。
- 捕获、预处理、检测、关联、深度、写盘任一基础设施错误都必须可见。
- 采集模式不得以丢帧换实时性；实时丢弃必须计数并报告连续缺口。
- 收到 SIGINT/SIGTERM 后停止新工作，先停触发，再排空和关闭相机/GPU/写入资源。
- 原始观测不可被滤波值或人工修正覆盖；派生结果必须记录输入哈希。

## 配置与执行证据

“配置选择”“SDK 可用”“二进制链接”“逐帧实际执行”是四层不同证据：

| 层级 | 证据 |
|---|---|
| L1 配置 | `config render-mainline`、最终 YAML、配置 SHA-256 |
| L2 环境 | `doctor` 的 MVS、CUDA、TensorRT、OpenCV、ROS2 与 GPIO 检查 |
| L3 二进制 | `doctor` 的 `pipeline_linkage`，且无 `not found` |
| L4 执行 | 初始化日志、逐阶段计数、真实帧结果和 Nsight 时间线 |

开发机的 L1–L3 不能替代 Orin NX 的 L4。目标机至少确认：左右 TensorRT engine
契约、CUDA stream 实际重叠、无意外整帧 D2H、队列水位、端到端 p50/p95/p99、
连续丢帧、温度与降频。

## 代码入口

| 职责 | 路径 |
|---|---|
| 进程装配与信号处理 | `apps/nx_mainline/` |
| 生命周期与有界队列 | `services/` |
| GPIO 上升沿 | `adapters/gpio/` |
| MVS、FrameSpecInfo 与双目配对 | `adapters/hik_mvs/` |
| CUDA 校正与 BGR/Gray | `adapters/cuda_preprocess/` |
| TensorRT 检测 | `adapters/tensorrt/` |
| 关联与几何契约 | `algorithms/stereo/` |
| CUDA ROI 深度 | `adapters/cuda_depth/` |
| v6 staging 与 v2.1 finalizer | `adapters/capture_stage/`、`python/volleyball_vision/mainline_capture.py` |
| ROS2 直接观测 | `adapters/ros2/` |

## 常用核对命令

```bash
volleyball config validate --profile nx-rdk
volleyball config render-mainline --profile nx-rdk
volleyball doctor --profile nx-rdk --json
volleyball run --profile nx-rdk --dry-run

PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s tests -v
cmake --preset host-core-release
cmake --build --preset host-core-release --parallel
ctest --preset host-core-release
python3 tools/check_repository.py
```

目标机的完整门禁见[验收标准](acceptance.md)，采集文件语义见[采集契约](capture.md)。
