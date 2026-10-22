# 验收标准

验收分为软件契约、NX 环境、相机同步、数据质量和性能五层。只有前一层通过，才进入下一层。主机测试通过不代表实机同步已经通过。

## A. 软件契约

```bash
python3 -m pip install -e '.[test]'
volleyball config validate --profile nx-rdk
volleyball config validate --profile free-flight
volleyball config validate --profile replay
volleyball config pipeline-report --profile nx-rdk --format json
PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s tests -v
cmake --preset host-core-release
cmake --build --preset host-core-release --parallel 2
ctest --preset host-core-release
python3 tools/check_repository.py
git diff --check
```

通过条件：全部命令返回 0；配置指纹稳定；同一 trial 回放两次得到同一派生指纹；乱序 USB 到达测试仍按 FrameCounter 正确配对；损坏哈希、计数缺口和不一致时序都被拒绝。

## B. NX 构建与只读自检

```bash
cmake --preset nx-legacy-release
cmake --build --preset nx-legacy-release --parallel
volleyball doctor --profile nx-rdk
volleyball doctor --profile free-flight
volleyball run --profile nx-rdk --dry-run
```

通过条件：目标机为 Jetson Orin NX；`doctor` 的配置意图、MVS/CUDA/TensorRT/VPI/OpenCV/ROS2、GPIO 和二进制动态链接检查全部通过；模型存在且 SHA-256 与清单一致；标定语义有效；生成配置可复现。这里证明的是 SDK/链接可用，不等于算子已经在目标 backend 实际执行；后者还必须按[运行依赖与实际启用判定](pipeline/07-runtime-enablement.md)保存启动日志和 Nsight 证据。非 Jetson CUDA 服务器只能验证通用 GPU 代码，不能替代此项。

`nx-legacy-release` 固定启用 `VOLLEYBALL_REQUIRE_HIK=ON` 和
`VOLLEYBALL_REQUIRE_ROS2=ON`。缺少 MVS 头文件、MVS 动态库或 ROS2 工作区时，
配置阶段必须失败；不允许通过静默禁用相机或发布器得到“成功构建”。

真实启动日志还必须证明左右 TensorRT engine 均成功反序列化、I/O 契约通过，且实际网络输入尺寸与 profile 一致。仅有 engine 文件和正确哈希不能证明其 tensor 布局受当前后处理支持。

## C. 触发与水印

先运行海康元数据探针，再录制至少 10 分钟静态场景和一批自由飞行：

```bash
adapters/legacy_nx/build/hik_frame_metadata_probe \
  --left-sn 00D39342665 --right-sn 00219471413 \
  --pwm-hz 90 --frames 300
```

探针必须返回 0，且 `FrameCounter-to-GPIO mapping` 的 `missing=0`；否则当前相机计数起点不满足采集适配器的映射前提，禁止正式采集。

- 两台相机在 PWM 启动前均已 `StartGrabbing`；
- TriggerMode/Source/Activation 回读分别为 On/Line0/RisingEdge；
- FrameSpecInfo 的 Timestamp、Framecounter、ExtTriggerCount 均已启用并可读取；
- 左右 `Width/Height/OffsetX/OffsetY` 回读与 profile 一致，正式运行中 payload 尺寸无漂移；
- 同步基线预热帧不会作为 `grab_failed` 进入采集回调，预热超限则明确失败；
- 左右 `FrameCounter` 始终同号且逐帧 +1，允许 uint32 自然回绕；
- USB 到达先后可变化，但不会改变配对；
- 水印像素在 SDK 解析后、视觉处理前被清除；
- raw VPI 锁定/解锁及 Stage 0 VPI/CUDA 任一步失败都会阻止该帧进入检测；
- 任何一侧断流、计数缺口或对齐超限均使采集失败，而不是静默继续。

必须用示波器同时观察 GPIO 和两台相机可用的曝光/闪光输出，测量：

- GPIO 写高到实际曝光开始的固定延迟；
- 左右曝光偏差的 p50/p95/p99/max；
- 触发周期抖动；
- 不同曝光、帧率和 USB 负载下的变化。

测量结果写入 `exposure_delay_ns`、`exposure_delay_uncertainty_ns` 和 `estimated_error_ns`，并把 `exposure_delay_status` 从 `assumed_zero_unmeasured` 改为 `measured`。在完成此测量前，采集器按延迟 0 计算曝光中心，但不确定度保持未知；只能声称“共用触发沿”，不能声称绝对曝光同步误差已经被证明。

## D. 数据包质量

对每个 trial 执行：

```bash
volleyball dataset validate /path/to/trial
volleyball replay --profile replay --trial /path/to/trial --json
```

通过条件：

- 没有未解释的帧、触发或原图缺口；
- 全部原始文件哈希通过；
- 每个 3D 样本都能追溯到左右帧、候选和已接受配对；
- 相机设备、NX 单调、NX 实时和轨迹参考时钟的转换唯一且覆盖有效区间；
- 所有拒绝项保留明确原因；
- 原始层没有滤波位置、预测状态或手工修正；
- trial ID、球 ID、标定、模型、代码和配置均可追溯。

## E. 几何与运行性能

几何测试至少覆盖标定距离上的静止球、工作空间网格和独立测量的移动轨迹。分别报告横向、纵深和 3D 的 bias、MAE、RMSE、p95、最大误差，以及协方差覆盖率。

性能测试至少报告：

- 曝光到左右 USB 到达的 p50/p95/p99/max；
- 左右到达差与乱序比例；
- 校正、检测、配对、三角化、写盘和发布各阶段延迟；
- 端到端延迟、队列水位、GPU/CPU/内存占用；
- 30 分钟连续运行的帧计数和温度降频情况。

阈值应根据控制闭环的可接受误差和延迟确定，不能先任意写一个数字再宣称通过。验收报告必须保留原始 trial、测试配置和统计脚本版本。

## 当前结论

仓库可以通过 CPU 软件门禁，并已具备严格记录与拒绝错误数据的结构。实际 NX 构建、模型权重固定、示波器同步误差和长时间相机录制属于目标硬件验收项；在这些证据产生前，项目不能宣称“实机全部验收通过”。

