# 配置系统

## 组合顺序

`configs/profiles/*.yaml` 只列出层。加载器按顺序合并：

1. `defaults/runtime.yaml`：完整且可校验的默认配置；
2. `hardware/*.yaml`：平台、相机、触发和加速器；
3. `calibration/*.yaml`：标定资产和坐标系；
4. `pipelines/*.yaml`：检测、配对和三角化算法；
5. `modes/*.yaml`：实时、采集或回放语义；
6. profile 显式列出的本机层（若有）；
7. 命令行 `--set` 临时覆盖。

每个 profile 合并后必须通过 `configs/schema/runtime.schema.json` 和跨字段语义检查，并生成规范化 SHA-256。`config explain` 可以回答一个值来自哪一层。

配置只保留会改变运行行为或采集证据的字段。CUDA 架构由 CMake 构建预设
管理；尚无实测变换的 `base_link`/世界坐标系不进入标定配置；数据校验是否跳过
哈希由对应 CLI 的显式参数决定，避免修改配置却不改变行为的空旋钮。

当前只有三组可执行组合：`runtime` 和 `capture` 都使用海康硬触发、CUDA 校正、
TensorRT 检测及 CUDA 深度；`replay` 只读取已录制事实，不启动这些后端。配置加载器
会在命令启动前拒绝跨模式混搭，也会拒绝非采集模式启用录制、禁用发布器却保留
ROS2 后端等不会生效的设置。

```bash
volleyball config validate --profile free-flight
volleyball config render --profile free-flight --format yaml
volleyball config explain --profile free-flight recording.loss_policy
volleyball config diff nx-rdk free-flight
volleyball config render-mainline --profile nx-rdk
```

`render-mainline` 只输出原生进程实际读取的最小配置，可作为 L1 配置证据；它不
表示目标机已经成功初始化或实际执行了相应 CUDA/TensorRT 算子。

## 修改原则

- 永久、可复用的差异写入对应层，不复制整份大 YAML；
- 私有序列号或本机路径放入未提交的 `configs/local.yaml`；加载器不会
  自动读取它，必须由仅本机 profile 显式列为最后一层；
- `--set` 只能覆盖已存在的键，防止拼写错误静默生成无效配置；
- 模型必须登记在 `models/manifest.yaml`，实机采集前固定 SHA-256；
- 模型清单的 artifact 格式、输入尺寸/格式、目标平台和 accelerator 必须与 live profile 完全一致；
- 模型清单还必须冻结 `output_contract`、有序 `class_names` 和 `target_class_id`，且 live profile 的 `detector.class_id` 必须指向同一目标；
- TensorRT engine 反序列化后的实际输入尺寸、输出契约和类别数必须通过检测器契约，且与 manifest/profile 一致；
- 标定必须有稳定 `calibration_id`、`created_at_utc` 和源文件 `sha256`，不得用文件 mtime 充当版本；
- 标定 YAML 的内容一旦变化就视为新资产：更换 `calibration_id`、更新时间和哈希，不能沿用旧 ID 覆盖；
- 单位写入键名或契约，时间统一为整数纳秒，物理量使用 SI。

本机覆盖的最小做法如下。`local-*.yaml` profile 和 `configs/local.yaml`
都已被 Git 忽略，不会静默影响正式 profile，也不会误提交设备私有信息：

```bash
cp configs/local.example.yaml configs/local.yaml
cp configs/profiles/nx-rdk.yaml configs/profiles/local-nx.yaml
```

然后在 `local-nx.yaml` 中把 `profile` 改为 `local-nx`，并在 `layers` 末尾
追加 `local.yaml`。用 `volleyball config validate --profile local-nx` 和
`volleyball config explain --profile local-nx <键>` 确认覆盖来源。正式采集记录的
配置指纹仍包含本机层的最终值，不会牺牲可追溯性。

## 模型资产

代码和 profile 只引用稳定的 `model_id`。`models/manifest.yaml` 把该 ID 映射到
仓库相对 artifact，并冻结输入尺寸、像素格式、输出布局、类别顺序、目标类别、
精度、目标平台、accelerator 和 SHA-256；禁止在配置中保存 `/home/nvidia/...`
一类机器绝对路径。

当前登记的 engine 契约是单类别 `volleyball=0`、单输出 YOLO pre-NMS。若换用
`player=0, volleyball=1` 等多类别权重，必须同时更新 `class_names`、
`target_class_id`、profile 的 `class_id`，并重新导出 engine 和固定哈希；不能只把
文件改名覆盖。原生进程会在启动时核对 engine 实际类别数和输出布局，错配立即失败。

ONNX、权重和 TensorRT engine 都不进入 Git。真实采集前必须在目标机上对最终
engine 重新计算 SHA-256，更新 manifest，并随 trial 记录 TensorRT、CUDA 和
JetPack 版本。源模型相同但重新构建的 engine 仍视为不同资产，不能复用旧哈希。

真实采集还要求 Git 工作树干净，并把 `apps/nx_mainline` 实际可执行文件 SHA-256
写入 `capture_context`。因此“HEAD 相同但本地源码有改动”或“源码相同但启动了另一份
旧二进制”都不能静默共享同一运行身份。dry-run 可以在开发状态执行，但 staging 永久
标记为不可发布；回放允许本地探索，不过代码 revision 会带 `-dirty` 并进入派生指纹。

## 硬件触发约束

当 `cameras.sync.method=hardware_trigger` 时，主线要求：

- 相机后端为 `hik_mvs`；
- 左右相机均为 `bayer8`，曝光和增益相同；
- 左右 ROI 均为零偏移、配置分辨率对应的完整画面；
- 传输策略为 `one_by_one`；
- 帧身份为 `hik_frame_spec_info`；
- 触发源、沿、GPIO chip 和 line 非空；
- 实机主线与相机硬件工具必须链接 `libgpiod`；请求 chip/line 失败时立即失败，不允许退回 sysfs 或硬编码全局 GPIO 号；
- 快门形式为 `global`，曝光时间来源为 `nx_gpio_rising_edge`；
- 未测量触发到曝光延迟时，必须使用 `assumed_zero_unmeasured`、`delay=0` 和未知不确定度；实测后改为 `measured`；
- 在线检测使用 TensorRT、正方形输入、`letterbox` 和 `bbox_center`；
- 在线校正与三角化后端均为 CUDA；
- 采集模式必须 `queue.overflow_policy=block`、`recording.loss_policy=fail_trial`、`raw_frames=all`。

这些限制不是相机能力的上限，而是当前原生主线已经实现并测试的契约。相机适配器
目前使用一组曝光和增益，并按 profile 分辨率预分配有界缓冲；打开相机时会回读
两侧 `Width/Height/OffsetX/OffsetY`，每帧复制前再次校验 payload 尺寸和目标
pitch，防止配置漂移造成越界写入。

## 原生进程配置

进程 YAML 是最终 profile 的最小投影，不再手工维护：

```bash
volleyball config render-mainline --profile nx-rdk
volleyball run --profile nx-rdk --generated-config /tmp/nx-runtime.yaml --dry-run
```

显式指定的生成路径必须不存在，避免无提示覆盖人工文件。实际启动前 `doctor` 会
检查标定和模型的内容哈希、原生可执行文件、SDK 和动态链接。生成的原生配置也携带
标定 SHA-256，C++ 进程在加载 OpenCV YAML 之前再次核对，防止配置生成后文件被替换。

## 常用可调项

| 类别 | 典型键 | 说明 |
|---|---|---|
| 曝光 | `cameras.left.exposure_us` | 必须小于触发周期并适应球速与照明 |
| 触发 | `runtime.nominal_fps` | 同时决定 PWM 目标周期和帧数验收 |
| 水印 | `cameras.transport.embedded_info_clear_rows` | 先解析后清除，须与相机配置一致 |
| 队列 | `runtime.queue.*` | 实时追求低延迟，采集追求不丢帧 |
| 检测 | `detector.*` | 模型、输入尺寸、阈值与中心方法 |
| 配对 | `stereo.association.*` | 曝光中心时差、极线、尺寸、视差和 IoU 门限 |
| 深度 | `stereo.triangulation.*` | 候选集合、选择顺序、圆拟合上限与 UVD 不确定度 |
| 录制 | `recording.*` | schema、原图、失败策略、哈希校验 |

`diagnostics.metrics_interval_frames` 控制主线进度统计周期。ROS2 使用传感器数据
QoS，并只发布直接观测；滤波、轨迹预测和控制参数不属于 NX 配置。

### 深度候选与最终选择

三类字段不能混用：

- `candidate_methods`：本次运行会实际计算的候选；
- `primary_method`：最终直接深度首先尝试的方法；
- `fallback_methods`：首选失败后的发布顺序，必须是候选集合的子集。

原生进程按 `primary_method` 后接 `fallback_methods` 的顺序选择，并要求这一顺序
恰好覆盖 `candidate_methods`；不允许计算一个候选后又不保存其选择语义。加载器会
在启动前拒绝重复、缺失或后端不支持的方法。

组装主线 C++ 关联服务时，必须把 `cameras.sync.estimated_error_ns` 作为左右曝光中心差的允许上限，写入 `AssociationConfig.max_capture_delta_ns`。该值表达经时钟映射后的同步误差预算，不是 USB 到达时间差；实际 `delta_t_lr_ns` 始终保留在配对记录中，供采集验收和后续噪声建模使用。

当前实际 P0/P1 路径中的 bbox 视差一致性惩罚由
`stereo.association.bbox_disparity_*` 管理，圆拟合采样上限位于
`stereo.triangulation.circle_fit`。`stereo.triangulation.measurement_uncertainty`
保存锚点和视差标准差先验，应用静态已知位置与独立轨迹的协方差覆盖率修订；
它不是纹理匹配调参。底层研究算子仍可在独立 API 中实验，其 patch/多点参数
不进入生产 profile。正式行为只能由 profile 改变，不能通过编辑生成的
进程 YAML 改变。

