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
- TensorRT engine 反序列化后的实际输入尺寸和 I/O 布局必须通过检测器契约，且与 profile 一致；
- 标定必须有稳定 `calibration_id` 和 `created_at_utc`，不得用文件 mtime 充当版本；
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
仓库相对 artifact，并冻结输入尺寸、像素格式、精度、目标平台、accelerator 和
SHA-256；禁止在配置中保存 `/home/nvidia/...` 一类机器绝对路径。

ONNX、权重和 TensorRT engine 都不进入 Git。真实采集前必须在目标机上对最终
engine 重新计算 SHA-256，更新 manifest，并随 trial 记录 TensorRT、CUDA 和
JetPack 版本。源模型相同但重新构建的 engine 仍视为不同资产，不能复用旧哈希。

## 硬件触发约束

当 `cameras.sync.method=hardware_trigger` 时，主线要求：

- 相机后端为 `hik_mvs`；
- 左右相机均为 `bayer8`，曝光和增益相同；
- 左右 ROI 均为零偏移、配置分辨率对应的完整画面；
- 传输策略为 `one_by_one`；
- 帧身份为 `hik_frame_spec_info`；
- 触发源、沿、GPIO chip 和 line 非空；
- 快门形式为 `global`，曝光时间来源为 `nx_gpio_rising_edge`；
- 未测量触发到曝光延迟时，必须使用 `assumed_zero_unmeasured`、`delay=0` 和未知不确定度；实测后改为 `measured`；
- 在线检测使用 TensorRT、正方形输入、`letterbox` 和 `bbox_center`；
- 在线校正与三角化后端均为 CUDA；
- 采集模式必须 `queue.overflow_policy=block`、`recording.loss_policy=fail_trial`、`raw_frames=all`。

这些限制不是相机能力的上限，而是当前旧 NX 适配器能够忠实表达的契约。适配器目前只有一组曝光和增益，并按 profile 分辨率预分配 VPI 图像；因此配置层会拒绝左右参数差异和不能投影到旧 YAML 的视觉选项。相机打开时还会回读两侧 `Width/Height/OffsetX/OffsetY`，每帧复制前再次校验 payload 尺寸和目标 pitch，防止配置漂移造成越界写入。

## 主线到旧 NX 配置

旧 NX YAML 是生成物，不再手工作为配置真相：

```bash
volleyball config render-legacy --profile nx-rdk
volleyball run --profile nx-rdk --generated-config /tmp/nx-runtime.yaml --dry-run
```

显式指定的生成路径必须不存在，避免无提示覆盖人工文件。实际启动前 `doctor` 会检查标定、模型及哈希、旧适配器可执行文件和基础配置。

## 常用可调项

| 类别 | 典型键 | 说明 |
|---|---|---|
| 曝光 | `cameras.left.exposure_us` | 必须小于触发周期并适应球速与照明 |
| 触发 | `runtime.nominal_fps` | 同时决定 PWM 目标周期和帧数验收 |
| 水印 | `cameras.transport.embedded_info_clear_rows` | 先解析后清除，须与相机配置一致 |
| 队列 | `runtime.queue.*` | 实时追求低延迟，采集追求不丢帧 |
| 检测 | `detector.*` | 模型、输入尺寸、阈值与中心方法 |
| 配对 | `stereo.association.*` | 曝光中心时差、极线、尺寸、视差和 IoU 门限 |
| 深度 | `stereo.triangulation.*` | 计算候选集合、主方法、回退顺序、亚像素与时间预算 |
| 录制 | `recording.*` | schema、原图、失败策略、哈希校验 |

`diagnostics.metrics_interval_frames` 会投影为旧管线的 `performance.log_interval`。当前旧 logger 输出带单调时钟、线程和级别前缀的文本，不是逐行 JSON，因此主线固定 `diagnostics.json_logs=false`；把它改成 `true` 会在配置校验阶段被拒绝。性能计时当前是管线固有观测能力，也不能用配置伪装为已关闭。

旧 ROI 管线当前固定初始化 3D `kalman_ca` 混合深度跟踪，NX 原始观测 publisher 固定使用 keep-last/best-effort 的 sensor-data QoS。因此 live profile 只接受 `tracking.enabled=true`、`tracking.filter=kalman_ca` 和 `publisher.qos=sensor_data`。已删除的旧 C++ 图像 SOT 与这项 3D 状态估计无关。

### 深度候选与最终选择

三类字段不能混用：

- `candidate_methods`：本次运行会实际计算的候选；
- `primary_method`：最终直接深度首先尝试的方法；
- `fallback_methods`：首选失败后的发布顺序，必须是候选集合的子集。

旧 NX 当前只支持 `roi_multi_point` 或 `circle_center` 作为首选，并按
`epipolar_fallback -> circle_center -> radial_center -> edge_pair_center -> edge_centroid -> bbox_center -> bbox_edges`
的固定相对顺序扫描回退。配置可以删去其中任意项，但不能伪造旧选择器无法执行的顺序。
加载器会在启动前拒绝不忠实的配置。

组装主线 C++ 关联服务时，必须把 `cameras.sync.estimated_error_ns` 作为左右曝光中心差的允许上限，写入 `AssociationConfig.max_capture_delta_ns`。该值表达经时钟映射后的同步误差预算，不是 USB 到达时间差；实际 `delta_t_lr_ns` 始终保留在配对记录中，供采集验收和后续噪声建模使用。

生成旧 YAML 时，适配器先关闭基础模板中的全部 `depth_modes`，再只开启
`candidate_methods` 显式列出的模式。因此修改旧模板或新增旧开关不会静默改变正式 profile。

当前实际 P0/P1 路径中的 bbox 视差一致性惩罚由
`stereo.association.bbox_disparity_*` 管理；多点结果相对初始几何的门限由
`stereo.triangulation.subpixel.max_*` 管理；单侧漏检搜索窗口和圆拟合采样上限分别位于
`stereo.triangulation.fallback_search` 与 `stereo.triangulation.circle_fit`。这些参数都会被
严格投影到旧 NX adapter，不能再通过修改 `base_pipeline.yaml` 改变正式 profile。

同样，`legacy_activation_policy.py` 会集中覆盖 DLA、双路 YOLO、全帧视差、
神经特征、异步 ROI、P2 诊断和显示等功能开关。实时 profile 才允许
异步/丢旧帧；采集 profile 即使只执行 `render-legacy` 也保持同步无丢帧语义。
旧模板暂时仍保存已禁用实验候选和诊断工具的兼容数值，但它们不再有权自行开启处理支路；
当前正式 P0/P1 几何路径使用的上述门限均由主线配置拥有。

