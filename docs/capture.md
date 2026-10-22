# 真实双目采集契约

## 采集目标

采集层保存尽可能接近传感器事实的内容，使后续可以重新检测、重新配对、重新三角化，或分别估计视觉噪声与动力学残差。规范的机器可读来源是 `contracts/capture/v2.1.0/`。

## 一次帧对的路径

1. 左右相机完成触发模式初始化并先进入抓取状态；
2. GPIO 产生一次上升沿并保存触发事件；
3. 左右图像独立经 USB 到达，各自记录到达时间；
4. SDK 从 payload 解析 FrameSpecInfo；
5. 在相机适配器内部消费同步基线预热帧，不把它们报告成管线丢帧；
6. 以 `FrameCounter` 对齐左右 `OneByOne` 队列；
7. 将同一计数器映射到同一 GPIO 触发序号；
8. 清除图像首部水印像素行，再去 Bayer、校正和检测；
9. 保存全部检测候选和全部左右配对尝试；
10. 保存通过门控的直接 3D 观测；
11. staging 完整关闭后校验并原子发布。

USB 到达顺序允许是 `L_n,R_n,R_(n+1),L_(n+1)`；只要每侧队列内部保持 payload 顺序、FrameCounter 连续，仍应正确形成第 n 和第 n+1 对。

启动时需要连续 3 对水印帧建立左右设备时间偏移基线。前两对只在相机适配器内部消费；第一个交给管线的帧必须已经完成同步初始化。若 16 对内仍不能稳定，启动后的第一次抓帧直接失败，采集 trial 不得发布。

## 曝光时间语义

当前两台相机均按全局快门建模。一次直接球心观测使用以下整数纳秒时间：

```text
exposure_start_ns = map(gpio_rising_edge_estimate_ns) + exposure_delay_ns
capture_timestamp_ns = exposure_start_ns + round(exposure_us * 1000 / 2)
```

`trigger_edge_*` 是 GPIO 写高调用前后时钟读数中点给出的上升沿估计，并用
`trigger_write_uncertainty_ns` 保存该调用区间宽度；它不是示波器测得的电气边沿。
当前 `exposure_delay_ns=0` 是尚未完成硬件测量时的显式假设，
`exposure_delay_status=assumed_zero_unmeasured` 且
`exposure_delay_uncertainty_ns=null`，不得解释为延迟已被证明为零。完成示波器或
相机曝光输出测量后，必须改为 `measured` 并写入延迟及不确定度。

`camera_frames.exposure_start_ns` 保留曝光开始；`capture_timestamp_ns` 保存全局曝光
中心。`trajectory.csv` 使用左右 `capture_timestamp_ns` 的中点，metadata 中声明
`timestamp_semantics=stereo_midpoint_capture`。旧 trial 若声明 `hardware_trigger`，仍按
其原始语义读取，不得静默套用曝光中心校正。

## 数据包

```text
dataset-root/
├── calibrations/<calibration_id>.json
└── trials/<trial_id>/
    ├── metadata.json
    ├── trajectory.csv
    ├── camera_frames.csv
    ├── detections_2d.csv
    ├── stereo_pairs.csv
    ├── vehicle_state.csv
    ├── surface_state.csv
    ├── contact_events.csv
    ├── environment.csv
    ├── clock_mappings.csv
    ├── processing_timing.csv
    ├── raw/rectified/{left,right}/
    └── raw/audit/
```

`trajectory.csv` 只保存直接三角化球心，并预留测量协方差列；当前协方差生产者尚未接入时允许整组留空。速度、加速度、等效旋转、Magnus 参数、滤波状态及落点都属于 `derived/`。

## 必须保留的证据

- 左右相机实际序列号、角色、内外参、基线和标定误差；
- 配置指纹、Git revision、模型权重 SHA-256；
- 每帧 `FrameCounter`、设备时间、GPIO 上升沿估计、曝光开始、曝光中心和两侧 USB 到达时间；
- 每个 2D 候选，包括低分和未选中候选；
- 每个双目配对尝试，包括拒绝原因；
- 配对的 `accepted` 只表示最终生成了直接 3D；通过 2D 门控但未被最终选择的候选仍以 `not_selected` 保留；
- 单球数据每个曝光时刻最多保留一个 `accepted` 配对：先取语义置信度最高者，再取几何配对代价最低者，完全相同时按稳定的原始顺序选择；staging 原表仍原样保存在 `raw/audit/`，便于复核或重新选择；
- `detections_2d.selected=true` 的集合必须与所有 accepted 配对引用的左右候选完全一致；
- 原始图像、staging 原表和生成的旧管线配置；
- 球 ID、可获得的球压、温湿度、气流和实验备注；
- 所有原始文件的路径、大小与 SHA-256。

## 发布门禁

`free-flight` profile 采用 `fail_trial`。以下任一情况禁止发布：

- 实际帧数与触发次数不符；
- 左右 FrameCounter 不相等、不连续或与触发序号不对齐；
- GPIO 周期偏离配置，或设备时间序列与触发序列不一致；
- USB 到达时间早于对应触发沿或曝光中心；
- 回调早于任一侧图像到达；
- 原始图像缺失、路径逃逸、表头变化、跨表引用断裂；
- 时间不单调、时钟映射不唯一、坐标系或协方差无效；
- 模型、配置、标定或文件哈希不一致；
- 同名 trial 已存在。

发布方式为同文件系统 staging 到正式目录的原子重命名。失败目录保留以便诊断，不能被当作有效 trial。

## 建议的 conditions.json

```json
{
  "ball_model": "V200W",
  "ball_pressure_kpa": null,
  "ball_temperature_c": null,
  "air_temperature_c": 24.1,
  "relative_humidity": 0.53,
  "static_pressure_pa": null,
  "wind_mps": null,
  "launch_method": "manual_throw",
  "operator": "operator-a"
}
```

不能测到的量保持未知。批量真实轨迹用于推断等效参数分布，而不是给每次轨迹虚构面板朝向、真实转速或精确气流。

## 实验操作

开发机可先预演最终无丢帧配置；预演不检查相机、GPIO、NX 可执行文件或模型 artifact，
生成的 staging 带 `planning_only=true`，后续即使补齐文件也不能发布为真实 trial：

```bash
volleyball capture --profile free-flight \
  --trial dry-run-001 --session planning --ball v200w-01 \
  --duration 3 --dataset-root /tmp/volleyball-capture-plan --dry-run
```

在输出的 `.capture-staging/<trial>/pipeline.yaml` 中确认 `capture_v21.enabled=true`、
`async_roi_stage2=false`、`drop_stale_roi_frames=false` 和阻塞队列。正式采集仍必须先通过 doctor：

```bash
volleyball doctor --profile free-flight
volleyball capture --profile free-flight \
  --trial trial-001 --session session-a --ball ball-01 \
  --duration 3 --conditions conditions.json
volleyball dataset validate data/captures/trials/trial-001
```

每个实验批次先拍静态标定板和静止球，再进行低速、常用速度和高速自由飞行。不要在采集期间改变曝光、增益、焦距、基线或相机安装；若改变，必须生成新标定/配置指纹。

回放报告的 `quality.transport_timing` 会自动给出左右曝光到 USB 接收、曝光到推理完成的 p50/p95/p99/max，以及 `left_first`、`right_first`、`tie` 次数和左右到达时间差。它用于判断 USB 长尾和乱序，报告中同时固定声明 `arrival_time_used_for_pairing=false`。

