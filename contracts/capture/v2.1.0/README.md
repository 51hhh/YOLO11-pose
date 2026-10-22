# 真实双目采集契约 2.1.0

该目录是视觉工程与排球动力学工程之间的冻结边界。契约源自 `volleyball-dynamics@db5255b4c6580a162e6ed5f6a1aaa8de3b6d8fab`，版本固定为 `2.1.0`。未提升 schema 版本时，不得改变列名、单位、时钟或坐标语义。

## 数据集布局

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
    ├── reference_trajectory.csv   # 可选
    ├── reference_spin.csv        # 可选
    └── raw/
```

`tables.json` 是 CSV 表头的机器可读唯一来源。必需 sidecar 即使没有数据也必须写入表头。`metadata.schema.json` 和 `calibration.schema.json` 提供 JSON 结构校验；跨表引用、时间单调性、协方差半正定、时钟映射覆盖及 SHA-256 由数据集校验器处理。

## 不可破坏的语义

- `trajectory.csv` 只保存直接三角化球心观测，不保存滤波位置、速度、加速度、动力学参数、预测轨迹或 RL 特征。
- `detections_2d.csv` 保留左右相机的全部候选，不只保留最终框。
- `stereo_pairs.csv` 保留接受和拒绝的每个关联尝试以及拒绝原因。
- 相机设备时钟、主机时钟和轨迹参考时钟不得混写。跨时钟转换必须使用 `clock_mappings.csv` 中的有效区间映射。
- 参考时钟转换用整数纳秒：`t_ref = a_ref + round((t_src - a_src) * (1 + drift_ppm * 1e-6))`。
- 左右相机角色必须绑定实际 `camera_id`；两台相机序列号、固定外参和时钟域必须可审计。
- 原始文件先写入临时目录并计算 SHA-256，全部成功后才原子发布；默认拒绝覆盖同名 trial。

## 派生数据

滤波、轨迹预测、等效旋转/Magnus 参数和 RL 观测必须写入独立的 `derived/<pipeline_fingerprint>/`，并引用原 trial ID、代码版本、模型哈希和配置哈希。派生结果不得回写或覆盖原始层。
