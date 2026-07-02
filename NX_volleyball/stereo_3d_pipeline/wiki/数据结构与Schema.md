# 数据结构与 Schema

最后核对: 2026-07-02

本页连接实时 C++ 数据结构、CSV 记录器和离线训练 schema。字段新增不能只改一个位置。

## FrameSlot

定义位置: `src/pipeline/frame_slot.h`

`FrameSlot` 是 ring buffer 的单帧状态，包含:

- 原始左右图 `rawL/rawR`。
- 校正灰度 `rectGray_vpiL/rectGray_vpiR`。
- 校正 BGR `rectBGR_vpiL/rectBGR_vpiR`。
- CUDA 指针缓存和 pitch。
- 左右 YOLO detections。
- tracker 状态和填充结果。
- disparity/confidence map。
- `Object3D` 结果。
- 左右相机水印、host timestamp、frame id。
- CUDA event: rect/detect/right detect 等完成标志。

设计约束:

- 每个 slot 必须在复用前 `reset()`。
- 新增 GPU buffer 要明确所有权、生命周期和是否需要 event 同步。
- hot path 不应在 Stage2 每帧 VPI lock/unlock 校正图，优先使用 slot 中缓存的 CUDA 指针。

## Object3D

`Object3D` 是单个球目标的在线观测和融合结果。主要字段类别:

| 类别 | 字段例子 |
|---|---|
| 原始/滤波 3D | `raw_x/y/z`, `x/y/z`, `vx/vy/vz`, `ax/ay/az` |
| 单目深度 | `z_mono` |
| bbox 双目 | `z_bbox_center`, `z_bbox_left_edge`, `z_bbox_right_edge` |
| 圆拟合双目 | `z_circle_center`, `z_circle_left_edge`, `z_circle_right_edge` |
| ROI 几何 | `z_roi_edge_centroid`, `z_roi_radial_center`, `z_roi_edge_pair_center` |
| ROI sparse | `z_roi_corner_points`, `z_roi_texture_points`, `z_roi_binary_points` |
| 描述子/近似描述子 | `z_roi_orb_points`, `z_roi_brisk_points`, `z_roi_akaze_points`, `z_roi_sift_points` |
| 彩色 IoU/patch | `z_roi_iou_region_color_patch`, `z_roi_patch_iou_color_edge` |
| 神经特征 | `z_roi_neural_feature` |
| patch/subpixel | `z_roi_center_patch`, `z_roi_multi_point` |
| fallback | `z_fallback`, `z_fallback_template`, `z_fallback_feature_points` |
| 诊断 | disparity、support、confidence、std、bbox/circle、sync watermark |

兼容旧 CSV 的 alias 字段仍存在: `z_yolo_bbox_pair`、`z_circle`、`z_subpixel`、`disparity_yolo`、`disparity_circle`、`disparity_subpixel`。它们分别对应 bbox center、circle center 和 ROI multi-point/subpixel，不应按新增测距方法重复统计。

`stereo_depth_source` 当前注释映射:

| 值 | 来源 |
|---:|---|
| 0 | 无 |
| 1 | 圆心/搜索 |
| 2 | ROI 多点 |
| 3 | bbox 中心 |
| 4 | 中心 patch |
| 5 | 边缘质心 |
| 6 | bbox 边缘 |
| 7 | 模板 fallback |
| 8 | 径向中心 |
| 9 | 边缘成对中心 |
| 10 | 角点特征 |
| 11 | 纹理特征 |
| 12 | 特征 fallback |
| 13 | 二值特征 |
| 14 | ORB |
| 15 | BRISK |
| 16 | AKAZE |
| 17 | SIFT |
| 18 | 彩色区域 IoU |
| 19 | 彩色边缘 IoU |
| 20 | 神经特征 |

新增深度候选时必须扩展该映射，并同步 [深度测量](深度测量.md)。

## 轨迹 CSV

定义位置: `src/utils/trajectory_recorder.*`

`TrajectoryRecordDetail`:

| 值 | 内容 |
|---|---|
| `LEGACY` | 旧基础字段 |
| `DEPTH_CANDIDATES` | 增加深度候选字段 |
| `EXTENDED` | 增加同步水印、bbox/circle 几何等训练诊断字段 |

实时配置建议:

```yaml
recording:
  enabled: true
  detail_level: "extended"
  raw_mode: true
  max_queue_frames: 1000
```

写入器是后台线程。队列满时应丢记录，不能反向阻塞 100fps 管线。

## 基准片段 CSV

定义位置: `src/utils/baseline_clip_recorder.*`

输出:

```text
clip_YYYYMMDD_HHMMSS_01/
├── left/
├── right/
├── left_bgr/      # image_mode=both 时存在
├── right_bgr/     # image_mode=both 时存在
├── frames.csv
└── metadata.yaml
```

`left/` 和 `right/` 保存 `image_mode` 选择的主图像；`bgr` 时是彩色图，`gray` 时是灰度图。默认 `write_after_capture=true`，实时阶段先缓存到内存，clip 完成后写盘。

`frames.csv` 记录:

- pipeline frame id。
- 左右图相对路径。
- 额外 BGR 图相对路径；仅 `image_mode=both` 时填写。
- 左右检测数量和最佳框。
- 左右 pair gate 结果。
- disparity、dy、size ratio。
- 左右 timestamp、frame number、frame counter、trigger index。
- delta 字段、fps、`grab_failed`、`is_detect_frame`。

## 离线 schema

权威训练 schema 在:

```text
trajectory_fusion/schema.md
```

同步要求:

1. C++ `Object3D` 新增字段。
2. `TrajectoryRecorder::writeHeader()` 和 `writeEntry()` 写出字段。
3. `trajectory_fusion/schema.md` 描述字段语义和单位。
4. `trajectory_fusion/dataset.py` 读取字段。
5. `trajectory_fusion/evaluate_fusion.py` 或相关评估脚本使用字段。
6. Wiki 的 [深度测量](深度测量.md) 和本页更新。
