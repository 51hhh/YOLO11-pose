# 数据结构与 Schema

最后核对: 2026-07-03

本页连接实时 C++ 数据结构、CSV 记录器和离线训练 schema。字段新增不能只改一个位置。

## FrameSlot

定义位置: `src/pipeline/object3d_types.h` 和 `src/pipeline/frame_slot.h`

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
| OpenCV/CUDA P2 | `z_roi_cuda_template_match`, `z_roi_cuda_stereo_bm`, `z_roi_cuda_stereo_sgm`, `z_roi_ring_edge_profile` |
| 神经特征 | `z_roi_neural_feature` |
| patch/subpixel | `z_roi_center_patch`, `z_roi_multi_point` |
| fallback | `z_fallback`, `z_fallback_epipolar`, `z_fallback_template`, `z_fallback_feature_points` |
| 诊断 | disparity、support、confidence、std、bbox/circle、`pair_*`、sync watermark |

兼容旧 CSV 的 alias 字段仍存在: `z_yolo_bbox_pair`、`z_circle`、`z_subpixel`、`disparity_yolo`、`disparity_circle`、`disparity_subpixel`。它们分别对应 bbox center、circle center 和 ROI multi-point/subpixel，不应按新增测距方法重复统计。

`z_stereo` 和 `z` 也是兼容字段，不是新增候选方法。当前实时管线用 `buildDepthCandidateObservations()` 收集候选，并由 `selectLegacyDepthOutputCandidate()` 从 P0 几何/bbox 和退化 fallback 中选择兼容输出，写入 `z_stereo/z/raw_z/stereo_depth_source`。训练可靠性模型时应读取各个原始 `z_*` 候选字段；`z_stereo/z` 只作为旧在线 baseline 或诊断字段。

直接左右 YOLO pair 的误匹配诊断字段为: `pair_initial_disparity`, `pair_epipolar_dy`, `pair_y_tolerance`, `pair_size_ratio`, `pair_shifted_iou`, `pair_score`, `pair_bbox_prior_penalty`, `pair_positive_disparity`。这些字段从 `DEPTH_CANDIDATES` 记录级别开始写入；fallback 行保持 `-1/0`，用 `stereo_match_source` 区分匹配来源。

左右 YOLO 直接配对、单侧 fallback、circle source 和 `z_fallback`/`z_circle_center` 的完整字段语义见 [YOLO左右匹配分支与字段语义](YOLO左右匹配分支与字段语义.md)。

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
| 21 | CUDA 模板匹配 |
| 22 | CUDA StereoBM |
| 23 | CUDA StereoSGM |
| 24 | CUDA ring/edge profile |

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

实时 recorder 还会生成同名前缀 `*.frames.csv` sidecar。它记录每帧同步水印、`result_count`、direct/fallback 数量、pair gate 统计、P2 候选 observed/valid 计数，以及 P2 调度状态字段 `p2_depth_modes_enabled`、`p2_depth_mode_mask`、`p2_realtime_requested`、`p2_diagnostic_requested`、`p2_feature_job_count` 和 trigger mask。该 sidecar 用于区分“未触发 P2”和“触发但候选无效”。

OpenCV CUDA GFTT/LK 当前不回写主 `Object3D` 和 trajectory CSV header。录制配置通过 P2 diagnostic lane 写同名前缀 `*.p2_diagnostic.csv`，再由 `trajectory_fusion/dataset.py` 合并为训练候选 `z_roi_opencv_cuda_gftt_lk`。因此新增或排查 diagnostic 字段时要同时看主 CSV、`.frames.csv` 和 `.p2_diagnostic.csv`，不能只检查 `Object3D`。

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
