# P2 算法 NX 测试排查指南

最后核对: 2026-07-04

本页用于明天在 NX 上测试 P2 算法时判断问题来源。P2 测试默认只证明“某个候选能否在实时管线里单项跑通”，不改变 P0/P1 默认采集集合。

## 是否已经完成本地准备

当前本地已完成:

- P2 默认关闭字段和矩阵 case 已准备好。
- `scripts/nx_algorithm_matrix_test.py` 已按单算法隔离生成临时 YAML。
- 报告会写出算法级 `algo_stage` 用时、完整 async worker 用时、deadline/drop 诊断、候选有效率和 CSV 行数；diagnostic-only case 会额外写 `p2_diag_*` 逐帧结果统计。
- `--debug-on-failure` 可在失败 case 后额外抓 feature debug 图和实时 zoom 图。

必须等 NX 实测才能确认:

- 新 P2 case 能否在 Jetson 编译后的真实 OpenCV CUDA/TensorRT 环境中运行。
- 单项是否稳定 100fps。
- 是否只是参数导致无效，还是算法本身在排球 ROI 上不稳定。

## 正式准入测试

准入测试先跑不带 debug 的矩阵。普通矩阵只运行 `./build/stereo_pipeline --config <临时yaml>`，脚本会关闭 `ros2.enable`，不启用 `--visualize`、`--debug-feature-matches` 或 `--debug-realtime-dump`。CSV 和 `.frames.csv` 会保留，用于统计候选有效率、FPS、deadline/drop 和 frame sidecar，这是采集路径的一部分。

不要在准入测速时打开 debug 图像开关。`--debug-on-failure` 会在失败/无有效/超 deadline case 后额外重新运行 `--debug-feature-matches` 和短时 `--debug-realtime-dump --debug-realtime-dump-stride 1`，会引入图像拷贝、PNG/JSON 写盘和 CPU 绘图。debug run 只能作为失败后的诊断图来源，不能把它的 FPS、`Stage2_AsyncRoiWorker` 或算法耗时当作准入结果。

推荐顺序:

1. 先跑不带 debug 的单项/矩阵，作为性能准入。
2. 如果 `diagnosis` 不是 `ok`，或候选有效率/深度质量异常，再对同一个 case 跑 `--debug-on-failure`。
3. debug 图只解释检测框、ROI、匹配效果和字段落点；真实性能仍以第一轮无 debug run 为准。

单项跑:

```bash
. /opt/ros/humble/setup.bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/12:${LD_LIBRARY_PATH:-}
python3 scripts/nx_algorithm_matrix_test.py \
  --out test_logs/nx_algorithm_matrix_one_$(date +%Y%m%d_%H%M) \
  --duration-sec 8 \
  --include-experimental \
  --cases opencv_cuda_template_match_patch9
```

失败后再诊断跑:

```bash
python3 scripts/nx_algorithm_matrix_test.py \
  --out test_logs/nx_algorithm_matrix_debug_$(date +%Y%m%d_%H%M) \
  --duration-sec 8 \
  --include-experimental \
  --debug-on-failure \
  --cases opencv_cuda_template_match_patch9
```

矩阵脚本里的 diagnostic-only case 会自动打开独立结果 CSV，路径形如:

```text
test_logs/<run>/<case>.p2_diagnostic.csv
```

需要手工逐 frame 对齐 diagnostic lane 迟到结果时，在临时 YAML 的 `performance` 中打开:

```yaml
p2_feature_job_scaffold_enabled: true
p2_realtime_lane_decision_enabled: false
p2_diagnostic_lane_decision_enabled: true
p2_diagnostic_stride: 10
p2_diagnostic_max_in_flight: 1
p2_diagnostic_results_enabled: true
p2_diagnostic_results_path: test_logs/p2_diag_template_stride10.csv
```

这个 CSV 记录 `frame_id`、算法 `mode/status`、视差/深度、`support/attempted`、左右 bbox、anchor、`queue_wait_ms`、`worker_elapsed_ms` 和 `over_deadline`。它不写主 trajectory CSV，也不回写 HybridDepth；第一次启用时要和不写 CSV 的同配置各跑一遍，确认文本落盘没有放大 diagnostic worker 尾延迟。

`--debug-on-failure` 会在对应目录下生成:

```text
debug/<case>/feature_matches/
debug/<case>/realtime_zoom/
logs/<case>.debug_feature_matches.log
logs/<case>.debug_realtime_dump.log
```

## 报告字段怎么读

`report.md` 的核心列:

| 字段 | 含义 | 判断 |
|---|---|---|
| `diagnosis` | 脚本按日志和 CSV 给出的第一层归因 | 先看这一列 |
| `algo_stage` | 当前 case 对应的算法 profiler 名 | 例如 `Stage2_OpenCVCudaStereoSGM` |
| `algo avg/p95/max` | 当前算法本身平均、p95 和最大耗时 | 判断算法是否慢，准入优先看 p95 |
| `worker avg/p95/max` | 完整 async ROI Stage2 平均、p95 和最大耗时 | 判断算法之外的 gate/拷贝/整理是否慢 |
| `over_deadline` | worker 超过 `async_roi_deadline_ms` 的次数 | 大于 0 表示结果可能迟到 |
| `stale/expired` | stale result、stale ready、expired pending、主线程 stale ROI 之和 | 大于 0 表示架构丢弃了过期结果 |
| `queue_drop` | pending/no-buffer/submit drop 之和 | 大于 0 表示 async 队列或 buffer 不够 |
| `candidate_valid/frames` | 目标 `z_*` 字段有效帧数/帧数 | 判断 gate 后是否真的产出深度 |
| `diag_valid/rows` | diagnostic 独立 CSV 中有效行/总行数 | diagnostic-only case 看这一列，不看 `candidate_valid` |
| `diag_over_deadline` | diagnostic worker 超过 `p2_diagnostic_deadline_ms` 的逐帧次数 | 大于 0 表示 diagnostic 结果迟到 |
| `accepted` | async 结果被主线程接收次数 | 为 0 时先看 detection/drop |
| `frame_cb_skip` | 原 ring slot 已复用，跳过图像 frame callback | 不影响 CSV 结果，但影响实时图像 debug |
| `host_gray` | 本 case 触发 host gray D2H 的次数 | GPU-only case 应接近 0 |

`summary.csv` 还保留所有详细字段，适合用脚本二次统计。

## 超时和架构丢弃

这里没有“强杀正在运行的 CUDA kernel/CPU 函数”。实际行为是:

1. worker 继续跑完当前 Stage2。
2. 如果完成时已经超过下一帧 YOLO-ready deadline，结果标记为过期。
3. 主线程不把过期结果写回在线输出。

判断方法:

- `algo avg/p95/max` 高，`worker avg/p95/max` 也高: 算法本身慢。
- `algo avg/p95/max` 低，`worker avg/p95/max` 高: 问题在图像拷贝、CPU gate、候选整理、写回或其他 stage。
- `over_deadline > 0`: worker 有实际超时。
- `stale/expired > 0`: 结果被 deadline/stale 规则丢弃。
- `fps` 接近 100 但 `candidate_valid` 很低: 主循环没被拖死，但候选迟到或 gate 后无效。
- `frame_cb_skip` 高但 `accepted` 高: 轨迹/CSV 结果正常接收，图像 frame callback 因 slot 复用被跳过。

常用日志定位:

```bash
rg -n "Stage2_OpenCVCuda|Stage2_NeuralFeatureMatch|Stage2_AsyncRoi|Stage2_DropStaleROI|Pipeline Performance" test_logs/<run>/logs/<case>.log
```

## 没有有效深度

按这个顺序看:

1. `status=failed`: 先看 `last error/warn` 和对应 log。
2. `diagnosis=no_detections`: 看 zoom 图中的左右检测框，先排查 YOLO/曝光/遮挡。
3. `accepted=0`: 看 async drop、no-buffer、stale、no-detections。
4. `candidate_valid=0` 且 `accepted>0`: 算法跑了但 gate 后无效，重点看匹配点、y 残差、视差 delta、sphere gate 和 confidence。
5. `target_rows=0`: 结果回调没有写目标 CSV，通常是无 accepted result 或配置/输出路径问题。

## Zoom 图怎么看

### 实时 zoom

`--debug-realtime-dump` 输出:

```text
frame_000123_zoom.png
frame_000123_summary.json
```

它显示:

- 左右检测框。
- circle/bbox 位置。
- 当前写出的 `z_*` 候选。
- `stereo_match_source`, `stereo_depth_source`, `pair_shifted_iou`, `epipolar_dy`。

限制:

- 它依赖 `frame_callback_`。重算法迟到导致原 ring slot 复用时，可能出现 `frame_cb_skip` 高、zoom 图少，但 CSV 结果仍存在。
- debug dump 会增加 D2H 和写盘，只用于诊断，不用于准入测速。

手动命令:

```bash
timeout 5 ./build/stereo_pipeline \
  --config test_logs/<run>/configs/<case>.yaml \
  --debug-realtime-dump \
  --debug-realtime-dump-dir test_logs/<run>/debug/<case>/realtime_zoom \
  --debug-realtime-dump-stride 1 \
  --debug-realtime-dump-max 20
```

### 单帧 feature debug

`--debug-feature-matches` 输出:

```text
left_detections.png
right_detections.png
selected_pair.png
feature_match_contact_sheet.png
summary.txt
*_matches.png
```

它用于确认:

- 左右 YOLO 框是否匹配错。
- ROI 是否截到球。
- 传统 CPU debug 点位是否集中在球面或背景。

限制:

- 当前 `debugFeatureMatchesOnce()` 主要画 sparse-lite 和 OpenCV CPU ORB/BRISK/AKAZE/SIFT debug 图。
- 它不等价于 OpenCV CUDA Template/BM/SGM、XFeat、SuperPoint 的内部匹配可视化。
- 对神经特征和离线 zoom，使用保存的 baseline pair 再跑 `tools/neural_feature_probe.py` 或 `tools/offline_volleyball_keypoint_probe.py`。

手动命令:

```bash
timeout 12 ./build/stereo_pipeline \
  --config test_logs/<run>/configs/<case>.yaml \
  --debug-feature-matches \
  --debug-feature-matches-dir test_logs/<run>/debug/<case>/feature_matches
```

## 离线复盘

如果实时 zoom 显示检测框正确，但候选仍无效，先录一小段彩色 baseline clip，再离线画匹配图:

```bash
./build/stereo_pipeline \
  --config config/pipeline_dual_yolo_roi.yaml \
  --record-baseline-clip \
  --baseline-out test_logs/p2_debug_clip \
  --baseline-frames 120 \
  --baseline-clips 1 \
  --baseline-image-mode bgr \
  --baseline-format png \
  --baseline-start-immediately
```

传统/patch 离线:

```bash
python3 tools/offline_volleyball_keypoint_probe.py \
  --left test_logs/p2_debug_clip/clip_000/left/0000.png \
  --right test_logs/p2_debug_clip/clip_000/right/0000.png \
  --out test_logs/p2_debug_clip/offline_probe
```

神经特征离线:

```bash
python3 tools/neural_feature_probe.py \
  --left test_logs/p2_debug_clip/clip_000/left/0000.png \
  --right test_logs/p2_debug_clip/clip_000/right/0000.png \
  --out test_logs/p2_debug_clip/neural_probe \
  --backends xfeat,superpoint_lightglue \
  --device cuda \
  --roi-size 128 \
  --top-k 64
```

离线结果只能解释点位和 gate，不代表 NX TensorRT/CUDA 实时性能。

## 快速归因表

| 现象 | 首看字段 | 下一步 |
|---|---|---|
| FPS 低 | `worker avg/p95/max`, `algo avg/p95/max` | 如果 `algo` 高，缩 ROI/top-k/点数；如果 worker 高，看 host gray、copy wait、CPU gate |
| FPS 正常但无候选 | `candidate_valid`, `accepted`, `diagnosis` | 跑 `--debug-on-failure`，看检测框和 ROI |
| `over_deadline` 高 | `algo p95/max`, `worker p95/max`, `stale/expired` | 算法进 diagnostic lane 或优化 GPU 后处理 |
| `host_gray` 非 0 | `async_need_host_gray_count` | 确认是否误开 CPU BRISK/AKAZE/SIFT/fallback |
| `accepted` 高、zoom 少 | `frame_cb_skip` | 正常 slot 复用现象；看 CSV，不用 frame callback 判断结果 |
| no engine | `status=skipped_missing_engine` | 先用模型转换脚本生成对应 TensorRT engine |
