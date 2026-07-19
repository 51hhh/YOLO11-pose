# P2 算法 NX 测试排查指南

最后核对: 2026-07-05

本页用于在 NX 上测试 P2 算法时判断问题来源。P2 测试默认只证明“某个新候选能否在实时管线里单项跑通”。当前 P1 已包含主 CSV `z_roi_multi_point` / `z_roi_center_patch` 和 sidecar `cuda_template` / `neural_xfeat` / `neural_superpoint`；color/SGM/VPI/GFTT-LK 等仍属于 P2 A/B 或历史项。

## 是否已经完成本地准备

当前本地已完成:

- P2 默认关闭字段和矩阵 case 已准备好；当前 `pipeline_record_p0p1.yaml` 保留 P0/P1 主 CSV 和 P1 sidecar 三算法。其它 P2 A/B 必须用临时 YAML 显式打开。
- `scripts/nx_algorithm_matrix_test.py` 已按单算法隔离生成临时 YAML。
- 报告会写出算法级 `algo_stage` 用时、完整 async worker 用时、deadline/drop 诊断、候选有效率和 CSV 行数；diagnostic-only case 会额外写 `p2_diag_*` 逐帧结果统计。
- `--debug-on-failure` 可在失败 case 后额外抓 feature debug 图和实时 zoom 图；`--debug-all` 可为每个选中 case 强制抓图。
- `p2_diagnostic_artifacts_enabled` 可为 diagnostic-only P2 输出真实算法级点对/峰值 PNG。

本轮已完成的 NX 有球实测:

- 08:30 全量性能矩阵: `test_logs/codex_p2_full_20260704_083048/`。
- 08:38 debug/zoom 矩阵: `test_logs/codex_p2_debug_20260704_083851/`，已同步本地，共 `226` 张 realtime zoom PNG。
- 10:49 targeted 性能复核: `test_logs/codex_p2_verify_20260704_104947/`。
- 10:53 artifact debug 抽样: `test_logs/codex_p2_artifact_debug_20260704_105356/`，已生成 OpenCV CUDA GFTT/LK、VPI Template、VPI ORB 的真实算法级样张。
- 11:08 final artifact 复测: NX run `codex_p2_artifacts_final_20260704_110837`，生成 `65` 张真实算法级 PNG；代表图同步到 `wiki/assets/p2_20260704_final/`。
- 13:02 inline artifact 重测: NX run `codex_p2_retest_inline_20260704_130245`，XFeat 生成真实匹配 PNG；color/color-edge 只生成 gate 后 sample overlay。代表图同步到 `wiki/assets/p2_20260704_inline/`。
- 13:06 dense artifact 重测: NX run `codex_p2_retest_dense_20260704_130617`，BM/SGM/VPI Stereo/libSGM 均生成 32x32 patch PNG；代表图同步到 `wiki/assets/p2_20260704_dense/`。
- 13:14 SGM valid artifact 单项复测: NX run `codex_p2_retest_sgm_valid_20260704_131452`，`opencv_cuda_stereo_sgm_diagnostic_only` 为 `96.6fps`、`3/549` diagnostic valid，并保留 valid 样张。
- Wiki 抽样图: `wiki/assets/p2_20260704/`，已在 [稳定100fps深度方法与训练入口](稳定100fps深度方法与训练入口.md) 的 P2 推荐排序总表中内联显示。注意这些图混合了 ROI/status zoom 和全帧 detection panel；性能/有效率以无 debug 矩阵 CSV/log 为准，图片只用于人工排查。逐项图片类型见 [P2算法效果与可视化审查](P2算法效果与可视化审查.md)。
- 算法级 artifact 样张: `wiki/assets/p2_20260704_artifacts/`、`wiki/assets/p2_20260704_final/`、`wiki/assets/p2_20260704_inline/` 和 `wiki/assets/p2_20260704_dense/`。这些图来自后端 `debug_matches`、peak 或 dense patch，不是 realtime status zoom。

历史参考:

- 核心 P2 inline: `test_logs/codex_ball_p2_core_20260704_063458/`。
- 神经特征 P2: `test_logs/codex_ball_p2_neural_20260704_063923/`。
- 可行候选 20s 长测: `test_logs/codex_ball_p2_viable_long_20260704_064200/`。
- 新增 diagnostic 后端: `test_logs/codex_ball_p2_experimental_diag_20260704_070148/`。
- GFTT/LK 修复复测: `test_logs/codex_ball_p2_gftt_lk_roi105_20260704_071106/`。
- selective 调度 smoke: `test_logs/codex_ball_p2_selective_after_exp_20260704_071202/`。
- 剩余 base/relaxed 变体: `test_logs/codex_ball_p2_remaining_variants_20260704_072216/`。
- base color/color-edge 20s 长测: `test_logs/codex_ball_p2_base_color_long_20260704_072551/`。

后续判断口径:

- `patch_iou_color_edge_wide_search` 10:49 targeted 为 `100.1fps`、`654/654` 有效、worker p95 `1.48ms`，但后续 artifact 显示错配，已退出默认配置。
- `iou_region_color_patch_wide_search` 10:49 targeted 为 `100.0fps`、`647/647` 有效、worker p95 `1.39ms`，但后续 artifact 显示错配，已退出默认配置。
- `diagnostic-only` case 不写主 CSV 候选字段，必须看 `diag_valid/rows`、`diag_over_deadline` 和算法 stage。
- VPI Template 10:49 targeted 为 `601/630` diagnostic 有效，但存在 `57.54ms` 长尾，只保留 diagnostic；11:08 artifact run 为 `46/81`，有 `20` 张峰值图。
- OpenCV CUDA GFTT/LK 10:49 targeted 为 `501/572` diagnostic 有效，但 FPS/长尾不准入；11:08 artifact run 为 `73/75`，有 `20` 张点对图。
- VPI Stereo、VPI Harris/LK、VPI ORB、Fixstars libSGM 都已经完成真实后端 isolated 实测；当前均不准入，详见 [实时特征算法矩阵](实时特征算法矩阵.md)。
- CUDA-SIFT 仍是 unsupported 入口，`status=unsupported` 不参与有效率判断。

## 正式准入测试

准入测试先跑不带 debug 的矩阵。普通矩阵只运行 `./build/stereo_pipeline --config <临时yaml>`，脚本会关闭 `ros2.enable`，不启用 `--visualize`、`--debug-feature-matches` 或 `--debug-realtime-dump`。CSV 和 `.frames.csv` 会保留，用于统计候选有效率、FPS、deadline/drop 和 frame sidecar，这是采集路径的一部分。

不要在准入测速时打开 debug 图像开关，也不要把写 artifact PNG 的 run 当作准入测速。`--debug-on-failure` 会在失败/无有效/超 deadline case 后额外重新运行 legacy CPU `--debug-feature-matches` 和短时 `--debug-realtime-dump --debug-realtime-dump-stride 1`，会引入图像拷贝、PNG/JSON 写盘和 CPU 绘图。`p2_diagnostic_artifacts_enabled` 还会额外下载/绘制算法级点对或峰值 PNG。debug run 只能作为失败后的诊断图来源，不能把它的 FPS、`Stage2_AsyncRoiWorker` 或算法耗时当作准入结果。

推荐顺序:

1. 先跑不带 debug 的单项/矩阵，作为性能准入。
2. diagnostic-only case 可直接看同轮 `<case>.p2_diagnostic.csv` 和 `<case>.p2_artifacts/`，但这轮不能作为 FPS 准入。
3. 如果 `diagnosis` 不是 `ok`，或候选有效率/深度质量异常，再对同一个 case 跑 `--debug-on-failure` 或 `--debug-all`。
4. debug 图只解释检测框、legacy CPU 点位、ROI、运行状态和字段落点；真实性能仍以第一轮无 debug run 为准。
5. GPU/VPI/TRT P2 的左右特征点对应只看 `p2_artifacts` 中的点对/峰值图，dense stereo 只看 `p2_artifacts` 中的 disparity/confidence patch；不能看 realtime status zoom。

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

强制每个选中 case 都抓 debug 和 P2 artifact:

```bash
python3 scripts/nx_algorithm_matrix_test.py \
  --out test_logs/nx_algorithm_matrix_debug_all_$(date +%Y%m%d_%H%M) \
  --duration-sec 4 \
  --include-experimental \
  --debug-all \
  --cases opencv_cuda_gftt_lk_diagnostic_only,vpi_template_match_diagnostic_only,vpi_orb_diagnostic_only
```

矩阵脚本里的 diagnostic-only case 会自动打开独立结果 CSV 和 artifact 目录，路径形如:

```text
test_logs/<run>/<case>.p2_diagnostic.csv
test_logs/<run>/<case>.p2_artifacts/
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
p2_diagnostic_artifacts_enabled: true
p2_diagnostic_artifacts_dir: test_logs/p2_diag_template_stride10_artifacts
p2_diagnostic_artifacts_max: 20
```

这个 CSV 记录 `frame_id`、算法 `mode/status`、视差/深度、`support/attempted`、左右 bbox、anchor、`debug_match_count`、`artifact_path`、`queue_wait_ms`、`worker_elapsed_ms` 和 `over_deadline`；`status` 为 `valid`、`invalid` 或 `unsupported`。它不写主 trajectory CSV，也不回写 HybridDepth。artifact PNG 会下载左右 gray snapshot 并绘制真实 `debug_matches` 点对/峰值；dense stereo 会额外画 32x32 disparity patch，VPI Stereo 还画 confidence patch。不允许用 anchor 均值伪造连线。

`--debug-on-failure` 会在对应目录下生成:

```text
debug/<case>/feature_matches/
debug/<case>/realtime_zoom/
debug/<case>/p2_artifacts/
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
| `debug match rows/median` | `debug_match_count > 0` 的行数和正样本中位数 | 大于 0 才说明有真实点对/峰值可画 |
| `artifacts` | 写出的算法级 PNG 数量 | 大于 0 时在 `debug_dirs` 的 `p2_artifacts=` 路径查看 |
| `p2 triggers` | selective trigger 命中计数 | 看 `low_iou/dy/low_conf/no_pair/skip`，区分为什么运行或未运行 P2 |
| `selective skip` | selective gating 实际跳过计数 | `inline/bgr/host` 分别表示跳过 inline P2、BGR snapshot、host gray D2H |
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
rg -n "Stage2_OpenCVCuda|Stage2_CudaRingEdge|Stage2_NeuralFeatureMatch|Stage2_AsyncRoi|Stage2_P2FeatureJob|Stage2_DropStaleROI|Pipeline Performance" test_logs/<run>/logs/<case>.log
```

## 没有有效深度

按这个顺序看:

1. `status=failed`: 先看 `last error/warn` 和对应 log。
2. `status=unsupported`: 说明只是配置/diagnostic 入口，目标库或真实后端还没接入，不参与有效率判断。
3. `diagnosis=no_detections`: 看 zoom 图中的左右检测框，先排查 YOLO/曝光/遮挡。
4. `accepted=0`: 看 async drop、no-buffer、stale、no-detections。
5. `candidate_valid=0` 且 `accepted>0`: 算法跑了但 gate 后无效，重点看匹配点、y 残差、视差 delta、sphere gate 和 confidence。
6. `target_rows=0`: 结果回调没有写目标 CSV，通常是无 accepted result 或配置/输出路径问题。

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
- 如果当前帧没有主 `Object3D` 回写，realtime dump 会输出左右全帧 detection panel，而不是 YOLO-IoU ROI crop。`vpi_*`、`fixstars_libsgm`、`cuda_hough_circle`、`cuda_ring_edge_profile`、`opencv_cuda_gftt_lk` 这类 diagnostic-only 抽样图常见这种情况。
- realtime zoom 是灰度图转 BGR 后画框；即使算法实际使用 BGR GPU snapshot，图像本身也不会显示彩色输入细节。
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
- 它不等价于自研 CUDA Template/NCC、OpenCV CUDA BM/SGM、XFeat、SuperPoint 的内部匹配可视化。
- 它也不等价于 VPI ORB、VPI Harris/LK、Fixstars libSGM、CUDA Hough 或自研 color patch/ring-edge 的内部可视化。
- 对神经特征和离线 zoom，使用保存的 baseline pair 再跑 `tools/neural_feature_probe.py` 或 `tools/offline_volleyball_keypoint_probe.py`。

### 需要算法级匹配图时

实时 P2 diagnostic CSV 会写聚合后的 `anchor_cx/right_anchor_cx/support/depth`，新版本还写 `debug_match_count` 和 `artifact_path`。`anchor_*` 字段只能定位候选代表点，不能还原每条左右点对；不要用 anchor 均值画假连线。

当前 artifact 状态:

- 已有: OpenCV CUDA GFTT/LK 左右点对 overlay。
- 已有: OpenCV CUDA ORB 左右点对 overlay。
- 已有: VPI ORB 左右点对 overlay，来自 10:53 debug run；11:08 final 段未复现。
- 已有: 自研 CUDA Template/NCC 单点峰值图、OpenCV CUDA Template baseline 和 VPI Template 单点峰值图；13:49 更新测试已补旧 OpenCV/VPI `SCORE PATCH`。
- 已有: OpenCV CUDA StereoSGM 有效 disparity 样本点和 32x32 disparity patch。
- 已有: OpenCV CUDA StereoBM、VPI Stereo、Fixstars libSGM 的 32x32 dense patch；VPI Stereo 额外有 confidence patch。
- 已有: XFeat TensorRT 真实左右 keypoint pair overlay。
- 已有: SuperPoint TensorRT 160/top64 真实左右 keypoint pair overlay；仅作调试证据，不准入。
- 部分已有: 自研 color patch / color-edge gate 后 inlier samples；现有图不含 search window、score/reject，也不区分 base / wide_search，不能作为完整匹配证明。
- 已有: CUDA ring-edge profile 最佳候选视差下的三圈采样点；当前候选仍 invalid。
- 已有: CUDA Hough circle 左右 refined center。
- 暂无: VPI Harris/LK 本轮没有有效 artifact；CUDA-SIFT 仍是 unsupported。

仍需继续补的 artifact:

- OpenCV CUDA ORB/GFTT-LK、VPI ORB: 已有样张，但还需要更多 failure/low-support 抽样。
- Template/VPI Template: 已有峰值点和 score map；后续只补单独 template patch/search window 裁剪。
- BM/SGM/VPI Stereo/libSGM: 已有 bounded ROI patch；后续只补更完整的 reject reason 和多帧 failure 抽样。
- color/color-edge: 重抓带 case 参数、search window、score/reject 的 artifact；现有 sample overlay 只保留为排错线索。
- ring-edge: 已有采样点和候选视差；后续补 gate 后 inlier/outlier 和 reject reason。

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
