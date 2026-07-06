# 深度后续TODO

最后核对: 2026-07-04

## Fallback 深度

- [x] 为 epipolar fallback 增加独立字段 `z_fallback_epipolar`。
- [x] 为 epipolar fallback 增加独立视差字段 `disparity_fallback_epipolar`。
- [x] 同步更新 `Object3D`、TrajectoryRecorder CSV、`schema.md`、`dataset.py` 和 `check_dataset.py` 的 fallback epipolar 字段读取与统计。
- [x] fallback 命中球后，降低在线深度更新权重。
- [x] fallback 圆心深度不写入 `z_circle_center`。
- [x] 增加 fallback center patch / multi-point 一致性检查。

## P2 CUDA 候选

- [x] 按 [架构调度优化](架构调度优化.md) 设计 realtime lane / diagnostic lane，P2 迟到结果不能阻塞 P0/P1。
- [ ] 评估是否可将 P2 FeatureJob 从完整 `runRoiStage2Core()` 中拆出，支持低频、迟到和按 frame id 落盘。
  - 已完成 FeatureJob 配置、决策结构、async ROI 提交流程观测点和 P2 diagnostic lane 独立 worker/队列/max-in-flight 限流。
  - NX 已验证 async ROI 路径的 OpenCV CUDA ORB/Template/BM/SGM diagnostic 执行体、独立 GPU snapshot 和独立 CUDA stream。Template stride=10 可维持 100fps；stride=1 会抢占 GPU，不作为准入。
  - 已编写并在 NX 验证 diagnostic lane 迟到结果独立 CSV 落盘，按 frame id 写算法状态、视差/深度、support/attempted、bbox、anchor 和 deadline 状态；最终 smoke 日志 `test_logs/codex_diag_csv_final_20260704_052610/`，Template diagnostic-only `diag_valid/rows=6/227`，`diag_over_deadline=1`。
  - 颜色 patch、神经特征和主结果融合仍未迁移。
- [ ] 缩小或移除 `roi_postprocess_mutex_` 对独立 P2 job 的影响。
- [ ] OpenCV CUDA P2 使用独立 stream/scratch/matcher 实例，测试多个 OpenCV CUDA 算法并行是否存在隐式同步或尾延迟放大。
- [ ] BM/SGM 尽量在 GPU 内完成 robust 聚合，只下载最终 `disparity/support/stddev/confidence/valid`。
  - Template 已迁移为自研 CUDA Template/NCC + GPU peak reduce；2026-07-05 NX A/B `100.1fps`、`1372/1374` 有效、algo p95 `1.06ms`、max `4.89ms`。BM/SGM dense map 聚合仍待迁移。
- [ ] 增加 P2 选择性触发: P0/P1 分歧、pair gate 变差、fallback、帧间跳变或运动残差异常时才运行。
  - 已接入 fallback/direct/host-gray/BGR 初始触发开关和 direct pair quality 触发；selective no-trigger 会跳过 inline P2、BGR snapshot 和 host gray D2H。
  - P0/P1 分歧、帧间跳变和运动残差触发仍待实现。
- [ ] 增加 P2 attempted/not_attempted/valid/reject reason 统计，避免把未触发当成无效。
  - 矩阵报告层已有 `candidate_attempted`、`candidate_not_attempted`、`candidate_valid` 和粗粒度 `candidate_reject_reason`；精确逐帧 reject reason 仍需新增实时字段。
  - TrajectoryRecorder `.frames.csv` 已有 P2 调度状态、`p2_depth_mode_mask`、trigger mask 和 `observed/valid/feature/cuda/neural` 粗统计；仍需补精确 gate reject reason 和 async stale/drop 未发布帧统计。
- [ ] 将整帧 async snapshot 优化为 realtime ROI pack / diagnostic full snapshot 分层。
- [ ] 评估固定自研 CUDA P2 pipeline 是否适合 CUDA Graph 降低 launch overhead。
- [x] profiler 增加 p50/p90/p95/p99，用于 P2 准入；drop/accepted ratio 继续由矩阵脚本统计。
- [ ] debug 特殊情况才下载 score map、disparity map、keypoint/match 可视化数据。
- [x] 增加 P2 算法级可视化 artifact 基础能力，不能再用 realtime status zoom 代替特征点/采样点匹配图。
  - 已输出 OpenCV CUDA ORB、OpenCV CUDA GFTT/LK、VPI ORB 点对 overlay。
  - 已输出自研 CUDA Template/NCC 峰值图、OpenCV CUDA Template baseline 和 VPI Template score patch。
  - 已输出 OpenCV CUDA StereoSGM 少量 disparity 样本点和 CUDA Hough refined center。
- [ ] 补齐剩余 P2 artifact。
  - XFeat 和 SuperPoint 160/top64 已输出真实左右点对 overlay；SuperPoint 仍不准入。
  - Template/VPI Template 已输出 score map；后续只补单独 template patch/search window 裁剪。
  - BM/SGM/libSGM 已输出 32x32 ROI disparity patch；VPI Stereo 已输出 32x32 disparity + confidence patch。
  - color patch/color edge 已输出 gate 后 inlier samples，但现有图不含 case 参数、search window、score/reject，不足以证明 `wide_search` 匹配正确。
  - ring-edge 已输出采样点和候选视差，仍需 gate 后 inlier/outlier 与 reject reason。
  - VPI Harris/LK 更新测试仍未捕获有效 artifact。
- [ ] P2 性能准入先跑不带 `--debug-on-failure` 的矩阵；失败后再单独跑 debug capture。
- [x] realtime P2 测试强制避免 CPU fallback 自动介入和 host gray D2H。
- [ ] 可行 P2 优先迁移到 `DualYoloDepthGpuMatcher` batch kernel 或自研小 ROI CUDA kernel，降低 OpenCV CUDA 调用粒度成本。
- [x] 实测 ROI ring/edge profile matcher。
  - 已接入 `roi_ring_edge_profile` 默认关闭字段、自研 CUDA kernel、TrajectoryRecorder/训练读取和 `cuda_ring_edge_profile_diagnostic_only` 矩阵 case；有球 NX diagnostic `100.0fps`、`0/652` 有效，support=0/low confidence，不准入。
- [x] 复测 `roi_iou_region_color_patch_offline_tuned`。
- [x] 复测 `roi_iou_region_color_patch_wide_search`。
- [x] 复测 `patch_iou_color_edge_offline_tuned`。
- [x] 复测 `patch_iou_color_edge_wide_search`。
- [x] 接入 `z_roi_cuda_template_match` 小 ROI 极线匹配字段。
- [x] 实测 OpenCV CUDA TemplateMatching 小 ROI 极线匹配 baseline。
- [x] 实现并实测自研 CUDA Template/NCC 默认后端。
- [x] 实测 VPI CUDA Template Matching / NCC 小 ROI 极线匹配。
  - 已接真实 VPI CUDA diagnostic-only 后端并通过 NX build；有球测试 `99.4fps`、`0/645` 有效，不准入。
  - 已修正 VPI scratch 复用和 GPU peak reduce；下一轮只需重测修正后耗时，不改变当前不准入结论。
- [x] 接入 OpenCV CUDA StereoBM 小 ROI dense disparity 字段。
- [x] 实测 OpenCV CUDA StereoBM 裁剪 ROI / 小 `numDisparities`。
- [x] 实测 `opencv_cuda_stereo_bm_patch9`。
- [x] 接入 OpenCV CUDA StereoSGM 小 ROI dense disparity 字段。
- [x] 实测 OpenCV CUDA StereoSGM 裁剪 ROI / 小 `numDisparities`。
- [x] 实测 `opencv_cuda_stereo_sgm_patch9`。
- [x] 实测 VPI CUDA Stereo Disparity 裁剪 ROI / 小 `maxdisp`。
  - 已接真实 VPI CUDA diagnostic-only 后端并通过 NX build；有球测试 `100.1fps`、`0/632` 有效，不准入。
  - 已修正 VPI scratch 复用和 `initial_disp ±32px` residual 搜索；下一轮重测有效率和耗时。
- [x] 实测 Fixstars libSGM 裁剪 ROI / 小 `maxdisp`。
  - 08:30 diagnostic `70.5fps`、`0/80` 有效、algo max `5210.15ms`；已构建/链接但当前不准入。
- [x] 复测 `opencv_cuda_orb_fast48`。
- [x] 复测 `opencv_cuda_orb_wide_y`。
- [x] 实测 VPI CUDA ORB + BruteForceMatcher。
  - 10:49 targeted `95.9fps`、`16/603` 有效、algo p95 `9.60ms`；10:53 有少量点对 artifact，但每帧运行仍不准入。
- [x] 实测 VPI CUDA Harris + Pyramidal LK。
  - 10:49 targeted `93.8fps`、`48/606` 有效；有效率太低，不准入。
- [x] 实测 OpenCV CUDA GFTT/Harris + SparsePyrLK。
  - 已接 `roi_cuda_gftt_lk` diagnostic-only 后端并通过 NX build；10:49 targeted `94.0fps`、`501/572` 有效，11:08 有真实点对 artifact，但 FPS/长尾不准入。
- [x] 暂停 CUDA-SIFT 当前阶段实现。
  - 保留 `roi_cuda_sift` 配置/diagnostic unsupported 入口；不引入第三方 CUDA-SIFT，不用 CPU SIFT 代替实时 GPU 结论。
- [x] 停止推进 BRISK/AKAZE 作为当前实时/P2 路线。
  - 当前 OpenCV 没有项目可用的官方 CUDA BRISK/AKAZE 后端；CPU 版只保留 debug/offline，不再进入 P2 矩阵。
- [x] 实测 CUDA Canny/HoughCircles fallback circle refinement。
  - 已接 `roi_cuda_hough_circle` diagnostic-only 后端并通过 NX build；有球 diagnostic `88.0fps`、`6/539` 有效，不准入。

## P2 TensorRT 神经特征

- [x] 构建并实测 `xfeat_extractor_96.engine`。
- [x] 实测 `neural_xfeat_96_top32`。
- [x] 实测 `neural_xfeat_96_top64`。
- [x] 实测 `neural_xfeat_128_top32`。
- [x] 实测 `neural_xfeat_128_top96`。
- [x] 实测 `neural_xfeat_160_top64`。
- [x] 构建并实测 `superpoint_extractor_128_top64.engine`。
- [x] 构建并实测 `superpoint_extractor_160_top64.engine`。
- [x] 构建并实测 `superpoint_extractor_224_top64.engine`。
- [x] 构建并实测官方 ALIKED-t16 DCN `128/top64/b2` TensorRT plugin engine。
- [ ] 若继续 ALIKED，优先调 `128/top64/b2` matching/gate/topK；`160/224` vanilla engine 不再作为默认方向。
- [ ] 将 XFeat NMS/descriptor sampling/mutual-NN 从 CPU 后处理迁移到 GPU 或 fused engine。
  - XFeat/SuperPoint/ALIKED 勾选仅表示 engine 构建和实时矩阵实测完成；当前有效率/FPS 未通过默认准入。2026-07-04 有球测试中，XFeat 96/top32 为 `97.9fps`、`3/627` 有效，128/top32 为 `94.5fps`、`44/604` 有效，160/top64 为 `93.5fps`、`32/601` 有效；SuperPoint 224/top64 为 `59.2fps`、`151/402` 有效。2026-07-06 ALIKED-t16 DCN current `0/572`，gate-off `68/572`，联合 `NCC+XFeat+ALIKED-DCN` 中 `0/1034`。

## 动态录制验证

- [ ] 各算法各录制一帧代表性效果 zoom 图，覆盖 P0 几何候选、P1 multi-point/center patch、fallback epipolar 和后续新增候选。
- [ ] 对实际录制的动态排球片段做效果分析，按 direct pair、fallback L2R/R2L、候选深度有效率、median/MAD、帧间跳变和误匹配案例分组。

## 测试准入

- [ ] 新 fallback 算法先单算法隔离测试。
- [ ] 确认排球 ROI 上稳定匹配到纹理/边缘响应。
- [ ] 确认单算法 `Stage2_AsyncRoiWorker` 不超过 `10ms` deadline。
- [ ] NX 实时管线确认 100fps。
- [ ] 统计有效率、median、MAD 和帧间抖动。
- [ ] 满足 100fps 后再新增实时候选字段。
- [ ] 新字段进入训练数据 schema。

## 下一轮测试状态

- [x] 轨迹录制使用固定 `config/pipeline_record_p0p1.yaml` 和 `--recording-out` 递增文件名，主 CSV、`.frames.csv`、metadata、log 的采集流程已明确。
- [x] 预热流程已明确: 先录一段 warmup 文件确认 FPS、水印和 Stage2 状态，再从下一个递增文件名开始正式采集；warmup 文件不进入训练 manifest。
- [x] 下一轮可以直接按 [轨迹模型数据采集流程](轨迹模型数据采集流程.md) 做实机测试。
