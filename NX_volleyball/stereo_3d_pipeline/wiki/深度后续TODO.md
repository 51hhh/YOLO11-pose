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
- [ ] Template/BM/SGM 尽量在 GPU 内完成 peak/robust 聚合，只下载最终 `disparity/support/stddev/confidence/valid`。
  - TemplateMatching 已新增 CUDA score peak reduction，只下载最终 peak 结构；NX `opencv_cuda_template_match_patch9` 实测仍 `0/414` 有效，p95 `3.91ms`、max `43.30ms`，不准入。BM/SGM dense map 聚合仍待迁移。
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
- [ ] P2 性能准入先跑不带 `--debug-on-failure` 的矩阵；失败后再单独跑 debug capture。
- [x] realtime P2 测试强制避免 CPU fallback 自动介入和 host gray D2H。
- [ ] 可行 P2 优先迁移到 `DualYoloDepthGpuMatcher` batch kernel 或自研小 ROI CUDA kernel，降低 OpenCV CUDA 调用粒度成本。
- [ ] 实测 ROI ring/edge profile matcher。
  - 已接入 `roi_ring_edge_profile` 默认关闭字段、自研 CUDA kernel、TrajectoryRecorder/训练读取和 `cuda_ring_edge_profile_diagnostic_only` 矩阵 case；剩余有球 NX 实测。
- [ ] 复测 `roi_iou_region_color_patch_offline_tuned`。
- [x] 复测 `roi_iou_region_color_patch_wide_search`。
- [ ] 复测 `patch_iou_color_edge_offline_tuned`。
- [x] 复测 `patch_iou_color_edge_wide_search`。
- [x] 接入 OpenCV CUDA TemplateMatching 小 ROI 极线匹配字段。
- [x] 实测 OpenCV CUDA TemplateMatching 小 ROI 极线匹配。
- [x] 实测 `opencv_cuda_template_match_patch9`。
- [ ] 实测 VPI CUDA Template Matching / NCC 小 ROI 极线匹配。
- [x] 接入 OpenCV CUDA StereoBM 小 ROI dense disparity 字段。
- [x] 实测 OpenCV CUDA StereoBM 裁剪 ROI / 小 `numDisparities`。
- [x] 实测 `opencv_cuda_stereo_bm_patch9`。
- [x] 接入 OpenCV CUDA StereoSGM 小 ROI dense disparity 字段。
- [x] 实测 OpenCV CUDA StereoSGM 裁剪 ROI / 小 `numDisparities`。
- [x] 实测 `opencv_cuda_stereo_sgm_patch9`。
- [ ] 实测 VPI CUDA Stereo Disparity 裁剪 ROI / 小 `maxdisp`。
- [ ] 实测 Fixstars libSGM 裁剪 ROI / 小 `maxdisp`。
- [x] 复测 `opencv_cuda_orb_fast48`。
- [x] 复测 `opencv_cuda_orb_wide_y`。
- [ ] 核对 NX VPI ORB 是否支持 CUDA backend；若支持则实现 VPI ORB P2 后端。
- [ ] 实测 VPI CUDA Harris + Pyramidal LK。
- [ ] 实测 OpenCV CUDA GFTT/Harris + SparsePyrLK。
- [ ] 调研并验证 CUDA-SIFT 第三方依赖是否能限制在 ROI/top-k 内稳定 10ms。
- [ ] 调研 BRISK/AKAZE 是否存在可维护 CUDA/VPI 后端；否则只保留 CPU debug。
- [ ] 实测 CUDA Canny/HoughCircles fallback circle refinement。

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
- [ ] 若可获得 ALIKED TensorRT engine，实测 `neural_aliked_160_top64`。
- [ ] 若可获得 ALIKED TensorRT engine，实测 `neural_aliked_224_top64`。
- [ ] 将 XFeat NMS/descriptor sampling/mutual-NN 从 CPU 后处理迁移到 GPU 或 fused engine。
  - XFeat/SuperPoint 勾选仅表示 engine 构建和实时矩阵实测完成；当前有效率/FPS 未通过默认准入。2026-07-04 追加单项测试中，XFeat 128/top32 为 `94.7fps`、`24/375` 有效，128/top96 为 `96.4fps`、`2/402` 有效，160/top64 为 `87.9fps`、`6/398` 有效。ALIKED 仍卡在 `torchvision::deform_conv2d`，没有可用 TensorRT engine。

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

## 明日测试状态

- [x] 轨迹录制使用固定 `config/pipeline_record_p0p1.yaml` 和 `--recording-out` 递增文件名，主 CSV、`.frames.csv`、metadata、log 的采集流程已明确。
- [x] 预热流程已明确: 先录一段 warmup 文件确认 FPS、水印和 Stage2 状态，再从下一个递增文件名开始正式采集；warmup 文件不进入训练 manifest。
- [x] 明天可以直接按 [轨迹模型数据采集流程](轨迹模型数据采集流程.md) 做实机测试。
