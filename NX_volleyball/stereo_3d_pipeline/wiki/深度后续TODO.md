# 深度后续TODO

最后核对: 2026-07-04

## Fallback 深度

- [ ] 为 epipolar fallback 增加独立字段 `z_fallback_epipolar`。
- [ ] 为 epipolar fallback 增加独立视差字段 `disparity_fallback_epipolar`。
- [ ] 同步更新 `Object3D`、TrajectoryRecorder CSV、`schema.md`、`dataset.py` 和 `check_dataset.py` 的 fallback epipolar 字段读取与统计。
- [ ] fallback 命中球后，降低在线深度更新权重。
- [ ] fallback 圆心深度不写入 `z_circle_center`。
- [ ] 增加 fallback center patch / multi-point 一致性检查。

## P2 CUDA 候选

- [ ] 按 [架构调度优化](架构调度优化.md) 设计 realtime lane / diagnostic lane，P2 迟到结果不能阻塞 P0/P1。
- [ ] 评估是否可将 P2 FeatureJob 从完整 `runRoiStage2Core()` 中拆出，支持低频、迟到和按 frame id 落盘。
- [ ] 缩小或移除 `roi_postprocess_mutex_` 对独立 P2 job 的影响。
- [ ] OpenCV CUDA P2 使用独立 stream/scratch/matcher 实例，测试多个 OpenCV CUDA 算法并行是否存在隐式同步或尾延迟放大。
- [ ] Template/BM/SGM 尽量在 GPU 内完成 peak/robust 聚合，只下载最终 `disparity/support/stddev/confidence/valid`。
- [ ] 增加 P2 选择性触发: P0/P1 分歧、pair gate 变差、fallback、帧间跳变或运动残差异常时才运行。
- [ ] 增加 P2 attempted/not_attempted/valid/reject reason 统计，避免把未触发当成无效。
- [ ] 将整帧 async snapshot 优化为 realtime ROI pack / diagnostic full snapshot 分层。
- [ ] 评估固定自研 CUDA P2 pipeline 是否适合 CUDA Graph 降低 launch overhead。
- [x] profiler 增加 p50/p90/p95/p99，用于 P2 准入；drop/accepted ratio 继续由矩阵脚本统计。
- [ ] debug 特殊情况才下载 score map、disparity map、keypoint/match 可视化数据。
- [ ] P2 性能准入先跑不带 `--debug-on-failure` 的矩阵；失败后再单独跑 debug capture。
- [ ] realtime P2 测试强制避免 CPU fallback 自动介入和 host gray D2H。
- [ ] 可行 P2 优先迁移到 `DualYoloDepthGpuMatcher` batch kernel 或自研小 ROI CUDA kernel，降低 OpenCV CUDA 调用粒度成本。
- [ ] 实测 ROI ring/edge profile matcher。
- [ ] 复测 `roi_iou_region_color_patch_offline_tuned`。
- [ ] 复测 `roi_iou_region_color_patch_wide_search`。
- [ ] 复测 `patch_iou_color_edge_offline_tuned`。
- [ ] 复测 `patch_iou_color_edge_wide_search`。
- [x] 接入 OpenCV CUDA TemplateMatching 小 ROI 极线匹配字段。
- [ ] 实测 OpenCV CUDA TemplateMatching 小 ROI 极线匹配。
- [ ] 实测 `opencv_cuda_template_match_patch9`。
- [ ] 实测 VPI CUDA Template Matching / NCC 小 ROI 极线匹配。
- [x] 接入 OpenCV CUDA StereoBM 小 ROI dense disparity 字段。
- [ ] 实测 OpenCV CUDA StereoBM 裁剪 ROI / 小 `numDisparities`。
- [ ] 实测 `opencv_cuda_stereo_bm_patch9`。
- [x] 接入 OpenCV CUDA StereoSGM 小 ROI dense disparity 字段。
- [ ] 实测 OpenCV CUDA StereoSGM 裁剪 ROI / 小 `numDisparities`。
- [ ] 实测 `opencv_cuda_stereo_sgm_patch9`。
- [ ] 实测 VPI CUDA Stereo Disparity 裁剪 ROI / 小 `maxdisp`。
- [ ] 实测 Fixstars libSGM 裁剪 ROI / 小 `maxdisp`。
- [ ] 复测 `opencv_cuda_orb_fast48`。
- [ ] 复测 `opencv_cuda_orb_wide_y`。
- [ ] 核对 NX VPI ORB 是否支持 CUDA backend；若支持则实现 VPI ORB P2 后端。
- [ ] 实测 VPI CUDA Harris + Pyramidal LK。
- [ ] 实测 OpenCV CUDA GFTT/Harris + SparsePyrLK。
- [ ] 调研并验证 CUDA-SIFT 第三方依赖是否能限制在 ROI/top-k 内稳定 10ms。
- [ ] 调研 BRISK/AKAZE 是否存在可维护 CUDA/VPI 后端；否则只保留 CPU debug。
- [ ] 实测 CUDA Canny/HoughCircles fallback circle refinement。

## P2 TensorRT 神经特征

- [ ] 构建并实测 `xfeat_extractor_96.engine`。
- [ ] 实测 `neural_xfeat_96_top32`。
- [ ] 实测 `neural_xfeat_96_top64`。
- [ ] 实测 `neural_xfeat_128_top32`。
- [ ] 实测 `neural_xfeat_128_top96`。
- [ ] 实测 `neural_xfeat_160_top64`。
- [ ] 构建并实测 `superpoint_extractor_128_top64.engine`。
- [ ] 构建并实测 `superpoint_extractor_160_top64.engine`。
- [ ] 构建并实测 `superpoint_extractor_224_top64.engine`。
- [ ] 若可获得 ALIKED TensorRT engine，实测 `neural_aliked_160_top64`。
- [ ] 若可获得 ALIKED TensorRT engine，实测 `neural_aliked_224_top64`。
- [ ] 将 XFeat NMS/descriptor sampling/mutual-NN 从 CPU 后处理迁移到 GPU 或 fused engine。

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
