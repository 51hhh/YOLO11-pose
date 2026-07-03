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

- [ ] 实测 ROI ring/edge profile matcher。
- [ ] 实测 OpenCV CUDA TemplateMatching 小 ROI 极线匹配。
- [ ] 实测 VPI CUDA Template Matching / NCC 小 ROI 极线匹配。
- [ ] 实测 OpenCV CUDA StereoBM 裁剪 ROI / 小 `numDisparities`。
- [ ] 实测 OpenCV CUDA StereoSGM 裁剪 ROI / 小 `numDisparities`。
- [ ] 实测 VPI CUDA Stereo Disparity 裁剪 ROI / 小 `maxdisp`。
- [ ] 实测 Fixstars libSGM 裁剪 ROI / 小 `maxdisp`。
- [ ] 实测 VPI CUDA Harris + Pyramidal LK。
- [ ] 实测 OpenCV CUDA GFTT/Harris + SparsePyrLK。
- [ ] 实测 CUDA Canny/HoughCircles fallback circle refinement。

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
