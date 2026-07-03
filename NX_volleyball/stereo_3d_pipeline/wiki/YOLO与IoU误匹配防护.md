# YOLO与IoU误匹配防护

本页记录左右 YOLO 配对、IoU/ROI 基线搜索和单侧漏检 fallback 的误匹配防护规则。目标是: 多候选、单侧退化或背景干扰时，宁可不输出深度，也不能把背景当成球。

## 风险来源

长基线下，同一球在左右图的 x 位置差异很大。只用“y 接近 + 尺寸接近”配对会有两个问题:

- 同一高度的背景误检可能通过左右 bbox gate。
- 单侧 YOLO 丢失时，极线搜索可能在预测窗口内找到背景边缘。
- IoU/patch 特征点数量多不等于可靠，弱纹理排球上容易出现成片离群点。
- 离线自动 ROI 如果直接使用颜色碎片 bbox，会把球面局部色块和背景色块混在一起。

## 实时防护顺序

直接双侧 YOLO 匹配:

1. 类别一致。
2. 正视差且小于 `stereo.max_disparity`。
3. 校正后中心 y 差小于 `epipolar_y_tolerance` 的基础/自适应门限。
4. 左右 bbox 宽高比例小于 `max_size_ratio`。
5. 把右框按中心视差平移到左目坐标后，shifted IoU 大于 `min_shifted_iou`。
6. 用 bbox 宽度和球直径反推物理视差，和左右中心视差做一致性排序惩罚。
7. 对全部左右候选做全局排序后再一对一占用，避免高置信假左框先抢走真右框。
8. 后续 circle/edge/feature/IoU 候选必须继续走各自的 y、互反查、MAD/RANSAC 和球体半径 gate。

这里的全局排序是实时路径的轻量 assignment: 先把所有可行左右 pair 按增强 score 排序，再按左右索引一对一占用。单球实时场景下开销近似为 `O(N log N)`，可以防止“按左框顺序局部贪心”造成的误占用。若后续进入多球、多目标跟踪，应把这一层升级成精确最小代价匹配，但仍保留同样的几何硬 gate。

单侧漏检 fallback:

1. 由 tracker/hybrid depth 或 bbox 宽度估计 expected disparity。
2. 只在 expected disparity 附近的有限极线窗口搜索。
3. 搜索结果必须满足中心偏移、半径比例、y 误差、正视差和深度范围。
4. 模板或特征 fallback 只能作为候选，不覆盖几何 gate。
5. `fallback_template` 和 `fallback_feature_points` 默认不进 100fps 主路径，需要专项 profile。

模板 fallback 不能只看最高相关峰。OpenCV `matchTemplate` 的教程也只是给出响应图和峰值位置；在本项目里必须额外检查 expected disparity、y 误差、中心误差、正视差、深度范围和二峰质量，防止背景纹理在极线窗口内胜出。二峰质量需要排除最佳峰附近的小邻域，否则相邻像素会被误当成第二峰。

离线 ROI 自动检测:

1. 先把排球颜色碎片合并为球尺寸粗 ROI。
2. 再使用左右 ROI pair gate。
3. Hough refine 只在粗 ROI 附近选择圆。
4. 候选排序加入球物理半径一致性，避免选到背景圆。

## 当前配置

位置: `config/pipeline_dual_yolo_roi.yaml`

```yaml
detector:
  dual_yolo:
    epipolar_y_tolerance: 12.0
    max_size_ratio: 2.0
    min_shifted_iou: 0.05
    bbox_disparity_consistency_ratio: 0.30
    bbox_disparity_consistency_min_px: 45.0
    bbox_disparity_penalty_scale: 0.75
    fallback_search_margin_px: 48
    fallback_max_width_px: 220
```

`bbox_disparity_*` 是排序惩罚，不是硬拒绝。单候选正常帧不会因为这个惩罚被直接丢弃；多候选时会优先选择物理视差更合理的左右框。

## 离线退化测试

灰度连续 clip 用于测试 YOLO/IoU 退化稳定性:

```bash
python3 NX_volleyball/stereo_3d_pipeline/tools/offline_yolo_iou_fallback_regression.py \
  --clip NX_volleyball/stereo_3d_pipeline/test_logs/nx_recordings/codex_2s_gray_20260702_142253/clip_20260702_142255_01 \
  --calib NX_volleyball/calibration/stereo_calib.yaml \
  --out NX_volleyball/stereo_3d_pipeline/test_logs/nx_recordings/codex_2s_gray_20260702_142253/yolo_iou_fallback_regression_20260703_full \
  --max-frames 0 \
  --template-min-score-gap 0.010 \
  --template-peak-exclusion-radius 12 \
  --fail-on-regression
```

测试内容:

- 正常左右 YOLO pair gate。
- 注入低视差假右框，确认 enhanced score 仍选真框。
- 注入高视差假右框，确认 enhanced score 仍选真框。
- 注入低/高视差假左框，并把假左框排在真左框之前，确认全局配对排序不会被左侧误检抢占。
- 模拟右侧 YOLO 丢失，用左侧模板在右侧极线窗口搜索。
- 模拟左侧 YOLO 丢失，用右侧模板在左侧极线窗口搜索。

当前 200 帧结果:

| 指标 | 结果 |
|---|---:|
| normal pair pass rate | `1.000` |
| fake right low selected true | `1.000` |
| fake right high selected true | `1.000` |
| fake left low selected true | `1.000` |
| fake left high selected true | `1.000` |
| right missing pass rate | `1.000` |
| left missing pass rate | `1.000` |
| right missing center p95 | `12.52px` |
| left missing center p95 | `12.82px` |
| right missing disparity p95 | `12.50px` |
| left missing disparity p95 | `12.69px` |
| right missing score gap min | `0.023` |
| left missing score gap min | `0.035` |
| template elapsed p95 | `206.73ms` |

注意: 这个灰度测试验证的是 YOLO/IoU/fallback 搜索稳定性，不验证彩色 LAB/label patch 特征质量。彩色特征质量必须使用 `--baseline-image-mode bgr --baseline-format png` 录制的 PNG clip。

## 连续特征评估

彩色 PNG clip 录制后，使用连续帧 probe 汇总稳定性:

```bash
python3 NX_volleyball/stereo_3d_pipeline/tools/offline_volleyball_sequence_probe.py \
  --clip baseline_clips/clip_YYYYMMDD_HHMMSS_01 \
  --calib NX_volleyball/calibration/stereo_calib.yaml \
  --out NX_volleyball/stereo_3d_pipeline/test_logs/sequence_probe_latest
```

输出:

- `frames.csv`: 每帧 probe 是否成功和总耗时。
- `per_frame_methods.csv`: 每帧每方法的点数、深度、validation 状态。
- `method_summary.csv`: pass rate、有效点数、深度 MAD、帧间抖动、耗时和运动残差。
- `report.md`: 人可读汇总。

验收建议:

| 指标 | 建议门限 |
|---|---:|
| validation pass rate | `>= 0.80` |
| valid points median | `>= 8` |
| depth MAD | `<= 0.03m` |
| frame jitter p95 | `<= 0.06m` |
| motion residual p95 | `<= 0.08m` |
| NX 实时 p95 耗时 | 按 100fps 预算单独 profile |

## 参考资料

- OpenCV Feature Matching + Homography: Lowe ratio test、RANSAC inlier mask 和“不够匹配点就拒绝”的工程模式。
  https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
- OpenCV Template Matching: 响应峰值只能定位候选，项目内还必须叠加极线、视差和物理尺寸 gate。
  https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- OpenCV calib3d `findHomography`: RANSAC/LMEDS/RHO 等鲁棒估计接口。
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- SciPy `linear_sum_assignment`: 标准线性和 assignment 问题。当前 C++ 为单球实时轻量全局排序，后续多球时可参考精确最小代价分配语义。
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- 立体校正后的基本约束: 正视差、极线 y 残差、视差深度范围和物理尺寸一致性应先于特征点平均值。
