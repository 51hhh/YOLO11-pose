# CUDA Template/NCC 深度候选

最后核对: 2026-07-05

本页记录 `z_roi_cuda_template_match` 当前实现。它已经不再默认调用 `cv::cuda::TemplateMatching`，而是使用项目内自研 CUDA 小窗口 Template/NCC kernel；OpenCV CUDA 版本仅作为 A/B baseline 保留。该字段当前归入 P1 sidecar 训练候选，不参与 legacy `z_stereo/obj.z`。

## 字段和入口

| 项 | 当前值 |
|---|---|
| 主 CSV 字段 | `z_roi_cuda_template_match` |
| Sidecar mode | `cuda_template` |
| 视差字段 | `disparity_roi_cuda_template_match` |
| 质量字段 | `roi_cuda_template_match_support`, `roi_cuda_template_match_std_px`, `roi_cuda_template_match_confidence` |
| 配置开关 | `detector.dual_yolo.depth_modes.roi_cuda_template_match` |
| realtime profile | `Stage2_CudaTemplateNccMatch` |
| 代码入口 | `matchCudaTemplateDisparityGPU()` |
| 自研 kernel | `computeCudaTemplateCcoeffNormedScoreMap()` + `findCudaTemplateScorePeak()` |
| legacy baseline | `STEREO_CUDA_TEMPLATE_BACKEND=opencv` |

该候选只在左右 YOLO direct pair 成功后运行；单侧漏检 fallback 不写这个字段。输入是校正后的左右灰度 GPU 图，不需要 BGR，也不触发 host gray D2H。

## 当前算法

当前实现不是通用关键点提取，也不是多点描述子匹配。它匹配的是一个单点模板:

```text
left YOLO bbox center
  -> 左图中心 11x11 gray patch, 由 subpixel_patch_radius=5 得到
  -> 右图 predicted_x = left_x - initial_disp
  -> 右图小极线窗口, 默认 x 方向 +-8px, y 方向 +-2px
  -> 每个候选位置计算 CCOEFF_NORMED / NCC score
  -> GPU peak reduce 只回传最终 peak
  -> y residual / overlap / sphere radius / disparity delta gate
  -> 写出单个深度候选
```

默认窗口来自现有 subpixel 参数: `subpixel_patch_radius=5` 时 patch 为 `11x11`；`subpixel_search_radius_px=8` 时右图 score map 为 `17x5`。如果模板方差很低，kernel 会退化为 RMS-SSD similarity，避免 OpenCV `TM_CCOEFF_NORMED` 在低纹理区域出现不稳定响应。score 里叠加了很小的中心先验，用于打破平坦响应时的左上角漂移。

输出语义要注意:

- `support=1` 表示单个模板峰值，不可和 GFTT/LK、XFeat、dense stereo 的多点 support 直接比较。
- `std_px=0` 表示单点没有离散度，不表示它比多点候选更稳定。
- `confidence` 是模板 peak score，仍需结合 gate 结果、深度偏置和连续帧统计使用。
- 该字段不参与 legacy `z_stereo/obj.z` 选择；训练应直接读取原始 `z_roi_cuda_template_match` 或 sidecar 合并后的同名字段。

## NX A/B 结果

本轮 A/B 使用同一实时管线和同一配置，仅切换 template 后端。

| 后端 | 数据路径 | FPS | 有效率 | 深度 median/MAD | 算法 avg/p95/p99/max | 结论 |
|---|---|---:|---:|---:|---:|---|
| 自研 CUDA Template/NCC | `/home/nvidia/trajectory_dataset/cuda_template_custom_tiebreak_20260705_074117/` | `100.1` | `1372/1374` | `3.4831/0.0019m` | `0.30/1.06/1.20/4.89ms` | 当前默认实现；满足单算法 100fps 准入 |
| OpenCV CUDA baseline | `/home/nvidia/trajectory_dataset/cuda_template_baseline_final_20260705_074215/` | `99.8` | `60/1374` | `3.4847/0.0105m` | `1.23/2.87/4.80/26.45ms` | 只保留 A/B baseline；不作为默认实现 |

2026-07-05 P0/P1+NCC+XFeat+SuperPoint 联合 run `/home/nvidia/trajectory_dataset/p0p1_xfeat_superpoint_ncc_review_20260705_103601`: 主线 `99.95fps`，`mode=cuda_template` sidecar `317/317` 有效，median/MAD `3.5032/0.0020m`，algo `avg/p95/max=1.08/1.07/4.32ms`。该结果是把 NCC 归入 P1 sidecar 的依据。

自研版本的改进点不是更复杂的匹配语义，而是去掉 OpenCV CUDA wrapper、score map 下载和隐式同步长尾，只在 GPU 内完成 score 计算和 peak reduce。

## 样张

以下图片来自 debug artifact run，只用于审查匹配峰值位置，不计入 FPS 准入:

```text
test_logs/cuda_template_custom_tiebreak_debug_20260705_074358/
```

| 样张 | 图 |
|---|---|
| valid peak 1 | <img src="assets/cuda_template_ncc_20260705/frame_000002_valid.png" width="360"> |
| valid peak 2 | <img src="assets/cuda_template_ncc_20260705/frame_000054_valid.png" width="360"> |

## 使用边界

- 该方法适合作为 P1 sidecar 训练候选或条件触发候选，不应直接提升为 legacy 在线主深度。
- 它依赖左右 YOLO direct pair 的 `initial_disp`，搜索窗口很小；如果左右 bbox 配对错了，它不会自行纠正大范围错配。
- 它只看球心附近局部灰度 patch，不能利用排球彩色条纹的全局结构。
- 后续如果要提高鲁棒性，应优先扩展为多锚点模板 batch 或 ring/edge profile，而不是把当前单点 `support=1` 当作强观测。
