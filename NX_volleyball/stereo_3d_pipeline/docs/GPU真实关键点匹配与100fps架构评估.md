# GPU 真实关键点匹配与 100fps 架构评估

日期: 2026-07-02

## 结论

当前 100fps 实时路径应继续使用默认几何候选: bbox/circle/edge/radial/edge-pair。它在 NX 实机上保持 100fps，连续帧深度稳定。

当前关键点/特征点路径分为五类:

1. 默认几何 CUDA 候选: bbox/circle/edge/radial/edge-pair，当前生产 100fps 主路径。
2. 自研 CUDA lite 特征: 跑在 GPU，速度快，但不是 ORB/BRISK/AKAZE/SIFT 的真实算法，历史实测连续帧有效率为 0。
3. OpenCV CUDA ORB: 当前 NX OpenCV 4.10 已有 `cudafeatures2d` 和 `cv::cuda::ORB`，实时 `roi_orb_points` 已接到真实 CUDA ORB，但同步路径实测 78-79fps 且 0/512 有效。
4. CPU/OpenCV debug 特征: 真实 ORB/BRISK/AKAZE，但只用于单帧 debug，不适合 100fps。
5. XFeat TensorRT extractor: 真实神经特征 extractor，但当前实现把输出拷回 CPU 做 NMS、descriptor 采样和互反查匹配，实测 75-90fps，不能作为同帧同步 100fps 主路径。

要实现“真实算法 + 100fps”，必须重构为异步 FeatureJob: YOLO 后只启动特征任务，不在 Stage2 等待；以下一帧 YOLO 完成为 deadline，feature 未完成就丢弃该候选，默认几何深度照常输出。

## 数据来源

实机:

- 主机: `nvidia@10.42.0.149`
- 平台: Jetson Orin NX Super
- 性能模式: `MAXN_SUPER` + `jetson_clocks`
- 配置: `config/pipeline_dual_yolo_roi.yaml`
- 左相机: `00D39342665`
- 右相机: `00219471413`
- 触发: 100Hz PWM

本地已保存数据:

```text
test_logs/nx_true_gpu_features_20260702/
├── report.md
├── default_after_true_gpu.log
├── opencv_cuda_orb_contig.csv
└── opencv_cuda_orb_contig.log
test_logs/nx_xfeat_trt_pipeline_fast_20260702/
├── build_xfeat_128.log
├── build_xfeat_160.log
├── xfeat_trt_roi128_top64.csv
├── xfeat_trt_roi160_top64.csv
└── xfeat_trt_top128.csv
test_logs/nx_keypoint_gpu_20260702/
├── keypoint_method_summary.csv
├── codex_algo_tests_ros_20260702_120523/
│   ├── report.md
│   ├── summary.csv
│   ├── default_geometry.csv
│   ├── roi_*.csv
│   └── neural_*.csv
├── codex_xfeat_engine_20260702c/
│   ├── neural_xfeat_engine.csv
│   └── neural_xfeat_engine.log
└── codex_feature_debug_20260702/
    └── summary.txt
```

水印探针:

- 300 对全部成功。
- `frame_counter_delta=0` 稳定。
- 左/右设备 timestamp 步长约 `9.999 ms`。
- 等效频率 `100.005/100.006 fps`。
- `trigger_index` 有少量 step anomaly，但左右 frame counter 同步可用。

## 连续帧实测

| 方法 | FPS | 有效率 | 关键耗时 | 深度表现 | 结论 |
|---|---:|---:|---:|---|---|
| default geometry | 100.1 | `z_stereo=666/666` | GPU candidates avg `0.26ms`, max `1.35ms` | median `3.604m`, MAD `0.0015m` | 生产可用 |
| `roi_center_patch` | 100.1 | `666/666` | avg `0.35ms` | median `3.529m`, jitter 明显高于几何 | 可实时，不建议主用 |
| `roi_subpixel` | 81.4 | `535/535` | avg `3.36ms` | median `3.529m` | 不能 100fps |
| `roi_corner_points` | 100.0 | `0/665` | avg `0.65ms` | 无有效点 | 快但当前无效 |
| `roi_texture_points` | 100.1 | `0/664` | avg `0.25ms` | 无有效点 | 快但当前无效 |
| `roi_binary_points` | 100.1 | `0/665` | avg `0.41ms` | 无有效点 | 快但当前无效 |
| `all_sparse_gpu` | 100.1 | `0/665` | avg `0.82ms` | 无有效点 | 快但当前无效 |
| `roi_orb_points` 历史 lite | 100.1 | `0/664` | avg `0.41ms` | 无有效点 | 历史 GPU-lite，不是真 ORB；当前已改走 OpenCV CUDA ORB |
| `roi_brisk_points` 历史 lite | 100.0 | `0/664` | avg `0.42ms` | 无有效点 | 历史 GPU-lite，不是真 BRISK |
| `roi_akaze_points` 历史 lite | 100.0 | `0/665` | avg `0.28ms` | 无有效点 | 历史 GPU-lite，不是真 AKAZE |
| `roi_sift_points` 历史 lite | 100.1 | `0/665` | avg `0.27ms` | 无有效点 | 历史 GPU-lite，不是真 SIFT |
| `roi_iou_region_color_patch` | 100.0 | `0/665` | avg `0.32ms` | 无有效点 | 当前无效 |
| `roi_patch_iou_color_edge` | 100.0 | `0/664` | avg `0.27ms` | 无有效点 | 当前无效 |
| OpenCV CUDA ORB | 78-79 | `0/512` | `Stage2_OpenCVCudaORB avg 3.24ms`, max `22.73ms` | 无有效 `z_roi_orb_points` | 真实 CUDA ORB，但当前场景质量和同步耗时都不过关 |
| XFeat TensorRT 224/top128 | 75 | `496/496` | `Stage2_NeuralFeatureMatch avg 3.92ms`, max `5.80ms` | median `3.5537m`, MAD `2.3mm`, support median `44` | 真实模型，质量最好，同步不能 100fps |
| XFeat TensorRT 160/top64 | 86 | `568/602` | `Stage2_NeuralFeatureMatch avg 2.24ms`, max `2.86ms` | median `3.5299m`, MAD `3.8mm`, support median `20` | 真实模型，同步不能 100fps |
| XFeat TensorRT 128/top64 | 89-90 | `582/582` | `Stage2_NeuralFeatureMatch avg 1.89ms`, max `2.66ms` | median `3.5316m`, MAD `4.2mm`, support median `19` | 真实模型，最快同步版本仍未到 100fps |

XFeat engine case 的最终 `stereo_depth_source` 仍全部是 `1`，说明在线最终深度仍选择 circle/搜索路径，没有采用神经特征作为主深度。

## 单帧 debug 结果

`--debug-feature-matches` 输出:

```text
initial_disp=365.016
corner  left_keypoints=5  right_keypoints=5  matches=4   disparity=374
texture left_keypoints=16 right_keypoints=16 matches=10  disparity=374
binary  matches=0
orb     matches=0
brisk   matches=0
akaze   matches=0
```

这解释了连续帧为什么 `corner/texture` 在线有效率为 0: 单帧确实能匹配到点，但视差约 `374px`，与初始 bbox/circle 视差约 `365px` 相差约 `9px`，超过当前 `max_disp_delta_px=2` 的几何一致性门限。不能简单放宽门限，否则会把错误球面点引入深度。

## 当前实现边界

### 已在 GPU 上运行

文件: `src/stereo/dual_yolo_depth_gpu.cu`

- bbox/circle/edge/radial/edge-pair。
- center patch。
- multi-point/subpixel。
- corner/texture/binary sparse。
- ORB/BRISK/AKAZE/SIFT-lite 历史实现仍在 kernel 内，但当前 pipeline 已把 `gpu_cfg.compute_orb_points/brisk/akaze/sift` 置为 `false`，避免把 lite 候选误报为真实算法。
- color IoU / color edge patch。

但 ORB/BRISK/AKAZE/SIFT-lite 是自研 CUDA 采样和 patch score，不是 OpenCV 原版算法，也不是论文定义的完整算法。

### OpenCV CUDA ORB 路径

文件: `src/stereo/roi_feature_match_cpu.*`

- `matchOpenCVORBDisparityGPU()` 使用 `cv::cuda::ORB` 和 CUDA `DescriptorMatcher`。
- 输入使用校正后灰度 CUDA 指针；外部 pitched ROI 会先拷到连续 `cv::cuda::GpuMat`，避免 OpenCV CUDA ORB 在 pitch ROI 上触发非法访存。
- 输出仍走原有 y 残差、重叠椭圆、球体半径、反向互检、视差 delta、MAD 聚合等 gate。
- NX 实测能稳定运行，但 512 帧有效候选为 0，且 `Stage2_OpenCVCudaORB` 平均 3.24ms，同步路径不满足 100fps。
- 当前 NX OpenCV/VPI 没有 BRISK/AKAZE/SIFT 的对应 CUDA API；不能因为 OpenCV 有 CUDA 就宣称这些算法已真实 GPU 化。

### CPU debug 路径

文件: `src/stereo/roi_feature_match_cpu.*`

- sparse corner/texture/binary。
- OpenCV ORB/BRISK/AKAZE。
- BFMatcher/Hamming。
- `drawMatches` 可视化。

这条路径用于单帧诊断，不应进入 100fps 默认管线。

### XFeat TensorRT 路径

文件: `src/stereo/neural_feature_matcher.*`

当前 uncommitted 实现已有 XFeat extractor-only fallback:

- TensorRT extractor 在 GPU 上跑。
- `feats/keypoints/heatmap` 通过 `cudaMemcpyAsync` 拷回 CPU。
- CPU 做 heatmap NMS、descriptor 采样、互反查匹配、MAD/final gate。

它是真 XFeat extractor，但不是完整 GPU 常驻特征匹配路径。实测 75-90fps，取决于 ROI size 和 top-k。`trtexec` 裸 extractor 128/160 的 GPU compute mean 分别约 0.467ms/0.546ms；管线内更慢是因为左右各跑一次 extractor、D2H 拷贝和 CPU 后处理仍在同步路径。

## 架构问题

当前 ROI 管线主循环是:

```text
requestGrab(N+1)
  -> Stage1 Detect(N)
  -> Stage2 Fuse(N-1)
  -> waitGrab(N+1)
  -> Stage0 Process(N+1)
```

`stage2_roi_match_fuse()` 会同步等待并执行 ROI 候选、神经特征和融合。也就是说，重 feature 后处理现在会阻塞 Stage2 输出和下一轮调度。

当前 `drop_stale_roi_frames` 的语义是: 如果旧检测帧进入 Stage2 时，后一帧 YOLO 已 ready，就跳过旧帧 ROI 后处理。它不是“特征任务异步运行，下一帧 YOLO 完成时超时取消”。因此它不能解决 XFeat/subpixel 这种 3-5ms 后处理拖慢全管线的问题。

另外，当前主配置是:

```yaml
detector:
  use_dla: false
  dual_yolo:
    right_use_dla: false
```

左右 YOLO 和神经特征都争用 GPU。即使 feature 改成异步 stream，也会和 YOLO 抢同一 GPU 资源。要真正并行，需要:

- YOLO 上 DLA/INT8，GPU 留给特征；或
- 使用 CUDA stream priority，让 YOLO 高优先级，feature 低优先级；或
- feature 任务只在 GPU 空隙中 opportunistic 运行，deadline 到即丢弃。

## 正确的 100fps FeatureJob 设计

建议新增独立异步特征队列:

```text
Stage2(YOLO collect for frame N):
  1. 立即生成默认几何深度候选。
  2. 若有 ROI pair，提交 FeatureJob(N) 到 cudaStreamFeature。
  3. 不等待 FeatureJob。

下一帧 YOLO 完成或 Stage2(N+1) 开始:
  1. cudaEventQuery(FeatureJob(N))。
  2. ready: 读取 feature 候选并追加记录/融合。
  3. not ready: 丢弃 FeatureJob(N)，默认几何已保证实时输出。
```

实现细节:

- FeatureJob 必须持有独立 ROI GPU buffer，不能引用即将复用的 `FrameSlot` 图像。
- feature 输出写入单独 ring buffer，避免覆盖。
- deadline 以“下一帧 YOLO complete event”为准，而不是固定 sleep。
- 所有 GPU 后处理必须 kernel 化: heatmap NMS、topK、descriptor gather、mutual check、y/disp gate、MAD/RANSAC。
- `cudaMemcpyDeviceToHost` 只能用于低频 debug，不允许在 100fps hot path 里每帧同步。
- 如果 feature 结果跨帧返回，只能作为候选记录和训练数据，不能阻塞当前帧主输出。

## 可行模型排序

| 模型/算法 | 是否真实算法 | 100fps 可能性 | 原因 |
|---|---|---|---|
| XFeat extractor + GPU postprocess/match | 是 | 高 | 当前已有 engine，ROI 小图可跑；瓶颈是 CPU 后处理和同步 |
| XFeat fused matcher | 是 | 高 | 如果导出端到端点对输出，可减少 CPU 和中间 D2H |
| ALIKED + LightGlue-lite | 是 | 中 | ALIKED 轻量，但 matcher 和动态 shape 工程复杂 |
| SuperPoint + LightGlue | 是 | 中低 | 模型成熟，但两段网络和匹配较重，NX 100fps 压力大 |
| DISK | 是 | 低 | 更偏高质量通用局部特征，小 ROI 100fps 不优先 |
| OpenCV CUDA ORB | 是 | 低 | NX 已可用且已接入；同步 avg 3.24ms，且排球场景 0/512 有效 |
| VPI ORB | 是 | 待测 | NX 有 `vpi/algo/ORB.h`，但本项目尚未实现/实测 VPI ORB 后端 |
| OpenCV/VPI BRISK/AKAZE/SIFT CUDA | 否 | 不成立 | 当前 NX API 未暴露这些真实 CUDA 后端 |
| VPI KLT/Optical flow | 是，但不是左右图匹配 descriptor | 中 | 更适合帧间跟踪，不直接解决左右同帧弱纹理匹配 |

## 外部资料

- XFeat: https://arxiv.org/abs/2404.19174
- XFeat implementation: https://github.com/verlab/accelerated_features
- LightGlue: https://arxiv.org/abs/2306.13643
- LightGlue implementation: https://github.com/cvg/LightGlue
- ALIKED: https://arxiv.org/abs/2304.03608
- DISK: https://arxiv.org/abs/2006.13566
- SuperPoint: https://arxiv.org/abs/1712.07629
- NVIDIA VPI KLT Tracker: https://docs.nvidia.com/vpi/algo_klt_tracker.html
- OpenCV CUDA DescriptorMatcher reference: https://docs.opencv.org/4.x/dd/dc5/classcv_1_1cuda_1_1DescriptorMatcher.html

## 下一步

1. 保持默认几何路径为 100fps 主路径。
2. 不再把 lite ORB/BRISK/AKAZE/SIFT 叫作真实算法；若保留，只作为自研 GPU 候选。
3. 优先实现 XFeat GPU 后处理和异步 FeatureJob。
4. 将 feature deadline 接到下一帧 YOLO complete event。
5. 重新测试:
   - feature job ready rate。
   - 被 deadline 丢弃比例。
   - `Stage2_NeuralFeatureMatch` 是否从主线程 profile 中消失。
   - 主输出 FPS 是否回到 100。
   - feature candidate 的有效率、深度 MAD、运动残差。
