# Neural ROI Feature Matching

目标是只在排球 IoU/ROI 小图上提取少量可靠点，用于双目三角测距。Python 只用于本地离线验证；NX 实时路径必须使用 TensorRT engine。

## 本地 CPU 验证

环境:

```bash
NX_volleyball/stereo_3d_pipeline/scripts/setup_neural_feature_env.sh
```

推荐测试:

```bash
.venv-stereo-neural/bin/python NX_volleyball/stereo_3d_pipeline/tools/neural_feature_probe.py \
  --xfeat-repo ~/.local/share/stereo_3d_pipeline/neural_repos/accelerated_features \
  --out NX_volleyball/stereo_3d_pipeline/test_logs/neural_feature_probe_recommended \
  --roi-size 224 \
  --top-k 128 \
  --max-y-error-px 2.0 \
  --max-disp-delta-px 32.0 \
  --final-disp-gate-px 2.0
```

当前单张排球 stereo pair 结果:

| backend | matches | valid points | depth | status |
| --- | ---: | ---: | ---: | --- |
| XFeat + descriptor NN | 11 | 8 | 3.399 m | pass |
| ALIKED + descriptor NN | 28 | 28 | 3.379 m | pass |
| SuperPoint + LightGlue | 20 | 9 | 3.400 m | pass |

注意: CPU 耗时不能代表 NX。该脚本用于验证匹配点是否几何正确，不用于评估实时性能。

## NX 实时推理方式

配置入口是 `config/pipeline_dual_yolo_roi.yaml` 的 `neural_feature_matching`，默认关闭。

实时路径原则:

- 固定 ROI 输入尺寸，当前推荐 `224x224`，`top_k=128`。
- 当前 C++ 实时实现支持 fused TensorRT engine:输入为左右 ROI(灰度 1/2 通道或 BGR 3/6 通道),输出 `[N,4]` 或 `[N,5]` 匹配点。
- XFeat extractor-only TensorRT split 路径已实现:输入单张固定 ROI gray/BGR,输出 `feats/keypoints/heatmap`;C++ 后处理做 keypoint 选择、descriptor 采样、互反查匹配和几何 gate。
- ALIKED/SuperPoint 等 direct extractor 路径已接入:若真实 TensorRT engine 输出固定 shape `keypoints/descriptors/scores`, C++ 会用 descriptor mutual NN 和几何 gate 生成候选。
- SuperPoint+LightGlue 优先使用 fused TensorRT engine；LightGlue extractor+matcher split engine 的多输入运行时仍待实现。
- 后处理必须保留 `max_y_error_px`、`max_disp_delta_px`、`final_disp_gate_px`，不能直接相信网络匹配。
- 目标 10ms 内时，优先评估 XFeat TensorRT；ALIKED 和 SuperPoint+LightGlue 作为质量对照。

## NX TensorRT 实测

实测数据位于 `test_logs/nx_true_gpu_features_20260702/` 和 `test_logs/nx_xfeat_trt_pipeline_fast_20260702/`。

| engine | ROI FPS | `Stage2_NeuralFeatureMatch` | 有效候选 | 深度数据 |
| --- | ---: | ---: | ---: | --- |
| XFeat 224/top128 | 75 | avg `3.92ms` | `496/496` | median `3.5537m`, MAD `2.3mm`, support median `44` |
| XFeat 160/top64 | 86 | avg `2.24ms` | `568/602` | median `3.5299m`, MAD `3.8mm`, support median `20` |
| XFeat 128/top64 | 89-90 | avg `1.89ms` | `582/582` | median `3.5316m`, MAD `4.2mm`, support median `19` |

`trtexec` 裸 extractor 不是瓶颈:128/160 的 GPU compute mean 分别约 `0.467ms`/`0.546ms`。管线内慢在左右各跑一次 extractor、输出 D2H 拷贝、CPU 后处理和 Stage2 同步等待。后续要守 100fps，需要 GPU 后处理或 fused matcher，并把 feature job 从 Stage2 同帧同步路径拆出去。
