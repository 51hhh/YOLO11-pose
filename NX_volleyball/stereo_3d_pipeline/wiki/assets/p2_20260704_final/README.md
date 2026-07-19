# P2 Final Diagnostic Artifacts

来源:

```text
NX run: codex_p2_artifacts_final_20260704_110837
```

这些 PNG 是 2026-07-04 11:08 final diagnostic artifact 复测的代表样张。它们来自 P2 diagnostic 后端返回的 `debug_matches` / peak 信息，用于确认算法级左右点对、模板峰值、disparity 样本或 refined center。它们不属于 realtime status zoom，也不用于 FPS 准入。

当前样张:

| 文件 | 含义 |
|---|---|
| `opencv_cuda_gftt_lk.png` | OpenCV CUDA GFTT/Harris + SparsePyrLK 左右 ROI 点对 |
| `opencv_cuda_orb.png` | OpenCV CUDA ORB + CUDA BF 左右 ROI 点对 |
| `opencv_cuda_template.png` | OpenCV CUDA TemplateMatching 单点峰值 |
| `opencv_cuda_stereo_sgm.png` | OpenCV CUDA StereoSGM 有效 disparity 样本点 |
| `cuda_hough_circle.png` | CUDA Hough circle 左右 refined center |
| `vpi_template.png` | VPI Template Matching 单点峰值 |

性能结论仍以 `test_logs/codex_p2_verify_20260704_104947/` 的无 debug targeted 复核为准。
