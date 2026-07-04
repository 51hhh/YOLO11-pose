# P2 Diagnostic Artifacts

来源:

```text
test_logs/codex_p2_artifact_debug_20260704_105356/
```

这些 PNG 是 P2 diagnostic 后端输出的算法级 artifact，来自同一帧 GPU gray snapshot 和 `SparseFeatureDisparityResult.debug_matches` / peak 信息。它们用于检查左右 ROI 点对或模板峰值位置，不是 realtime status zoom，也不用于 FPS 准入。

当前样张:

| 文件 | 含义 |
|---|---|
| `opencv_cuda_gftt_lk_valid.png` | OpenCV CUDA GFTT/Harris + SparsePyrLK 左右 ROI 点对 |
| `vpi_template_match_valid.png` | VPI Template Matching 单点峰值 |
| `vpi_orb_valid.png` | VPI ORB + BruteForceMatcher 左右 ROI 点对 |

完整 debug 输出仍保留在:

```text
test_logs/codex_p2_artifact_debug_20260704_105356/debug/<case>/p2_artifacts/
```
