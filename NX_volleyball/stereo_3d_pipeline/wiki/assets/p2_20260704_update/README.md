# P2 2026-07-04 update artifacts

来源:

```text
NX run: test_logs/codex_p2_update_artifacts_20260704_134915/
NX run: test_logs/codex_p2_superpoint_artifacts_20260704_135156/
```

这些 PNG 只用于算法级 artifact 审查，不作为 FPS 准入结果。

- `opencv_cuda_template_score_patch.png`: OpenCV CUDA TemplateMatching 峰值连线 + score patch。
- `vpi_template_score_patch.png`: VPI CUDA TemplateMatching 峰值连线 + score patch。
- `cuda_ring_edge_profile_samples.png`: CUDA ring-edge profile 最佳候选视差下的三圈采样点；该帧仍为 invalid。
- `neural_superpoint_160_top64.png`: SuperPoint 160/top64 真实左右 keypoint overlay；该配置不准入。
