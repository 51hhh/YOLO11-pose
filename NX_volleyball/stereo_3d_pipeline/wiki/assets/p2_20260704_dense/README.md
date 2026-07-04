# P2 dense artifacts 2026-07-04

Source runs:

```text
NX run: test_logs/codex_p2_retest_dense_20260704_130617/
NX run: test_logs/codex_p2_retest_dense_priority_20260704_131027/
NX run: test_logs/codex_p2_retest_sgm_valid_20260704_131452/
```

These are algorithm-level dense stereo artifacts. The top panels show rectified left/right crops or frames; the bottom panels show bounded 32x32 debug patches.

| File | Case | Meaning |
|---|---|---|
| `opencv_cuda_stereo_bm_patch.png` | `opencv_cuda_stereo_bm_diagnostic_only` | Invalid OpenCV CUDA StereoBM frame with disparity patch |
| `opencv_cuda_stereo_sgm_valid_patch.png` | `opencv_cuda_stereo_sgm_diagnostic_only` | Valid OpenCV CUDA StereoSGM frame with inlier samples and disparity patch |
| `vpi_stereo_disparity_confidence_patch.png` | `vpi_stereo_disparity_diagnostic_only` | Invalid VPI Stereo frame with disparity and confidence patches |
| `fixstars_libsgm_patch.png` | `fixstars_libsgm_diagnostic_only` | Invalid Fixstars libSGM frame with disparity patch |
