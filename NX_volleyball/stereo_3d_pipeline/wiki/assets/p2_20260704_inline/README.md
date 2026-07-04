# P2 inline artifacts 2026-07-04

Source run:

```text
NX run: test_logs/codex_p2_retest_inline_20260704_130245/
```

These are algorithm-level P2 artifacts, not realtime status zooms.

| File | Case | Meaning |
|---|---|---|
| `iou_region_color_patch.png` | `iou_region_color_patch_wide_search` | CUDA color patch inlier samples on rectified left/right crops |
| `patch_iou_color_edge.png` | `patch_iou_color_edge_wide_search` | CUDA color-edge patch inlier samples on rectified left/right crops |
| `neural_xfeat.png` | `neural_xfeat_128_top32` | TensorRT XFeat keypoint matches mapped back to rectified image coordinates |
