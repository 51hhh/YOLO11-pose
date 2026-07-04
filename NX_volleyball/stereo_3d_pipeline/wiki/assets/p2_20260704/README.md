# P2 2026-07-04 Debug Assets

These images are debug samples, not algorithm-level match overlays.

| Image type | Size | Files | Meaning |
|---|---|---|---|
| realtime ROI/status zoom | `510-523 x 220` | `iou_region_color_patch*.png`, `patch_iou_color_edge*.png`, `opencv_cuda_orb_wide_y.png`, `opencv_cuda_template_match_patch9.png`, `opencv_cuda_stereo_bm_patch9.png`, `opencv_cuda_stereo_sgm_patch9.png`, `neural_xfeat_128_top32.png`, `neural_superpoint_224_top64.png` | Shows bbox/circle/field state for a realtime result. It does not draw internal feature matches. |
| full-frame detection panel | `2560 x 720` | `vpi_*.png`, `fixstars_libsgm.png`, `cuda_hough_circle.png`, `cuda_ring_edge_profile.png`, `opencv_cuda_gftt_lk.png` | Shows full left/right detection state when no main `Object3D` zoom was available. It is not a YOLO-IoU ROI match view. |

For the current review and per-algorithm effect summary, see `../../P2算法效果与可视化审查.md`.
