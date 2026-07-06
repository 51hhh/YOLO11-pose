# ALIKED-t16 DCN artifacts 2026-07-06

来源:

```text
NX: /home/nvidia/NX_volleyball/stereo_3d_pipeline/test_logs/neural_aliked_dcn_zoom_20260706_095209/
```

内容:

| 文件 | 说明 |
|---|---|
| `report.md` | isolated `current` / `gate_off` 矩阵报告 |
| `aliked_dcn_gate_off_valid_000217.png` | 文档引用用样张，等同 `p2_artifacts/frame_000217_00_neural_aliked_valid.png` |
| `p2_artifacts/frame_000217_00_neural_aliked_valid.png` | official DCN gate-off valid artifact |
| `p2_artifacts/frame_000222_00_neural_aliked_valid.png` | official DCN gate-off valid artifact |
| `p2_artifacts/frame_000225_00_neural_aliked_valid.png` | official DCN gate-off valid artifact |
| `p2_artifacts/frame_000230_00_neural_aliked_valid.png` | official DCN gate-off valid artifact |

结论摘要:

- `current`: `0/572` 有效。
- `gate_off`: `68/572` 有效，median/MAD `3.4333/0.0370m`，support 中位 `2.0`。
- `NCC + XFeat + ALIKED-DCN` 联合 run 中 `z_roi_neural_aliked=0/1034`。
