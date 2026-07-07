# Reliability Sweep Configs

These JSON files are meant to make post-recording trajectory experiments repeatable.

Use them through `run_dataset_workflow.py`:

```bash
.venv-stereo-neural/bin/python trajectory_fusion/run_dataset_workflow.py \
  ~/mid360_datasets/trajectory \
  -o ~/mid360_datasets/trajectory/workflow_YYYYMMDD \
  --configs trajectory_fusion/configs/sweep_known_distance_selection.json \
  --gravity-y 0
```

Files:

- `sweep_smoke.json`: one quick run to check PyTorch, manifest, suite, method audit, and model selection wiring.
- `sweep_known_distance_selection.json`: first real model-selection set for multi-distance static clips with train/val split and `known_z`.
- `sweep_dynamic_regularization.json`: second-pass exploratory set for dynamic clips after the fixed calibration baseline and known-distance set are already reviewed.

Each config pins `seed` so repeated runs on the same dataset are comparable. A good model still needs heldout `known_z` and dynamic validation; a stable seeded run on one no-label clip is not evidence of final accuracy.

For ordinary multi-distance static selection, generate the manifest with `--stratify-known-z` so each measured distance has train/val coverage when enough clips exist. For leave-one-distance-out checks, use `--holdout-known-z <distance> --holdout-split val` to keep one distance bucket fully held out.

Selection rule:

- Treat `calibrated_smoother` as the non-neural baseline.
- Treat `robust_smooth` as the uncalibrated physical baseline.
- Prefer `reliability_smoother` only when heldout `known_z` bias/MAD improves without method-audit warnings.
- Do not select a model from no-`known_z` static clips; those clips only validate stability and toolchain behavior.
