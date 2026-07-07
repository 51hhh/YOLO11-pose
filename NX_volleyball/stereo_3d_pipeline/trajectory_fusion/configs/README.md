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
- `sweep_p0p1_ncc_xfeat_smoke.json`: one quick run using the current formal `p0p1_ncc_xfeat` candidate set; use this before the full known-distance sweep.
- `sweep_known_distance_selection.json`: first real model-selection set for multi-distance static clips with train/val split and `known_z`; every config is pinned to the current formal `p0p1_ncc_xfeat` candidate set.
- `sweep_method_ablation.json`: method-set ablation for P0-only, P0+P1, P0+P1+NCC, and P0+P1+NCC+XFeat. Use it after the basic known-distance workflow is clean to test whether NCC/XFeat actually improve heldout distance accuracy.
- `sweep_dynamic_regularization.json`: second-pass exploratory set for dynamic clips after the fixed calibration baseline and known-distance set are already reviewed; every config is pinned to `p0p1_ncc_xfeat`.

Each config pins `seed` so repeated runs on the same dataset are comparable. A good model still needs heldout `known_z` and dynamic validation; a stable seeded run on one no-label clip is not evidence of final accuracy.

Configs may include `methods`, either as a preset string or comma-separated method names. Supported presets include `p0`, `p0p1`, `p0p1_ncc`, `p0p1_xfeat`, and `p0p1_ncc_xfeat`. The allowlist is applied consistently to ReliabilityNet training, direct consensus, robust smoother, calibrated smoother, ReliabilityNet smoother evaluation, and training-input audit when the workflow can infer a single method set. The formal known-distance/dynamic configs should stay pinned to `p0p1_ncc_xfeat`; use `sweep_method_ablation.json` when comparing method subsets.

For ordinary multi-distance static selection, generate the manifest with `--stratify-known-z` so each measured distance has train/val coverage when enough clips exist. For leave-one-distance-out checks, use `--holdout-known-z <distance> --holdout-split val` to keep one distance bucket fully held out.

`known_z` is a training label only for clips marked `static: true` by default. Dynamic clips may still carry `known_z` metadata for evaluation/grouping, but ReliabilityNet ignores it for training unless `known_z_training: true` is explicitly set.

Selection rule:

- Treat `calibrated_smoother` as the non-neural baseline.
- Treat `robust_smooth` as the uncalibrated physical baseline.
- Prefer `reliability_smoother` only when heldout `known_z` bias/MAD improves without method-audit warnings and it beats the best non-neural baseline with the same method allowlist.
- `select_reliability_model.py` rejects candidates with heldout `mean_abs_known_z_bias > 0.08m` or `mean_known_z_mad > 0.03m` by default.
- `select_reliability_model.py` also rejects a candidate as `baseline_variant_better` when a matching `robust_smooth`/`calibrated_smoother`/RTS/depth-polyfit baseline has a lower heldout score.
- Do not select a model from no-`known_z` static clips; `select_reliability_model.py` hard-rejects `known_clip_count=0` as `no_known_z`.
