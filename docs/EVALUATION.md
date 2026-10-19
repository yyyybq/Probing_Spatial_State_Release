# Evaluation

This document describes how trained probes are scored: which evaluator to run, what
it writes, the metrics, when a result is accepted, how the confidence intervals are
built, how to read the C1 controls, and the C3-from-C1 diagnostic. The data, split and
training recipe are in [PROTOCOL.md](PROTOCOL.md). The accepted tables are described
in [results/README.md](../results/README.md).

## 1. Ground rules

- **Split:** `eval_split=val` only. That is 276 windows, or 171 on the MiniMax
  all-layers subset. **There is no held-out test split**, so every reported number is
  a validation result.
- **Weights:** the final-epoch checkpoint of each run: epoch 99 for A1, epoch 49 for
  all other probes. There is no checkpoint selection and no early stopping.
- **Seed:** training seed 42. The bootstrap in `eval_diag.py` is also seeded with
  `cfg.seed`, which is 42.
- **Same settings as training:** evaluate with the same experiment config, the same
  overrides (model, layer, timestep, `prefix_len`, batch sizes, identity overrides for
  A1 mask), the same `INSCENE_MANIFEST` and the same `INSCENE_PROBE_TASK` and
  `PROBE_FP16_PASSTHROUGH` as the training job.

## 2. Evaluators and outputs

| Probe | Entry point | Files written under `<run>/eval/` | `evaluator_version` |
| --- | --- | --- | --- |
| A1 `streaming_depth`, `streaming_mask` | `vidfm3d/eval_a1_streaming.py` | `val_summary.json`, `val_predictions.pt` | `a1_streaming_eval_v2_nonfinite_aware` |
| A2, B1, B2, C1, C2 | `vidfm3d/eval_diag.py` | `val_summary.json`, `val_predictions.pt` | `eval_diag_v2_nonfinite_aware` |
| C3-from-C1 | `vidfm3d/eval_diag.py` on a C1 checkpoint, with `++evaluation_protocol=c3_action_sensitivity` | `val_c3_action_sensitivity_summary.json`, `val_c3_action_sensitivity_predictions.pt` | `eval_diag_v2_nonfinite_aware` |

- **Output location.** `<run>` is the Hydra output directory of the evaluation call,
  `logs/inscene15k_streaming/runs/<paths.run_folder_name>`. Pass the training job's
  `paths.run_folder_name` so that the summary lands next to the checkpoint. The C3 files
  have their own names, so they never overwrite the ordinary C1 summary. By default
  Hydra rewrites `<run>/.hydra/` on every call. Pass `hydra.output_subdir=null` to
  every evaluation, as all commands below and `scripts/probe_recipe.py` do. Otherwise
  the evaluation overwrites the training run's `.hydra/config.yaml` with its own
  settings, for example the `action_horizons=[1,2,4]` of a C3 call on an h1-trained
  C1. `scripts/export_results.py` reads the trained horizons from
  `<run>/run_manifest.json`, which only training writes, and falls back to
  `.hydra/config.yaml`.
- **Optional evaluator switches.** All are off by default.
  - Both evaluators read `PROBE_EVAL_REQUIRE_CUDA=1`, which fails instead of falling
    back to CPU.
  - Only `eval_diag.py` reads `PROBE_EVAL_MODE` and
    `PROBE_EVAL_LEGACY_CONSTANTS_COMPAT`. Setting `PROBE_EVAL_MODE=LEGACY_AS_IS_LOOSE`
    together with `PROBE_EVAL_LEGACY_CONSTANTS_COMPAT=1` lets it load older C1/C2
    checkpoints that lack the constant-context parameters.
  - Both evaluators copy the `PROBE_EVAL_*` provenance labels (for example
    `PROBE_EVAL_PROTOCOL`, `PROBE_EVAL_CONFIG_SHA256`) into the summary.
  - The older `Q00_*` names are still read as aliases.
- **A1 evaluator.** It reads only the validation loader and refuses any `eval_split`
  other than `val`. It loads the checkpoint strictly and runs under bf16 autocast on
  CUDA. The heads it scores come from `model.probe.active_heads`. A depth config used
  as A1 mask therefore needs the identity overrides at evaluation too, or the strict
  load fails.
- **`eval_diag.py`.** It runs the model in fp32.

## 3. Commands

Use the training command, replace the entry point, and add the checkpoint and the
evaluation flags. `scripts/reproduce_rows.py` prints the exact train, eval and C3
commands for any accepted row, and `scripts/run_streaming_probe_sweep.sh` wraps
extraction, training and evaluation for a whole sweep.

```bash
RUN=logs/inscene15k_streaming/runs/c1_dino_l0_p12
CKPT=$(ls -t $RUN/VideoProbe3D/*/checkpoints/epoch=*-step=*.ckpt | head -n 1)
export INSCENE_MANIFEST=$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix12.jsonl
export INSCENE_PROBE_TASK=action_dynamics PROBE_FP16_PASSTHROUGH=1
OVR="seed=42 vfm_name=dino video_channels=1024 feature_layer=0 feature_timestep=749 \
  feat_postfix=_t749_layer0 prefix_len=12 \
  data.data_module.batch_size_per_device=2 data.data_module.batch_size_per_device_val=2 \
  trainer.devices=[0] trainer.num_nodes=1 autoresume=false logger.wandb.offline=true \
  paths.run_folder_name=c1_dino_l0_p12"

# C1 (and likewise A2, B1, B2, C2)
python vidfm3d/eval_diag.py experiment=inscene15k_streaming/action_dynamics_dino_h124_v1 $OVR \
  "ckpt_path='$CKPT'" +eval_split=val train=false test=false hydra.output_subdir=null

# C3-from-C1 on the same canonical (h1/h2/h4-trained) checkpoint
python vidfm3d/eval_diag.py experiment=inscene15k_streaming/action_dynamics_dino_h124_v1 $OVR \
  "ckpt_path='$CKPT'" ++evaluation_protocol=c3_action_sensitivity \
  +eval_split=val train=false test=false hydra.output_subdir=null
```

```bash
# A1 (here A1 mask through the Aether depth config, so the identity overrides are needed)
python vidfm3d/eval_a1_streaming.py experiment=inscene15k_streaming/streaming_depth_aether_v1 \
  <training overrides, including the four identity overrides> \
  "ckpt_path='$CKPT'" +eval_split=val train=false test=false hydra.output_subdir=null

# C3 diagnostic on an h1-trained C1 (timestep line or MiniMax subset): expand to h1/h2/h4
python vidfm3d/eval_diag.py experiment=inscene15k_streaming/action_dynamics_wan_v1 \
  <training overrides> "ckpt_path='$CKPT'" ++evaluation_protocol=c3_action_sensitivity \
  "action_horizons=[1,2,4]" +eval_split=val train=false test=false hydra.output_subdir=null
```

Quoting:

- Hydra needs the checkpoint path in single quotes because the file name contains `=`.
  Without them the override does not parse.
- Quote `"trainer.devices=[0]"` and `"action_horizons=[1,2,4]"` when you pass them
  directly on the command line, so the shell does not expand the brackets.

`tests/test_probe_recipe.py` composes every command form that
`scripts/probe_recipe.py` prints with Hydra; those commands use these override forms
and `hydra.output_subdir=null`.
`tests/test_entrypoints_compose.py` checks that the three entry points compose from
any working directory.

## 4. Metrics

`n_loader` and `n_scored` are the typical counts on the 276-window val split.

| Probe | Primary key | Better | Secondary keys in the result tables | `n_loader` / `n_scored` |
| --- | --- | --- | --- | --- |
| A1 depth | `mean_loss_depth` | lower | `mean_Auc_30`, `mean_Auc_5` (camera pose AUC, higher is better) | 276 / 276 |
| A1 mask | `mean_loss_identity` | lower | none | 276 / 276 |
| A2 | `balanced_acc` | higher | `pos_acc`, `neg_acc` | 276 / 276 |
| B1 | `mean_az_err_deg` | lower | `mean_el_err_deg`, `mean_log_dist_err` | 276 / 80 at prefix 8, 171 at prefixes 12-24 |
| B2 | `mean_ang_err_deg` | lower | `mean_az_err_deg`, `mean_el_err_deg`, `mean_log_dist_err`, `top1`, `top3` | 276 / 80 or 171 |
| C1, canonical (h1/h2/h4) | `global_R@1` | higher | `global_R@5`, `mean_rank`, `mean_cos`, `no_action_global_R@1`, `shuffled_action_global_R@1` | 828 / 828 |
| C1, h1 only | `global_R@1` | higher | same | 276 / 276 |
| C2 | `global_R@1` | higher | `global_R@5`, `mean_rank`, `mean_cos`, `final_pose_error` (lower is better) | 276 / 276 |
| C3-from-C1 | `intervention_validity` | higher; chance 1/3 | `intervention_chance`, `intervention_validity_chance_adjusted`, `exact_candidate_validity`, `intervention_margin`, `counterfactual_consistency`, `action_sensitivity_cosine`, `target_sensitivity_cosine`, `target_to_prediction_sensitivity_pearson`, `sensitivity_calibration_mae`, `n_valid_action_sets`, `n_valid_candidates` | 828 candidates / 276 action sets |

On the MiniMax subset the counts are 171 instead of 276: B probes score 48 or 114 of
171, and C3 has 513 candidates in 171 action sets.

Notes on the metrics:

- The A2 entry `overlap_mae` in `secondary_json` is empty. The evaluator writes this
  value as `mean_overlap_mae`. The A2 summary also has `roc_auc`, `pr_auc` and
  `trivial_neg_acc` (the accuracy of always predicting no overlap).
- The full summaries hold more keys than the tables, for example
  `last_observation_global_R@1`, `copy_anchor_isolated_*`, `h1_*`/`h2_*`/`h4_*`,
  `macro_horizon_*`, `drift_rate` and `pose_error_h{1,2,4}`.
- `loader_manifest_ids`, `valid_manifest_ids` and `invalid_manifest_ids` list exactly
  which windows were loaded and which were scored.

**Where the table counts come from.** The table columns map to these summary keys:

| Table column | `eval_diag.py` | `eval_a1_streaming.py` | C3 summary |
| --- | --- | --- | --- |
| `n_loader` | `loader_n_samples` | `n_samples` | `loader_n_samples` (candidates) |
| `n_scored` | `n_scored` (C2: `n_valid_records`) | `n_valid_prediction` | `n_valid_action_sets` |
| `n_nonfinite_pred` | `n_nonfinite_pred` | `n_nonfinite_prediction` | `n_nonfinite_pred_records` |

**Non-finite handling (v2 evaluators).** A prediction with NaN or inf is counted as
invalid and left out of every mean. It is never scored as class 0, as "no overlap" or
as a zero loss. A C3 action set with any non-finite candidate is not scored at all.

## 5. Acceptance

A summary is `ACCEPTED` when all of these hold:

1. `evaluator_version` is a v2 evaluator (`eval_diag_v2_nonfinite_aware` or
   `a1_streaming_eval_v2_nonfinite_aware`);
2. the primary metric is finite;
3. the non-finite count is 0;
4. `n_scored > 0`;
5. the sha256 of the summary file matches the hash recorded when it was evaluated.

Otherwise the summary is `REJECTED`, with a reason. All 1,300 rows in the three
accepted tables (804 canonical, 316 timestep and 180 C3) passed.
`scripts/export_results.py` applies checks 1-4 to your own summaries and writes them
in the public schema. It also rejects a summary whose checkpoint is not the final
epoch, or whose checkpoint name has no `epoch=E` (`REJECTED_NOT_FINAL_EPOCH`). Check 5 needs the evaluation receipts
of the original runs, which local runs do not have.

## 6. Confidence intervals

`eval_diag.py` writes `bootstrap_95ci` into each summary. It uses 200 resamples
(`bootstrap_reps`), seeded with `cfg.seed`, and reports the 2.5th and 97.5th
percentiles.

| Evaluator path | Resampling unit | Metrics with an interval |
| --- | --- | --- |
| A2 | Scenes | `balanced_acc`, `roc_auc`, `mean_overlap_mae` |
| B1 | Scenes | `mean_az_err_deg`, `mean_el_err_deg`, `mean_log_dist_err` |
| B2 | Scenes | `mean_ang_err_deg`, `top1`, `mean_log_dist_err` |
| C1 | Scenes | `mean_cos`, `action_gain_cos` (not `global_R@1`) |
| C3-from-C1 | Action sets | `intervention_validity`, `intervention_validity_chance_adjusted`, `exact_candidate_validity`, `intervention_margin`, `action_sensitivity_cosine`, `target_to_prediction_sensitivity_pearson`, `equivalence_separation_gap` |
| A1, C2 | none | none |

In the accepted tables, `primary_ci_low` and `primary_ci_high` are filled only for B1,
B2 and C3.

## 7. Reading C1 and C2

- **Pool size.** `global_R@1` ranks each prediction against the targets of all
  evaluated records. Canonical C1 ranks against 828 targets that mix the three
  horizons. h1-only C1 ranks against 276. The two are not comparable: chance is 1/828
  in one case and 1/276 in the other. Compare canonical C1 with h1-only C1 through
  `h1_global_R@1`, which ranks only the h1 records (276).
- **Controls.** Report C1 against its no-action, shuffled-action and copy-last controls
  for the same model, layer and prefix ([PROTOCOL.md, section 6](PROTOCOL.md#6-controls)).
  A high absolute `global_R@1` alone does not show that the probe uses the action.
- **Shuffled-action control.** It rolls actions inside the evaluation batch, so it
  depends on `batch_size_per_device_val`, which is 2 in the canonical recipe and 4 for
  Wan C1. In a batch of one it equals the no-action control.
- **C2** has no shortcut controls. Its `final_pose_error` is computed from the pose of
  the retrieved target, not from a pose regressor.

## 8. C3-from-C1

### What it measures

C3 is not a trained probe. It evaluates a frozen C1 one-step predictor
(`protocols/C_PROBES.md`):

- The anchor (prefix tail) of each val window has three future endpoints on the same
  recorded trajectory: h1, h2 and h4. Together they form one **action set** with three
  candidate actions (anchor to h1, anchor to h2, anchor to h4) and their three isolated
  target features.
- Each candidate action goes through C1 on its own, with the same prefix. No candidate
  target or id is ever an input to C1.
- `intervention_validity`: for each candidate's prediction, find the closest of the
  three targets in its set by cosine similarity. The candidate is a hit if that target
  is its own. The score is the hit rate over all candidates. Higher is better.
- `intervention_margin`: the cosine to the candidate's own target minus the highest
  cosine to another target in the set.
- `target_to_prediction_sensitivity_pearson`: how well the changes in the predictions
  track the changes in the targets across a set.

These endpoints belong to one observed trajectory. **They are not counterfactual
branches**, where the same state is restored and different actions are executed. C3
is therefore a multi-action sensitivity diagnostic. Do not describe its results in
causal or counterfactual terms (caveat
`C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL`).

### How it is run

Use the same experiment, overrides and checkpoint as the C1 evaluation, and add
`++evaluation_protocol=c3_action_sensitivity`. For an h1-trained C1, also add
`action_horizons=[1,2,4]` so that the val loader expands each prefix to all three
horizons. Pass `hydra.output_subdir=null` as well (section 2). The evaluator groups
the flat C1 records by `manifest_sample_id` into action sets. The results go to
`val_c3_action_sensitivity_summary.json` (section 3).

The accepted C3 rows were evaluated with a development commit of
`vidfm3d/eval_diag.py`, named in their `eval_pack` (`main_eval_v2_from_main_206c5de`)
and not part of this repository's history. On the C3 path the shipped evaluator
differs from that commit only in non-finite handling: an action set with a non-finite
candidate prediction is no longer scored. No accepted C3 row has a non-finite
prediction, so this does not change any accepted row.

| Variant (C3 table `variant`) | C1 training horizons | Rows | Status |
| --- | --- | ---: | --- |
| `c3_from_c1` | h1, h2, h4 (canonical `*_h124_v1`) | 120 | Protocol C3: evaluated on the horizons it was trained on |
| `h1_trained_diagnostic` | h1 only (44 timestep and 16 MiniMax-subset C1) | 60 | Not protocol C3: h2 and h4 were never seen in training. Report separately (caveat `H1_TRAINED_NOT_PROTOCOL_C3`). |

### How to read it

- **Chance is 1/3.** A predictor that ignores the action produces the same prediction
  for all three candidates. Exactly one candidate in each set is then a hit, so it
  scores exactly 1/3 with zero variance. "CI above 1/3" is therefore a weak test.
- **The intervals resample action sets, not scenes.** The 276 action sets come from 69
  scenes, so these intervals are narrower than the scene-level intervals of the other
  probes.
- **The margin is near 0.** All 120 protocol rows and 55 of the 60 diagnostic rows
  have `primary_ci_low > 1/3`. Per-model mean validity ranges from 0.355 (Aether) to
  0.620 (DINOv2). Even so, `intervention_margin` is close to 0 for most models: the
  predictions move very little when the action changes, and mostly only their
  direction leans toward the correct target.
- **Report alongside C1.** Report C3 together with the identity of the C1 checkpoint
  and the C1 controls (`c1_unique_key` in the C3 table).

## 9. Comparability

- **MiniMax all-layers subset.** 96 canonical rows and 16 C3 rows were trained on
  1,523 windows and evaluated on 171 val windows (`split_id`
  `canonical_v2_minimax_owner_subset_r2`). The subset membership is not public. Do
  not put these rows side by side with rows on the 276-window val split (caveat
  `VAL_171_SAMPLES_NOT_COMPARABLE_TO_276`). MiniMax-H3 layer 30 A1, A2 and B2 rows use
  the full split.
- **B probes across prefixes.** Prefix 8 scores 80 windows and prefixes 12-24 score
  171, so they cover different windows.
- **Batch size.** Some canonical rows were trained at batch 1 or 4 instead of 2, so
  cross-model comparisons mix different numbers of optimizer steps (caveat
  `TRAIN_BATCH_SIZE_n_NOT_CANONICAL_2`).
- **Timesteps.** Timestep values are scheduler indices (Wan, CogVideoX) or `t * 1000`
  (MiniMax-H3). They are not noise levels and are not comparable across models.
- **Timestep sweep against canonical rows.** The canonical rows are not the middle
  point of the sweep. The CogVideoX sweep layers (15, 20, 25) and the MiniMax-H3 sweep
  layer (15) have no canonical rows. For Wan layers 15, 20 and 25, only A1 uses the
  same batch size (1) in both lines: canonical Wan B2, C1 and C2 used 1, 4 and 4
  against 2 in the sweep, and canonical C1 is trained on h1/h2/h4 while sweep C1 is
  h1 only. The lines were also trained with different code snapshots
  (`training_code_tag`). Do not join the two tables on (model, probe, layer, prefix)
  to build timestep curves. See
  [results/README.md, section 4](../results/README.md#4-coverage).

## 10. Comparing your own runs

```bash
python scripts/export_results.py --runs-root logs/inscene15k_streaming/runs \
  --out results/local --compare results/
```

`results/local/` is ignored by git. The exporter collects `eval/val_summary.json` and
`eval/val_c3_action_sensitivity_summary.json`, applies acceptance checks 1-4 of
section 5 and the final-epoch check, writes the rows to `--out`, and joins them with the accepted tables on
(model, probe, layer, prefix, timestep). It flags joined rows whose validation split
or C3 variant differs from the published row (`SPLIT_MISMATCH`, `SPLIT_UNKNOWN`,
`VARIANT_MISMATCH`); their deltas do not compare like with like. Expect small
differences rather than identical numbers
([PROTOCOL.md, section 9](PROTOCOL.md#9-what-a-rerun-reproduces)).
