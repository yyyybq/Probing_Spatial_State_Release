# Results

Static tables of the accepted probe evaluations. Every number is a validation-split
result for training seed 42. There is no held-out test split, so none of these rows
is a test result.

| File | Rows | Content |
| --- | ---: | --- |
| `canonical_v2_val_seed42.csv` | 804 | Canonical V2 matrix: 10 models, six probes, prefixes 8/12/16/24 |
| `timestep_val_seed42.csv` | 316 | Feature timestep sweep for CogVideoX, Wan and MiniMax-H3 |
| `c3_from_c1_val_seed42.csv` | 180 | C3 action-sensitivity evaluation of trained C1 heads: 120 protocol C3 rows and 60 `h1_trained_diagnostic` rows |
| `SOURCES.json` | | sha256 of each table and of the internal table it was exported from, and the dropped and derived columns |

The protocol is described in [docs/PROTOCOL.md](../docs/PROTOCOL.md) and the evaluators
in [docs/EVALUATION.md](../docs/EVALUATION.md).

## 1. What ACCEPTED means

A row is `ACCEPTED` when all of these hold:

- it was produced by a v2 evaluator (`evaluator_version` is `eval_diag_v2_nonfinite_aware`
  or `a1_streaming_eval_v2_nonfinite_aware`), which marks non-finite predictions as
  invalid instead of scoring them;
- the primary metric is finite, `n_nonfinite_pred == 0` and `n_scored > 0`;
- the sha256 of the evaluator summary matched the evaluation receipt.

All rows use `eval_split=val` and seed 42. ACCEPTED means that the evaluation is valid.
It does not mean that the number is comparable with every other row (see the caveats).

## 2. Columns

| Column | Meaning |
| --- | --- |
| `unique_key` | `<model>__scannetpp__t<timestep>__l<layer>__p<prefix>__<probe>__seed42`. In the C3 table the probe part is `c3_from_c1`. |
| `line` | `canonical_v2`, `timestep` or `c3_from_c1` |
| `model`, `probe` | Feature model and probe (section 3) |
| `variant` | C3 table only: `c3_from_c1` (C1 trained on horizons 1/2/4) or `h1_trained_diagnostic` (C1 trained on horizon 1 only). Empty in the other tables. |
| `layer` | Table layer: the layer numbering of these tables and of the `--layer` option of the scripts. `lingbot` layer 48 and `minimax_h3` layer 50 count the last block from 1 and are disk layers 47 and 49; use 47 and 49 when extracting features. All other layers are disk layers. `qwen2_5_vl` uses -1, the vision merger output tokens (3,584 channels). |
| `prefix` | Number of observed frames: 8, 12, 16 or 24 |
| `timestep` | Feature timestep from the key. Canonical rows use 749, or 250 for `lingbot` and `minimax_h3`. For the encoders (`dino`, `f3r`, `videomae`, `vjepa2`) and `qwen2_5_vl`, 749 is only a placeholder. |
| `seed`, `eval_split` | Always 42 and `val` |
| `split_id` | `canonical_v2` (2,445 train / 276 val windows, `data/splits/canonical_v2`), or `canonical_v2_minimax_owner_subset_r2` (1,523 train / 171 val windows) for scope `canonical_v2_minimax_all_layers_20261011`. The second id is a historical label, kept verbatim; it names the MiniMax-H3 all-layers subset. |
| `scope` | Training campaign the run belongs to (see the glossary below); informational only |
| `status` | Always `ACCEPTED` |
| `primary_metric`, `primary_value` | Section 3 |
| `primary_ci_low`, `primary_ci_high` | 95% bootstrap interval, given for B1/B2 (resampling scenes) and C3 (resampling action sets) only; empty otherwise |
| `secondary_json` | Secondary metrics of the evaluator summary, as JSON |
| `n_loader`, `n_scored` | Evaluation records loaded and scored. C1 trained on horizons 1/2/4 has 3 records per window (828 = 3 x 276). |
| `n_nonfinite_pred` | Records with non-finite predictions (always 0) |
| `evaluator_version` | Evaluator that wrote the summary |
| `eval_revision` | Evaluation pass: canonical `r2` (668 rows), `r3` (8 rows re-run with corrected inputs), `r4` (128 supplementary rows); `t1` for timestep; `c3` for C3 |
| `eval_pack`, `training_code_tag` | Names of the frozen code snapshots used for evaluation and training (caveat `TRAINING_AND_EVAL_PACK_COMMITS_NOT_ON_GITHUB`) |
| `ckpt_epoch`, `ckpt_step` | From the checkpoint name `epoch=E-step=S.ckpt` |
| `train_batch_size_inferred` | Training batch size B with `floor(N_train / B) * (ckpt_epoch + 1) == ckpt_step`. N_train is 2,445, or 1,523 for the MiniMax all-layers subset. Lightning drops the last partial batch. |
| `checkpoint_replaced` | `true` when an fp32 retrain replaced a checkpoint whose weights were non-finite |
| `checkpoint_sha256`, `summary_file`, `summary_sha256` | Identity of the evaluated checkpoint and of the evaluator summary |
| `caveats` | `;`-separated caveat codes (section 5) |
| `c1_unique_key` | C3 table only: the C1 row in `canonical_v2_val_seed42.csv` or `timestep_val_seed42.csv` whose checkpoint was evaluated |

**Glossary of `scope` labels.** The labels are historical campaign names, kept verbatim
from the accepted runs. A number in a label is the planned run count of the campaign,
not its row count here; a date names the campaign.

| `scope` | Rows (canonical / timestep / C3) | Meaning |
| --- | ---: | --- |
| `canonical_v2_main552` | 536 / 0 / 92 | Canonical matrix of `aether`, `cogvideox`, `dino`, `f3r`, `lingbot`, `qwen2_5_vl`, `videomae` and `vjepa2` at batch 2: 23 (model, layer) pairs x 6 probes x 4 prefixes = 552 runs. 16 of its B2 runs were replaced by fp32 retrains. |
| `canonical_v2_wan168` | 140 / 0 / 28 | Wan canonical runs at the Wan config batch sizes: 7 layers x 6 probes x 4 prefixes = 168 runs. The 28 Wan A2 runs are not released. |
| `canonical_v2_minimax_l30_ab16` | 16 / 0 / 0 | MiniMax-H3 layer 30: A1 depth, A1 mask, A2 and B2 at 4 prefixes (16 runs), on the full 276-window val split |
| `canonical_v2_minimax_all_layers_20261011` | 96 / 0 / 16 | MiniMax-H3 all-layers runs on the 1,523 / 171-window subset |
| `canonical_v2_b2_fp32_retrain_20261016`, `tstep_b2_fp32_retrain_20261016` | 16 / 32 / 0 | fp32 retrains of B2 runs whose checkpoints had non-finite weights |
| `tstep_probe_opt_v3`, `tstep_probe_pack_v4_1` | 0 / 284 / 44 | Timestep runs, named after their training code snapshot (`training_code_tag`) |

**Recipe run groups.** `scripts/probe_recipe.py` prints the run group of a cell in its
header (`# run group: ...`). Each group shares one set of recipe rules:

| Run group | Cells | Scopes of the accepted rows |
| --- | --- | --- |
| `canonical_main` | Canonical cells at batch 2, on the public split | `canonical_v2_main552` and its fp32 B2 retrains |
| `canonical_wan` | Wan canonical cells | `canonical_v2_wan168` |
| `canonical_minimax_l30` | MiniMax-H3 layer 30 A1, A2 and B2 on the public split | `canonical_v2_minimax_l30_ab16` |
| `minimax_all_layers_subset` | MiniMax-H3 cells on the non-public subset split (`--split minimax_subset`) | `canonical_v2_minimax_all_layers_20261011` |
| `timestep` | Timestep sweep | `tstep_*` |

## 3. Primary metric per probe

| Probe | Name | Primary metric | Better |
| --- | --- | --- | --- |
| `streaming_depth` | A1 depth and camera | `mean_loss_depth` | lower |
| `streaming_mask` | A1 instance identity mask | `mean_loss_identity` | lower |
| `view_consistency` | A2 view overlap | `balanced_acc` | higher |
| `ego_belief` | B1 hidden-object direction, mask-conditioned (timestep table only) | `mean_az_err_deg` | lower |
| `ego_belief_v2` | B2 hidden-object direction, query-conditioned | `mean_ang_err_deg` | lower |
| `action_dynamics` | C1 one-step action dynamics | `global_R@1` | higher |
| `path_integration` | C2 path integration | `global_R@1` | higher |
| `c3_from_c1` | C3 action sensitivity of a trained C1 head | `intervention_validity` (chance 1/3) | higher |

## 4. Coverage

Canonical V2 (`canonical_v2_val_seed42.csv`), all at prefixes 8/12/16/24:

| Model | Layers | Rows | Notes |
| --- | --- | ---: | --- |
| `aether` | 1 | 24 | |
| `cogvideox` | 0, 5, 10, 30, 35, 40 | 144 | |
| `dino` | 0 | 24 | |
| `f3r` | 24 | 24 | |
| `lingbot` | 10, 20, 30, 40, 48 | 120 | 48 is disk layer 47 |
| `minimax_h3` | 10, 20, 30, 40, 50 | 112 | 50 is disk layer 49. No C1/C2 at layer 10. C1/C2 at layers 20-50 and all probes at layers 10, 20, 40, 50 use the 171-window val subset. A1/A2/B2 at layer 30 use the full 276-window val split. |
| `qwen2_5_vl` | -1 | 24 | |
| `videomae` | 5, 11, 17, 23 | 96 | |
| `vjepa2` | 5, 11, 17, 23 | 96 | |
| `wan` | 0, 5, 10, 15, 20, 25, 29 | 140 | No A2 (`view_consistency`) |

Timestep sweep (`timestep_val_seed42.csv`), seven probes (B1 included):

| Model | Timesteps | Layers | Rows | Notes |
| --- | --- | --- | ---: | --- |
| `cogvideox` | 499, 999 | 15, 20, 25 | 168 | Complete |
| `wan` | 999 | 15, 20, 25 | 84 | Complete |
| `wan` | 499 | 15, 20, 25 | 8 | A1 only: 4 depth and 4 mask rows |
| `minimax_h3` | 500, 1000 | 15 | 56 | Layers 20 and 25 are not available |

**Joining the sweep with canonical rows.** The canonical rows are not the middle point
of the timestep sweep. The CogVideoX sweep layers (15, 20, 25) and the MiniMax-H3 sweep
layer (15) have no canonical rows, and canonical MiniMax-H3 uses timestep 250, not
500 or 1000. For Wan layers 15, 20 and 25, only A1 has the same batch size (1) in both
lines. Canonical Wan B2, C1 and C2 used 1, 4 and 4 against 2 in the sweep, canonical C1
is trained on h1/h2/h4 (828 records) while sweep C1 is h1 only (276 records), Wan has
no canonical A2, and B1 exists only in the sweep. The two lines were also trained with
different code snapshots (`training_code_tag`). Do not join the two tables on (model,
probe, layer, prefix) to build timestep curves.

C3 (`c3_from_c1_val_seed42.csv`): one row per accepted C1 row, 136 canonical and 44 timestep.

## 5. Caveat codes

Counts are rows per table.

| Code | Meaning | Canonical | Timestep | C3 |
| --- | --- | ---: | ---: | ---: |
| `VAL_SPLIT_SEED42_ONLY` | Validation split and one training seed; no test split, no seed variance | 804 | 316 | 180 |
| `TRAINING_AND_EVAL_PACK_COMMITS_NOT_ON_GITHUB` | The frozen code snapshots that trained and evaluated the row are named in `training_code_tag` and `eval_pack` but are not published. The 32 timestep rows without this code name `commit e4e4886`, a commit of the development repository that is not in this repository's history either. The C3 rows name `main_eval_v2_from_main_206c5de`: they were evaluated from development commit 206c5de, which is not in this history. On the C3 path the shipped `vidfm3d/eval_diag.py` differs from that commit only in non-finite handling, which does not change any accepted row. | 804 | 284 | 180 |
| `TRAINING_CODE_SNAPSHOT_IDENTITY_20261017` | The timestep training code is identified by a snapshot taken on 2026-10-17; no file hash manifest was recorded at training time | 0 | 284 | 44 |
| `REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS` | C1 (`action_dynamics`): read `global_R@1` against the zero-action, shuffled-action and copy-last-observation controls; `secondary_json` carries `no_action_global_R@1` and `shuffled_action_global_R@1`, and the copy-last value is in the evaluator summary. C2 (`path_integration`) rows carry the code too, but no C2 controls are computed and none are in `secondary_json`. | 272 | 88 | 0 |
| `HIDDEN_OBJECT_SCORED_x_OF_y` | B1/B2: only x of y val windows have a valid hidden object and are scored: 80/276 at prefix 8 and 171/276 at prefixes 12-24; on the MiniMax subset 48/171 and 114/171 | 140 | 88 | 0 |
| `TRAIN_BATCH_SIZE_n_NOT_CANONICAL_2` | Trained at batch size n instead of the canonical 2 (Wan: 1 for A1 and B2, 4 for C1 and C2; MiniMax layer 30: 1 for A1). Cross-model comparisons mix different numbers of optimizer steps. | 148 | 0 | 28 |
| `VAL_171_SAMPLES_NOT_COMPARABLE_TO_276` | MiniMax all-layers subset: 171 val windows, not comparable with rows on the 276-window split | 96 | 0 | 16 |
| `FP32_RETRAIN_REPLACES_NONFINITE_CHECKPOINT` | The original checkpoint had non-finite weights; the row evaluates its fp32 retrain | 16 | 32 | 0 |
| `C1_TRAINED_AND_EVALUATED_H1_ONLY` | C1 trained and evaluated on horizon 1 only (one record per window), unlike the canonical 1/2/4 horizons | 16 | 44 | 0 |
| `C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL` | The three C3 candidates are the h1/h2/h4 endpoints of the same trajectory, not counterfactual branches | 0 | 0 | 180 |
| `H1_TRAINED_NOT_PROTOCOL_C3` | C3 of a C1 head that never saw horizons 2 and 4 in training; report separately from protocol C3 | 0 | 0 | 60 |

Batch sizes: canonical 656 rows at 2, 92 at 1 and 56 at 4; timestep 1 for A1 and 2 for
all other probes (the timestep protocol, so not flagged). C3 rows inherit the caveats of
their C1 run with two exceptions: the C1/C2 control caveat, which does not apply to
`intervention_validity`, and `C1_TRAINED_AND_EVALUATED_H1_ONLY`, which
`H1_TRAINED_NOT_PROTOCOL_C3` replaces on the 60 diagnostic rows. Every C3 row also
carries `C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL`.

## 6. Known gaps

- 1,444 runs were trained (868 canonical, 576 timestep) and 1,120 are accepted here.
  Not released: 128 runs await a retrain (MiniMax-H3 timestep layers 20 and 25, whose
  cached features overflowed in fp16); 88 runs are not reproducible because their
  training code cannot be recovered (Wan timestep 499); 44 are legacy C3 heads trained
  separately under an older protocol; 28 are Wan A2 runs with conflicting sanity checks;
  36 are MiniMax-H3 probes outside the six-probe protocol.
- Final-epoch weights were evaluated, with no checkpoint selection: epoch 99 for A1 and
  epoch 49 for all other probes.
- Timestep values are scheduler indices for Wan and CogVideoX and `t * 1000` for
  MiniMax-H3 (the shipped MiniMax-H3 extractor maps 1000 to t = 0.999). They are not
  noise levels and are not comparable across models.
- The 56 timestep MiniMax-H3 rows (layer 15) and the 8 C3 rows built on them read fp16
  feature caches. The shipped extractor writes MiniMax-H3 features in bf16, so a
  re-extracted cache is not the one those runs read.
- The membership of the MiniMax all-layers 1,523/171 subset is not part of
  `data/splits/canonical_v2`.
- C3: all 120 protocol rows and 55 of the 60 diagnostic rows have `primary_ci_low > 1/3`,
  but `intervention_margin` in `secondary_json` is close to 0, so predictions move very
  little when the action changes. A predictor that ignores the action scores exactly 1/3.
  The intervals resample action sets, not scenes, and are therefore narrow.

## 7. Compare your own run

Train and evaluate with the recipe in the top-level README, then export the evaluator
summaries under your run directory in this schema and print the differences to these tables:

```bash
python scripts/export_results.py --runs-root <dir> --compare results/
```

Rows are joined on (model, probe, layer, prefix, timestep). Joined rows whose
validation split or C3 variant differs from the published row are flagged
(`SPLIT_MISMATCH`, `SPLIT_UNKNOWN`, `VARIANT_MISMATCH`), because their deltas do not
compare like with like.
