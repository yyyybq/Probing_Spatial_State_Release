# C Probes Protocol

Source of truth for current C1/C2/C3 definitions. Code evidence is in
`vidfm3d/eval_diag.py`. C3 is an evaluation protocol over a frozen C1 one-step
predictor; it is not a separately trained probe. Accepted results:

- C1 and C2: `results/canonical_v2_val_seed42.csv` (canonical line) and
  `results/timestep_val_seed42.csv` (timestep line).
- C3 from C1: `results/c3_from_c1_val_seed42.csv`.

Earlier trained-C3 results are not part of this release; label them
`legacy_trained_c3` if you cite them.

## Shared Setup

C probes use the canonical V2 streaming-prefix protocol
(`protocols/inscene_streaming_canonical_v2.json`) over ScanNet++ streaming
features. Inputs are prefix/context features and target-isolated future
features. All accepted rows use `train_split=train`, `eval_split=val`, seed `42`,
and the canonical split in `data/splits/canonical_v2/` (2,721 shared units per
prefix: 2,445 train / 276 val). Build the per-prefix manifests with
`python scripts/build_tstep_canonical_manifests.py build --output-dir <MAN>`.

Accepted canonical C1 and C2 rows cover 10 models over prefixes 8/12/16/24:
`aether`, `cogvideox`, `dino`, `f3r`, `lingbot`, `minimax_h3`, `qwen2_5_vl`,
`videomae`, `vjepa2` and `wan` (136 rows per probe; multi-layer models have one
row per layer and prefix). The 16 `minimax_h3` rows per probe come from the
MiniMax all-layers subset (`split_id=canonical_v2_minimax_owner_subset_r2`, 171
val units) and are not directly comparable to the 276-unit rows. The timestep
line covers `cogvideox`, `minimax_h3` and `wan` (44 C1 and 44 C2 rows). Runs
train C1 and C2; C3 reuses the matched C1 checkpoint at evaluation time.

C1 horizons differ by line:

- Canonical C1 (`action_dynamics_<model>_h124_v1`) trains on horizons 1/2/4,
  with one horizon assigned per prefix, and evaluates every prefix at all three
  horizons (828 val records from 276 units). The accepted Wan C1 runs used
  `action_dynamics_wan_v1` with `action_horizons=[1,2,4]`, which composes to
  `action_dynamics_wan_h124_v1` apart from `job_name`.
- Timestep C1 and the MiniMax all-layers C1 use the h1 configs
  (`action_dynamics_<model>_v1`): trained and evaluated at horizon 1 only.

## C1: Action-conditioned Prediction

Scientific question: given a prefix feature and a relative camera action, can the probe predict the isolated future target feature better than shortcut controls?

Actual input and target:

- Input: `input_feat` from the streaming prefix.
- Action: 9D relative camera action from the batch.
- Target: pooled `target_feat`, computed as the spatial mean of the isolated target feature.
- Probe call: `probe(input_feat, action)`.

Metrics and controls:

- True-action retrieval: `global_R@1`, `global_R@5`, `mean_cos`, `mean_rank`.
- No-action baseline: `probe(input_feat, zeros_like(action))`.
- Shuffled-action baseline: `probe(input_feat, rolled_action)`.
- Last-observation/copy baseline: pooled last prefix observation.
- Derived gains (compute them from the summary fields; the evaluator does not
  write them under these names):
  - `action_gain_vs_no_action = global_R@1 - no_action_global_R@1`
  - `action_gain_vs_shuffle = global_R@1 - shuffled_action_global_R@1`
  - `gain_over_copy = global_R@1 - last_observation_global_R@1`

Interpretation rule:

Absolute C1 `global_R@1` is not sufficient evidence for action reasoning. A row is stronger evidence only if true-action performance beats no-action, shuffled-action, and copy baselines under the same manifest/prefix/model/layer. Report C1 with these baseline-normalized gains, not with absolute `global_R@1` alone.

The results tables carry `no_action_global_R@1` and `shuffled_action_global_R@1`
in `secondary_json`. The copy baseline (`last_observation_global_R@1`) is in
each run's `eval/val_summary.json`.

## C2: Path Integration

Scientific question: can the probe recurrently roll forward through an action sequence and recover waypoint target features/poses?

Actual input and target:

- Input: `input_feat_seq`.
- Action sequence: `path_actions`.
- Horizons: `action_horizons`.
- Target: pooled `target_feat_seq`.
- Pose target: `target_extrinsics`.
- Probe call: `probe(input_feat_seq, path_actions)`.

Metrics:

- Aggregate retrieval: `global_R@1`, `global_R@5`, `mean_cos`, `mean_rank`.
- Pose metrics: `final_pose_error`, `drift_rate`, `mean_step_pose_error`, `loop_closure_error`.
- Horizon metrics: `pose_error_h1`, `pose_error_h2`, `pose_error_h4` when present.

Result structure:

- `results/canonical_v2_val_seed42.csv` and `results/timestep_val_seed42.csv`
  contain one row per experiment with primary metric `global_R@1`;
  `secondary_json` holds `global_R@5`, `mean_cos`, `mean_rank` and
  `final_pose_error`. The other pose and per-horizon metrics are in each run's
  `eval/val_summary.json`.

Open analysis:

Action-shuffle or trajectory-control C2 baselines are not yet canonicalized. Do not claim C2 shortcut controls are solved unless those controls are added to the registry and canonical tables.

## C3: C1 Action-Sensitivity Evaluation (No C3 Training)

Scientific question: with the representation, anchor state, and trained C1
readout fixed, does changing the candidate action change the one-step prediction
in the way required by the matched candidate target?

Required eval-set unit:

- One `action_set_id` identifies a single frozen anchor/context.
- The set contains at least two candidate actions and one matched isolated
  target feature per action.
- Optional `action_equivalence_group` labels actions that should be equivalent
  under the eval-set definition.
- Every candidate is evaluated independently with the C1 call
  `c1_probe(shared_context, candidate_action)`.
- No candidate target, candidate ID, equivalence label, or other branch result
  is an input to C1.

The preferred batched keys are `input_feat`, `candidate_actions`,
`candidate_target_feat`, `candidate_valid`, `action_set_id`, and optional
`action_equivalence_group`. The evaluator also accepts the historical data-only
names `input_feat_seq`, `counterfactual_actions`, `target_feat_seq`, and
`counterfactual_valid`, but still requires an `action_dynamics` checkpoint.

Metrics:

- `intervention_validity`: within-set matching accuracy; when equivalence labels
  exist, retrieving any target in the action's equivalence class is correct.
- `exact_candidate_validity`: stricter diagonal-only matching, retained as a
  diagnostic when equivalent candidates have near-identical targets.
- `intervention_margin`: best same-equivalence-class cosine minus the strongest
  different-class cosine (or diagonal minus off-diagonal without equivalence
  labels).
- `counterfactual_consistency`: mean matched prediction-target cosine. The name
  is retained for table compatibility; it is causal only when the eval targets
  come from restored and executed environment branches.
- `action_sensitivity_cosine`: mean pairwise cosine distance between predictions
  under different candidate actions from the same anchor.
- `target_sensitivity_cosine`: corresponding mean pairwise target distance.
- `target_to_prediction_sensitivity_pearson` and
  `sensitivity_calibration_mae`: whether predicted changes track target changes,
  rather than merely changing by an arbitrary amount.
- When equivalence labels exist: `equivalent_action_prediction_distance`,
  `distinct_action_prediction_distance`, and `equivalence_separation_gap`.

Interpretation rule:

C3 is a held-out capability analysis of C1, not an additional learnable
capacity. Report it together with the C1 checkpoint identity and C1 shortcut
controls. High output change alone is not success: it must align with the
matched target change. For equivalent actions, lower within-class prediction
distance is desirable; for distinct actions, matching accuracy/margin and
between-class separation are desirable.

Do not call same-trajectory h1/h2/h4 endpoints counterfactual branches. They may
be used as a multi-action sensitivity diagnostic, but causal language requires
an eval set produced by restoring the exact anchor state and executing each
candidate action. The earlier trained-C3 head, its configs and its tables are
not part of this release and are not the current protocol. Every row in
`results/c3_from_c1_val_seed42.csv` carries the caveat
`C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL`.

Run with the C1 experiment, the same overrides and environment as the C1 training
run, the final C1 checkpoint and the evaluation-only switch. Pass
`hydra.output_subdir=null`, because the evaluation reuses the C1 run folder and
would otherwise overwrite the C1 run's Hydra record `.hydra/config.yaml` with the
evaluation settings:

```bash
python vidfm3d/eval_diag.py \
  experiment=inscene15k_streaming/action_dynamics_wan_h124_v1 \
  <same overrides as the C1 training run, including paths.run_folder_name> \
  "ckpt_path='/path/to/epoch=49-step=N.ckpt'" \
  ++evaluation_protocol=c3_action_sensitivity \
  +eval_split=val train=false test=false hydra.output_subdir=null
```

For a C1 checkpoint trained with an h1 config, also pass
`action_horizons=[1,2,4]` so that each action set has three candidates.

### Variants in the C3 results table

- `c3_from_c1` (120 rows): protocol C3. Canonical C1 checkpoints trained on
  horizons 1/2/4 (`action_dynamics_<model>_h124_v1`) for nine models (all except
  `minimax_h3`), evaluated on 276 action sets of three candidates each.
- `h1_trained_diagnostic` (60 rows): C1 checkpoints trained on horizon 1 only
  and evaluated with `action_horizons=[1,2,4]`: the 44 timestep-line C1 rows and
  the 16 MiniMax all-layers C1 rows. The probe never saw horizons 2 and 4 during
  training, so these rows are not protocol C3. They carry the caveat
  `H1_TRAINED_NOT_PROTOCOL_C3`.

The primary metric is `intervention_validity` (chance 1/3 with three
candidates).

The ordinary mixed-horizon C1 loader groups rows by `manifest_sample_id`. A
dedicated alternative-action eval dataset should instead emit an explicit
`action_set_id` and the preferred candidate batch keys above.

For a frozen `.pt` artifact, store either a list of action-set dictionaries or
`{"records": [...]}` and use the checked loader directly:

```bash
python vidfm3d/eval_diag.py \
  experiment=inscene15k_streaming/action_dynamics_wan_h124_v1 \
  ckpt_path=/path/to/c1.ckpt \
  +evaluation_protocol=c3_action_sensitivity \
  "data.data_module.validation_datasets=[\"C1ActionSetDataset(path='/path/to/action_sets.pt',split='val')\"]" \
  data.data_module.batch_size_per_device_val=4 \
  data.data_module.num_workers_val=0 \
  +eval_split=val train=false test=false hydra.output_subdir=null
```

Each artifact record contains unbatched `input_feat (T,H,W,C)`,
`candidate_actions (K,9)`, `candidate_target_feat (K,H,W,C)` or `(K,C)`,
`candidate_valid (K)`, `candidate_ids (K)`, and the lineage strings
`action_set_id`, `manifest_sample_id`, and `scene_path`. Optional
`action_equivalence_group (K)` must use integer class IDs. The loader rejects
duplicate sets/candidates, fewer than two valid candidates, shape/channel
mismatches, and non-finite tensors before inference.

Protocol-specific outputs are written as
`eval/<split>_c3_action_sensitivity_predictions.pt` and
`eval/<split>_c3_action_sensitivity_summary.json`, so running C3 analysis cannot
overwrite the ordinary C1 evaluation dump.
