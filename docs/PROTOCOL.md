# Probing protocol (canonical V2)

This document defines the protocol behind every accepted number in `results/`: data,
windows, split, feature extraction, probe definitions, controls, and the training
recipe. Installation and the dataset download are in the top-level `README.md`.
Evaluators, metrics and acceptance rules are in [EVALUATION.md](EVALUATION.md). The
result tables and their caveat codes are in [results/README.md](../results/README.md).

Each value below names the release file that defines it. If this document and that
file disagree, the file is correct.

## At a glance

| Item | Value | Defined in |
| --- | --- | --- |
| Protocol id | `inscene_streaming_canonical_v2_obs28_h124` | `protocols/inscene_streaming_canonical_v2.json` |
| Protocol sha256 | `215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f` | same file |
| Data | InsScene-15K, ScanNet++ subset only | same file (`source: scannetpp`) |
| Window | 28 motion-uniform observations | same file (`canonical_window`) |
| Prefixes | 8, 12, 16, 24 observations | same file (`streaming`) |
| Future horizons (C probes) | 1, 2, 4 observations after the prefix tail | same file (`target`) |
| Split | 612 train / 69 val scenes; 2,445 train / 276 val windows | `data/splits/canonical_v2/SPLIT.json` |
| Test split | none | `SPLIT.json` (`test_split: null`) |
| Training seed | 42 | every `configs/experiment/inscene15k_streaming/*.yaml` |
| Reported weights | final epoch (99 for A1, 49 otherwise), no checkpoint selection | `results/*.csv` (`ckpt_epoch`) |

## Scope

The release covers three lines of experiments:

- **Canonical V2.** Six probes (A1 depth, A1 mask, A2, B2, C1, C2) on ten frozen
  feature models, at prefixes 8/12/16/24. V2 is the version label of the canonical
  window protocol (`inscene_streaming_canonical_v2_obs28_h124`).
- **Timestep sweep.** The same windows, split and probes plus B1, with features taken
  at diffusion timesteps 499/999 of CogVideoX and Wan and 500/1000 of MiniMax-H3,
  partly at other layers than the canonical line. C1 is trained on horizon 1 only in
  this line.
- **C3-from-C1.** An evaluation of trained C1 checkpoints. Nothing is trained for it.
  See [EVALUATION.md](EVALUATION.md#8-c3-from-c1).

**There is no held-out test split.** Every accepted number is a validation-split result
for training seed 42. Each one comes from the final-epoch weights of its run, with no
checkpoint selection. Read them as validation results, not test results.

## 1. Data

- **Dataset.** InsScene-15K, ScanNet++ subset (`processed_scannetpp_v2`) only. Each
  scene directory has `images/`, instance masks in `refined_ins_ids/`, `depth/` and
  `scene_iphone_metadata.npz`, which holds the frame list and the camera trajectory
  (`data/download_inscene15k.py`).
- **No Infinigen.** Infinigen frames are independently rendered views, not a video
  trajectory. They cannot act as time steps for temporal, memory, action or
  path-integration probes. The extractor refuses the temporal modes for any source
  other than `scannetpp` unless `ALLOW_INFINIGEN_TEMPORAL=1` is set
  (`features/run_inscene15k.py`). No accepted row uses Infinigen.
- **Frame index** (`frame_index_schema: image_mask_intersection_v1`). Frame names
  include the image extension, for example `frame_000000.jpg`. A frame `<name>` is
  valid only if both `images/<name>` and its mask `refined_ins_ids/<name>.npy` exist,
  for example `images/frame_000000.jpg` and `refined_ins_ids/frame_000000.jpg.npy`.
  Valid frames stay in metadata order, so they line up with the pose rows
  (`_collect_scannetpp_scenes` in `vidfm3d/data/components/inscene15k_dataset.py`).
  All window indices below index this valid-frame list. Every manifest row carries the
  list as `valid_frames`.

## 2. Canonical V2 windows

Window parameters (`canonical_window` in the protocol JSON; the implementation is
`vidfm3d/utils/temporal_sampling.py`):

| Parameter | Value |
| --- | --- |
| `observations` | 28 |
| `sampling` | `motion_uniform` |
| `motion_step` | 0.35 |
| `rotation_weight` | 0.5 |
| `window_stride` | 8 observations |
| `max_windows_per_scene` | 4 |

A window is built in three steps:

1. Cumulative camera motion along the trajectory is the sum of
   `translation + 0.5 * rotation_angle_rad` between consecutive valid frames.
2. One observation is taken every 0.35 units of cumulative motion: the valid frame
   closest to each step, plus the last frame.
3. A window is 28 consecutive observations. Windows start every 8 observations, plus
   the last possible start. When more than 4 starts exist, 4 evenly spaced ones are
   kept.

The window length is 28 because the largest prefix is 24 and the largest horizon is 4.

**Prefixes and targets** (`streaming` and `target` in the protocol JSON):

- Input for prefix `p`: `window[:p]`, the first `p` observations.
- Anchor, which is the current frame or prefix tail: `window[p-1]`.
- Future target for horizon `h` in {1, 2, 4}: `window[p-1+h]`. A horizon counts
  observations, not raw frames. C1 records the raw frame gap separately as
  `raw_frame_offset`.

| Prefix `p` | Input observations | Anchor | h = 1 | h = 2 | h = 4 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 0-7 | 7 | 8 | 9 | 11 |
| 12 | 0-11 | 11 | 12 | 13 | 15 |
| 16 | 0-15 | 15 | 16 | 17 | 19 |
| 24 | 0-23 | 23 | 24 | 25 | 27 |

All probes and all prefixes use the same windows, so a window id refers to the same
frames at every prefix. Each prefix is trained as a separate job with a fixed input
length (`prefix_len`).

**Universe** (`expected_universe` in the protocol JSON): 2,721 windows from 681 scenes.
At 4 prefixes each, this gives 10,884 prefix samples.

## 3. Split

The source of truth is
`data/splits/canonical_v2/inscene_c_h124_prefix24.public.jsonl.gz`. It has one row per
(scene, window), with `split`, the 28 window indices, the valid-frame list and the pose
indices, and no storage paths. `SPLIT.json` in the same directory documents the rule
and the counts. `scenes.csv` lists the 681 scenes and their split.

**Rule** (`SPLIT.json`, `rule`). Sort the 683 ScanNet++ scene directory names. Permute
them with `numpy.random.default_rng(42)`. The first `int(0.9 * 683) = 614` scenes are
train and the remaining 69 are val. Two scenes are then dropped: `a003a6585e` (not
enough valid frames) and `c24f94007b` (no images directory). Both fall on the train
side. The split is by scene, so no scene has windows in both splits.

| | Train | Val |
| --- | ---: | ---: |
| Scenes | 612 | 69 |
| Windows | 2,445 | 276 |
| Windows with id 0 / 1 / 2 / 3 | 612 / 612 / 611 / 610 | 69 / 69 / 69 / 69 |

**Integrity.** `universe_sha256 = 4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b`
(`EXPECTED_UNIVERSE_SHA256` in `scripts/build_tstep_canonical_manifests.py`). The hash
covers `scene_id`, `split`, window id and window indices of every window.

**Per-prefix manifests.** Build them once and audit them:

```bash
python scripts/build_tstep_canonical_manifests.py build --output-dir "$CANONICAL_MANIFEST_DIR"
python scripts/build_tstep_canonical_manifests.py audit --manifest-dir "$CANONICAL_MANIFEST_DIR"
```

This writes `inscene_c_h124_prefix{8,12,16,24}.jsonl` (each with a `.meta.json`) and
`canonical_summary.json`. The build fails unless the universe hash matches. Each row
gets `sample_id = <scene>_w<window>_p<prefix>`, `streaming_indices = window[:p]` and
`streaming_tail = p - 1`.

**Using the manifests.** Set these for every training and evaluation job:

| Variable | Value | Effect |
| --- | --- | --- |
| `INSCENE_MANIFEST` | `$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix<P>.jsonl`, with `<P>` equal to `prefix_len` | The loader takes samples and split from the manifest. |
| `INSCENE_LOCAL_SCENE_ROOT` | Directory that holds the scene directories | Maps each `scene_id` to a local scene directory. |
| `INSCENE_TRUST_EXPANDED_MANIFEST` | `1` | Keeps every manifest row. Without it, the loader drops rows whose prefix is missing from the feature cache's `prefix_index.npy`, which silently shrinks the sample count. |

Without `INSCENE_MANIFEST`, the loader logs a warning and applies the seed-42 rule to
the scenes it finds at runtime. That puts 8 of the 681 scenes in the other split
(`SPLIT.json`, notes). Always train and evaluate from the built manifests.

**No test split.** The manifests contain only train and val rows.
`vidfm3d/eval_a1_streaming.py` refuses any `eval_split` other than `val`.
`vidfm3d/eval_diag.py` accepts `eval_split=test` only when the loader finds test rows,
and no shipped file contains any.

**MiniMax all-layers subset.** 96 canonical rows and 16 C3 rows used a smaller window
set: 1,523 train and 171 val windows. Their `split_id` in the result tables is
`canonical_v2_minimax_owner_subset_r2`. The subset membership is not published, so
these rows cannot be re-run from `data/splits/canonical_v2`. Their numbers are not
comparable with rows on the 276-window val split (caveat
`VAL_171_SAMPLES_NOT_COMPARABLE_TO_276`).

## 4. Feature extraction

The feature models are frozen. Features are computed once and cached as safetensors
files (`.sft`), each with a JSON sidecar. The probes only ever read these caches.

Each model needs two extraction passes. `FEATURE_NAMESPACE` must equal the basename of
`--out-root`, and the extractor exits otherwise (`features/run_inscene15k.py`).

| Pass | `--mode` | Namespace (`--out-root` basename) | What is forwarded |
| --- | --- | --- | --- |
| Streaming inputs | `streaming_prefix` | `FEAT_STREAMING_CANONICAL_V2_SEEDED_V1` | Each prefix `window[:p]`, p = 8/12/16/24, in its own forward pass. Prefixes are never sliced from a longer clip. |
| C targets | `target_isolated` with `--target-from-streaming-windows` | `FEAT_TARGET_CANONICAL_V2_SEEDED_V1` | Each anchor and future target frame in its own forward pass, with the frame repeated to fill the model's clip |

A1, A2, B1 and B2 read only the streaming pass. C1 and C2 also read the target pass.
The loader appends `<vfm>/<source>/<scene>` to `INSCENE_STREAMING_FEAT_ROOT` and
`INSCENE_TARGET_FEAT_ROOT`. Set `INSCENE_TARGET_LAYOUT=flat` for target caches written
this way. Qwen2.5-VL uses `features/run_inscene15k_mllm.py` with the same two
namespaces. The README and `scripts/run_streaming_probe_sweep.sh` give the full
extraction commands.

The extractor recomputes the windows from the camera poses with the parameters of
section 2. They match the shipped rows only if your copy of the dataset matches the
snapshot behind the split. The dataset revision is not pinned (see the README).

**Deterministic noise.** Feature generation follows the contract
`inscene_deterministic_noise_v1` (`features/noise_seed.py`; model groups from
`features/run_inscene15k.py`):

| Models | Seed scheme | Noise |
| --- | --- | --- |
| Wan, CogVideoX, Aether, Fast3R, LingBot | `inscene_noise_seed_v1` | A fresh torch generator per logical sample, seeded from a hash of the sample identity |
| MiniMax-H3 | `minimax_h3_fixed_seed_v1` | Fixed internal seed 0, reset before every forward pass |
| DINOv2, V-JEPA2, VideoMAE | `deterministic_forward_no_noise_v1` | No random draws |

The seed is the first 8 bytes (big-endian) of the SHA-256 of the compact, key-sorted
JSON `{noise_seed_scheme, identity}`, masked to 63 bits. The identity contains:

- the canonical protocol id and sha256;
- `vfm`, `model_id`, `scene_id` and the sample kind;
- for a streaming prefix, `window_id` and `prefix_len`; for an isolated target,
  `frame_index`.

The identity leaves out timestep, layer, process rank and sample order. A sample
therefore sees the same noise at every timestep and layer, under any sharding.

**Storage.** Features are stored in fp16. MiniMax-H3 features are stored in bf16
because its activations exceed the fp16 range. They are checked for finiteness before
and after the cast (`MINIMAX_H3_FINITE_BF16_V1`, `prepare_feature_for_storage` in
`features/run_inscene15k.py`). The accepted timestep MiniMax-H3 layer-15 rows (56
timestep rows and the 8 C3 rows built on them) read fp16 caches that were written
before this rule and were finite at that layer. A fresh extraction writes bf16, so it
does not reproduce those caches byte for byte; `scripts/probe_recipe.py` notes this
for those cells.

**Timesteps.** `--t` means different things for different models:

- Wan, CogVideoX and Aether: a scheduler index. Canonical value 749.
- MiniMax-H3: `t * 1000`. LingBot: `sigma * 1000`. Canonical value 250 for both.
- Encoders (DINOv2, V-JEPA2, VideoMAE, Fast3R) and Qwen2.5-VL: no timestep. The value
  749 in configs and result keys is only a placeholder.

The timestep sweep uses 499/999 for CogVideoX and Wan, and 500/1000 for MiniMax-H3.
These values are not comparable across models.

**Layers and file names.** Cache file names come from `feature_layer` and
`feature_timestep` (`vidfm3d/utils/feature_layers.py`):

- diffusion models: `feature_t<T>_layer<L>.sft`;
- V-JEPA2, VideoMAE and Qwen2.5-VL: `feature_layer<L>.sft`;
- Fast3R: `feature_l<L>.sft`;
- DINOv2: `feature.sft`.

The result tables call the last LingBot block L48 and the last MiniMax-H3 block L50.
On disk these are layers 47 and 49, so use 47 and 49 for extraction and for
`feature_layer`.

**Provenance of the original features.** The original canonical runs read their
features through the owner overlay (section 8). The overlay also used gap-fill and
MiniMax bf16 caches, whose namespaces end in `_GAP_DELTA_V1` and `_MINIMAX_BF16_V2`
(both listed in `features/run_inscene15k.py`). A fresh extraction into the two seeded
namespaces follows the same protocol but is not guaranteed to be byte-identical to
those caches.

**Optional provenance tooling.** The extractors keep a few hooks that were used to
produce the original caches: `--canonical-gap-worklist` and
`--canonical-gap-worklist-sha256` (gap-fill worklists, `features/canonical_gap_worklist.py`),
the `CANONICAL_GAP_{COMPLETED,LOCAL_READY}_{SCENES,UNITS}_FILE` exclusion lists with
their `_SHA256` bindings, `EXTRACT_DONE_INDEX`, and `MINIMAX_H3_MODEL_LOAD_ID` with
`MINIMAX_H3_MIRROR_*` for a local copy of the MiniMax-H3 weights. All of them are off
by default, the worklists are not published, and none of them is needed to reproduce
the results.

## 5. Probes

Each probe is a small trainable head on frozen features. Unless a probe says
otherwise, its input is the cached streaming feature of the `p` prefix observations,
with shape `(p, H, W, C)`. "Current frame" means the prefix tail `window[p-1]`. No probe
receives camera poses as input, except the C-probe actions described below. The
configs are in `configs/experiment/inscene15k_streaming/` and the heads in
`vidfm3d/models/components/`.

| ID | Probe (result name) | Target | Primary metric | Better |
| --- | --- | --- | --- | --- |
| A1 | `streaming_depth` | Depth map and camera pose of every prefix frame | `mean_loss_depth` | lower |
| A1 | `streaming_mask` | Instance identity of every pixel across the prefix | `mean_loss_identity` | lower |
| A2 | `view_consistency` | View overlap of every pair of prefix frames | `balanced_acc` | higher |
| B1 | `ego_belief` | Direction of a hidden object, from its past masks | `mean_az_err_deg` | lower |
| B2 | `ego_belief_v2` | Direction of a hidden object, from an appearance query | `mean_ang_err_deg` | lower |
| C1 | `action_dynamics` | Pooled feature of a future frame, given the camera action | `global_R@1` | higher |
| C2 | `path_integration` | Pooled features at h1, h2, h4, given the action sequence | `global_R@1` | higher |

Section 7 lists the config for each probe. EVALUATION.md lists the secondary metrics.

### A1 depth and camera (`streaming_depth`)

- **Input:** prefix features. The head is `ProbeModelPA`, a pixel-aligned DPT-style
  decoder, with its depth and camera heads active (`active_heads: [depth, camera]`).
- **Target:** the ground-truth depth map of every prefix frame, kept in metric units
  and valid where depth > 0. The camera head predicts the poses of all prefix frames,
  rebased to the current frame.
- **Metric:** `mean_loss_depth`, the absolute depth error weighted by the ground-truth
  validity map and averaged over the prefix frames. Lower is better. Camera accuracy is
  reported as `mean_Auc_30` (pose AUC at 30 degrees). Higher is better. Pose is this
  camera head; there is no separate camera probe.

### A1 instance mask (`streaming_mask`)

- **Input:** prefix features, with the same head as A1 depth but only the identity head
  active (`active_heads: [identity]`).
- **Target:** the instance id of every pixel in every prefix frame, from
  `refined_ins_ids`.
- **Metric:** `mean_loss_identity`. It is a per-pixel cross-entropy of each pixel
  embedding against per-instance mean embeddings (prototypes) computed over the whole
  prefix (`identity_loss` in `vidfm3d/utils/loss.py`). Lower is better.

### A2 view consistency (`view_consistency`)

- **Input:** prefix features. The head (`ViewConsistencyProbe`) predicts an overlap
  probability for every pair of prefix frames.
- **Target:** the geometric overlap of each pair. For frames i and j, it is the
  fraction of frame i's valid pixels that project inside frame j with positive depth.
  The value is made symmetric by taking the minimum of the two directions
  (`compute_overlap_ratio` in `vidfm3d/utils/spatial_diag.py`).
- **Metric:** pairs with overlap >= 0.4 are positives and pairs with overlap <= 0.05
  are negatives; all other pairs are ignored (`configs/model/probe_ext.yaml`). A
  prediction above 0.5 means "overlapping". `balanced_acc` is the mean of the positive
  and negative accuracies. Higher is better, and 0.5 is chance.

### Hidden-object selection (B1 and B2)

Both B probes use one object per (scene, window), shared by all four prefixes
(`select_common_history_hidden_object_id` in `vidfm3d/utils/spatial_diag.py`; the
thresholds are `InsScene15KDataset` defaults). The object is chosen from the common
history, observations 0-7, and must meet all of these conditions:

- it is visible (at least 200 pixels) in at least 3 history observations;
- it covers at least 1,024 pixels in its best history observation, and its mask there
  stays at least 16 pixels from the image border;
- it is hidden (fewer than 200 pixels) at observations 11, 15 and 23, the tails of
  prefixes 12, 16 and 24.

Among the candidates, the selector prefers objects visible at observation 7, then
objects visible in more history frames, then larger ones.

For each prefix, a sample is scored only if the object is hidden at that prefix's tail
and visible in an earlier frame of the prefix. Selection prefers objects visible at
observation 7, which is the tail of prefix 8, so most prefix-8 samples are not scored.
Of the 276 val windows, 80 are scored at prefix 8 and 171 at prefixes 12, 16 and 24. On
the MiniMax subset the counts are 48 and 114 of 171 (caveat
`HIDDEN_OBJECT_SCORED_x_OF_y`).

The target is the object's 3D centroid in the current camera frame, expressed as
azimuth, elevation and log distance. Neither B probe receives camera pose or a flag
that marks the current frame.

### B1 (`ego_belief`), timestep line only

- **Input:** prefix features, the object's mask in each prefix frame where it is
  visible (the current frame excluded, resized to the feature grid), and the global
  feature of the final frame (`use_final_global_feature: true`).
- **Output:** a regression of azimuth, elevation and log distance.
- **Metric:** `mean_az_err_deg`, the absolute azimuth error wrapped to 180 degrees.
  Lower is better.

### B2 (`ego_belief_v2`)

- **Input:** an object query plus all patch tokens of all prefix frames. The query is
  the object's feature, mask-pooled from the prefix frame where the object is most
  visible. This is never the current frame. The probe gets no masks.
- **Output:** a class over 16 azimuth x 8 elevation bins, plus a log-distance
  regression.
- **Metric:** `mean_ang_err_deg`, the angle between the center of the predicted bin and
  the true direction. Lower is better.

### C1 action dynamics (`action_dynamics`)

- **Input:** prefix features plus a 9-D action: the relative camera pose from the
  anchor to the target frame, as a 6-D rotation and a 3-D translation. The head is
  `ActionDynamicsProbe`, a small transformer.
- **Target:** the spatial mean of the isolated feature of the target frame
  `window[p-1+h]`, read from the target pass.
- **Loss:** MSE + (1 - cosine similarity) (`configs/model/probe_ext.yaml`).
- **Metric:** `global_R@1`. Each prediction is ranked by cosine similarity against the
  targets of all evaluated records, and it is a hit when its own target ranks first.
  Higher is better. Chance is 1 / pool size: 1/828 for canonical C1 and 1/276 for
  h1-only C1. The batch-local `dyn_inbatch_R@1` logged during training is not a
  reported metric.

Horizons differ between training and evaluation (`_expand_c1_horizon_samples` in
`vidfm3d/data/components/inscene15k_dataset.py`):

| Config | `action_horizons` | Training (2,445 prefixes) | Evaluation (276 prefixes) | `n_loader` |
| --- | --- | --- | --- | ---: |
| `action_dynamics_<m>_h124_v1` (canonical) | `[1, 2, 4]` | One horizon per prefix, assigned by a fixed seeded random partition (815 prefixes per horizon), so the training set keeps its size | Every prefix is expanded to all three horizons | 828 |
| `action_dynamics_<m>_v1` (timestep line, MiniMax all-layers subset) | `[1]` | h1 | h1 | 276 (171 on the subset) |

The partition uses the dataset seed when one is set, and otherwise the torch seed set by
`seed=42`. The evaluation summary also reports per-horizon metrics (`h1_*`, `h2_*`,
`h4_*`, each ranked against its own pool of 276) and their means (`macro_horizon_*`).

### C2 path integration (`path_integration`)

- **Input:** the prefix features are mean-pooled over time and space to initialize a
  two-layer GRU. The GRU then applies three actions: anchor to h1, h1 to h2, and h2 to
  h4 (9-D each).
- **Target:** the pooled isolated features at h1, h2 and h4. Each prefix gives one
  record with three steps.
- **Metric:** `global_R@1` over all valid steps of all records, with the same ranking
  as C1. Higher is better. Secondary metrics include `final_pose_error`: the
  translation error between the pose of the retrieved target and the true final
  waypoint. Lower is better.

### Rule for C targets

C1 and C2 targets come only from `target_isolated` features. They are never sliced
from the streaming forward pass of the prefix. Streaming inputs come from the
independently forwarded `streaming_prefix` caches.

### Legacy probes

The legacy probe types `abnormal`, `counterfactual`, `sae_spatial`, `streaming_camera`
and `streaming_movement` are not part of this protocol, and no accepted row uses them.
Some of them are still wired into the modules and evaluators, but no shipped
experiment config uses them.

## 6. Controls

**C1 shortcut controls.** `vidfm3d/eval_diag.py` computes these for every C1
evaluation and scores each one with the same retrieval metrics as C1:

| Control | What it predicts | Summary key prefix |
| --- | --- | --- |
| No action | The same probe and prefix with an all-zero action | `no_action_` |
| Shuffled action | The same probe and prefix with the actions rolled by one position inside the evaluation batch | `shuffled_action_` |
| Copy last | The spatial mean of the last prefix observation's streaming feature, used directly as the prediction | `last_observation_` |
| Copy anchor (isolated) | The anchor frame's isolated feature from the target pass | `copy_anchor_isolated_` |

Reading rule (`protocols/C_PROBES.md`): a C1 row shows that the probe uses the action
only if its `global_R@1` beats the no-action, shuffled-action and copy-last controls
for the same model, layer and prefix. Report the differences, for example
`global_R@1 - no_action_global_R@1`. The evaluator does not write these R@1 gains.
It writes only the cosine gain `action_gain_cos = mean_cos - no_action_mean_cos`. The result tables carry
`no_action_global_R@1` and `shuffled_action_global_R@1` in `secondary_json`. The
copy-last values are in the evaluator summary (caveat
`REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS`).

The shuffled-action control depends on the evaluation batch size
(`batch_size_per_device_val`). With a batch of one, rolling has nothing to swap, so
the control falls back to a zero action, the same as no action.

**Other controls.**

- The A2 summary reports `trivial_neg_acc`, the accuracy of always predicting no
  overlap, together with `roc_auc` and `pr_auc`.
- For B1/B2, the loader can replace features with random ones (`scramble_feat`) or
  zero the visual and object conditions (`unconditional_baseline`). No accepted row
  uses either.
- C2 has no shortcut controls.

**C3-from-C1.** This diagnostic feeds the h1, h2 and h4 endpoints of the same
trajectory to a frozen C1 head and checks whether each prediction is closest to its
own endpoint. These endpoints are not counterfactual branches. Chance is 1/3, and the
confidence intervals resample action sets, not scenes. See
[EVALUATION.md](EVALUATION.md#8-c3-from-c1).

## 7. Training recipe

Entry point: `python vidfm3d/train.py experiment=inscene15k_streaming/<config> <overrides>`.
Each job trains one (model, probe, layer, prefix) cell, plus a timestep in the
timestep line. `scripts/probe_recipe.py` prints the exact experiment, overrides and
environment for any cell, with the batch size of the accepted runs of that cell, and
`scripts/reproduce_rows.py` does so for any accepted row.

### Settings fixed by the configs

| Setting | A1 (`streaming_depth`, `streaming_mask`) | All other probes | Defined in |
| --- | --- | --- | --- |
| Module | `VideoProbeLitModule` | `ProbeExtensionLitModule` | `configs/model/probe.yaml`, `configs/model/probe_ext.yaml` |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.95), weight decay 0.05 | AdamW, lr 3e-4, betas (0.9, 0.95), weight decay 0.05 | same |
| Schedule | Linear warmup over 10 epochs, then cosine decay to 1e-6 | Linear warmup over 5 epochs, then cosine decay to 1e-6 | same |
| `trainer.max_epochs` | 100 | 50 | experiment YAMLs |
| `trainer.precision` | `bf16-mixed` | `bf16-mixed` | experiment YAMLs |
| Devices | One GPU, `trainer.devices=[0]` | same | experiment YAMLs |
| `seed` | 42 | 42 | experiment YAMLs |
| Data loading | Train: shuffled, last partial batch dropped. Val: in order, all samples kept. | same | `vidfm3d/data/video_probe_datamodule.py` |

There is no early stopping and no checkpoint selection.

### Choosing the experiment config

`<m>` is the model key: `aether`, `cogvideox`, `dino`, `f3r`, `lingbot`,
`minimax_h3`, `videomae`, `vjepa2` or `wan`. Qwen2.5-VL (`qwen2_5_vl`) uses the
`direct_vlm_*` configs.

| Line | Probe | Models | Experiment |
| --- | --- | --- | --- |
| canonical | A1 depth | all except Qwen | `streaming_depth_<m>_v1` |
| canonical | A1 depth | Qwen | `direct_vlm_streaming_depth_v1` |
| canonical | A1 mask | cogvideox, lingbot, minimax_h3, videomae, vjepa2, wan | `streaming_mask_<m>_v1` |
| canonical | A1 mask | dino, f3r, aether | `streaming_depth_<m>_v1` + identity overrides |
| canonical | A1 mask | Qwen | `direct_vlm_streaming_depth_v1` + identity overrides |
| canonical | A2, B2, C2 | all except Qwen | `view_consistency_<m>_v1`, `ego_belief_v2_<m>_v1`, `path_integration_<m>_v1` |
| canonical | A2, B2, C2 | Qwen | `direct_vlm_view_consistency_v1`, `direct_vlm_ego_belief_v2_v1`, `direct_vlm_path_integration_v1` |
| canonical | C1 | all except Qwen and Wan | `action_dynamics_<m>_h124_v1` |
| canonical | C1 | wan | `action_dynamics_wan_v1` + `"action_horizons=[1,2,4]"`, as in the accepted Wan runs. This composes to `action_dynamics_wan_h124_v1` apart from `job_name`. |
| canonical | C1 | Qwen | `action_dynamics_qwen2_5_vl_h124_v1` |
| canonical, MiniMax all-layers subset | C1 | minimax_h3 | `action_dynamics_minimax_h3_v1` (h1 only) |
| timestep | all seven probes | cogvideox, wan, minimax_h3 | `<probe>_<m>_v1`. B1 uses `ego_belief_<m>_v1`; C1 uses `action_dynamics_<m>_v1` (h1 only). |

The identity overrides turn a depth config into the A1 mask probe. Pass them at both
training and evaluation:

```text
"model.probe.active_heads=[identity]" model.loss_weights.dmap=0.0 model.loss_weights.camera=0.0 model.loss_weights.identity=1.0
```

### Overrides for every job

| Override | Value |
| --- | --- |
| `seed` | `42` |
| `vfm_name` | Model key |
| `video_channels` | dino, f3r, videomae, vjepa2: 1024; wan: 1536; lingbot: 2048; cogvideox, aether: 3072; qwen2_5_vl: 3584; minimax_h3: 5376 |
| `feature_layer` | The disk layer. LingBot L48 is 47 and MiniMax-H3 L50 is 49. DINOv2 uses 0 and Qwen2.5-VL uses -1. The accepted LingBot L48 runs passed `feature_layer=48` and read disk layer 47 through the owner overlay (section 8), which maps 48 to 47. Without the overlay the cache file name follows `feature_layer`, so pass 47; the features are the same. |
| `feature_timestep` | 749; 250 for lingbot and minimax_h3; the swept value in the timestep line |
| `feat_postfix` | `_t<T>_layer<L>`, as in the original runs. It has no effect on which file is read, because file names follow `feature_layer` and `feature_timestep`. |
| `prefix_len` | 8, 12, 16 or 24. It must match `INSCENE_MANIFEST`. |
| `data.data_module.batch_size_per_device`, `data.data_module.batch_size_per_device_val` | `B` (see below) |
| `trainer.devices`, `trainer.num_nodes` | `[0]`, `1` |
| `autoresume` | `false` |
| `logger.wandb.offline` | `true` |
| `paths.run_folder_name` | A unique name per job (recommended) |

The default run folder is `inscene15k_streaming_<job_name>`. Without a unique
`paths.run_folder_name`, all prefixes and layers of one experiment share a folder, and
each evaluation overwrites the previous one's summary.

### Batch size

- **Canonical:** `B = 2` for training and validation, with these exceptions. Wan A1
  depth, A1 mask and B2 used 1. Wan C1 and C2 used 4. MiniMax-H3 layer 30 A1 depth and
  A1 mask used 1. The exceptions equal the experiment-config defaults. Wan A2 has no
  released rows; the printer gives it the Wan config value 4. In total, 656
  accepted rows used 2, 92 used 1 and 56 used 4 (caveat
  `TRAIN_BATCH_SIZE_n_NOT_CANONICAL_2`).
- **Timestep line:** 1 for A1 and 2 for all other probes.

`scripts/probe_recipe.py` uses these accepted values unless `--batch-size` is given,
and then notes the accepted value. `scripts/reproduce_rows.py` uses each row's
recorded `train_batch_size_inferred`.

Every result row records `train_batch_size_inferred`, which satisfies
`floor(N_train / B) * (ckpt_epoch + 1) == ckpt_step`. `N_train` is 2,445, or 1,523 on
the MiniMax subset. For example, `B = 2` gives 1,222 optimizer steps per epoch.

### Environment

| Variable | Needed for | Value |
| --- | --- | --- |
| `INSCENE_DATA_ROOT`, `INSCENE_LOCAL_SCENE_ROOT` | all probes | Dataset paths (README) |
| `INSCENE_MANIFEST`, `INSCENE_TRUST_EXPANDED_MANIFEST` | all probes | Section 3 |
| `INSCENE_STREAMING_FEAT_ROOT` | all probes | Root of the streaming namespace |
| `INSCENE_TARGET_FEAT_ROOT`, `INSCENE_TARGET_LAYOUT=flat` | C1, C2 and C3-from-C1 evaluation | Root of the target namespace |
| `PROBE_FP16_PASSTHROUGH` | all probes (part of the recipe) | `0` (the loader default) casts the cached features to fp32 in the data workers. `1` keeps them in their storage dtype (fp16, or bf16 for MiniMax-H3) into the probe during training: nothing widens them later. Both evaluators cast batch tensors to fp32 on the device (`vidfm3d/eval_diag.py`, `vidfm3d/eval_a1_streaming.py`), so at evaluation only values the loader derives differ. Values the loader derives from them, such as the B2 query token (a masked mean), are computed in that dtype too (`_feat_cast` in `vidfm3d/data/components/inscene15k_dataset.py`). Under `bf16-mixed` the two settings therefore give different values, and the setting changed training outcomes: the B2 heads that ended with non-finite weights had been trained with `1`, and their retrains with `0` are the accepted rows (caveat `FP32_RETRAIN_REPLACES_NONFINITE_CHECKPOINT`, 16 canonical and 32 timestep rows). The accepted runs used `0` for MiniMax-H3 on the canonical line, for the canonical CogVideoX layer 30/35/40 B2 runs and for every fp32 B2 retrain, and `1` for the other timestep runs. For the other canonical runs `1` is inferred, not recorded. Use the value that `scripts/probe_recipe.py` prints for each cell. |
| `INSCENE_PROBE_TASK` | optional; the recipe sets it | The probe's result name, for example `streaming_mask` or `ego_belief_v2`. It selects a lean loading path that reads only what that probe needs. When it is unset, the loader keeps the inputs of every probe, for example those of both B1 and B2. When it is set, it must name the probe actually trained: an A1 mask job run through a depth config needs `streaming_mask`, not `streaming_depth`. `scripts/probe_recipe.py` sets it. |

### Example commands

Each example sets the environment that `scripts/probe_recipe.py` prints for its cell.

```bash
# Canonical C1, DINOv2, prefix 12
export INSCENE_MANIFEST=$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix12.jsonl
export INSCENE_PROBE_TASK=action_dynamics PROBE_FP16_PASSTHROUGH=1
python vidfm3d/train.py experiment=inscene15k_streaming/action_dynamics_dino_h124_v1 \
  seed=42 vfm_name=dino video_channels=1024 feature_layer=0 feature_timestep=749 \
  feat_postfix=_t749_layer0 prefix_len=12 \
  data.data_module.batch_size_per_device=2 data.data_module.batch_size_per_device_val=2 \
  "trainer.devices=[0]" trainer.num_nodes=1 autoresume=false logger.wandb.offline=true \
  paths.run_folder_name=c1_dino_l0_p12

# Canonical A1 mask, Aether layer 1, prefix 8 (depth config + identity overrides)
export INSCENE_MANIFEST=$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix8.jsonl
export INSCENE_PROBE_TASK=streaming_mask PROBE_FP16_PASSTHROUGH=1
python vidfm3d/train.py experiment=inscene15k_streaming/streaming_depth_aether_v1 \
  seed=42 vfm_name=aether video_channels=3072 feature_layer=1 feature_timestep=749 \
  feat_postfix=_t749_layer1 prefix_len=8 \
  "model.probe.active_heads=[identity]" model.loss_weights.dmap=0.0 \
  model.loss_weights.camera=0.0 model.loss_weights.identity=1.0 \
  data.data_module.batch_size_per_device=2 data.data_module.batch_size_per_device_val=2 \
  "trainer.devices=[0]" trainer.num_nodes=1 autoresume=false logger.wandb.offline=true \
  paths.run_folder_name=mask_aether_l1_p8

# Timestep B1, Wan t999 layer 20, prefix 16
export INSCENE_MANIFEST=$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix16.jsonl
export INSCENE_PROBE_TASK=ego_belief PROBE_FP16_PASSTHROUGH=1
python vidfm3d/train.py experiment=inscene15k_streaming/ego_belief_wan_v1 \
  seed=42 vfm_name=wan video_channels=1536 feature_layer=20 feature_timestep=999 \
  feat_postfix=_t999_layer20 prefix_len=16 \
  data.data_module.batch_size_per_device=2 data.data_module.batch_size_per_device_val=2 \
  "trainer.devices=[0]" trainer.num_nodes=1 autoresume=false logger.wandb.offline=true \
  paths.run_folder_name=b1_wan_t999_l20_p16
```

Evaluation commands are in [EVALUATION.md](EVALUATION.md#3-commands).

### Outputs and checkpoints

- **Run folder:** `logs/inscene15k_streaming/runs/<paths.run_folder_name>/` under the
  repository root (`configs/hydra/default.yaml`, `configs/paths/default.yaml`).
  `train.py` writes `run_manifest.json` there.
- **Checkpoint:** `<run>/VideoProbe3D/<wandb-run-id>/checkpoints/epoch=<E>-step=<S>.ckpt`.
  This is Lightning's default checkpointing, which keeps only the most recent epoch.
- **Why there is no `last.ckpt`:** every streaming experiment uses
  `configs/callbacks/v3_checkpoint.yaml`, which has no `_target_`, so
  `vidfm3d/utils/instantiators.py` skips it. As a result, there are no
  every-5-epochs checkpoints and no `last.ckpt`. `autoresume` looks for
  `<run>/checkpoints/last.ckpt` and therefore never resumes. The accepted runs behaved
  the same way.
- **Evaluated checkpoint:** the final one, with E = 99 for A1 and E = 49 for all other
  probes.

## 8. Owner overlay and owner catalog (optional)

`PROBE_OWNER_OVERLAY_MANIFEST` (read in `InsScene15KDataset.__init__`; implemented in
`vidfm3d/utils/probe_owner_overlay.py`) records how the original canonical runs
selected cached features. When it is set:

- the loader verifies a READY overlay manifest and two SQLite databases bound by
  SHA-256;
- it resolves every streaming feature by (vfm, source, scene, layer, timestep, window,
  prefix) and every target feature by (vfm, source, scene, layer, timestep, target
  frame) to the exact cached file that the original runs read;
- it skips the `prefix_index.npy` filter.

`PROBE_OWNER_CANONICAL_PREFIX` and `PROBE_OWNER_LOCAL_MIRROR_ROOT` can map the
overlay's storage URIs to a local mirror. While the prefix is empty, which is the
default, no mapping happens.

`INSCENE_OWNER_CATALOG` (or the dataset argument `owner_catalog`; implemented in
`vidfm3d/utils/owner_catalog.py`) is a second, separate owner source. When it is set,
the loader accepts only a READY catalog whose route file still matches its recorded
SHA-256, and resolves each streaming feature through the catalog instead of
`INSCENE_STREAMING_FEAT_ROOT`. Samples without a catalog entry are dropped when the
prefix-index filter runs, and an entry whose feature file is missing is an error. The
catalog cannot be combined with `PROBE_OWNER_OVERLAY_MANIFEST`.

The overlay manifest, its databases and the owner catalog are not published, and the
protocol does not need them. Leave these variables unset. The loader then reads
features directly from `INSCENE_STREAMING_FEAT_ROOT` and `INSCENE_TARGET_FEAT_ROOT`.

## 9. What a rerun reproduces

Following this protocol reproduces the experiment, but not the exact bits of the
accepted numbers, for these reasons:

- The accepted numbers came from frozen code snapshots that are named in the
  `training_code_tag` and `eval_pack` columns and are not published (caveat
  `TRAINING_AND_EVAL_PACK_COMMITS_NOT_ON_GITHUB`).
- The C3 rows were evaluated with a development commit of `vidfm3d/eval_diag.py`
  (named in `eval_pack`). On the C3 path the shipped evaluator differs from it only in
  non-finite handling, which does not change any accepted row.
- The canonical features were read through the owner overlay (section 8). The
  accepted timestep MiniMax-H3 layer-15 rows read fp16 caches that the shipped
  extractor does not write (section 4).
- The dataset revision is not pinned.
- Some GPU kernels are not deterministic. For example, the identity loss sums with
  `scatter_add_`.

To compare a rerun with the accepted tables, use
`python scripts/export_results.py --runs-root <dir> --compare results/`.
