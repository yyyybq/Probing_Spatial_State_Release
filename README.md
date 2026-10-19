# Probing Spatial State

Lightweight diagnostic probes trained on **frozen** video and vision foundation-model
features. They measure how much spatial state those features carry about a camera
moving through a 3D scene: depth and camera pose, object identity, view overlap,
the direction of objects that are currently out of view, and the effect of camera
actions. All probes run on streaming video windows from the ScanNet++ part of
[InsScene-15K](https://huggingface.co/datasets/lifuguan/InsScene-15K).

The code is derived from **VidFM3D**, *How Much 3D Do Video Foundation Models Encode?*
([arXiv:2512.19949](https://arxiv.org/abs/2512.19949),
[code](https://github.com/zxhuang1698/VidFM3D)) by Zixuan Huang, Xiang Li, Zhaoyang Lv
and James M. Rehg. Please credit and cite their work (see [Citation](#citation)).

This repository contains three experimental lines and nothing else:

- **Canonical V2**: six probes on ten frozen feature models at prefixes of 8, 12, 16
  and 24 observations (V2 is the version label of the canonical window protocol);
- **Timestep sweep**: the same probes plus B1 at diffusion timesteps 499 and 999 (500
  and 1000 for MiniMax-H3) of CogVideoX, Wan and MiniMax-H3, partly at other layers
  than the canonical line;
- **C3-from-C1**: an evaluation of trained C1 heads under alternative actions.

The accepted validation results of all three are in [`results/`](results/README.md).

## Contents

- [What is measured](#what-is-measured)
- [Feature models](#feature-models)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data and split](#data-and-split)
- [Feature extraction](#feature-extraction)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing the published tables](#reproducing-the-published-tables)
- [Results](#results)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Changes relative to VidFM3D](#changes-relative-to-vidfm3d)
- [License](#license)
- [Citation](#citation)

## What is measured

Every probe is a small trainable head on top of cached, frozen features of a
**prefix**: the first *p* observations of a 28-observation window, with
*p* ∈ {8, 12, 16, 24}. The full definitions are in [docs/PROTOCOL.md](docs/PROTOCOL.md).

| ID | Probe key | What the head predicts | Primary metric | Better |
| --- | --- | --- | --- | --- |
| A1 | `streaming_depth` | Depth and camera of the prefix frames | `mean_loss_depth` | lower |
| A1 | `streaming_mask` | Instance identity masks of the prefix frames | `mean_loss_identity` | lower |
| A2 | `view_consistency` | Which prefix views overlap | `balanced_acc` | higher |
| B1 | `ego_belief` | Direction of an object that is hidden at the last frame, given its mask (timestep line only) | `mean_az_err_deg` | lower |
| B2 | `ego_belief_v2` | Direction of a hidden object, given an object query | `mean_ang_err_deg` | lower |
| C1 | `action_dynamics` | Feature of a future frame, given the prefix and a camera action | `global_R@1` | higher |
| C2 | `path_integration` | Features of future frames along the camera path | `global_R@1` | higher |
| C3 | `c3_from_c1` (evaluation only) | Whether a trained C1 head picks the right future among the h1/h2/h4 endpoints | `intervention_validity` (chance 1/3) | higher |

C1 must be read against its controls: a zero action, a shuffled action and a copy of
the last observation. C2 has no shortcut controls. C3-from-C1 trains nothing. It
scores frozen C1 checkpoints on three endpoints of the **same** trajectory, so it
measures action sensitivity and is not a counterfactual test. See
[docs/EVALUATION.md](docs/EVALUATION.md) for all metrics, controls and the acceptance
rule. Older probe types from the development history (`abnormal`, `counterfactual`,
`sae_spatial`, `streaming_camera`, `streaming_movement`) are not part of this
protocol. Some of them are still wired into the code, but no shipped experiment config
uses them.

## Feature models

| Key | Extractor | Weights (downloaded by you) | Layers in the canonical tables |
| --- | --- | --- | --- |
| `dino` | `features/dino` | `facebook/dinov2-large` | 0 (last hidden state) |
| `vjepa2` | `features/vjepa2` | `facebook/vjepa2-vitl-fpc64-256` | 5, 11, 17, 23 |
| `videomae` | `features/run_inscene15k.py` | `MCG-NJU/videomae-large` | 5, 11, 17, 23 |
| `f3r` | `features/f3r` (vendored Fast3R) | `jedyang97/Fast3R_ViT_Large_512` | 24 |
| `qwen2_5_vl` | `features/run_inscene15k_mllm.py` | `Qwen/Qwen2.5-VL-7B-Instruct` | -1 (visual merger output) |
| `wan` | `features/wan` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 0, 5, 10, 15, 20, 25, 29 |
| `cogvideox` | `features/cogvideox` | `THUDM/CogVideoX-5b-I2V` | 0, 5, 10, 30, 35, 40 |
| `aether` | `features/aether` | `AetherWorldModel/AetherV1` plus CogVideoX-5b-I2V components | 1 |
| `minimax_h3` | `features/minimax_h3` | `MiniMaxAI/MiniMax-H3`, local path in `MINIMAX_H3_MODEL_PATH` | 10, 20, 30, 40, 50 |
| `lingbot` | `features/lingbot` | `robbyant/lingbot-video-moe-30b-a3b` in `LINGBOT_MODEL_PATH`, plus a clone of [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video) in `LINGBOT_ROOT` | 10, 20, 30, 40, 48 |

- **Timesteps.** Canonical features are taken at scheduler timestep 749 for Wan,
  CogVideoX and Aether, and at 250 for MiniMax-H3 and LingBot. For the encoders and
  Qwen2.5-VL the value 749 in configs and result keys is only a placeholder.
- **Last-block layer names.** The tables call the last LingBot block L48 and the last
  MiniMax-H3 block L50. On disk, and for `feature_layer`, these are layers 47 and 49.
- **Licenses.** Each set of weights has its own license, several of them
  non-commercial. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Repository layout

| Path | Content |
| --- | --- |
| `vidfm3d/` | Dataset, probe heads and Lightning modules, plus the entry points `train.py`, `eval_diag.py` and `eval_a1_streaming.py` |
| `features/` | Feature extractors: `run_inscene15k.py` for all video and vision models, `run_inscene15k_mllm.py` for Qwen2.5-VL |
| `configs/` | Hydra configs. Experiments are `configs/experiment/inscene15k_streaming/<probe>_<model>_v1.yaml`. The base data config is named `data/dl3dv.yaml` for historical reasons. |
| `data/` | `download_inscene15k.py` and the shipped split in `data/splits/canonical_v2/` |
| `protocols/` | Machine-readable window protocol and the C-probe notes |
| `scripts/` | Manifest builder, recipe printer, row reproducer, results exporter, sweep wrapper and checks |
| `docs/` | [PROTOCOL.md](docs/PROTOCOL.md) and [EVALUATION.md](docs/EVALUATION.md) |
| `results/` | Accepted validation results and their caveats |
| `tests/` | CPU-only test suite |

## Installation

Run every command in this README from the repository root. The extractors in
`features/` are not installed as a package, and the commands use paths relative to
the root.

**GPU environment** for extraction, training and evaluation, on Linux x86_64 with an
NVIDIA driver that supports CUDA 12.4. `environment.yml` creates a Python 3.11
environment from conda-forge. pip then installs PyTorch 2.6.0 and torchvision 0.21.0
from the PyTorch CUDA 12.4 wheel index (`--extra-index-url
https://download.pytorch.org/whl/cu124`), the pinned packages in
`requirements-lock.txt` and this repository in editable mode. Check that PyTorch sees
the GPU before you extract or train:

```bash
conda env create -f environment.yml
conda activate probe-spatial
python -c "import torch; assert torch.cuda.is_available()"
```

**CPU environment** for the tests, the same steps as CI:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

Optional packages, imported only when needed:

- `pytorch3d` and `trimesh` for point-cloud metrics;
- `filterpy` for Aether camera smoothing;
- an `aoss_client` installation if feature roots live in an S3-style object store.
  Set `AOSS_CONF`, and optionally `AOSS_CLUSTER` and `AOSS_CACHE_DIR`. Local paths need
  none of this.

Training logs to Weights & Biases. Pass `logger.wandb.offline=true` or set
`WANDB_MODE=offline` to keep everything local. Machine-specific Hydra settings can go
in `configs/local/default.yaml`, which `configs/train.yaml` loads when it exists and
git ignores.

**MiniMax-H3 needs a special diffusers build.** Its extractor imports
`diffusers.modular_pipelines.minimax_h3`, which public diffusers releases do not
provide. See [Known limitations](#known-limitations).

**Resources.** No trained probe checkpoints and no feature caches are published.
`checkpoint_sha256` in the result tables identifies a checkpoint but does not let you
download it, so reproducing any row, C3 included, means extracting the features and
training the probe yourself. Each training and evaluation job runs on one GPU. Feature extraction for CogVideoX-5B,
Aether (which also loads CogVideoX-5B components), MiniMax-H3 and the LingBot 30B MoE
needs large-memory GPUs. The feature caches grow with the channel count of the model
(from 1,024 for the encoders to 5,376 for MiniMax-H3) and with every layer and
timestep you extract.

## Data and split

**1. Download the ScanNet++ part of InsScene-15K.** The InsScene-15K and ScanNet++
terms apply.

```bash
python data/download_inscene15k.py --base-dir <BASE>      # download + extract ScanNet++
export INSCENE_DATA_ROOT=<BASE>/data
export INSCENE_LOCAL_SCENE_ROOT=$INSCENE_DATA_ROOT/processed_scannetpp_v2/processed_scannetpp_v2
```

**2. Build the canonical split manifests** from the shipped public split file:

```bash
python scripts/build_tstep_canonical_manifests.py build --output-dir <MANIFESTS>
python scripts/build_tstep_canonical_manifests.py audit --manifest-dir <MANIFESTS>
export CANONICAL_MANIFEST_DIR=<MANIFESTS>
export INSCENE_TRUST_EXPANDED_MANIFEST=1
export INSCENE_TARGET_LAYOUT=flat
```

The builder reads `data/splits/canonical_v2/inscene_c_h124_prefix24.public.jsonl.gz`.
It writes `inscene_c_h124_prefix{8,12,16,24}.jsonl` and fails unless the window set
matches the published one:

| | Windows | Scenes |
| --- | ---: | ---: |
| Train | 2,445 | 612 |
| Val | 276 | 69 |
| Total | 2,721 | 681 |

The window-set fingerprint (`universe_sha256`) is
`4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b`. The rule behind
the split is in [`data/splits/canonical_v2/SPLIT.json`](data/splits/canonical_v2/SPLIT.json).
There is **no held-out test split**.

**Every training and evaluation job must set `INSCENE_MANIFEST`** to the manifest of
its prefix, for example `$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix12.jsonl`.
Without it, the dataset class falls back to a runtime split that differs from the
canonical one.

## Feature extraction

Features are extracted once per model and cached as `.sft` files with JSON sidecars.
Each model needs two passes, and `FEATURE_NAMESPACE` must equal the basename of
`--out-root`:

```bash
# Streaming inputs: every prefix in its own forward pass (all probes)
FEATURE_NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1 \
python -m features.run_inscene15k --vfm wan --mode streaming_prefix \
    --data-root $INSCENE_DATA_ROOT --source scannetpp \
    --out-root $FEAT_BASE/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1 \
    --prefix-lengths 8,12,16,24 --t 749 --output-layers 20

# Isolated future targets (C1 and C2 only)
FEATURE_NAMESPACE=FEAT_TARGET_CANONICAL_V2_SEEDED_V1 \
python -m features.run_inscene15k --vfm wan --mode target_isolated \
    --target-from-streaming-windows --target-prefix-lengths 8,12,16,24 \
    --target-horizons 1,2,4 --num-targets 0 \
    --data-root $INSCENE_DATA_ROOT --source scannetpp \
    --out-root $FEAT_BASE/FEAT_TARGET_CANONICAL_V2_SEEDED_V1 \
    --t 749 --output-layers 20

export INSCENE_STREAMING_FEAT_ROOT=$FEAT_BASE/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1
export INSCENE_TARGET_FEAT_ROOT=$FEAT_BASE/FEAT_TARGET_CANONICAL_V2_SEEDED_V1
```

- Use `--vfm` with any key from [Feature models](#feature-models), except Qwen2.5-VL,
  which uses `python -m features.run_inscene15k_mllm` with the same namespaces.
- Use the timestep of the model (`--t`) and its layers (`--output-layers`), with
  disk layers 47 and 49 for the last LingBot and MiniMax-H3 blocks.
- All models share the two namespace roots. The extractor and the loader both append
  `<vfm>/<source>/<scene>`.
- Noise is seeded per sample, so extraction is deterministic. Features are stored in
  fp16, and MiniMax-H3 features in bf16.
- `python scripts/validate_feature_cache.py <root>` checks the sidecars.

## Training

One job trains one (model, probe, layer, prefix) cell; timestep runs also fix the
timestep. `scripts/probe_recipe.py` prints the exact experiment, overrides and
environment of any cell:

```bash
python scripts/probe_recipe.py --line canonical --model dino --probe ego_belief_v2 --layer 0 --prefix 12
```

Its header also names the run group whose recipe rules it applies, for example
`canonical_main`. When the published rows of a cell used another split, a `# note:`
line says so. [results/README.md](results/README.md#2-columns) explains the run
groups and the `scope` labels in the tables.

A canonical example, written out with the environment that the printer gives for this
cell:

```bash
export INSCENE_MANIFEST=$CANONICAL_MANIFEST_DIR/inscene_c_h124_prefix12.jsonl
export INSCENE_PROBE_TASK=ego_belief_v2
export PROBE_FP16_PASSTHROUGH=1
python vidfm3d/train.py experiment=inscene15k_streaming/ego_belief_v2_dino_v1 \
    seed=42 vfm_name=dino video_channels=1024 feature_layer=0 feature_timestep=749 \
    feat_postfix=_t749_layer0 prefix_len=12 \
    data.data_module.batch_size_per_device=2 data.data_module.batch_size_per_device_val=2 \
    "trainer.devices=[0]" trainer.num_nodes=1 autoresume=false logger.wandb.offline=true \
    paths.run_folder_name=b2_dino_l0_p12
```

Rules that the recipe printer applies:

- **C1.** Canonical C1 uses the `action_dynamics_<model>_h124_v1` configs, which train
  on horizons 1, 2 and 4. For Wan the printer uses `action_dynamics_wan_v1` plus
  `"action_horizons=[1,2,4]"`, as the accepted Wan runs did; this composes to the same
  config as `action_dynamics_wan_h124_v1` apart from `job_name`. The timestep line and
  the MiniMax all-layers subset use the horizon-1 `action_dynamics_<model>_v1` configs.
- **A1 mask.** DINOv2, Fast3R, Aether and Qwen2.5-VL have no `streaming_mask` config.
  They use their `streaming_depth` config plus four identity-head overrides.
- **Batch size.** Without `--batch-size`, the printer uses the batch size of the
  accepted runs of the cell. On the canonical line this is the protocol value 2, except
  Wan A1 and B2 at 1, Wan C1 and C2 at 4, and MiniMax-H3 layer 30 A1 at 1 on the public
  split. Wan A2, which has no released rows, gets its config value 4. The timestep line
  uses 1 for A1 and 2 otherwise. `--batch-size` overrides the value, and the printer
  then notes the accepted one.
- **MiniMax-H3 all-layers rows.** The published MiniMax-H3 rows at layers 10, 20, 40
  and 50, and C1/C2 at layer 30, were trained and evaluated on a non-public subset of
  the split, with C1 on horizon 1 only. For these cells the printer gives the
  public-split recipe with a `# note:`. `--split minimax_subset` prints the published
  recipe, which needs `MINIMAX_SUBSET_MANIFEST_DIR`.
- **Environment.** `INSCENE_PROBE_TASK` names the trained probe, and
  `PROBE_FP16_PASSTHROUGH` is set to the value of the accepted runs. Both are part of
  the recipe. With `PROBE_FP16_PASSTHROUGH=1` the features reach the probe in their
  fp16 or bf16 storage dtype during training, and this changed training outcomes (see
  [docs/PROTOCOL.md](docs/PROTOCOL.md#environment)). Use the values that the printer
  gives.
- **Epochs.** A1 trains for 100 epochs and every other probe for 50. Training uses
  bf16 mixed precision on one GPU with seed 42.
- **Run folder.** Give every job a unique `paths.run_folder_name`. Otherwise
  evaluations of one experiment overwrite each other's summaries.

Runs land in `logs/inscene15k_streaming/runs/<run folder>/`. The checkpoint is
`VideoProbe3D/<run id>/checkpoints/epoch=<E>-step=<S>.ckpt`. Only the final epoch is
kept, and no `last.ckpt` is written, as in the original runs.
[docs/PROTOCOL.md](docs/PROTOCOL.md#7-training-recipe) has the full recipe.

## Evaluation

Evaluation always uses the validation split and the final checkpoint, with the same
experiment, overrides and environment as training. Quote the checkpoint path as shown,
because it contains `=`. Pass `hydra.output_subdir=null` to every evaluation, because
an evaluation that reuses the training run folder otherwise overwrites the training
run's Hydra record `.hydra/config.yaml` with its own settings, for example the
`action_horizons=[1,2,4]` of a C3 call.

```bash
# A1 (streaming_depth, streaming_mask)
python vidfm3d/eval_a1_streaming.py <same experiment and overrides as training> \
    "ckpt_path='<run>/VideoProbe3D/<id>/checkpoints/epoch=99-step=<S>.ckpt'" +eval_split=val \
    hydra.output_subdir=null

# A2, B1, B2, C1, C2
python vidfm3d/eval_diag.py <same experiment and overrides as training> \
    "ckpt_path='<checkpoint>'" +eval_split=val train=false test=false hydra.output_subdir=null

# C3-from-C1, on a C1 checkpoint
python vidfm3d/eval_diag.py <same experiment and overrides as the C1 run> \
    "ckpt_path='<C1 checkpoint>'" +eval_split=val train=false test=false \
    ++evaluation_protocol=c3_action_sensitivity hydra.output_subdir=null
```

For a C1 head trained on horizon 1 only, add `"action_horizons=[1,2,4]"` to the C3
command. Summaries are written to `<run>/eval/val_summary.json`, and C3 to
`<run>/eval/val_c3_action_sensitivity_summary.json`. `probe_recipe.py --stage eval`
and `--stage c3` print these commands. [docs/EVALUATION.md](docs/EVALUATION.md)
describes the outputs, metrics and acceptance rule.

## Reproducing the published tables

`scripts/reproduce_rows.py` turns any selection of published rows into the
training, evaluation and C3 commands that produced them, using each row's recorded
batch size:

```bash
python scripts/reproduce_rows.py --line canonical --model dino > dino.sh   # print only
python scripts/reproduce_rows.py --line canonical --model dino --run        # execute
```

`--run` needs `INSCENE_DATA_ROOT`, `INSCENE_STREAMING_FEAT_ROOT` and
`CANONICAL_MANIFEST_DIR`, plus `INSCENE_TARGET_FEAT_ROOT` when C1, C2 or C3 rows are
selected. With `--skip-existing`, training is skipped only when the newest checkpoint
is the final epoch. Afterwards, export your evaluation summaries into the published schema and
compare them with the tables:

```bash
python scripts/export_results.py --runs-root logs/inscene15k_streaming/runs \
    --out results/local --compare results/
```

`results/local/` is ignored by git, so local exports stay separate from the published
tables.

`scripts/run_streaming_probe_sweep.sh` runs extraction, training and evaluation for
whole models in one go, with the recipe printer's settings for every cell; its header
lists the environment variables it reads.

A rerun reproduces the protocol, not the exact bits. The reasons are listed under
[Known limitations](#known-limitations).

## Results

| File | Rows | Content |
| --- | ---: | --- |
| [`results/canonical_v2_val_seed42.csv`](results/canonical_v2_val_seed42.csv) | 804 | Canonical V2, 10 models |
| [`results/timestep_val_seed42.csv`](results/timestep_val_seed42.csv) | 316 | Timestep sweep, CogVideoX, Wan, MiniMax-H3 |
| [`results/c3_from_c1_val_seed42.csv`](results/c3_from_c1_val_seed42.csv) | 180 | C3-from-C1: 120 protocol rows and 60 diagnostic rows on horizon-1 C1 heads |

A row is accepted when a non-finite-aware evaluator produced it, its primary metric
is finite, no prediction was non-finite and at least one record was scored. All rows
are validation results for seed 42 from final-epoch weights, with no checkpoint
selection.

Read the caveats before comparing rows. They are listed per row in the `caveats`
column and explained in [results/README.md](results/README.md):

- batch sizes differ between models;
- the MiniMax-H3 all-layers runs use a 171-window validation subset that is not
  comparable with the 276-window split;
- B1 and B2 score only windows with a hidden object;
- Wan has no accepted A2 rows, and Wan timestep 499 has only A1 rows;
- MiniMax-H3 timestep layers 20 and 25 are missing;
- timestep values are not comparable across models;
- timestep rows are not extra points of the canonical rows: most sweep layers have no
  canonical row, and for Wan the two lines use different batch sizes and C1 horizons;
- C1 rows must be read against their controls, and C2 has no controls;
- C3 margins are close to zero, and the 60 C3 rows on horizon-1 C1 heads are
  diagnostics, not protocol C3.

## Tests

```bash
python -m pytest -q
```

The suite runs on CPU and needs no data, features or weights. It covers the probe
contracts, the evaluators including C3, non-finite handling, the split and manifest
builder, the entry points, the recipe printer, the results tables and a scan for
private paths. `pytorch3d` is optional, and one test is skipped without it.
[tests/README.md](tests/README.md) lists the files.

## Known limitations

- **Code identity.** The accepted numbers came from frozen code snapshots named in the
  `training_code_tag` and `eval_pack` columns. These snapshots are not published, so a
  rerun reproduces the protocol, not bit-identical numbers. The C3 rows were evaluated
  with a development commit of `vidfm3d/eval_diag.py`. On the C3 path the shipped
  evaluator differs from it only in non-finite handling, which does not change any
  accepted row (see [results/README.md](results/README.md#5-caveat-codes)).
- **Feature provenance.** The original canonical runs read cached features through an
  internal index (`PROBE_OWNER_OVERLAY_MANIFEST`). It is optional and not published.
  Fresh extraction into the two seeded namespaces follows the same protocol.
- **MiniMax-H3.** The extractor needs `diffusers.modular_pipelines.minimax_h3`, which
  public diffusers releases do not include. The membership of the MiniMax all-layers
  subset (1,523 train and 171 val windows) is not published either. Its 96 canonical
  and 16 C3 rows therefore cannot be re-run from this repository, and
  `scripts/reproduce_rows.py` refuses them unless `MINIMAX_SUBSET_MANIFEST_DIR` is set.
  The accepted timestep MiniMax-H3 layer-15 rows read fp16 caches written before
  MiniMax-H3 features were stored in bf16. The shipped extractor writes bf16, so a
  re-extracted cache is not the one those runs read.
- **LingBot.** The source commit of `Robbyant/lingbot-video` used for extraction is not
  recorded.
- **Dataset revision.** The InsScene-15K revision is not pinned.
  `data/download_inscene15k.py --revision` accepts one. A different revision can change
  frame lists and break the shipped split.
- **Nondeterminism.** Some GPU kernels, such as the `scatter_add_` in the identity
  loss, are not deterministic.

## Changes relative to VidFM3D

This repository keeps the VidFM3D probe framework and adds:

- the InsScene-15K streaming dataset with the canonical windows, split and targets;
- new probes (A1 depth and mask on streaming prefixes, A2, B1, B2, C1, C2), their
  Lightning modules, evaluators and the C3-from-C1 evaluation;
- extractors for V-JEPA2, VideoMAE, Qwen2.5-VL, MiniMax-H3 and LingBot, plus streaming
  and target modes for all extractors, with deterministic per-sample noise;
- reproducibility tooling: the shipped split, manifest builder, recipe printer,
  results exporter and tests.

The original CO3D and DL3DV pipelines were removed. Some optional internal provenance
tooling remains: the gap-fill worklist hooks of the extractors, and the owner overlay
and owner catalog of the loader. It is off by default, its inputs are not published,
and it is not needed to reproduce the results ([docs/PROTOCOL.md](docs/PROTOCOL.md),
sections 4 and 8).

## License

The repository is released under **CC BY-NC 4.0**, the license of VidFM3D, from which
it is derived (see [LICENSE](LICENSE)). Vendored and adapted third-party code keeps
its own license. The table is a summary; [THIRD_PARTY.md](THIRD_PARTY.md) is the
complete list:

| Component | License |
| --- | --- |
| Fast3R | FAIR Noncommercial Research License |
| DUSt3R and CroCo | CC BY-NC-SA 4.0; ShareAlike applies to the adapted DUSt3R and CroCo files |
| VGGT, including the VGGT-derived probe components `vidfm3d/models/components/{dpt_head,backbone_pixalign,probe_pixalign,movement_head}.py` | VGGT License v1 |
| DINOv2 layers | Apache-2.0 |
| Aether | MIT |
| Code adapted from diffusers | Apache-2.0 |
| PyTorch3D (adapted helpers) | BSD-3-Clause |
| VGGT evaluation and PoseDiffusion pose-error helpers (`vidfm3d/utils/metrics.py`) | CC BY-NC 4.0 |
| lightning-hydra-template | MIT |

Model weights, InsScene-15K and ScanNet++ follow their own terms. The repository as a
whole is for non-commercial research use. License files, copyright holders and
modification notes are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

If you use this code, please cite VidFM3D, on which it builds:

```bibtex
@article{huang2025vidfm3d,
  title   = {How Much 3D Do Video Foundation Models Encode?},
  author  = {Huang, Zixuan and Li, Xiang and Lv, Zhaoyang and Rehg, James M.},
  journal = {arXiv preprint arXiv:2512.19949},
  year    = {2025}
}
```

If you use the split or the result tables, please also cite the datasets as their
pages request: [InsScene-15K](https://huggingface.co/datasets/lifuguan/InsScene-15K)
and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/).

We also thank the authors of Fast3R, VGGT, DUSt3R and CroCo, DINOv2, Aether,
InsScene-15K and ScanNet++ for releasing their code and data.
