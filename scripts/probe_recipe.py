#!/usr/bin/env python3
"""Training and evaluation recipe for one probe cell.

This module is the single source of the mapping

    (line, model, probe, layer, prefix[, timestep, batch size, split])
        -> (experiment config, Hydra overrides, environment, evaluator)

used by the accepted rows in ``results/``.  ``scripts/reproduce_rows.py`` and
``scripts/run_streaming_probe_sweep.sh`` both build their commands from it.
It needs only the Python standard library.

Lines
-----
``canonical``  canonical-V2 matrix (one fixed scheduler timestep per model).
``timestep``   the timestep sweep for Wan, CogVideoX and MiniMax-H3.

Rules encoded here (derived from the campaign tooling that trained the accepted
runs; see ``docs`` for the protocol):

* Overrides: ``seed vfm_name video_channels feature_layer feature_timestep
  feat_postfix prefix_len`` plus both data-module batch sizes and
  ``trainer.devices=[0]`` (``trainer.num_nodes=1`` on the canonical line).
* Table layers (as in results/*.csv) ``lingbot 48`` and ``minimax_h3 50`` name the last block by its
  1-based count; the cached files are disk layers 47 and 49, so
  ``feature_layer`` and ``feat_postfix`` use the disk layer.  (The accepted
  LingBot layer-48 runs passed ``feature_layer=48`` and read disk layer 47
  through the optional owner overlay, which does this mapping itself; the
  features are the same.)
* C1 (``action_dynamics``) on the canonical line trains on horizons 1/2/4:
  ``action_dynamics_<model>_h124_v1`` plus ``action_horizons=[1,2,4]``.  The Wan
  runs used ``action_dynamics_wan_v1`` plus the same override, which composes to
  the same configuration apart from ``job_name``.  The timestep line and the
  MiniMax all-layers subset trained C1 on h1 only (``*_v1`` configs).
* ``streaming_mask`` for DINO, Fast3R and Aether uses the
  ``streaming_depth_<model>_v1`` config with the identity-head overrides; Qwen
  uses ``direct_vlm_streaming_depth_v1`` with the same overrides.  The Wan runs
  carried the same four overrides on top of ``streaming_mask_wan_v1``, where they
  restate the config's own values.
* MiniMax-H3 layer 30 A1/A2/B2 on the public split used the config batch sizes
  (1 for A1, 2 otherwise) and B2 added ``streaming_hidden_prefix_lengths=[<prefix>]``.
  The other published MiniMax-H3 canonical rows (layers 10/20/40/50, and C1/C2
  at layer 30) used the non-public owner subset split (``--split
  minimax_subset``) and trained C1 on h1 only; without ``--split`` the recipe is
  the public-split one and says so in a note.
* Batch size: unless given, the batch size of the accepted runs
  (``accepted_batch_size``): 2 on the canonical line except Wan A1/B2 (1),
  Wan A2/C1/C2 (4) and MiniMax-H3 layer 30 A1 (1); on the timestep line 1 for A1
  and 2 otherwise.  ``default_batch_size`` is the protocol value.
* Evaluation: A1 (``streaming_depth``, ``streaming_mask``) uses
  ``vidfm3d/eval_a1_streaming.py`` (validation split only); every other probe
  uses ``vidfm3d/eval_diag.py``.  C3 is an evaluation of a C1 checkpoint with
  ``++evaluation_protocol=c3_action_sensitivity``, plus
  ``action_horizons=[1,2,4]`` when the C1 head was trained on h1 only.
* Environment: the per-prefix split manifest, ``INSCENE_PROBE_TASK=<probe>``
  (it selects which inputs the loader builds for the probe; a few accepted runs
  did not set it, see the notes),
  ``INSCENE_TRUST_EXPANDED_MANIFEST=1``, ``INSCENE_LOCAL_SCENE_ROOT`` (a value
  that is already set is kept; the default is the nested ScanNet++ folder under
  ``INSCENE_DATA_ROOT``), ``INSCENE_TARGET_LAYOUT=flat`` and
  ``PROBE_FP16_PASSTHROUGH`` (see ``fp16_passthrough``).  The target-isolated
  cache (``INSCENE_TARGET_FEAT_ROOT``) is required for C1 and C2, and so for C3,
  only.

Usage::

    python scripts/probe_recipe.py --line canonical --model dino \\
        --probe ego_belief_v2 --layer 0 --prefix 12
    python scripts/probe_recipe.py --line timestep --model wan --probe streaming_depth \\
        --layer 20 --prefix 8 --timestep 999 --format json
    python scripts/probe_recipe.py --line canonical --model minimax_h3 --probe action_dynamics \\
        --layer 50 --prefix 8 --split minimax_subset     # the published (non-public split) recipe
    python scripts/probe_recipe.py ... --stage eval --ckpt <path/to/epoch=E-step=S.ckpt>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_GROUP = "inscene15k_streaming"
CONFIG_DIR = ROOT / "configs" / "experiment" / EXPERIMENT_GROUP

LINES = ("canonical", "timestep")
CHANNELS = {
    "aether": 3072,
    "cogvideox": 3072,
    "dino": 1024,
    "f3r": 1024,
    "lingbot": 2048,
    "minimax_h3": 5376,
    "qwen2_5_vl": 3584,
    "videomae": 1024,
    "vjepa2": 1024,
    "wan": 1536,
}
# Number of cached backbone layers (0-based disk indices 0..n-1); Qwen also has -1
# (the visual-merger output).
NUM_LAYERS = {
    "aether": 42, "cogvideox": 42, "dino": 1, "f3r": 25, "lingbot": 48,
    "minimax_h3": 50, "qwen2_5_vl": 32, "videomae": 24, "vjepa2": 24, "wan": 30,
}
TIMESTEP_MODELS = ("cogvideox", "minimax_h3", "wan")
CANONICAL_TIMESTEP = {m: (250 if m in {"lingbot", "minimax_h3"} else 749) for m in CHANNELS}
DISK_LAYER = {("lingbot", 48): 47, ("minimax_h3", 50): 49}
LEDGER_LAYER = {(model, disk): ledger for (model, ledger), disk in DISK_LAYER.items()}
PREFIXES = (8, 12, 16, 24)
A1_PROBES = ("streaming_depth", "streaming_mask")
CANONICAL_PROBES = (
    "streaming_depth", "streaming_mask", "view_consistency",
    "ego_belief_v2", "action_dynamics", "path_integration",
)
TIMESTEP_PROBES = (
    "streaming_depth", "streaming_mask", "view_consistency", "ego_belief",
    "ego_belief_v2", "action_dynamics", "path_integration",
)
PROBES = {"canonical": CANONICAL_PROBES, "timestep": TIMESTEP_PROBES}
IDENTITY_HEAD_OVERRIDES = (
    "model.probe.active_heads=[identity]",
    "model.loss_weights.dmap=0.0",
    "model.loss_weights.camera=0.0",
    "model.loss_weights.identity=1.0",
)
# streaming_mask models without their own streaming_mask_<model>_v1 config.
MASK_FROM_DEPTH_MODELS = frozenset({"dino", "f3r", "aether"})
H124_OVERRIDE = "action_horizons=[1,2,4]"
C3_PROTOCOL_OVERRIDE = "++evaluation_protocol=c3_action_sensitivity"
PRIMARY_METRIC = {
    "streaming_depth": "mean_loss_depth",
    "streaming_mask": "mean_loss_identity",
    "view_consistency": "balanced_acc",
    "ego_belief": "mean_az_err_deg",
    "ego_belief_v2": "mean_ang_err_deg",
    "action_dynamics": "global_R@1",
    "path_integration": "global_R@1",
    "c3_from_c1": "intervention_validity",
}
EVAL_A1 = "vidfm3d/eval_a1_streaming.py"
EVAL_DIAG = "vidfm3d/eval_diag.py"
TRAIN_ENTRY = "vidfm3d/train.py"
MAX_EPOCHS = {probe: (100 if probe in A1_PROBES else 50) for probe in TIMESTEP_PROBES}

PUBLIC_SPLIT = "canonical_v2"
MINIMAX_SUBSET_SPLIT = "canonical_v2_minimax_owner_subset_r2"
SPLITS = (PUBLIC_SPLIT, MINIMAX_SUBSET_SPLIT)
SPLIT_ALIASES = {"public": PUBLIC_SPLIT, "minimax_subset": MINIMAX_SUBSET_SPLIT}
MANIFEST_NAME = "inscene_c_h124_prefix{prefix}.jsonl"
MANIFEST_DIR_VAR = {PUBLIC_SPLIT: "CANONICAL_MANIFEST_DIR", MINIMAX_SUBSET_SPLIT: "MINIMAX_SUBSET_MANIFEST_DIR"}
N_TRAIN_WINDOWS = {PUBLIC_SPLIT: 2445, MINIMAX_SUBSET_SPLIT: 1523}
N_VAL_WINDOWS = {PUBLIC_SPLIT: 276, MINIMAX_SUBSET_SPLIT: 171}
REQUIRED_ENV = ("INSCENE_DATA_ROOT", "INSCENE_STREAMING_FEAT_ROOT")
# Only the C1/C2 configs (and C3, an evaluation of a C1 checkpoint) read the
# target-isolated cache.
TARGET_ENV = "INSCENE_TARGET_FEAT_ROOT"
C_PROBES = ("action_dynamics", "path_integration")
SCENE_SUBDIR = "processed_scannetpp_v2/processed_scannetpp_v2"
# A scene root the caller already exported wins over the nested default.
SCENE_ROOT_DEFAULT = f"${{INSCENE_LOCAL_SCENE_ROOT:-${{INSCENE_DATA_ROOT}}/{SCENE_SUBDIR}}}"
RUNS_DIR = "logs/inscene15k_streaming/runs"

# Layers of the accepted rows (table numbering, as in results/*.csv).
ACCEPTED_LAYERS = {
    "canonical": {
        "aether": (1,), "cogvideox": (0, 5, 10, 30, 35, 40), "dino": (0,), "f3r": (24,),
        "lingbot": (10, 20, 30, 40, 48), "minimax_h3": (10, 20, 30, 40, 50),
        "qwen2_5_vl": (-1,), "videomae": (5, 11, 17, 23), "vjepa2": (5, 11, 17, 23),
        "wan": (0, 5, 10, 15, 20, 25, 29),
    },
    "timestep": {"cogvideox": (15, 20, 25), "minimax_h3": (15,), "wan": (15, 20, 25)},
}
# Table layers whose published MiniMax-H3 canonical rows are on the owner subset
# split (all other published MiniMax-H3 canonical rows are on the public split).
MINIMAX_SUBSET_ROW_LAYERS = {
    probe: ((20, 30, 40, 50) if probe in C_PROBES else (10, 20, 40, 50)) for probe in CANONICAL_PROBES
}

# Run groups of the accepted rows; each one shares one set of recipe rules.
FAMILY_DESCRIPTIONS = {
    "canonical_main": "canonical matrix at the protocol batch size 2",
    "canonical_wan": "Wan canonical runs at the Wan config batch sizes (1 for A1/B2, 4 for A2/C1/C2)",
    "canonical_minimax_l30": "MiniMax-H3 layer-30 A1/A2/B2 runs on the public split at the config batch sizes",
    "minimax_all_layers_subset": "MiniMax-H3 all-layers runs on the non-public owner subset split",
    "timestep": "timestep sweep",
}
PROBE_TASK_NOTE = (
    "the accepted runs of this cell {how} launched without INSCENE_PROBE_TASK; the recipe sets it "
    "anyway because it only selects which inputs the loader builds, which is not expected to change values"
)

# PROBE_FP16_PASSTHROUGH bases.  The text before the first colon says whether
# the value is verified from the accepted runs' launch settings or inferred.
FP16_INFERRED = (
    "inferred, not verified: the launch settings of these runs are not in the published records; "
    "1 is inferred because the later fp32 B2 retrains changed exactly this setting from 1 to 0"
)
FP16_VERIFIED_RETRAIN = "verified: the accepted fp32 B2 retrains set 0, so the loader widens features to fp32"
FP16_VERIFIED_COG_B2_RERUN = (
    "verified: the accepted CogVideoX layer 30/35/40 B2 runs were rerun with 0 before release, "
    "so the loader widens features to fp32"
)
FP16_VERIFIED_MINIMAX = "verified: the accepted MiniMax-H3 canonical runs read their bf16 caches with 0"
FP16_VERIFIED_TIMESTEP = "verified: the accepted timestep runs set 1"
FP16_VERIFIED_FP16_CACHE = (
    "verified with fp16 caches only: the accepted timestep MiniMax-H3 layer-15 runs set 1 and read fp16 caches, "
    "which the released extractor does not write (see the note)"
)
FP16_MINIMAX_BF16 = (
    "no accepted run: timestep MiniMax-H3 caches other than the original layer-15 ones are bf16 (as the released "
    "extractor writes them), and the bf16 MiniMax-H3 runs read them with 0"
)


class RecipeError(ValueError):
    """Raised for a cell outside the released protocol."""


@dataclass
class Recipe:
    line: str
    model: str
    probe: str
    layer: int
    disk_layer: int
    prefix: int
    timestep: int
    batch_size: int
    split: str
    family: str
    experiment: str
    overrides: list[str]
    train_extra: list[str]
    env: dict[str, str]
    required_env: list[str]
    evaluator: str
    primary_metric: str
    max_epochs: int
    fp16_passthrough_basis: str
    trained_action_horizons: list[int] | None = None
    c3_overrides: list[str] = field(default_factory=list)
    c3_variant: str | None = None
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- derived
    @property
    def unique_key(self) -> str:
        return unique_key(self.model, self.timestep, self.layer, self.prefix, self.probe)

    @property
    def run_name(self) -> str:
        """Hydra ``paths.run_folder_name`` used by the release scripts."""
        group = "timestep" if self.line == "timestep" else self.split
        return f"{group}/{self.unique_key}"

    @property
    def run_dir(self) -> str:
        return f"{RUNS_DIR}/{self.run_name}"

    @property
    def manifest_name(self) -> str:
        return MANIFEST_NAME.format(prefix=self.prefix)

    def train_args(self) -> list[str]:
        """``vidfm3d/train.py`` arguments (experiment first)."""
        return [*self.overrides, *self.train_extra]

    def eval_args(self, ckpt: str | None = None) -> list[str]:
        """Evaluator arguments; ``ckpt`` is omitted when ``None`` (callers add it)."""
        args = list(self.overrides)
        if ckpt is not None:
            args.append(ckpt_override(ckpt))
        # Evaluations write into the training run folder (<run>/eval/); keep the
        # training run's .hydra/ record instead of replacing it with the eval config.
        args += ["++eval_split=val", "hydra.output_subdir=null"]
        if self.evaluator == EVAL_DIAG:
            args += ["train=false", "test=false"]
        return args

    def c3_args(self, ckpt: str | None = None) -> list[str]:
        if not self.c3_overrides:
            raise RecipeError("C3 is evaluated from action_dynamics (C1) checkpoints only")
        return [*self.eval_args(ckpt), *self.c3_overrides]

    def to_dict(self, ckpt: str | None = None) -> dict:
        out = asdict(self)
        out.update(
            unique_key=self.unique_key,
            run_name=self.run_name,
            run_dir=self.run_dir,
            manifest_name=self.manifest_name,
            train_entry=TRAIN_ENTRY,
            train_args=self.train_args(),
            eval_args=self.eval_args(ckpt if ckpt is not None else CKPT_PLACEHOLDER),
            c3_entry=EVAL_DIAG if self.c3_overrides else None,
            c3_args=self.c3_args(ckpt if ckpt is not None else CKPT_PLACEHOLDER) if self.c3_overrides else None,
            checkpoint_glob=checkpoint_globs(self.run_dir)[0],
            family_description=FAMILY_DESCRIPTIONS[self.family],
        )
        return out


CKPT_PLACEHOLDER = f"<{RUNS_DIR}/<run>/VideoProbe3D/<id>/checkpoints/epoch=E-step=S.ckpt>"


# ----------------------------------------------------------------- helpers
def unique_key(model: str, timestep: int, layer: int, prefix: int, probe: str, seed: int = 42) -> str:
    return f"{model}__scannetpp__t{int(timestep)}__l{int(layer)}__p{int(prefix)}__{probe}__seed{int(seed)}"


def disk_layer(model: str, layer: int) -> int:
    """Disk (0-based) layer for a table layer (results/*.csv): lingbot 48 -> 47, minimax_h3 50 -> 49."""
    return DISK_LAYER.get((model, int(layer)), int(layer))


def ledger_layer(model: str, disk: int) -> int:
    """Inverse of ``disk_layer`` for the two last-block aliases."""
    return LEDGER_LAYER.get((model, int(disk)), int(disk))


def ckpt_override(ckpt: str) -> str:
    """Lightning names contain '=', so the value must be a quoted Hydra string."""
    if "'" in str(ckpt):
        raise RecipeError(f"checkpoint path must not contain a single quote: {ckpt!r}")
    return f"ckpt_path='{ckpt}'"


def checkpoint_globs(run_dir: str | Path) -> list[str]:
    """Final-epoch checkpoint patterns, in order of preference.

    No ModelCheckpoint callback is instantiated (``callbacks/v3_checkpoint.yaml``
    has no ``_target_``), so Lightning's default checkpoint is the only one:
    ``<run>/VideoProbe3D/<id>/checkpoints/epoch=E-step=S.ckpt``.  There is no
    ``last.ckpt``.
    """
    run_dir = str(run_dir).rstrip("/")
    return [f"{run_dir}/VideoProbe3D/*/checkpoints/epoch=*-step=*.ckpt",
            f"{run_dir}/checkpoints/epoch=*-step=*.ckpt"]


CKPT_NAME_RE = re.compile(r"epoch=(\d+)-step=(\d+)")


def checkpoint_epoch(ckpt: str | Path) -> int | None:
    """Epoch ``E`` of an ``epoch=E-step=S.ckpt`` name (``None`` for other names)."""
    match = CKPT_NAME_RE.search(Path(str(ckpt)).name)
    return int(match.group(1)) if match else None


def newest_checkpoint(run_dir: str | Path) -> Path | None:
    """The newest ``epoch=*-step=*.ckpt`` of a run, or ``None``."""
    import glob

    for pattern in checkpoint_globs(run_dir):
        found = [Path(p) for p in glob.glob(pattern)]
        if found:
            return max(found, key=lambda p: (p.stat().st_mtime, p.name))
    return None


def normalize_split(split: str | None) -> str:
    split = SPLIT_ALIASES.get(split or PUBLIC_SPLIT, split or PUBLIC_SPLIT)
    if split not in SPLITS:
        raise RecipeError(f"unknown split {split!r}; expected one of {SPLITS}")
    return split


def default_batch_size(line: str, probe: str) -> int:
    """Protocol batch size: 2 on the canonical line; timestep 1 for A1 and 2 otherwise."""
    if line == "timestep":
        return 1 if probe in A1_PROBES else 2
    return 2


def accepted_batch_size(line: str, model: str, probe: str, layer: int, split: str = PUBLIC_SPLIT) -> int:
    """Per-device batch size of the accepted runs for this cell (the recipe default).

    Wan used its config batch sizes on the canonical line (1 for A1 and B2, 4
    otherwise; Wan A2 has no published row), and so did MiniMax-H3 layer 30
    A1/A2/B2 on the public split (1 for A1, 2 otherwise).
    """
    if line == "timestep":
        return default_batch_size(line, probe)
    if model == "wan":
        return 1 if probe in A1_PROBES + ("ego_belief_v2",) else 4
    if _is_minimax_l30(model, probe, layer, normalize_split(split)):
        return 1 if probe in A1_PROBES else 2
    return 2


def fp16_passthrough(line: str, model: str, probe: str, layer: int) -> tuple[str, str]:
    """``PROBE_FP16_PASSTHROUGH`` value of the accepted runs and how it is known."""
    if line == "timestep":
        if model == "minimax_h3" and int(layer) != 15:
            return "0", FP16_MINIMAX_BF16
        if probe == "ego_belief_v2" and model in {"cogvideox", "minimax_h3"}:
            return "0", FP16_VERIFIED_RETRAIN
        if model == "minimax_h3":
            return "1", FP16_VERIFIED_FP16_CACHE
        return "1", FP16_VERIFIED_TIMESTEP
    if model == "minimax_h3":
        return "0", FP16_VERIFIED_MINIMAX
    if probe == "ego_belief_v2" and model == "cogvideox" and int(layer) in {30, 35, 40}:
        return "0", FP16_VERIFIED_COG_B2_RERUN
    if probe == "ego_belief_v2" and model in {"cogvideox", "aether"}:
        return "0", FP16_VERIFIED_RETRAIN
    return "1", FP16_INFERRED


def _is_minimax_l30(model: str, probe: str, layer: int, split: str) -> bool:
    return (model == "minimax_h3" and int(layer) == 30 and split == PUBLIC_SPLIT
            and probe in {"streaming_depth", "streaming_mask", "view_consistency", "ego_belief_v2"})


def is_minimax_subset_cell(model: str, probe: str, layer: int) -> bool:
    """True when the published canonical row of this cell is on the MiniMax-H3 owner subset split."""
    return model == "minimax_h3" and int(layer) in MINIMAX_SUBSET_ROW_LAYERS.get(probe, ())


def experiment_path(experiment: str) -> Path:
    return ROOT / "configs" / "experiment" / f"{experiment}.yaml"


def _validate(line: str, model: str, probe: str, layer: int, prefix: int, timestep: int | None, split: str) -> int:
    if line not in LINES:
        raise RecipeError(f"unknown line {line!r}; expected one of {LINES}")
    if model not in CHANNELS:
        raise RecipeError(f"unknown model {model!r}; expected one of {sorted(CHANNELS)}")
    if probe not in PROBES[line]:
        raise RecipeError(f"probe {probe!r} is not part of the {line} line; expected one of {PROBES[line]}")
    if int(prefix) not in PREFIXES:
        raise RecipeError(f"prefix must be one of {PREFIXES}, got {prefix}")
    disk = disk_layer(model, layer)
    lowest = -1 if model == "qwen2_5_vl" else 0
    if not lowest <= disk < NUM_LAYERS[model]:
        raise RecipeError(f"layer {layer} is outside {model}'s cached layers {lowest}..{NUM_LAYERS[model] - 1}")
    if split != PUBLIC_SPLIT and (line != "canonical" or model != "minimax_h3"):
        raise RecipeError(f"split {split} exists only for canonical minimax_h3 rows")
    if line == "timestep":
        if model not in TIMESTEP_MODELS:
            raise RecipeError(f"the timestep line covers {TIMESTEP_MODELS}, not {model!r}")
        if timestep is None:
            raise RecipeError("the timestep line needs --timestep")
        return int(timestep)
    canonical_t = CANONICAL_TIMESTEP[model]
    if timestep is not None and int(timestep) != canonical_t:
        raise RecipeError(f"the canonical line fixes {model} at timestep {canonical_t}, got {timestep}")
    return canonical_t


# ------------------------------------------------------------------ recipe
def recipe(
    line: str,
    model: str,
    probe: str,
    layer: int,
    prefix: int,
    timestep: int | None = None,
    batch_size: int | None = None,
    split: str | None = None,
    fp16: str | None = None,
) -> Recipe:
    """Build the recipe for one cell.  Raises ``RecipeError`` for unknown cells.

    ``batch_size`` defaults to ``accepted_batch_size``.  ``split`` defaults to
    the public split; for a MiniMax-H3 cell whose published row is on the owner
    subset split, the recipe then carries a note saying so.
    """
    if line == "canonical_v2":
        line = "canonical"
    split_given = split is not None
    split = normalize_split(split)
    layer, prefix = int(layer), int(prefix)
    t = _validate(line, model, probe, layer, prefix, timestep, split)
    disk = disk_layer(model, layer)
    accepted_batch = accepted_batch_size(line, model, probe, layer, split)
    batch = int(batch_size) if batch_size is not None else accepted_batch
    if batch < 1:
        raise RecipeError("batch size must be positive")
    notes: list[str] = []
    extra: list[str] = []
    horizons: list[int] | None = None
    if batch != accepted_batch:
        notes.append(f"batch size {batch} was given; the accepted runs' batch size for this cell is {accepted_batch}")

    if line == "timestep":
        family = "timestep"
        stem = f"{probe}_{model}_v1"
        if probe == "action_dynamics":
            horizons = [1]
        if (probe == "ego_belief_v2" and model in {"cogvideox", "minimax_h3"}
                and layer in ACCEPTED_LAYERS["timestep"][model]):
            notes.append(PROBE_TASK_NOTE.format(how="(fp32 B2 retrains) were"))
    elif split == MINIMAX_SUBSET_SPLIT:
        family = "minimax_all_layers_subset"
        stem = f"{probe}_{model}_v1"
        if probe == "action_dynamics":
            horizons = [1]
        notes.append("trained and evaluated on the MiniMax-H3 owner subset split (1,523 train / 171 val "
                     "windows), which is not public; set MINIMAX_SUBSET_MANIFEST_DIR to its manifests")
    else:
        if model == "wan":
            family = "canonical_wan"
        elif _is_minimax_l30(model, probe, layer, split):
            family = "canonical_minimax_l30"
            # The A1/A2 launch records lack INSCENE_PROBE_TASK; the B2 record is incomplete.
            notes.append(PROBE_TASK_NOTE.format(how="may have been" if probe == "ego_belief_v2" else "were"))
        else:
            family = "canonical_main"
        if not split_given and is_minimax_subset_cell(model, probe, layer):
            used = (f"the non-public MiniMax-H3 owner subset split ({MINIMAX_SUBSET_SPLIT}, "
                    f"{N_VAL_WINDOWS[MINIMAX_SUBSET_SPLIT]} val windows)")
            if probe == "action_dynamics":
                used += " and the h1-only action_dynamics_minimax_h3_v1 C1 config"
                here = f"the public {PUBLIC_SPLIT} split and the h124 C1 config"
            else:
                here = f"the public {PUBLIC_SPLIT} split"
            notes.append(f"the published row of this cell used {used}; this recipe uses {here}, so it matches "
                         f"no published row; pass --split minimax_subset for the published recipe")
        if probe == "action_dynamics":
            horizons = [1, 2, 4]
            if model == "wan":
                stem = "action_dynamics_wan_v1"
                notes.append("the Wan C1 runs used action_dynamics_wan_v1 with action_horizons=[1,2,4], "
                             "which composes to action_dynamics_wan_h124_v1 apart from job_name")
            else:
                stem = f"action_dynamics_{model}_h124_v1"
            extra.append(H124_OVERRIDE)
        elif model == "qwen2_5_vl":
            if probe == "streaming_mask":
                stem = "direct_vlm_streaming_depth_v1"
                extra.extend(IDENTITY_HEAD_OVERRIDES)
            else:
                stem = f"direct_vlm_{probe}_v1"
        elif probe == "streaming_mask" and model in MASK_FROM_DEPTH_MODELS:
            stem = f"streaming_depth_{model}_v1"
            extra.extend(IDENTITY_HEAD_OVERRIDES)
        else:
            stem = f"{probe}_{model}_v1"
            if probe == "streaming_mask" and model == "wan":
                extra.extend(IDENTITY_HEAD_OVERRIDES)
        if family == "canonical_minimax_l30" and probe == "ego_belief_v2":
            extra.append(f"streaming_hidden_prefix_lengths=[{prefix}]")
        if disk != layer:
            notes.append(f"table layer {layer} is disk layer {disk} (feature_t{t}_layer{disk}.sft)")
        if model == "lingbot" and disk != layer:
            notes.append(f"the accepted LingBot layer-{layer} runs passed feature_layer={layer} and read disk layer "
                         f"{disk} through the optional owner overlay (PROBE_OWNER_OVERLAY_MANIFEST), which maps "
                         f"{layer} to {disk}; without it the cache file follows feature_layer, so this recipe "
                         f"passes {disk}; the features are the same and export_results.py reports layer {layer}")

    experiment = f"{EXPERIMENT_GROUP}/{stem}"
    if not experiment_path(experiment).is_file():
        raise RecipeError(f"missing experiment config {experiment_path(experiment).relative_to(ROOT)}")

    overrides = [
        f"experiment={experiment}",
        "seed=42",
        f"vfm_name={model}",
        f"video_channels={CHANNELS[model]}",
        f"feature_layer={disk}",
        f"feature_timestep={t}",
        f"feat_postfix=_t{t}_layer{disk}",
        f"prefix_len={prefix}",
        f"data.data_module.batch_size_per_device={batch}",
        f"data.data_module.batch_size_per_device_val={batch}",
        "trainer.devices=[0]",
    ]
    if line == "canonical":
        overrides.append("trainer.num_nodes=1")
    overrides += extra

    fp16_value, fp16_basis = fp16_passthrough(line, model, probe, layer)
    if fp16 is not None:
        if str(fp16) not in {"0", "1"}:
            raise RecipeError("PROBE_FP16_PASSTHROUGH must be 0 or 1")
        if str(fp16) != fp16_value:
            fp16_basis = f"overridden by the caller (accepted runs: {fp16_value}; {fp16_basis})"
        fp16_value = str(fp16)
    if line == "timestep" and model == "minimax_h3" and layer == 15:
        note = ("the accepted timestep MiniMax-H3 layer-15 runs read fp16 feature caches, but the released extractor "
                "writes MiniMax-H3 features in bf16, so a re-extracted cache is not the one those runs read")
        if fp16_value == "1":
            note += ("; PROBE_FP16_PASSTHROUGH=1 has only been run on fp16 caches, and the bf16 MiniMax-H3 runs "
                     "used 0 (`scripts/probe_recipe.py --fp16-passthrough 0` prints that recipe)")
        notes.append(note)

    manifest_var = MANIFEST_DIR_VAR[split]
    env = {
        "INSCENE_LOCAL_SCENE_ROOT": SCENE_ROOT_DEFAULT,
        "INSCENE_MANIFEST": f"${{{manifest_var}}}/{MANIFEST_NAME.format(prefix=prefix)}",
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "INSCENE_TARGET_LAYOUT": "flat",
        "INSCENE_PROBE_TASK": probe,
        "PROBE_FP16_PASSTHROUGH": fp16_value,
    }
    c3: list[str] = []
    variant = None
    if probe == "action_dynamics":
        c3 = [C3_PROTOCOL_OVERRIDE]
        if horizons == [1]:
            c3.append(H124_OVERRIDE)
            variant = "h1_trained_diagnostic"
        else:
            variant = "c3_from_c1"
    return Recipe(
        line=line, model=model, probe=probe, layer=layer, disk_layer=disk, prefix=prefix, timestep=t,
        batch_size=batch, split=split, family=family, experiment=experiment, overrides=overrides,
        train_extra=["autoresume=false", "logger.wandb.offline=true"], env=env,
        required_env=[*REQUIRED_ENV, *((TARGET_ENV,) if probe in C_PROBES else ()), manifest_var],
        evaluator=EVAL_A1 if probe in A1_PROBES else EVAL_DIAG,
        primary_metric=PRIMARY_METRIC[probe], max_epochs=MAX_EPOCHS[probe],
        fp16_passthrough_basis=fp16_basis, trained_action_horizons=horizons,
        c3_overrides=c3, c3_variant=variant, notes=notes,
    )


KEY_RE = re.compile(
    r"^(?P<model>.+?)__scannetpp__t(?P<timestep>\d+)__l(?P<layer>-?\d+)__p(?P<prefix>\d+)"
    r"__(?P<probe>.+)__seed(?P<seed>\d+)$"
)


def parse_key(key: str) -> dict:
    """Split a ``<model>__scannetpp__t<T>__l<L>__p<P>__<probe>__seed<S>`` key."""
    match = KEY_RE.match(key or "")
    if not match:
        raise RecipeError(f"not a probe key: {key!r}")
    out = match.groupdict()
    for name in ("timestep", "layer", "prefix", "seed"):
        out[name] = int(out[name])
    return out


def line_of_key(key: str) -> str:
    """``canonical`` when the key's timestep is the model's canonical one, else ``timestep``."""
    parts = parse_key(key)
    return "canonical" if CANONICAL_TIMESTEP.get(parts["model"]) == parts["timestep"] else "timestep"


def recipe_for_row(row: dict, *, fp16: str | None = None) -> Recipe:
    """Recipe for one row of ``results/*_val_seed42.csv``.

    A C3 row (``line == c3_from_c1``) returns the recipe of the C1 run named in
    its ``c1_unique_key``; its ``c3_args`` give the C3 evaluation.
    """
    batch = row.get("train_batch_size_inferred") or None
    if row["line"] == "c3_from_c1":
        c1 = parse_key(row["c1_unique_key"])
        if c1["probe"] != "action_dynamics":
            raise RecipeError(f"C3 row does not name a C1 run: {row['c1_unique_key']}")
        line = line_of_key(row["c1_unique_key"])
        model, probe, layer, prefix, timestep = c1["model"], "action_dynamics", c1["layer"], c1["prefix"], c1["timestep"]
    else:
        line = "canonical" if row["line"] == "canonical_v2" else row["line"]
        model, probe, layer, prefix = row["model"], row["probe"], int(row["layer"]), int(row["prefix"])
        timestep = int(row["timestep"])
    return recipe(
        line, model, probe, layer, prefix,
        timestep=timestep if line == "timestep" else None,
        batch_size=int(batch) if batch else None,
        split=row.get("split_id") or PUBLIC_SPLIT, fp16=fp16,
    )


# --------------------------------------------------------------- rendering
def shell_quote(arg: str) -> str:
    """Quote for bash, preferring "..." for Hydra values that carry single quotes."""
    quoted = shlex.quote(arg)
    if quoted == arg:
        return arg
    if "'" in arg and not any(ch in arg for ch in '"$`\\!'):
        return f'"{arg}"'
    return quoted


_VAR_RE = re.compile(r"\$(\w+)")
_BRACED_RE = re.compile(r"^(\w+)(?::-(.*))?$", re.S)


def expand(value: str, environ: dict[str, str] | None = None) -> str:
    """Expand ``$VAR``, ``${VAR}`` and ``${VAR:-default}`` from ``environ`` as bash does.

    ``${VAR:-default}`` takes the default (itself expanded) when VAR is unset or
    empty.  Unset variables without a default are kept as written.
    """
    env = os.environ if environ is None else environ
    out: list[str] = []
    i = 0
    while i < len(value):
        if value.startswith("${", i):
            depth, j = 0, i
            while j < len(value):
                if value.startswith("${", j):
                    depth, j = depth + 1, j + 2
                    continue
                if value[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= len(value):          # unterminated: keep the rest as written
                out.append(value[i:])
                break
            match = _BRACED_RE.match(value[i + 2:j])
            if match is None:
                out.append(value[i:j + 1])
            else:
                name, default = match.groups()
                current = env.get(name)
                if default is not None:
                    out.append(current if current else expand(default, env))
                else:
                    out.append(current if current is not None else value[i:j + 1])
            i = j + 1
            continue
        match = _VAR_RE.match(value, i)
        if match:
            out.append(env.get(match.group(1), match.group(0)))
            i = match.end()
            continue
        out.append(value[i])
        i += 1
    return "".join(out)


def _stage_args(rec: Recipe, stage: str, ckpt: str | None) -> tuple[str, list[str]]:
    if stage == "train":
        return TRAIN_ENTRY, rec.train_args()
    if stage == "eval":
        return rec.evaluator, rec.eval_args(ckpt)
    return EVAL_DIAG, rec.c3_args(ckpt)


def render_shell(rec: Recipe, stage: str, ckpt: str | None, run_name: str | None = None) -> str:
    run_name = run_name or rec.run_name
    entry, args = _stage_args(rec, stage, ckpt if ckpt is not None else CKPT_PLACEHOLDER)
    args = [*args, f"job_name={rec.unique_key}", f"paths.run_folder_name={run_name}"]
    lines = [
        f"# probe_recipe: line={rec.line} model={rec.model} probe={rec.probe} layer={rec.layer} "
        f"disk_layer={rec.disk_layer} prefix={rec.prefix} timestep={rec.timestep} batch={rec.batch_size}",
        f"# split={rec.split} stage={stage} evaluator={rec.evaluator} "
        f"max_epochs={rec.max_epochs} primary_metric={rec.primary_metric}",
        f"# run group: {FAMILY_DESCRIPTIONS[rec.family]} ({rec.family})",
        f"# PROBE_FP16_PASSTHROUGH={rec.env['PROBE_FP16_PASSTHROUGH']} ({rec.fp16_passthrough_basis})",
        *(f"# note: {note}" for note in rec.notes),
        ": " + " ".join(f'"${{{name}:?set {name}}}"' for name in rec.required_env),
    ]
    for key, value in rec.env.items():
        lines.append(f'export {key}="{value}"')
    lines.append(" ".join(["python", entry, *(shell_quote(a) for a in args)]))
    return "\n".join(lines) + "\n"


def render_bash_vars(rec: Recipe) -> str:
    """Bash assignments for ``eval`` in shell wrappers (no checkpoint in the eval args)."""
    def arr(values: Iterable[str]) -> str:
        return "(" + " ".join(shlex.quote(v) for v in values) + ")"

    scalars = {
        "RECIPE_EXPERIMENT": rec.experiment, "RECIPE_FAMILY": rec.family,
        "RECIPE_DISK_LAYER": str(rec.disk_layer), "RECIPE_TIMESTEP": str(rec.timestep),
        "RECIPE_BATCH": str(rec.batch_size), "RECIPE_EVALUATOR": rec.evaluator,
        "RECIPE_UNIQUE_KEY": rec.unique_key, "RECIPE_RUN_NAME": rec.run_name,
        "RECIPE_MANIFEST_NAME": rec.manifest_name, "RECIPE_CHANNELS": str(CHANNELS[rec.model]),
    }
    lines = [f"{k}={shlex.quote(v)}" for k, v in scalars.items()]
    lines += [
        f"RECIPE_TRAIN_ARGS={arr(rec.train_args())}",
        f"RECIPE_EVAL_ARGS={arr(rec.eval_args(None))}",
        f"RECIPE_C3_ARGS={arr(rec.c3_overrides)}",
        "RECIPE_ENV=" + arr(f"{k}={rec.env[k]}" for k in ("INSCENE_PROBE_TASK", "PROBE_FP16_PASSTHROUGH")),
        f"RECIPE_NOTES={arr(rec.notes)}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the training/evaluation recipe of one probe cell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Formats: shell (sourceable exports + command), json (everything), "
               "args (one Hydra argument per line), bash (RECIPE_* assignments for wrappers).",
    )
    parser.add_argument("--line", required=True, choices=["canonical", "canonical_v2", "timestep"])
    parser.add_argument("--model", required=True, choices=sorted(CHANNELS))
    parser.add_argument("--probe", required=True, choices=TIMESTEP_PROBES)
    parser.add_argument("--layer", required=True, type=int, help="table layer as in results/*.csv (lingbot 48 / minimax_h3 50 = last block)")
    parser.add_argument("--prefix", required=True, type=int, choices=PREFIXES)
    parser.add_argument("--timestep", type=int, default=None, help="required for --line timestep")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="per-device batch size (default: the accepted runs' batch size: canonical 2 except "
                             "Wan A1/B2 1, Wan A2/C1/C2 4 and MiniMax-H3 layer 30 A1 1; timestep 1 for A1, "
                             "2 otherwise)")
    parser.add_argument("--split", default=None, choices=[*SPLITS, *SPLIT_ALIASES],
                        help=f"canonical minimax_h3 only: minimax_subset is the non-public owner subset split of "
                             f"the published all-layers rows (default: the public {PUBLIC_SPLIT} split)")
    parser.add_argument("--fp16-passthrough", choices=["0", "1"], default=None,
                        help="override PROBE_FP16_PASSTHROUGH (default: the value of the accepted runs)")
    parser.add_argument("--stage", choices=["train", "eval", "c3"], default="train")
    parser.add_argument("--ckpt", default=None, help="checkpoint for --stage eval/c3")
    parser.add_argument("--format", choices=["shell", "json", "args", "bash"], default="shell")
    args = parser.parse_args(argv)
    try:
        rec = recipe(args.line, args.model, args.probe, args.layer, args.prefix, timestep=args.timestep,
                     batch_size=args.batch_size, split=args.split, fp16=args.fp16_passthrough)
        if args.format == "json":
            payload = rec.to_dict(args.ckpt)
            payload["stage"] = args.stage
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.format == "args":
            _, stage_args = _stage_args(rec, args.stage, args.ckpt)
            print("\n".join(stage_args))
        elif args.format == "bash":
            sys.stdout.write(render_bash_vars(rec))
        else:
            sys.stdout.write(render_shell(rec, args.stage, args.ckpt))
    except RecipeError as exc:
        print(f"probe_recipe: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
