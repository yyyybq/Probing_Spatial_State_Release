"""scripts/probe_recipe.py reproduces the recipe of every accepted row in results/.

Every row of results/{canonical_v2,timestep,c3_from_c1}_val_seed42.csv is mapped
to its recipe and checked against what the row records (batch size inferred from
the checkpoint step, also as the recipe default; final epoch, loader size per
trained horizon, C3 variant, key, split).  Each unique (experiment, override
template) pair is then composed with Hydra in-process, and three evaluator
commands go through the real decorated entry points with ``--cfg job``.  The
sweep wrapper is dry-run as in CI, and ``reproduce_rows.py --run`` is driven
with a stub interpreter.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_recipe as pr  # noqa: E402
import reproduce_rows  # noqa: E402

RESULTS = ROOT / "results"
TABLES = ("canonical_v2_val_seed42.csv", "timestep_val_seed42.csv", "c3_from_c1_val_seed42.csv")
CKPT = "/x/run/VideoProbe3D/abc/checkpoints/epoch=49-step=61100.ckpt"
# Override values that vary inside one template; the composition checks them per row.
VOLATILE = re.compile(
    r"^(prefix_len|feature_layer|feat_postfix|feature_timestep|"
    r"data\.data_module\.batch_size_per_device(_val)?|streaming_hidden_prefix_lengths)="
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    out = []
    for name in TABLES:
        with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
            out += [dict(row, _table=name) for row in csv.DictReader(handle)]
    assert out and all(row["status"] == "ACCEPTED" for row in out)
    return out


@pytest.fixture(scope="module")
def recipes(rows) -> list[tuple[dict, pr.Recipe]]:
    return [(row, pr.recipe_for_row(row)) for row in rows]


def _template(args: list[str]) -> tuple[str, ...]:
    return tuple(a.split("=", 1)[0] + "=*" if VOLATILE.match(a) else a for a in args)


def test_every_accepted_row_maps_to_its_recipe(recipes):
    problems = []
    for row, rec in recipes:
        key = row["unique_key"]
        c3 = row["line"] == "c3_from_c1"
        n_val = pr.N_VAL_WINDOWS[rec.split]
        default = pr.recipe_for_row(dict(row, train_batch_size_inferred=""))
        expected = {
            "split": (rec.split, row["split_id"]),
            "subset cell": (pr.is_minimax_subset_cell(rec.model, rec.probe, rec.layer) and rec.line == "canonical",
                            row["split_id"] == pr.MINIMAX_SUBSET_SPLIT),
            "config exists": (pr.experiment_path(rec.experiment).is_file(), True),
            "batch": (pr.accepted_batch_size(rec.line, rec.model, rec.probe, rec.layer, rec.split),
                      int(row["train_batch_size_inferred"])),
            "recipe batch": (rec.batch_size, int(row["train_batch_size_inferred"])),
            "default recipe batch": (default.batch_size, int(row["train_batch_size_inferred"])),
            "default recipe overrides": (default.overrides, rec.overrides),
            "target cache required": (pr.TARGET_ENV in rec.required_env, rec.probe in pr.C_PROBES),
            "final epoch": (rec.max_epochs - 1, int(row["ckpt_epoch"])),
            "steps": ((pr.N_TRAIN_WINDOWS[rec.split] // rec.batch_size) * rec.max_epochs, int(row["ckpt_step"])),
            "primary metric": (pr.PRIMARY_METRIC["c3_from_c1" if c3 else rec.probe], row["primary_metric"]),
            "evaluator": (rec.evaluator, pr.EVAL_A1 if rec.probe in pr.A1_PROBES else pr.EVAL_DIAG),
            "key": (rec.unique_key, row["c1_unique_key"] if c3 else key),
            "disk layer": (rec.disk_layer, pr.disk_layer(rec.model, int(row["layer"]))),
            "timestep": (rec.timestep, int(row["timestep"])),
            "INSCENE_PROBE_TASK": (rec.env["INSCENE_PROBE_TASK"], rec.probe),
        }
        if c3:
            expected["variant"] = (rec.c3_variant, row["variant"])
            expected["candidates"] = (3 * n_val, int(row["n_loader"]))
            expected["c3 key"] = (pr.unique_key(rec.model, rec.timestep, rec.layer, rec.prefix, "c3_from_c1"), key)
        elif rec.probe == "action_dynamics":
            expected["C1 loader = val windows x trained horizons"] = (
                n_val * len(rec.trained_action_horizons), int(row["n_loader"]))
        else:
            expected["loader"] = (n_val, int(row["n_loader"]))
        for label, (got, want) in expected.items():
            if got != want:
                problems.append(f"{key}: {label}: recipe {got!r} != table {want!r}")
    assert not problems, "\n".join(problems[:20])


def test_recipe_families_and_fp16_cover_the_accepted_scopes(recipes):
    by_scope: dict[str, set] = {}
    for row, rec in recipes:
        by_scope.setdefault(row["scope"], set()).add((rec.family, rec.env["PROBE_FP16_PASSTHROUGH"]))
    expected_family = {
        "canonical_v2_main552": "canonical_main", "canonical_v2_b2_fp32_retrain_20261016": "canonical_main",
        "canonical_v2_wan168": "canonical_wan", "canonical_v2_minimax_l30_ab16": "canonical_minimax_l30",
        "canonical_v2_minimax_all_layers_20261011": "minimax_all_layers_subset",
    }
    for scope, pairs in by_scope.items():
        families = {family for family, _ in pairs}
        assert families == {expected_family.get(scope, "timestep")}, (scope, families)
    assert set(pr.FAMILY_DESCRIPTIONS) == set(expected_family.values()) | {"timestep"}
    # B2 retrains and every MiniMax-H3 canonical run widened features to fp32.
    assert {fp for _, fp in by_scope["canonical_v2_b2_fp32_retrain_20261016"]} == {"0"}
    assert {fp for _, fp in by_scope["tstep_b2_fp32_retrain_20261016"]} == {"0"}
    assert {fp for _, fp in by_scope["canonical_v2_minimax_all_layers_20261011"]} == {"0"}
    assert {fp for _, fp in by_scope["tstep_probe_opt_v3"]} == {"1"}


def test_unique_recipe_templates_compose(recipes):
    """One in-process Hydra composition per unique (experiment, override template)."""
    train, evals = {}, {}
    for _, rec in recipes:
        train.setdefault(_template(rec.train_args()), rec)
        evals.setdefault(_template(rec.eval_args(CKPT)), (rec, rec.eval_args(CKPT)))
        if rec.c3_overrides:
            evals.setdefault(_template(rec.c3_args(CKPT)), (rec, rec.c3_args(CKPT)))
    assert len({t[0] for t in train}) >= 50
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        for rec in train.values():
            cfg = compose(config_name="train.yaml", overrides=rec.train_args())
            where = f"{rec.unique_key} ({rec.experiment})"
            assert cfg.vfm_name == rec.model and cfg.video_channels == pr.CHANNELS[rec.model], where
            assert cfg.feature_layer == rec.disk_layer and cfg.feature_timestep == rec.timestep, where
            assert cfg.feat_postfix == f"_t{rec.timestep}_layer{rec.disk_layer}", where
            assert cfg.prefix_len == rec.prefix and cfg.seed == 42 and cfg.autoresume is False, where
            dm = cfg.data.data_module
            assert dm.batch_size_per_device == dm.batch_size_per_device_val == rec.batch_size, where
            assert cfg.trainer.max_epochs == rec.max_epochs and list(cfg.trainer.devices) == [0], where
            assert cfg.trainer.get("num_nodes", 1) == 1, where
            expected_type = {"streaming_depth": "pixalign", "streaming_mask": "pixalign"}.get(rec.probe, rec.probe)
            assert cfg.model.probe_type == expected_type, where
            if rec.probe == "streaming_mask":
                assert list(cfg.model.probe.active_heads) == ["identity"], where
                weights = cfg.model.loss_weights
                assert (weights.dmap, weights.camera, weights.identity) == (0.0, 0.0, 1.0), where
            elif rec.probe == "streaming_depth":
                assert list(cfg.model.probe.active_heads) == ["depth", "camera"], where
            if rec.probe == "action_dynamics":
                assert list(cfg.action_horizons) == rec.trained_action_horizons, where
            if rec.family == "canonical_minimax_l30" and rec.probe == "ego_belief_v2":
                assert list(cfg.streaming_hidden_prefix_lengths) == [rec.prefix], where
        for rec, args in evals.values():
            cfg = compose(config_name="train.yaml", overrides=args, return_hydra_config=True)
            where = f"{rec.unique_key}: {' '.join(args[-4:])}"
            assert cfg.ckpt_path == CKPT and cfg.eval_split == "val", where
            assert cfg.hydra.output_subdir is None, where
            if "++evaluation_protocol=c3_action_sensitivity" in args:
                assert cfg.evaluation_protocol == "c3_action_sensitivity", where
                assert list(cfg.action_horizons) == [1, 2, 4], where
            if rec.evaluator == pr.EVAL_DIAG:
                assert cfg.train is False and cfg.test is False, where


def test_wan_c1_override_equals_the_h124_config():
    base = ["seed=42", "prefix_len=12", "feature_layer=20"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        literal = compose(config_name="train.yaml", overrides=[
            "experiment=inscene15k_streaming/action_dynamics_wan_v1", pr.H124_OVERRIDE, *base])
        h124 = compose(config_name="train.yaml", overrides=[
            "experiment=inscene15k_streaming/action_dynamics_wan_h124_v1", *base])
    left, right = (OmegaConf.to_container(c, resolve=False) for c in (literal, h124))
    for cfg in (left, right):
        cfg.pop("job_name")
        cfg.pop("autoresume")
    assert left == right


def test_disk_layers_and_streaming_mask_configs():
    assert pr.disk_layer("lingbot", 48) == 47 and pr.disk_layer("minimax_h3", 50) == 49
    assert pr.disk_layer("lingbot", 40) == 40 and pr.disk_layer("wan", 29) == 29
    assert pr.ledger_layer("lingbot", 47) == 48 and pr.ledger_layer("minimax_h3", 49) == 50
    rec = pr.recipe("canonical", "lingbot", "streaming_depth", 48, 8)
    assert "feature_layer=47" in rec.overrides and "feat_postfix=_t250_layer47" in rec.overrides
    own = {p.stem[len("streaming_mask_"):-len("_v1")] for p in pr.CONFIG_DIR.glob("streaming_mask_*_v1.yaml")}
    assert own == {"cogvideox", "lingbot", "minimax_h3", "videomae", "vjepa2", "wan"}
    for model in ("dino", "f3r", "aether", "qwen2_5_vl"):
        rec = pr.recipe("canonical", model, "streaming_mask", pr.ACCEPTED_LAYERS["canonical"][model][0], 12)
        stem = "direct_vlm_streaming_depth_v1" if model == "qwen2_5_vl" else f"streaming_depth_{model}_v1"
        assert rec.experiment == f"{pr.EXPERIMENT_GROUP}/{stem}"
        assert rec.overrides[-4:] == list(pr.IDENTITY_HEAD_OVERRIDES)
    for model in own - {"wan"}:
        rec = pr.recipe("canonical", model, "streaming_mask", pr.ACCEPTED_LAYERS["canonical"][model][0], 12)
        assert rec.experiment.endswith(f"streaming_mask_{model}_v1")
        assert not set(pr.IDENTITY_HEAD_OVERRIDES) & set(rec.overrides)


def test_c1_horizons_and_c3_overrides():
    canonical = pr.recipe("canonical", "dino", "action_dynamics", 0, 8)
    assert canonical.experiment.endswith("action_dynamics_dino_h124_v1")
    assert pr.H124_OVERRIDE in canonical.overrides
    assert canonical.c3_overrides == [pr.C3_PROTOCOL_OVERRIDE] and canonical.c3_variant == "c3_from_c1"
    for rec in (pr.recipe("timestep", "wan", "action_dynamics", 20, 8, timestep=999),
                pr.recipe("canonical", "minimax_h3", "action_dynamics", 20, 8, split="minimax_subset")):
        assert rec.experiment.endswith(f"action_dynamics_{rec.model}_v1")
        assert pr.H124_OVERRIDE not in rec.overrides
        assert rec.c3_overrides == [pr.C3_PROTOCOL_OVERRIDE, pr.H124_OVERRIDE]
        assert rec.c3_variant == "h1_trained_diagnostic"
    with pytest.raises(pr.RecipeError):
        pr.recipe("timestep", "dino", "streaming_depth", 0, 8, timestep=999)
    with pytest.raises(pr.RecipeError):
        pr.recipe("canonical", "dino", "streaming_depth", 0, 8, timestep=999)
    with pytest.raises(pr.RecipeError):
        pr.recipe("canonical", "dino", "streaming_depth", 0, 8, split="minimax_subset")


def _shell(capsys, *argv: str) -> str:
    assert pr.main(list(argv)) == 0
    return capsys.readouterr().out


def _batch_overrides(text: str) -> set[str]:
    return set(re.findall(r"batch_size_per_device(?:_val)?=(\d+)", text))


def test_default_batch_size_is_the_accepted_one(capsys):
    """Without --batch-size the recipe (CLI and sweep) uses the accepted runs' batch size."""
    cells = {
        ("wan", "streaming_depth", 20): 1, ("wan", "streaming_mask", 0): 1, ("wan", "ego_belief_v2", 29): 1,
        ("wan", "action_dynamics", 20): 4, ("wan", "path_integration", 5): 4,
        ("minimax_h3", "streaming_depth", 30): 1, ("minimax_h3", "streaming_mask", 30): 1,
        ("minimax_h3", "view_consistency", 30): 2, ("minimax_h3", "ego_belief_v2", 30): 2,
        ("dino", "streaming_depth", 0): 2, ("dino", "ego_belief_v2", 0): 2, ("dino", "action_dynamics", 0): 2,
        ("lingbot", "streaming_mask", 48): 2,
    }
    for (model, probe, layer), batch in cells.items():
        rec = pr.recipe("canonical", model, probe, layer, 8)
        assert rec.batch_size == batch, (model, probe, layer)
        assert rec.split == pr.PUBLIC_SPLIT and not any(n.startswith("batch size") for n in rec.notes)
        for stage in ("train", "eval"):
            text = _shell(capsys, "--line", "canonical", "--model", model, "--probe", probe, "--layer", str(layer),
                          "--prefix", "8", "--stage", stage)
            assert f"batch={batch}" in text.splitlines()[0] and _batch_overrides(text) == {str(batch)}, text
    # MiniMax-H3 layer 30 A1 on the public split is 1; the subset split uses 2.
    assert pr.recipe("canonical", "minimax_h3", "streaming_depth", 30, 8, split="public").batch_size == 1
    assert pr.recipe("canonical", "minimax_h3", "streaming_depth", 20, 8, split="minimax_subset").batch_size == 2
    # The timestep line keeps its default.
    assert pr.recipe("timestep", "cogvideox", "streaming_mask", 15, 8, timestep=499).batch_size == 1
    assert pr.recipe("timestep", "cogvideox", "view_consistency", 15, 8, timestep=499).batch_size == 2
    assert pr.recipe("timestep", "wan", "action_dynamics", 20, 8, timestep=999).batch_size == 2
    # --batch-size stays an explicit override and is noted.
    text = _shell(capsys, "--line", "canonical", "--model", "wan", "--probe", "action_dynamics", "--layer", "20",
                  "--prefix", "8", "--batch-size", "2")
    assert _batch_overrides(text) == {"2"}
    assert "# note: batch size 2 was given; the accepted runs' batch size for this cell is 4" in text
    assert pr.default_batch_size("canonical", "streaming_depth") == 2


def test_minimax_subset_cells_note_the_split(capsys):
    published = {(int(row["layer"]), row["probe"]) for row in csv.DictReader(
        (RESULTS / "canonical_v2_val_seed42.csv").open(newline="", encoding="utf-8"))
        if row["model"] == "minimax_h3" and row["split_id"] == pr.MINIMAX_SUBSET_SPLIT}
    flagged = {(layer, probe) for layer in range(pr.NUM_LAYERS["minimax_h3"] + 1) for probe in pr.CANONICAL_PROBES
               if pr.is_minimax_subset_cell("minimax_h3", probe, layer)}
    assert flagged == published
    marker = "pass --split minimax_subset for the published recipe"
    for probe, layer in (("action_dynamics", 50), ("path_integration", 30), ("streaming_mask", 10)):
        base = ["--line", "canonical", "--model", "minimax_h3", "--probe", probe, "--layer", str(layer), "--prefix", "8"]
        text = _shell(capsys, *base)
        note = next(line for line in text.splitlines() if marker in line)
        assert note.startswith("# note: ") and pr.MINIMAX_SUBSET_SPLIT in note and "non-public" in note
        assert f"public {pr.PUBLIC_SPLIT} split" in note
        assert "split=canonical_v2 " in text
        if probe == "action_dynamics":
            assert "action_dynamics_minimax_h3_h124_v1" in text and "h124 C1 config" in note
            assert "h1-only" in note
        else:
            # Only C1 notes mention horizons; A/B/C2 cells are not h1-only.
            assert "h1" not in note
        assert marker not in _shell(capsys, *base, "--split", "public")
        subset = _shell(capsys, *base, "--split", "minimax_subset")
        assert marker not in subset and "${MINIMAX_SUBSET_MANIFEST_DIR}" in subset
        bash = _shell(capsys, *base, "--format", "bash")
        assert "RECIPE_NOTES=(" in bash and marker in bash
    # Layer 30 A1/A2/B2 and layer 10 C1 have no subset row: no note.
    for probe, layer in (("streaming_depth", 30), ("ego_belief_v2", 30), ("action_dynamics", 10)):
        assert not any(marker in note for note in pr.recipe("canonical", "minimax_h3", probe, layer, 8).notes)


def test_environment_scene_root_and_required_variables(capsys):
    a1 = pr.recipe("canonical", "dino", "streaming_depth", 0, 12)
    c2 = pr.recipe("canonical", "dino", "path_integration", 0, 12)
    assert pr.TARGET_ENV not in a1.required_env and pr.TARGET_ENV in c2.required_env
    for probe in ("streaming_mask", "view_consistency", "ego_belief_v2"):
        assert pr.TARGET_ENV not in pr.recipe("canonical", "qwen2_5_vl", probe, -1, 8).required_env
    assert pr.TARGET_ENV in pr.recipe("canonical", "dino", "action_dynamics", 0, 8).required_env
    text = _shell(capsys, "--line", "canonical", "--model", "dino", "--probe", "streaming_depth", "--layer", "0",
                  "--prefix", "12")
    assert "INSCENE_TARGET_FEAT_ROOT" not in text
    assert ('export INSCENE_LOCAL_SCENE_ROOT="${INSCENE_LOCAL_SCENE_ROOT:-${INSCENE_DATA_ROOT}/'
            'processed_scannetpp_v2/processed_scannetpp_v2}"') in text
    scene = a1.env["INSCENE_LOCAL_SCENE_ROOT"]
    nested = "/d/processed_scannetpp_v2/processed_scannetpp_v2"
    assert pr.expand(scene, {"INSCENE_DATA_ROOT": "/d"}) == nested
    assert pr.expand(scene, {"INSCENE_DATA_ROOT": "/d", "INSCENE_LOCAL_SCENE_ROOT": ""}) == nested
    assert pr.expand(scene, {"INSCENE_DATA_ROOT": "/d", "INSCENE_LOCAL_SCENE_ROOT": "/flat"}) == "/flat"
    assert pr.expand(scene, {}) == "${INSCENE_DATA_ROOT}/processed_scannetpp_v2/processed_scannetpp_v2"
    assert pr.expand("${A}/$B/${C}x", {"A": "1", "B": "2"}) == "1/2/${C}x"
    if shutil.which("bash"):  # the rendered default behaves the same in bash
        for preset, want in ((None, nested), ("/flat", "/flat")):
            env = {k: v for k, v in os.environ.items() if k != "INSCENE_LOCAL_SCENE_ROOT"}
            env["INSCENE_DATA_ROOT"] = "/d"
            if preset is not None:
                env["INSCENE_LOCAL_SCENE_ROOT"] = preset
            out = subprocess.run(["bash", "-c", f'printf %s "{scene}"'], env=env, capture_output=True, text=True)
            assert out.stdout == want == pr.expand(scene, env)


def test_recipe_text_has_no_internal_labels_and_states_the_fp16_basis(recipes, capsys):
    labels = re.compile(r"main552|wan168|ab16|20\d\d-\d\d|runner")
    bases = set()
    for _, rec in recipes:
        bases.add(rec.fp16_passthrough_basis)
        for text in (rec.fp16_passthrough_basis, *rec.notes, rec.family, pr.FAMILY_DESCRIPTIONS[rec.family]):
            assert not labels.search(text), text
    assert {basis.split(":", 1)[0] for basis in bases} == {
        "verified", "inferred, not verified", "verified with fp16 caches only"}
    text = _shell(capsys, "--line", "canonical", "--model", "dino", "--probe", "ego_belief_v2", "--layer", "0",
                  "--prefix", "12", "--format", "json")
    assert not labels.search(text) and json.loads(text)["family"] == "canonical_main"
    # Timestep MiniMax-H3 layer 15: accepted runs read fp16 caches; the released extractor writes bf16.
    for probe, value in (("view_consistency", "1"), ("ego_belief_v2", "0")):
        rec = pr.recipe("timestep", "minimax_h3", probe, 15, 8, timestep=1000)
        assert rec.env["PROBE_FP16_PASSTHROUGH"] == value
        note = next(n for n in rec.notes if "fp16 feature caches" in n)
        assert "bf16" in note and (("--fp16-passthrough 0" in note) == (value == "1"))
    other = pr.recipe("timestep", "minimax_h3", "view_consistency", 30, 8, timestep=1000)
    assert other.env["PROBE_FP16_PASSTHROUGH"] == "0" and other.fp16_passthrough_basis.startswith("no accepted run")
    # Notes for the accepted runs launched without INSCENE_PROBE_TASK and for LingBot layer 48.
    for rec in (pr.recipe("canonical", "minimax_h3", "streaming_mask", 30, 8),
                pr.recipe("timestep", "cogvideox", "ego_belief_v2", 20, 8, timestep=499)):
        assert any("without INSCENE_PROBE_TASK" in n for n in rec.notes)
    assert not any("INSCENE_PROBE_TASK" in n for n in pr.recipe("canonical", "dino", "ego_belief_v2", 0, 8).notes)
    lingbot = pr.recipe("canonical", "lingbot", "ego_belief_v2", 48, 12)
    assert any("passed feature_layer=48" in n for n in lingbot.notes)
    assert "feature_layer=47" in lingbot.overrides


def test_cli_formats(capsys):
    assert pr.main(["--line", "canonical", "--model", "lingbot", "--probe", "action_dynamics", "--layer", "48",
                    "--prefix", "8", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["disk_layer"] == 47 and payload["env"]["INSCENE_PROBE_TASK"] == "action_dynamics"
    assert payload["env"]["INSCENE_MANIFEST"] == "${CANONICAL_MANIFEST_DIR}/inscene_c_h124_prefix8.jsonl"
    assert payload["c3_args"][-1] == pr.C3_PROTOCOL_OVERRIDE
    assert pr.main(["--line", "timestep", "--model", "wan", "--probe", "ego_belief_v2", "--layer", "20",
                    "--prefix", "12", "--timestep", "999", "--format", "args"]) == 0
    args = capsys.readouterr().out.splitlines()
    assert args[0] == "experiment=inscene15k_streaming/ego_belief_v2_wan_v1" and "autoresume=false" in args
    assert pr.main(["--line", "canonical", "--model", "dino", "--probe", "streaming_mask", "--layer", "0",
                    "--prefix", "8", "--stage", "eval", "--ckpt", CKPT]) == 0
    shell = capsys.readouterr().out
    assert "export INSCENE_PROBE_TASK=\"streaming_mask\"" in shell
    assert f"python {pr.EVAL_A1} " in shell and f"\"ckpt_path='{CKPT}'\"" in shell
    assert pr.main(["--line", "timestep", "--model", "dino", "--probe", "streaming_depth", "--layer", "0",
                    "--prefix", "8", "--timestep", "999"]) == 2


def _cfg_job(entry: str, args: list[str], tmp_path: Path) -> str:
    env = {k: v for k, v in os.environ.items() if k != "PROJECT_ROOT"}
    env.update(INSCENE_DATA_ROOT=str(tmp_path), INSCENE_STREAMING_FEAT_ROOT=str(tmp_path),
               INSCENE_TARGET_FEAT_ROOT=str(tmp_path), WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, str(ROOT / entry), *args, "--cfg", "job"], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_recipe_commands_compose_through_the_entry_points(tmp_path):
    a1 = pr.recipe("canonical", "dino", "streaming_mask", 0, 16)
    text = _cfg_job(a1.evaluator, a1.eval_args(CKPT), tmp_path)
    assert f"ckpt_path: {CKPT}" in text and "eval_split: val" in text and "- identity" in text
    c3 = pr.recipe("timestep", "minimax_h3", "action_dynamics", 15, 8, timestep=1000)
    text = _cfg_job(pr.EVAL_DIAG, c3.c3_args(CKPT), tmp_path)
    assert "evaluation_protocol: c3_action_sensitivity" in text and "feature_timestep: 1000" in text
    assert re.search(r"action_horizons:\n- 1\n- 2\n- 4", text)
    qwen = pr.recipe("canonical", "qwen2_5_vl", "streaming_mask", -1, 24)
    text = _cfg_job(pr.TRAIN_ENTRY, qwen.train_args(), tmp_path)
    assert "feature_layer: -1" in text and "- identity" in text


def test_reproduce_rows_prints_a_runnable_script(monkeypatch):
    monkeypatch.delenv("MINIMAX_SUBSET_MANIFEST_DIR", raising=False)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert reproduce_rows.main(["--line", "canonical", "c3", "--model", "dino"]) == 0
    script = buffer.getvalue()
    assert script.count("vidfm3d/train.py") == 24
    assert script.count("vidfm3d/eval_diag.py") == 16 + 4   # A2/B2/C1/C2 evals + C3 of the 4 C1 runs
    assert script.count("vidfm3d/eval_a1_streaming.py") == 8
    assert "\"ckpt_path='${CKPT}'\"" in script and "newest_ckpt() {" in script
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    with redirect_stdout(io.StringIO()):
        assert reproduce_rows.main(["--line", "canonical", "--model", "minimax_h3", "--layer", "50"]) == 2
    monkeypatch.setenv("MINIMAX_SUBSET_MANIFEST_DIR", "/tmp/subset")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert reproduce_rows.main(["--line", "canonical", "--model", "minimax_h3", "--layer", "50",
                                    "--prefix", "8", "--probe", "action_dynamics"]) == 0
    assert "${MINIMAX_SUBSET_MANIFEST_DIR}/inscene_c_h124_prefix8.jsonl" in buffer.getvalue()
    assert "action_dynamics_minimax_h3_v1" in buffer.getvalue()


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="needs /bin/sh for the stub interpreter")
def test_reproduce_rows_run_resolves_paths_and_skips_only_final_checkpoints(tmp_path, monkeypatch, capsys):
    """``--run`` with a stub interpreter: relative paths, scene root, --skip-existing."""
    repo, caller = tmp_path / "repo", tmp_path / "caller"
    (caller / "canon").mkdir(parents=True)
    (caller / "canon" / pr.MANIFEST_NAME.format(prefix=12)).write_text("")
    log = tmp_path / "calls.log"
    stub = tmp_path / "stub.sh"
    stub.write_text('#!/bin/sh\nprintf \'%s|%s|%s|%s\\n\' "$PWD" "$INSCENE_MANIFEST" "$INSCENE_LOCAL_SCENE_ROOT" '
                    '"$*" >> "$STUB_LOG"\n')
    stub.chmod(0o755)
    monkeypatch.setattr(reproduce_rows, "ROOT", repo)
    monkeypatch.chdir(caller)
    for name in ("INSCENE_TARGET_FEAT_ROOT", "INSCENE_LOCAL_SCENE_ROOT", "MINIMAX_SUBSET_MANIFEST_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("INSCENE_DATA_ROOT", "data")
    monkeypatch.setenv("INSCENE_STREAMING_FEAT_ROOT", "feat")
    monkeypatch.setenv("CANONICAL_MANIFEST_DIR", "canon")
    key = "dino__scannetpp__t749__l0__p12__view_consistency__seed42"   # A2: no target cache needed
    rec = pr.recipe("canonical", "dino", "view_consistency", 0, 12)
    ckpts = repo / rec.run_dir / "VideoProbe3D" / "abc" / "checkpoints"
    ckpts.mkdir(parents=True)

    def calls(*steps: str) -> list[list[str]]:
        log.write_text("")
        argv = ["--results-dir", str(RESULTS), "--key", key, "--run", "--skip-existing", "--python", str(stub),
                "--steps", ",".join(steps)]
        assert reproduce_rows.main(argv) == 0
        return [line.split("|") for line in log.read_text().splitlines()]

    # An interrupted run (epoch 3 of 50) is trained again, with absolute paths and cwd=ROOT.
    (ckpts / "epoch=3-step=4888.ckpt").write_text("")
    (call,) = calls("train")
    assert call[0] == str(repo) and pr.TRAIN_ENTRY in call[3]
    assert call[1] == str(caller / "canon" / pr.MANIFEST_NAME.format(prefix=12))
    assert call[2] == str(caller / "data" / pr.SCENE_SUBDIR)
    assert "not the final epoch 49; training again" in capsys.readouterr().out
    # Its evaluation runs, with a warning, because the checkpoint is not final.
    (call,) = calls("eval")
    assert "epoch=3-step=4888.ckpt" in call[3]
    assert "is not the final-epoch checkpoint" in capsys.readouterr().err
    # A final-epoch checkpoint skips training; a summary of that checkpoint skips the evaluation.
    final = ckpts / "epoch=49-step=61100.ckpt"
    final.write_text("")
    os.utime(final, (os.path.getmtime(ckpts / "epoch=3-step=4888.ckpt") + 10,) * 2)
    assert calls("train") == []
    summary = repo / rec.run_dir / "eval" / "val_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"ckpt_path": str(ckpts / "epoch=3-step=4888.ckpt")}))
    (call,) = calls("eval")                       # summary of an older checkpoint: evaluate again
    assert f"ckpt_path='{final}'" in call[3]
    summary.write_text(json.dumps({"ckpt_path": str(final)}))
    assert calls("eval") == []
    # A scene root set by the caller is kept (and made absolute).
    monkeypatch.setenv("INSCENE_LOCAL_SCENE_ROOT", "scenes")
    final.unlink()
    (call,) = calls("train")
    assert call[2] == str(caller / "scenes")
    # C rows still need the target cache.
    with pytest.raises(SystemExit, match="INSCENE_TARGET_FEAT_ROOT"):
        reproduce_rows.main(["--results-dir", str(RESULTS), "--key",
                             "dino__scannetpp__t749__l0__p12__path_integration__seed42", "--run",
                             "--python", str(stub), "--steps", "train"])


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sha256sum") is None,
                    reason="needs bash and coreutils")
def test_sweep_dry_run_as_in_ci(tmp_path):
    script = ROOT / "scripts" / "run_streaming_probe_sweep.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    canon = tmp_path / "canon"
    canon.mkdir()
    for prefix in pr.PREFIXES:
        (canon / pr.MANIFEST_NAME.format(prefix=prefix)).write_text("")
    env = dict(os.environ, DRY_RUN="1", VFMS="dino", LINE="canonical", FEAT_BASE=str(tmp_path / "feat"),
               INSCENE_DATA_ROOT=str(tmp_path / "data"), CANONICAL_MANIFEST_DIR=str(canon),
               EXTRACT_STREAMING="0", EXTRACT_TARGETS="0", PYTHON=sys.executable)
    out = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    commands = [line for line in out.stdout.splitlines() if line.startswith("[dry-run]")]
    assert sum("vidfm3d/train.py" in c for c in commands) == 24
    assert sum("vidfm3d/eval_a1_streaming.py" in c for c in commands) == 8
    assert sum("vidfm3d/eval_diag.py" in c for c in commands) == 16 + 4
    assert all("INSCENE_MANIFEST=" in c and "INSCENE_PROBE_TASK=" in c for c in commands)
    assert not any("last.ckpt" in c for c in commands)


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sha256sum") is None,
                    reason="needs bash and coreutils")
def test_sweep_uses_accepted_batches_ledger_layers_and_notes(tmp_path):
    script = ROOT / "scripts" / "run_streaming_probe_sweep.sh"
    canon = tmp_path / "canon"
    canon.mkdir()
    for prefix in pr.PREFIXES:
        (canon / pr.MANIFEST_NAME.format(prefix=prefix)).write_text("")

    def dry_run(**extra: str) -> str:
        env = dict(os.environ, DRY_RUN="1", LINE="canonical", FEAT_BASE=str(tmp_path / "feat"),
                   INSCENE_DATA_ROOT=str(tmp_path / "data"), CANONICAL_MANIFEST_DIR=str(canon), PREFIX_LENGTHS="8",
                   EXTRACT_STREAMING="0", EXTRACT_TARGETS="0", EVAL="0", EVAL_C3="0", PYTHON=sys.executable, **extra)
        out = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout

    def batches(text: str) -> dict[str, int]:
        found = re.findall(r"==== train (\S+) \(\S+, batch (\d+)\)", text)
        return {key.split("__")[0] + ":" + key.split("__")[5] + ":" + key.split("__")[3]: int(b) for key, b in found}

    # LAYERS=default: wan 20, dino 0.
    assert batches(dry_run(VFMS="wan dino", LAYERS="default",
                           PROBES="streaming_depth streaming_mask ego_belief_v2 action_dynamics path_integration")) == {
        "wan:streaming_depth:l20": 1, "wan:streaming_mask:l20": 1, "wan:ego_belief_v2:l20": 1,
        "wan:action_dynamics:l20": 4, "wan:path_integration:l20": 4,
        "dino:streaming_depth:l0": 2, "dino:streaming_mask:l0": 2, "dino:ego_belief_v2:l0": 2,
        "dino:action_dynamics:l0": 2, "dino:path_integration:l0": 2,
    }
    text = dry_run(VFMS="minimax_h3", LAYERS="30 last", PROBES="streaming_depth action_dynamics")
    assert batches(text) == {"minimax_h3:streaming_depth:l30": 1, "minimax_h3:action_dynamics:l30": 2,
                             "minimax_h3:streaming_depth:l50": 2, "minimax_h3:action_dynamics:l50": 2}
    notes = [line for line in text.splitlines() if line.startswith("[note]")]
    subset = [n for n in notes if "--split minimax_subset" in n]
    assert {n.split(":")[0] for n in subset} == {
        "[note] minimax_h3__scannetpp__t250__l30__p8__action_dynamics__seed42",
        "[note] minimax_h3__scannetpp__t250__l50__p8__streaming_depth__seed42",
        "[note] minimax_h3__scannetpp__t250__l50__p8__action_dynamics__seed42",
    }
    text = dry_run(VFMS="lingbot", LAYERS="last", PROBES="streaming_mask", BATCH_SIZE="3")
    assert "lingbot__scannetpp__t250__l48__p8__streaming_mask__seed42" in text and "__l47__" not in text
    assert batches(text) == {"lingbot:streaming_mask:l48": 3}
