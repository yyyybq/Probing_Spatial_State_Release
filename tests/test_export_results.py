"""scripts/export_results.py turns evaluator summaries back into public table rows.

Synthetic summaries (the fields vidfm3d/eval_a1_streaming.py and
vidfm3d/eval_diag.py write) are placed in a temporary runs root; the exporter
must reproduce the public schema, table layers, lines, inferred batch sizes,
C3 variants, caveats and acceptance codes, read the trained horizons from
run_manifest.json before the Hydra record, and ``--compare`` must join them with
results/, report the primary-metric deltas and flag split or variant mismatches.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_results as er  # noqa: E402
import probe_recipe as pr  # noqa: E402

RESULTS = ROOT / "results"
A1_VERSION = "a1_streaming_eval_v2_nonfinite_aware"
DIAG_VERSION = "eval_diag_v2_nonfinite_aware"


def _published(name: str, key: str) -> dict:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return next(row for row in csv.DictReader(handle) if row["unique_key"] == key)


def _header(name: str) -> list[str]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _run(root: Path, rec: pr.Recipe, *, horizons=None, seed=42, manifest=None, eval_record=False) -> Path:
    """A run folder with a Hydra record (optionally rewritten by an evaluation) and run_manifest.json."""
    run = root / rec.run_name
    (run / "eval").mkdir(parents=True)
    (run / ".hydra").mkdir()
    lines = [f"seed: {seed}", f"prefix_len: {rec.prefix}"]
    if horizons is not None:
        lines.append("action_horizons:")
        lines += [f"- {h}" for h in horizons]
    if eval_record:
        lines += ["train: false", "eval_split: val", "evaluation_protocol: c3_action_sensitivity"]
    (run / ".hydra" / "config.yaml").write_text("\n".join(lines) + "\n")
    if manifest is not None:
        _write(run / "run_manifest.json", {"schema_version": 1, "config": manifest})
    return run


def _identity(rec: pr.Recipe, ckpt: Path) -> dict:
    return {"vfm_name": rec.model, "feature_layer": rec.disk_layer, "feature_timestep": rec.timestep,
            "feat_postfix": f"_t{rec.timestep}_layer{rec.disk_layer}", "prefix_len": rec.prefix,
            "ckpt_path": str(ckpt)}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    # Timestep A1 (Wan t999 L20 p8, batch 1): published value + 0.01.
    a1 = pr.recipe("timestep", "wan", "streaming_depth", 20, 8, timestep=999)
    a1_pub = _published("timestep_val_seed42.csv", a1.unique_key)
    run = _run(root, a1)
    ckpt = run / "VideoProbe3D" / "abc" / "checkpoints" / "epoch=99-step=244500.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    _write(run / "eval" / "val_summary.json", {
        **_identity(a1, ckpt), "probe_type": "streaming_depth", "evaluator_version": A1_VERSION,
        "mean_loss_depth": float(a1_pub["primary_value"]) + 0.01, "mean_Auc_30": 0.1, "mean_Auc_5": 0.01,
        "n_samples": 276, "n_valid_prediction": 276, "n_nonfinite_prediction": 0})
    # Canonical LingBot table layer L48 (disk 47) C1 with its C3, trained on h1/h2/h4.
    c1 = pr.recipe("canonical", "lingbot", "action_dynamics", 48, 8)
    c1_pub = _published("canonical_v2_val_seed42.csv", c1.unique_key)
    run = _run(root, c1, horizons=[1, 2, 4])
    ckpt = run / "VideoProbe3D" / "x" / "checkpoints" / "epoch=49-step=61100.ckpt"
    _write(run / "eval" / "val_summary.json", {
        **_identity(c1, ckpt), "probe_type": "action_dynamics", "evaluation_protocol": "action_dynamics",
        "evaluator_version": DIAG_VERSION, "global_R@1": float(c1_pub["primary_value"]), "global_R@5": 0.1,
        "mean_rank": 3.0, "mean_cos": 0.9, "no_action_global_R@1": 0.0, "shuffled_action_global_R@1": 0.0,
        "bootstrap_95ci": {"global_R@1": {"low": 0.0, "high": 1.0}},
        "loader_n_samples": 828, "n_scored": 828, "n_nonfinite_pred": 0})
    c3_key = pr.unique_key("lingbot", 250, 48, 8, "c3_from_c1")
    c3_pub = _published("c3_from_c1_val_seed42.csv", c3_key)
    _write(run / "eval" / "val_c3_action_sensitivity_summary.json", {
        **_identity(c1, ckpt), "probe_type": "action_dynamics", "evaluation_protocol": "c3_action_sensitivity",
        "evaluator_version": DIAG_VERSION, "intervention_validity": float(c3_pub["primary_value"]) + 0.05,
        "n_valid_action_sets": 276, "n_valid_candidates": 828, "n_nonfinite_pred_records": 0,
        "loader_n_samples": 828, "bootstrap_95ci": {"intervention_validity": [0.3, 0.4]}})
    # Timestep C1 trained on h1 only, evaluated as C3 with action_horizons=[1,2,4].
    h1 = pr.recipe("timestep", "cogvideox", "action_dynamics", 15, 12, timestep=499)
    run = _run(root, h1, horizons=[1])
    ckpt = run / "VideoProbe3D" / "y" / "checkpoints" / "epoch=49-step=61100.ckpt"
    _write(run / "eval" / "val_c3_action_sensitivity_summary.json", {
        **_identity(h1, ckpt), "probe_type": "action_dynamics", "evaluation_protocol": "c3_action_sensitivity",
        "evaluator_version": DIAG_VERSION, "intervention_validity": 0.38, "n_valid_action_sets": 276,
        "n_nonfinite_pred_records": 0, "loader_n_samples": 828})
    # Canonical DINO B2 with non-finite predictions, batch 4 (step 611 per epoch).
    b2 = pr.recipe("canonical", "dino", "ego_belief_v2", 0, 12)
    run = _run(root, b2)
    ckpt = run / "VideoProbe3D" / "z" / "checkpoints" / "epoch=49-step=30550.ckpt"
    _write(run / "eval" / "val_summary.json", {
        **_identity(b2, ckpt), "probe_type": "ego_belief_v2", "evaluation_protocol": "ego_belief_v2",
        "evaluator_version": DIAG_VERSION, "mean_ang_err_deg": float("nan"), "loader_n_samples": 276,
        "n_scored": 171, "n_nonfinite_pred": 2})
    return root


def test_export_reproduces_the_public_schema(runs, tmp_path):
    tables = er.export(runs)
    assert {name: len(rows) for name, rows in tables.items()} == {"canonical_v2": 2, "timestep": 1, "c3_from_c1": 2}
    out = tmp_path / "out"
    written = er.write_tables(tables, out)
    for path in written:
        public = "c3_from_c1_val_seed42.csv" if path.name == "c3_from_c1.csv" else "canonical_v2_val_seed42.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == _header(public)

    a1 = tables["timestep"][0]
    assert a1["unique_key"] == "wan__scannetpp__t999__l20__p8__streaming_depth__seed42"
    assert (a1["line"], a1["split_id"], a1["status"]) == ("timestep", "canonical_v2", "ACCEPTED")
    assert (a1["ckpt_epoch"], a1["ckpt_step"], a1["train_batch_size_inferred"]) == (99, 244500, 1)
    assert a1["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert a1["n_loader"] == 276 and a1["n_scored"] == 276 and a1["n_nonfinite_pred"] == 0
    assert json.loads(a1["secondary_json"]) == {"mean_Auc_30": 0.1, "mean_Auc_5": 0.01}

    by_key = {row["unique_key"]: row for row in tables["canonical_v2"]}
    c1 = by_key["lingbot__scannetpp__t250__l48__p8__action_dynamics__seed42"]
    assert (c1["layer"], c1["train_batch_size_inferred"], c1["status"]) == (48, 2, "ACCEPTED")
    assert (c1["primary_ci_low"], c1["primary_ci_high"]) == (0.0, 1.0)
    assert c1["checkpoint_sha256"] == ""  # checkpoint file absent
    assert "REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS" in c1["caveats"].split(";")

    b2 = by_key["dino__scannetpp__t749__l0__p12__ego_belief_v2__seed42"]
    codes = b2["caveats"].split(";")
    assert b2["status"] == "REJECTED"
    assert {"REJECTED_PRIMARY_NOT_FINITE", "REJECTED_2_NONFINITE_PREDICTIONS"} <= set(codes)
    assert {"HIDDEN_OBJECT_SCORED_171_OF_276", "TRAIN_BATCH_SIZE_4_NOT_CANONICAL_2"} <= set(codes)

    c3 = {row["unique_key"]: row for row in tables["c3_from_c1"]}
    protocol = c3["lingbot__scannetpp__t250__l48__p8__c3_from_c1__seed42"]
    assert protocol["c1_unique_key"] == c1["unique_key"] and protocol["variant"] == "c3_from_c1"
    assert (protocol["primary_metric"], protocol["n_scored"]) == ("intervention_validity", 276)
    assert (protocol["primary_ci_low"], protocol["primary_ci_high"]) == (0.3, 0.4)
    diagnostic = c3["cogvideox__scannetpp__t499__l15__p12__c3_from_c1__seed42"]
    assert diagnostic["variant"] == "h1_trained_diagnostic"
    assert "H1_TRAINED_NOT_PROTOCOL_C3" in diagnostic["caveats"].split(";")


def test_compare_joins_with_the_published_tables(runs, capsys):
    deltas = {e["unique_key"]: e for e in er.compare(er.export(runs, hash_checkpoints=False), RESULTS)}
    assert deltas["wan__scannetpp__t999__l20__p8__streaming_depth__seed42"]["delta"] == pytest.approx(0.01)
    assert deltas["lingbot__scannetpp__t250__l48__p8__action_dynamics__seed42"]["delta"] == pytest.approx(0.0)
    assert deltas["lingbot__scannetpp__t250__l48__p8__c3_from_c1__seed42"]["delta"] == pytest.approx(0.05)
    assert all(e["published_key"] == key for key, e in deltas.items())

    assert er.main(["--runs-root", str(runs), "--no-checkpoint-hash", "--compare", str(RESULTS)]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].split("\t")[:5] == ["unique_key", "primary_metric", "local", "published", "delta"]
    assert len(lines) == 1 + 5
    assert "5 matched, 0 without a published row" in captured.err


PACK_CODES = {"TRAINING_AND_EVAL_PACK_COMMITS_NOT_ON_GITHUB", "TRAINING_CODE_SNAPSHOT_IDENTITY_20261017",
              "FP32_RETRAIN_REPLACES_NONFINITE_CHECKPOINT"}


def _codes(row: dict) -> list[str]:
    return [code for code in row["caveats"].split(";") if code and code not in PACK_CODES]


def _summary_like(pub: dict, rec: pr.Recipe, ckpt: Path, *, c3: bool) -> dict:
    """A summary that reproduces one published row."""
    out = {**_identity(rec, ckpt), "probe_type": rec.probe, "evaluator_version": pub["evaluator_version"],
           pub["primary_metric"]: float(pub["primary_value"]), "loader_n_samples": int(pub["n_loader"]),
           "n_scored": int(pub["n_scored"]), "n_nonfinite_pred": 0}
    if rec.evaluator == pr.EVAL_DIAG:
        out["evaluation_protocol"] = "c3_action_sensitivity" if c3 else rec.probe
    return out


def test_exported_caveats_follow_the_published_convention(tmp_path):
    """Control caveat on C1/C2 rows only (not C3); same codes and order as results/ minus pack codes."""
    root = tmp_path / "runs"
    cells = [  # (recipe, publishes a C3 row from this C1 run)
        (pr.recipe("canonical", "dino", "path_integration", 0, 8), False),
        (pr.recipe("canonical", "dino", "ego_belief_v2", 0, 12), False),
        (pr.recipe("canonical", "wan", "streaming_depth", 20, 8), False),
        (pr.recipe("canonical", "wan", "action_dynamics", 20, 12), True),
        (pr.recipe("canonical", "minimax_h3", "streaming_depth", 10, 8, split="minimax_subset"), False),
        (pr.recipe("canonical", "minimax_h3", "action_dynamics", 20, 8, split="minimax_subset"), True),
        (pr.recipe("timestep", "cogvideox", "action_dynamics", 15, 12, timestep=499), True),
        (pr.recipe("timestep", "wan", "path_integration", 20, 8, timestep=999), False),
    ]
    published = {}
    for rec, has_c3 in cells:
        table = "timestep_val_seed42.csv" if rec.line == "timestep" else "canonical_v2_val_seed42.csv"
        pub = _published(table, rec.unique_key)
        run = _run(root, rec, manifest={"seed": 42, "action_horizons": rec.trained_action_horizons}
                   if rec.trained_action_horizons else {"seed": 42})
        ckpt = run / "VideoProbe3D" / "id" / "checkpoints" / f"epoch={pub['ckpt_epoch']}-step={pub['ckpt_step']}.ckpt"
        _write(run / "eval" / "val_summary.json", _summary_like(pub, rec, ckpt, c3=False))
        published[rec.unique_key] = pub
        if has_c3:
            c3_key = pr.unique_key(rec.model, rec.timestep, rec.layer, rec.prefix, "c3_from_c1")
            c3_pub = _published("c3_from_c1_val_seed42.csv", c3_key)
            _write(run / "eval" / "val_c3_action_sensitivity_summary.json", {
                **_summary_like(c3_pub, rec, ckpt, c3=True), "n_valid_action_sets": int(c3_pub["n_scored"])})
            published[c3_key] = c3_pub
    rows = [row for rows in er.export(root, hash_checkpoints=False).values() for row in rows]
    assert {row["unique_key"] for row in rows} == set(published)
    control = "REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS"
    for row in rows:
        pub = published[row["unique_key"]]
        assert row["status"] == "ACCEPTED", row
        for field in ("split_id", "variant", "train_batch_size_inferred", "ckpt_epoch", "n_loader"):
            assert str(row[field]) == pub[field], (row["unique_key"], field)
        assert _codes(row) == _codes(pub), row["unique_key"]
        assert (control in _codes(row)) == (row["probe"] in pr.C_PROBES), row["unique_key"]


def test_trained_horizons_come_from_the_run_manifest(tmp_path):
    """An evaluation that rewrote .hydra/config.yaml does not relabel an h1-trained C1 run."""
    def export_one(**run_kwargs) -> dict[str, dict]:
        root = tmp_path / f"runs{len(list(tmp_path.iterdir()))}"
        rec = pr.recipe("timestep", "wan", "action_dynamics", 20, 8, timestep=999)
        run = _run(root, rec, **run_kwargs)
        ckpt = run / "VideoProbe3D" / "id" / "checkpoints" / "epoch=49-step=61100.ckpt"
        _write(run / "eval" / "val_summary.json", {
            **_identity(rec, ckpt), "probe_type": "action_dynamics", "evaluation_protocol": "action_dynamics",
            "evaluator_version": DIAG_VERSION, "global_R@1": 0.1, "loader_n_samples": 276, "n_scored": 276,
            "n_nonfinite_pred": 0})
        _write(run / "eval" / "val_c3_action_sensitivity_summary.json", {
            **_identity(rec, ckpt), "probe_type": "action_dynamics", "evaluation_protocol": "c3_action_sensitivity",
            "evaluator_version": DIAG_VERSION, "intervention_validity": 0.4, "n_valid_action_sets": 276,
            "n_nonfinite_pred_records": 0, "loader_n_samples": 828})
        tables = er.export(root, hash_checkpoints=False)
        return {"c1": tables["timestep"][0], "c3": tables["c3_from_c1"][0]}

    def h1_labelled(rows: dict[str, dict]) -> bool:
        return (rows["c3"]["variant"] == "h1_trained_diagnostic"
                and "H1_TRAINED_NOT_PROTOCOL_C3" in rows["c3"]["caveats"].split(";")
                and "C1_TRAINED_AND_EVALUATED_H1_ONLY" in rows["c1"]["caveats"].split(";"))

    # The documented C3 command without hydra.output_subdir=null leaves [1, 2, 4] in .hydra/config.yaml.
    rewritten = dict(horizons=[1, 2, 4], eval_record=True)
    assert h1_labelled(export_one(**rewritten, manifest={"seed": 42, "action_horizons": [1]}))
    assert h1_labelled(export_one(**rewritten))            # no manifest: the record is ignored, recipe horizons
    assert er.training_config(tmp_path / "missing") == {}
    # A manifest wins over the Hydra record; an untouched Hydra record is still read.
    protocol = export_one(horizons=[1], manifest={"seed": 7, "action_horizons": [1, 2, 4]})
    assert protocol["c3"]["variant"] == "c3_from_c1" and protocol["c3"]["seed"] == 7
    assert "C1_TRAINED_AND_EVALUATED_H1_ONLY" not in protocol["c1"]["caveats"]
    assert export_one(horizons=[1, 2, 4])["c3"]["variant"] == "c3_from_c1"


def test_non_final_checkpoints_are_rejected(tmp_path):
    root = tmp_path / "runs"
    rec = pr.recipe("canonical", "dino", "view_consistency", 0, 12)
    pub = _published("canonical_v2_val_seed42.csv", rec.unique_key)
    run = _run(root, rec)
    for name, rejected in (("epoch=3-step=4888.ckpt", True), ("epoch=49-step=61100.ckpt", False),
                           ("weights.ckpt", True)):
        _write(run / "eval" / "val_summary.json", _summary_like(pub, rec, run / "ck" / name, c3=False))
        (row,) = er.export(root, hash_checkpoints=False)["canonical_v2"]
        assert ("REJECTED_NOT_FINAL_EPOCH" in row["caveats"].split(";")) == rejected, name
        assert row["status"] == ("REJECTED" if rejected else "ACCEPTED"), name
    a1 = pr.recipe("canonical", "dino", "streaming_depth", 0, 12)
    run = _run(root, a1)
    _write(run / "eval" / "val_summary.json", {
        **_identity(a1, run / "ck" / "epoch=49-step=61100.ckpt"), "probe_type": "streaming_depth",
        "evaluator_version": A1_VERSION, "mean_loss_depth": 0.3, "n_samples": 276, "n_valid_prediction": 276,
        "n_nonfinite_prediction": 0})
    row = next(r for r in er.export(root, hash_checkpoints=False)["canonical_v2"] if r["probe"] == "streaming_depth")
    assert row["status"] == "REJECTED" and "REJECTED_NOT_FINAL_EPOCH" in row["caveats"]   # A1 ends at epoch 99


def test_compare_flags_split_and_variant_mismatches(tmp_path, capsys):
    """A public-split rerun of a MiniMax-H3 subset row is matched but flagged SPLIT_MISMATCH."""
    root = tmp_path / "runs"
    rec = pr.recipe("canonical", "minimax_h3", "view_consistency", 10, 8)          # public split
    assert rec.split == pr.PUBLIC_SPLIT
    run = _run(root, rec)
    _write(run / "eval" / "val_summary.json", {
        **_identity(rec, run / "ck" / "epoch=49-step=61100.ckpt"), "probe_type": "view_consistency",
        "evaluation_protocol": "view_consistency", "evaluator_version": DIAG_VERSION, "balanced_acc": 0.7,
        "loader_n_samples": 276, "n_scored": 276, "n_nonfinite_pred": 0})
    # An h1-trained C1 on the public split, compared with the protocol C3 row of the same cell.
    c1 = pr.recipe("canonical", "dino", "action_dynamics", 0, 8)
    run = _run(root, c1, manifest={"seed": 42, "action_horizons": [1]})
    _write(run / "eval" / "val_c3_action_sensitivity_summary.json", {
        **_identity(c1, run / "ck" / "epoch=49-step=61100.ckpt"), "probe_type": "action_dynamics",
        "evaluation_protocol": "c3_action_sensitivity", "evaluator_version": DIAG_VERSION,
        "intervention_validity": 0.4, "n_valid_action_sets": 276, "n_nonfinite_pred_records": 0,
        "loader_n_samples": 828})
    entries = {e["unique_key"]: e for e in er.compare(er.export(root, hash_checkpoints=False), RESULTS)}
    split = entries[rec.unique_key]
    assert split["published_key"] == rec.unique_key and split["flags"] == "SPLIT_MISMATCH"
    assert (split["split_id"], split["published_split_id"]) == (pr.PUBLIC_SPLIT, pr.MINIMAX_SUBSET_SPLIT)
    variant = entries[pr.unique_key("dino", 749, 0, 8, "c3_from_c1")]
    assert (variant["variant"], variant["published_variant"]) == ("h1_trained_diagnostic", "c3_from_c1")
    assert variant["flags"] == "VARIANT_MISMATCH"

    assert er.main(["--runs-root", str(root), "--no-checkpoint-hash", "--compare", str(RESULTS)]) == 0
    captured = capsys.readouterr()
    header, *lines = captured.out.splitlines()
    fields = header.split("\t")
    assert fields[-5:] == ["split_id", "published_split_id", "variant", "published_variant", "flags"]
    printed = {line.split("\t")[0]: dict(zip(fields, line.split("\t"))) for line in lines}
    assert printed[rec.unique_key]["flags"] == "SPLIT_MISMATCH"
    assert printed[rec.unique_key]["published_split_id"] == pr.MINIMAX_SUBSET_SPLIT
    assert "2 matched, 0 without a published row, 2 flagged" in captured.err


def test_batch_inference_and_layer_mapping():
    assert er.infer_batch_size(49, 61100, 2445) == 2
    assert er.infer_batch_size(99, 244500, 2445) == 1
    assert er.infer_batch_size(49, 30550, 2445) == 4
    assert er.infer_batch_size(49, 761 * 50, 1523) == 2
    assert er.infer_batch_size(49, 12345, 2445) is None
    assert pr.ledger_layer("minimax_h3", 49) == 50 and pr.ledger_layer("minimax_h3", 30) == 30
