#!/usr/bin/env python3
"""Export your own evaluation summaries into the public results schema.

Finds ``<runs-root>/**/eval/val_summary.json`` (A1, A2, B1, B2, C1, C2) and
``<runs-root>/**/eval/val_c3_action_sensitivity_summary.json`` (C3 from C1),
written by ``vidfm3d/eval_a1_streaming.py`` and ``vidfm3d/eval_diag.py``, and
writes one CSV per table in the schema of ``results/*_val_seed42.csv``:

    <out>/canonical_v2.csv   <out>/timestep.csv   <out>/c3_from_c1.csv

Acceptance follows the published tables: a row is ACCEPTED when the summary
comes from a v2 evaluator, the primary metric is finite, it records zero
non-finite predictions and at least one scored record, and the evaluated
checkpoint is the final epoch (epoch 99 for A1, 49 otherwise; a checkpoint name
without ``epoch=E`` cannot be checked and is rejected too).  Rejected rows carry
``REJECTED_*`` codes in ``caveats``.  (The published tables also checked each
summary hash against an evaluation receipt; local runs have no receipts.)

Row identity comes from the summary (``vfm_name``, ``probe_type`` /
``evaluation_protocol``, ``feature_layer``, ``prefix_len``,
``feature_timestep``, ``ckpt_path``) and from the training run: the seed and
the trained ``action_horizons`` are read from ``<run>/run_manifest.json``, which
only ``vidfm3d/train.py`` writes, and otherwise from ``<run>/.hydra/config.yaml``.
An evaluation that reuses the run folder without ``hydra.output_subdir=null``
rewrites that Hydra record, so its ``action_horizons`` are ignored when it is an
evaluation record, and the horizons the released recipe trains are used.  Disk
layers 47 (LingBot) and 49 (MiniMax-H3) are reported as table layers 48 and
50.  The line is ``canonical_v2`` when the timestep is the model's canonical one
(749, or 250 for LingBot and MiniMax-H3) and ``timestep`` otherwise.  Caveat
codes follow the published convention; codes that describe the original
evaluation packs are not added.  Fields that a local run cannot know
(evaluation pack, code tag, campaign scope, revision, checkpoint replacement)
are left empty.

``--compare <dir>`` joins the exported rows with the tables in ``<dir>`` (for
example ``results/``) on (model, probe, layer, prefix, timestep), prints the
primary-metric deltas and the ``split_id`` and ``variant`` of both sides, and
flags ``SPLIT_MISMATCH`` (different validation splits, e.g. a public-split
rerun of a MiniMax-H3 subset row), ``SPLIT_UNKNOWN`` (the local split could not
be determined) and ``VARIANT_MISMATCH`` (a C3 row compared with the other C3
variant).  Flagged deltas do not compare like with like.

Example::

    python scripts/export_results.py --runs-root logs/inscene15k_streaming/runs \\
        --out results/local --compare results/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_recipe as pr  # noqa: E402

COLUMNS = [
    "unique_key", "line", "model", "probe", "variant", "layer", "prefix", "timestep", "seed",
    "eval_split", "split_id", "scope", "status", "primary_metric", "primary_value",
    "primary_ci_low", "primary_ci_high", "secondary_json", "n_loader", "n_scored",
    "n_nonfinite_pred", "evaluator_version", "eval_revision", "eval_pack", "training_code_tag",
    "ckpt_epoch", "ckpt_step", "train_batch_size_inferred", "checkpoint_replaced",
    "checkpoint_sha256", "summary_file", "summary_sha256", "caveats",
]
C3_COLUMNS = COLUMNS + ["c1_unique_key"]
TABLE_FILES = {"canonical_v2": "canonical_v2.csv", "timestep": "timestep.csv", "c3_from_c1": "c3_from_c1.csv"}
SUMMARY_FILES = {"val_summary.json": False, "val_c3_action_sensitivity_summary.json": True}
SECONDARY = {
    "streaming_depth": ["mean_Auc_30", "mean_Auc_5"],
    "streaming_mask": [],
    "view_consistency": ["overlap_mae", "pos_acc", "neg_acc"],
    "ego_belief_v2": ["mean_az_err_deg", "mean_el_err_deg", "mean_log_dist_err", "top1", "top3"],
    "ego_belief": ["mean_el_err_deg", "mean_log_dist_err"],
    "action_dynamics": ["global_R@5", "mean_rank", "mean_cos", "no_action_global_R@1", "shuffled_action_global_R@1"],
    "path_integration": ["global_R@5", "mean_rank", "mean_cos", "final_pose_error"],
    "c3_from_c1": ["intervention_chance", "intervention_validity_chance_adjusted", "exact_candidate_validity",
                   "intervention_margin", "counterfactual_consistency", "action_sensitivity_cosine",
                   "target_sensitivity_cosine", "target_to_prediction_sensitivity_pearson",
                   "sensitivity_calibration_mae", "n_valid_action_sets", "n_valid_candidates"],
}
# Field names differ by evaluator and probe.
NONFINITE_FIELDS = ("n_nonfinite_pred", "n_nonfinite_pred_records", "n_nonfinite_prediction")
SCORED_FIELDS = ("n_scored", "n_valid_prediction", "n_valid_records", "n_valid_action_sets")
CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)")
JOIN_FIELDS = ("model", "probe", "layer", "prefix", "timestep")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def first_present(summary: dict, names: tuple[str, ...]):
    return next((summary[name] for name in names if name in summary), None)


def infer_batch_size(epoch: int, step: int, n_train: int, max_batch: int = 64) -> int | None:
    """Batch B with floor(n_train / B) * (epoch + 1) == step (last partial batch dropped)."""
    matches = [b for b in range(1, max_batch + 1) if (n_train // b) * (epoch + 1) == step]
    return matches[0] if len(matches) == 1 else None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _manifest_config(run_dir: Path) -> dict:
    """Seed and trained horizons from ``<run>/run_manifest.json`` (written by train.py only)."""
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8")).get("config")
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(config, dict):
        return {}
    out = {}
    if _is_int(config.get("seed")):
        out["seed"] = config["seed"]
    horizons = config.get("action_horizons")
    if isinstance(horizons, list) and horizons and all(_is_int(h) for h in horizons):
        out["action_horizons"] = list(horizons)
    return out


def _hydra_config(run_dir: Path) -> dict:
    """Seed and horizons from ``<run>/.hydra/config.yaml``.

    Evaluations that reuse the run folder without ``hydra.output_subdir=null``
    overwrite this file with their own config (it then has ``eval_split`` or
    ``evaluation_protocol``); its ``action_horizons`` are then the evaluation's
    and are ignored.
    """
    path = run_dir / ".hydra" / "config.yaml"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    seed = re.search(r"^seed:\s*(-?\d+)\s*$", text, re.M)
    if seed:
        out["seed"] = int(seed.group(1))
    if re.search(r"^(eval_split|evaluation_protocol):", text, re.M):
        return out
    horizons = re.search(r"^action_horizons:\s*\[([^\]]*)\]", text, re.M)
    if horizons:
        out["action_horizons"] = [int(v) for v in re.findall(r"-?\d+", horizons.group(1))]
    else:
        block = re.search(r"^action_horizons:\s*\n((?:- .*\n)+)", text, re.M)
        if block:
            out["action_horizons"] = [int(v) for v in re.findall(r"-?\d+", block.group(1))]
    return out


def training_config(run_dir: Path) -> dict:
    """Seed and trained horizons of the training run: run_manifest.json first, then .hydra/config.yaml."""
    out = _manifest_config(run_dir)
    for key, value in _hydra_config(run_dir).items():
        out.setdefault(key, value)
    return out


def split_of(run_dir: Path, n_loader, is_c3: bool) -> str:
    for part in run_dir.parts:
        if part in pr.SPLITS:
            return part
        if part == "timestep":
            return pr.PUBLIC_SPLIT
    for split, n_val in pr.N_VAL_WINDOWS.items():
        if n_loader in ({n_val * 3} if is_c3 else {n_val, n_val * 3}):
            return split
    return ""


def export_summary(path: Path, *, hash_checkpoints: bool = True, scope: str = "") -> dict:
    """One public-schema row for one evaluator summary."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    is_c3 = SUMMARY_FILES[path.name] or summary.get("evaluation_protocol") == "c3_action_sensitivity"
    run_dir = path.parent.parent
    model = summary.get("vfm_name")
    probe = summary.get("probe_type")
    if not model or not probe:
        raise ValueError("summary lacks vfm_name or probe_type")
    disk = summary.get("feature_layer")
    if disk is None and summary.get("feat_postfix") is not None:
        match = re.search(r"layer(-?\d+)", str(summary["feat_postfix"]))
        disk = int(match.group(1)) if match else None
    if disk is None:
        raise ValueError("summary lacks feature_layer")
    layer = pr.ledger_layer(model, int(disk))
    prefix = int(summary["prefix_len"])
    timestep = int(summary["feature_timestep"])
    canonical = pr.CANONICAL_TIMESTEP.get(model) == timestep
    config = training_config(run_dir)
    seed = config.get("seed")
    caveats: list[str] = []

    n_loader = summary.get("loader_n_samples", summary.get("n_samples"))
    split = split_of(run_dir, n_loader, is_c3)
    horizons = config.get("action_horizons")
    if probe == "action_dynamics" and horizons is None:
        try:  # fall back to how the released recipe trains this cell
            horizons = pr.recipe("canonical" if canonical else "timestep", model, "action_dynamics", layer, prefix,
                                 timestep=None if canonical else timestep,
                                 split=split or pr.PUBLIC_SPLIT).trained_action_horizons
        except pr.RecipeError:
            horizons = None

    table_probe = "c3_from_c1" if is_c3 else probe
    line = "c3_from_c1" if is_c3 else ("canonical_v2" if canonical else "timestep")
    metric = pr.PRIMARY_METRIC[table_probe]
    value = summary.get(metric)
    ci_entry = (summary.get("bootstrap_95ci") or {}).get(metric)
    if isinstance(ci_entry, dict):
        ci = [ci_entry.get("low"), ci_entry.get("high")]
    elif isinstance(ci_entry, (list, tuple)) and len(ci_entry) == 2:
        ci = list(ci_entry)
    else:
        ci = ["", ""]
    n_scored = first_present(summary, SCORED_FIELDS)
    n_nonfinite = first_present(summary, NONFINITE_FIELDS)

    ckpt_text = str(summary.get("ckpt_path") or "")
    ckpt_match = CKPT_RE.search(Path(ckpt_text).name)
    epoch, step = (int(ckpt_match.group(1)), int(ckpt_match.group(2))) if ckpt_match else (None, None)
    batch = (infer_batch_size(epoch, step, pr.N_TRAIN_WINDOWS.get(split, pr.N_TRAIN_WINDOWS[pr.PUBLIC_SPLIT]))
             if ckpt_match else None)
    ckpt_sha = ""
    if hash_checkpoints and ckpt_text:
        ckpt = Path(ckpt_text)
        if not ckpt.is_absolute():
            ckpt = (pr.ROOT / ckpt) if (pr.ROOT / ckpt).is_file() else Path.cwd() / ckpt
        if ckpt.is_file():
            ckpt_sha = sha256_file(ckpt)

    # Same codes and order as the published tables (pack-specific codes omitted).
    if seed is None:
        seed = 42
        caveats.append("SEED_NOT_RECORDED_ASSUMED_42")
    if seed == 42:
        caveats.append("VAL_SPLIT_SEED42_ONLY")
    if canonical and batch is not None and batch != 2:
        caveats.append(f"TRAIN_BATCH_SIZE_{batch}_NOT_CANONICAL_2")
    if split == pr.MINIMAX_SUBSET_SPLIT:
        caveats.append("VAL_171_SAMPLES_NOT_COMPARABLE_TO_276")
    # The control caveat belongs to C1 and C2 rows; C3 rows do not carry it.
    if probe in pr.C_PROBES and not is_c3:
        caveats.append("REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS")
    if probe in {"ego_belief", "ego_belief_v2"} and not is_c3 and n_scored is not None and n_loader:
        caveats.append(f"HIDDEN_OBJECT_SCORED_{n_scored}_OF_{n_loader}")
    variant = ""
    if is_c3:
        variant = "c3_from_c1" if horizons and len(horizons) >= 2 else "h1_trained_diagnostic"
        caveats.append("C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL")
        if variant == "h1_trained_diagnostic":
            caveats.append("H1_TRAINED_NOT_PROTOCOL_C3")
    elif probe == "action_dynamics" and horizons == [1]:
        caveats.append("C1_TRAINED_AND_EVALUATED_H1_ONLY")

    reasons = []
    max_epochs = pr.MAX_EPOCHS.get(probe)
    if max_epochs is not None and epoch != max_epochs - 1:
        reasons.append("REJECTED_NOT_FINAL_EPOCH")
    if "_v2_" not in str(summary.get("evaluator_version", "")):
        reasons.append("REJECTED_NOT_V2_EVALUATOR")
    if not finite(value):
        reasons.append("REJECTED_PRIMARY_NOT_FINITE")
    if n_nonfinite is None:
        reasons.append("REJECTED_NO_NONFINITE_COUNT")
    elif n_nonfinite != 0:
        reasons.append(f"REJECTED_{n_nonfinite}_NONFINITE_PREDICTIONS")
    if not n_scored:
        reasons.append("REJECTED_NO_SCORED_RECORDS")

    row = {
        "unique_key": pr.unique_key(model, timestep, layer, prefix, table_probe, seed),
        "line": line, "model": model, "probe": table_probe, "variant": variant,
        "layer": layer, "prefix": prefix, "timestep": timestep, "seed": seed,
        "eval_split": path.name.split("_", 1)[0], "split_id": split, "scope": scope,
        "status": "REJECTED" if reasons else "ACCEPTED",
        "primary_metric": metric, "primary_value": value if value is not None else "",
        "primary_ci_low": "" if ci[0] is None else ci[0], "primary_ci_high": "" if ci[1] is None else ci[1],
        "secondary_json": json.dumps({k: summary.get(k) for k in SECONDARY[table_probe]}, sort_keys=True),
        "n_loader": "" if n_loader is None else n_loader, "n_scored": "" if n_scored is None else n_scored,
        "n_nonfinite_pred": "" if n_nonfinite is None else n_nonfinite,
        "evaluator_version": summary.get("evaluator_version", ""),
        "eval_revision": "", "eval_pack": "", "training_code_tag": "",
        "ckpt_epoch": "" if epoch is None else epoch, "ckpt_step": "" if step is None else step,
        "train_batch_size_inferred": "" if batch is None else batch, "checkpoint_replaced": "",
        "checkpoint_sha256": ckpt_sha, "summary_file": path.name, "summary_sha256": sha256_file(path),
        "caveats": ";".join(reasons + caveats),
    }
    if is_c3:
        row["c1_unique_key"] = pr.unique_key(model, timestep, layer, prefix, "action_dynamics", seed)
    return row


def find_summaries(runs_root: Path) -> list[Path]:
    found = []
    for name in SUMMARY_FILES:
        found += [p for p in runs_root.rglob(name) if p.parent.name == "eval"]
    return sorted(found)


def export(runs_root: Path, *, hash_checkpoints: bool = True, scope: str = "") -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_FILES}
    for path in find_summaries(runs_root):
        try:
            row = export_summary(path, hash_checkpoints=hash_checkpoints, scope=scope)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"export_results: skipped {path}: {exc}", file=sys.stderr)
            continue
        tables[row["line"]].append(row)
    for rows in tables.values():
        rows.sort(key=lambda r: (r["unique_key"], r["summary_file"]))
    return tables


def write_tables(tables: dict[str, list[dict]], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in tables.items():
        path = out / TABLE_FILES[name]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=C3_COLUMNS if name == "c3_from_c1" else COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def join_key(row: dict) -> tuple:
    return (row["model"], row["probe"], int(row["layer"]), int(row["prefix"]), int(row["timestep"]))


def load_published(directory: Path) -> dict[tuple, list[dict]]:
    published: dict[tuple, list[dict]] = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not set(JOIN_FIELDS) | {"primary_value"} <= set(reader.fieldnames or []):
                continue
            for row in reader:
                published.setdefault(join_key(row), []).append(row)
    return published


def compare(tables: dict[str, list[dict]], directory: Path) -> list[dict]:
    """Primary-metric deltas of exported rows against the tables in ``directory``.

    A published row on the same split is preferred.  ``flags`` names what makes
    the two sides differ: ``SPLIT_MISMATCH``, ``SPLIT_UNKNOWN`` (no local split)
    and ``VARIANT_MISMATCH``.
    """
    published = load_published(directory)
    out = []
    for rows in tables.values():
        for row in rows:
            local_split = str(row.get("split_id") or "")
            candidates = published.get(join_key(row), [])
            ref = next((c for c in candidates if (c.get("split_id") or "") == local_split),
                       candidates[0] if candidates else None)
            entry = {"unique_key": row["unique_key"], "primary_metric": row["primary_metric"],
                     "local": row["primary_value"], "status": row["status"], "published": "", "delta": "",
                     "rel_delta": "", "inside_published_ci": "", "published_key": "",
                     "split_id": local_split, "published_split_id": "", "variant": row.get("variant") or "",
                     "published_variant": "", "flags": ""}
            if ref is not None:
                entry["published_key"] = ref["unique_key"]
                entry["published"] = ref["primary_value"]
                entry["published_split_id"] = ref.get("split_id") or ""
                entry["published_variant"] = ref.get("variant") or ""
                flags = []
                if not local_split:
                    flags.append("SPLIT_UNKNOWN")
                elif local_split != entry["published_split_id"]:
                    flags.append("SPLIT_MISMATCH")
                if entry["variant"] != entry["published_variant"]:
                    flags.append("VARIANT_MISMATCH")
                entry["flags"] = ";".join(flags)
                try:
                    local, pub = float(row["primary_value"]), float(ref["primary_value"])
                    if not (math.isfinite(local) and math.isfinite(pub)):
                        raise ValueError("non-finite value")
                    entry["delta"] = local - pub
                    entry["rel_delta"] = (local - pub) / abs(pub) if pub else ""
                    if ref.get("primary_ci_low") not in (None, "") and ref.get("primary_ci_high") not in (None, ""):
                        entry["inside_published_ci"] = (
                            float(ref["primary_ci_low"]) <= local <= float(ref["primary_ci_high"]))
                except (TypeError, ValueError):
                    pass
            out.append(entry)
    return out


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_comparison(entries: list[dict], directory: Path) -> None:
    fields = ["unique_key", "primary_metric", "local", "published", "delta", "rel_delta",
              "inside_published_ci", "status", "split_id", "published_split_id", "variant",
              "published_variant", "flags"]
    print("\t".join(fields))
    for entry in entries:
        print("\t".join(_fmt(entry[f]) for f in fields))
    matched = sum(1 for e in entries if e["published_key"])
    flagged = sum(1 for e in entries if e["flags"])
    print(f"# compared with {directory}: {matched} matched, {len(entries) - matched} without a published row, "
          f"{flagged} flagged (split or variant differs; such deltas do not compare like with like)",
          file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=pr.ROOT / pr.RUNS_DIR,
                        help="searched recursively for eval/val_*summary.json")
    parser.add_argument("--out", type=Path, default=None, help="write canonical_v2.csv, timestep.csv, c3_from_c1.csv here")
    parser.add_argument("--compare", type=Path, default=None, metavar="DIR",
                        help="published tables to compare against, e.g. results/")
    parser.add_argument("--scope", default="", help="value for the scope column (default empty)")
    parser.add_argument("--no-checkpoint-hash", action="store_true", help="leave checkpoint_sha256 empty")
    args = parser.parse_args(argv)
    if not args.runs_root.is_dir():
        print(f"export_results: no such directory {args.runs_root}", file=sys.stderr)
        return 2
    tables = export(args.runs_root, hash_checkpoints=not args.no_checkpoint_hash, scope=args.scope)
    counts = {name: len(rows) for name, rows in tables.items()}
    accepted = sum(r["status"] == "ACCEPTED" for rows in tables.values() for r in rows)
    print(f"# exported {sum(counts.values())} summaries ({accepted} ACCEPTED): {counts}", file=sys.stderr)
    if args.out is not None:
        for path in write_tables(tables, args.out):
            print(f"# wrote {path}", file=sys.stderr)
    if args.compare is not None:
        print_comparison(compare(tables, args.compare), args.compare)
    elif args.out is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=C3_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rows in tables.values():
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
