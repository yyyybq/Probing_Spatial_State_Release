#!/usr/bin/env python3
"""Print (default) or run the commands that reproduce rows of the results tables.

Reads ``results/{canonical_v2,timestep,c3_from_c1}_val_seed42.csv``, selects rows
with the filters below and, through ``scripts/probe_recipe.py``, builds for each row

* the training command, with the row's ``train_batch_size_inferred``;
* the evaluation command (A1 -> ``vidfm3d/eval_a1_streaming.py``, otherwise
  ``vidfm3d/eval_diag.py``) on the newest ``epoch=*-step=*.ckpt`` of the run;
* for C1 (``action_dynamics``) rows and C3 rows, the C3-from-C1 evaluation.

A C3 row trains the C1 run named in its ``c1_unique_key``.  Commands shared by
several selected rows are emitted once.  Outputs land in
``logs/inscene15k_streaming/runs/<split or timestep>/<C1 or row key>/`` with
``eval/val_summary.json`` and ``eval/val_c3_action_sensitivity_summary.json``;
``scripts/export_results.py`` turns them back into table rows.

Without ``--run`` the output is a bash script that uses placeholders for the
data and feature roots, so nothing needs to exist.  ``--run`` executes the
commands and needs INSCENE_DATA_ROOT, INSCENE_STREAMING_FEAT_ROOT and
CANONICAL_MANIFEST_DIR (built by ``scripts/build_tstep_canonical_manifests.py``),
plus INSCENE_TARGET_FEAT_ROOT when C1, C2 or C3 rows are selected.  Relative
paths in these variables (and in MINIMAX_SUBSET_MANIFEST_DIR and
INSCENE_LOCAL_SCENE_ROOT) are resolved against the current directory, because
the jobs run from the repository root.  An INSCENE_LOCAL_SCENE_ROOT that is
already set is kept.  With ``--skip-existing``, training is skipped only when
the newest checkpoint is the final epoch, and an evaluation only when its
summary was written for that checkpoint.

The 96 canonical and 16 C3 MiniMax-H3 all-layers rows were trained and evaluated
on an owner subset of the canonical split (1,523 train / 171 val windows) that
is not public.  They are refused unless MINIMAX_SUBSET_MANIFEST_DIR points to its
``inscene_c_h124_prefix{8,12,16,24}.jsonl`` manifests.

Examples::

    python scripts/reproduce_rows.py --line canonical --model dino > dino.sh
    python scripts/reproduce_rows.py --key wan__scannetpp__t999__l20__p8__streaming_depth__seed42
    python scripts/reproduce_rows.py --line timestep --model cogvideox --probe ego_belief_v2 --run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_recipe as pr  # noqa: E402

ROOT = pr.ROOT
TABLES = {
    "canonical_v2": "canonical_v2_val_seed42.csv",
    "timestep": "timestep_val_seed42.csv",
    "c3_from_c1": "c3_from_c1_val_seed42.csv",
}
LINE_ALIASES = {"canonical": "canonical_v2", "canonical_v2": "canonical_v2", "timestep": "timestep",
                "c3": "c3_from_c1", "c3_from_c1": "c3_from_c1"}
STAGES = ("train", "eval", "c3")
SUMMARY = {"eval": "val_summary.json", "c3": "val_c3_action_sensitivity_summary.json"}
SUBSET_VAR = pr.MANIFEST_DIR_VAR[pr.MINIMAX_SUBSET_SPLIT]
# Path-valued variables; --run resolves relative values against the caller's cwd
# because the jobs run with cwd=ROOT.
PATH_VARS = ("CANONICAL_MANIFEST_DIR", SUBSET_VAR, "INSCENE_DATA_ROOT", "INSCENE_STREAMING_FEAT_ROOT",
             "INSCENE_TARGET_FEAT_ROOT", "INSCENE_LOCAL_SCENE_ROOT")


@dataclass
class Task:
    stage: str
    recipe: pr.Recipe
    rows: list[str]


def load_rows(results_dir: Path, lines: list[str] | None) -> list[dict]:
    wanted = {LINE_ALIASES[line] for line in lines} if lines else set(TABLES)
    rows = []
    for line, name in TABLES.items():
        if line not in wanted:
            continue
        path = results_dir / name
        if not path.is_file():
            raise SystemExit(f"reproduce_rows: missing results table {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def select(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    def keep(row: dict) -> bool:
        if args.model and row["model"] not in args.model:
            return False
        if args.probe and row["probe"] not in args.probe:
            return False
        if args.layer and int(row["layer"]) not in args.layer:
            return False
        if args.prefix and int(row["prefix"]) not in args.prefix:
            return False
        if args.timestep and int(row["timestep"]) not in args.timestep:
            return False
        if args.key and not ({row["unique_key"], row.get("c1_unique_key", "")} & set(args.key)):
            return False
        return True

    return [row for row in rows if keep(row)]


def build_tasks(rows: list[dict], stages: tuple[str, ...]) -> tuple[list[Task], list[str]]:
    """Ordered, de-duplicated tasks plus the keys of refused rows."""
    tasks: dict[tuple[str, str], Task] = {}
    refused = []
    for row in rows:
        rec = pr.recipe_for_row(row)
        if rec.split == pr.MINIMAX_SUBSET_SPLIT and not os.environ.get(SUBSET_VAR):
            refused.append(row["unique_key"])
            continue
        if row["line"] == "c3_from_c1":
            wanted = ("train", "c3")
        else:
            wanted = ("train", "eval") + (("c3",) if rec.probe == "action_dynamics" else ())
        for stage in wanted:
            if stage not in stages:
                continue
            task = tasks.setdefault((stage, rec.run_name), Task(stage, rec, []))
            task.rows.append(row["unique_key"])
    # Group by run (first-seen order), then train -> eval -> c3 within a run.
    first_seen: dict[str, int] = {}
    for _, run_name in tasks:
        first_seen.setdefault(run_name, len(first_seen))
    ordered = sorted(tasks.values(), key=lambda t: (first_seen[t.recipe.run_name], STAGES.index(t.stage)))
    return ordered, refused


def stage_command(task: Task, ckpt: str | None) -> tuple[str, list[str]]:
    rec = task.recipe
    if task.stage == "train":
        entry, args = pr.TRAIN_ENTRY, rec.train_args()
    elif task.stage == "eval":
        entry, args = rec.evaluator, rec.eval_args(ckpt)
    else:
        entry, args = pr.EVAL_DIAG, rec.c3_args(ckpt)
    return entry, [*args, f"job_name={rec.unique_key}", f"paths.run_folder_name={rec.run_name}"]


# --------------------------------------------------------------------- print
def render(tasks: list[Task], n_rows: int) -> str:
    required = sorted({name for task in tasks for name in task.recipe.required_env})
    out = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/reproduce_rows.py: {n_rows} table row(s), {len(tasks)} command(s).",
        "# Run from the repository root (or set REPO_ROOT). Outputs: logs/inscene15k_streaming/runs/.",
        "set -euo pipefail",
        'cd "${REPO_ROOT:-.}"',
        'PYTHON="${PYTHON:-python}"',
        ": " + " ".join(f'"${{{name}:?set {name}}}"' for name in required),
        f'export INSCENE_LOCAL_SCENE_ROOT="{pr.SCENE_ROOT_DEFAULT}"',
        "export INSCENE_TRUST_EXPANDED_MANIFEST=1 INSCENE_TARGET_LAYOUT=flat",
        'export WANDB_MODE="${WANDB_MODE:-offline}"',
        "",
        "# Newest checkpoint of a run (Lightning default layout; there is no last.ckpt). After a",
        "# complete training run it is the final epoch.",
        "newest_ckpt() {",
        '  local run="$1" best="" f',
        '  for f in "$run"/VideoProbe3D/*/checkpoints/epoch=*-step=*.ckpt; do',
        '    [[ -f "$f" ]] || continue',
        '    if [[ -z "$best" || "$f" -nt "$best" ]]; then best="$f"; fi',
        "  done",
        '  if [[ -z "$best" ]]; then',
        '    for f in "$run"/checkpoints/epoch=*-step=*.ckpt; do',
        '      [[ -f "$f" ]] || continue',
        '      if [[ -z "$best" || "$f" -nt "$best" ]]; then best="$f"; fi',
        "    done",
        "  fi",
        '  [[ -n "$best" ]] || { echo "no epoch=*-step=*.ckpt under $run" >&2; return 1; }',
        '  case "$best" in /*) ;; *) best="$PWD/$best" ;; esac',
        "  printf '%s\\n' \"$best\"",
        "}",
    ]
    current = None
    for task in tasks:
        rec = task.recipe
        if rec.run_name != current:
            current = rec.run_name
            out += [
                "",
                f"# ---- {rec.unique_key}",
                f"# line={rec.line} group={rec.family} split={rec.split} batch={rec.batch_size} "
                f"PROBE_FP16_PASSTHROUGH={rec.env['PROBE_FP16_PASSTHROUGH']}"
                f" ({rec.fp16_passthrough_basis.split(':', 1)[0]})",
                *(f"# note: {note}" for note in rec.notes),
                f'RUN="{rec.run_dir}"',
            ]
        out.append(f"# {task.stage}: " + ", ".join(task.rows))
        env = " ".join(f'{key}="{rec.env[key]}"' for key in ("INSCENE_MANIFEST", "INSCENE_PROBE_TASK",
                                                              "PROBE_FP16_PASSTHROUGH"))
        if task.stage == "train":
            entry, args = stage_command(task, None)
            quoted = " ".join(pr.shell_quote(a) for a in args)
        else:
            out.append('CKPT="$(newest_ckpt "$RUN")"')
            entry, args = stage_command(task, None)
            quoted = " ".join(pr.shell_quote(a) for a in args) + ' "ckpt_path=\'${CKPT}\'"'
        out.append(f'env {env} "$PYTHON" {entry} {quoted}')
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------- run
def absolute_paths(environ: dict[str, str]) -> dict[str, str]:
    """``environ`` with relative ``PATH_VARS`` resolved against the current directory."""
    out = dict(environ)
    for name in PATH_VARS:
        value = out.get(name)
        if value and "://" not in value:          # leave object-store URIs alone
            out[name] = os.path.abspath(os.path.expanduser(value))
    return out


def summary_matches(summary: Path, ckpt: Path) -> bool:
    """True when ``summary`` exists and records ``ckpt`` as its checkpoint."""
    if not summary.is_file():
        return False
    try:
        recorded = json.loads(summary.read_text(encoding="utf-8")).get("ckpt_path")
    except (OSError, ValueError, AttributeError):
        return False
    if not recorded:
        return False
    recorded = Path(str(recorded))
    if not recorded.is_absolute():
        recorded = ROOT / recorded
    return os.path.realpath(recorded) == os.path.realpath(ckpt)


def run(tasks: list[Task], python: str, skip_existing: bool) -> None:
    required = sorted({name for task in tasks for name in task.recipe.required_env})
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("reproduce_rows --run needs " + ", ".join(missing))
    base_env = absolute_paths(dict(os.environ))
    for task in tasks:
        rec = task.recipe
        env = dict(base_env)
        env.update({key: pr.expand(value, env) for key, value in rec.env.items()})
        env.setdefault("WANDB_MODE", "offline")
        manifest = Path(env["INSCENE_MANIFEST"])
        if not manifest.is_file():
            raise SystemExit(f"reproduce_rows: missing split manifest {manifest}")
        run_dir = ROOT / rec.run_dir
        ckpt = pr.newest_checkpoint(run_dir)
        final_epoch = rec.max_epochs - 1
        epoch = pr.checkpoint_epoch(ckpt) if ckpt is not None else None
        if task.stage == "train":
            if skip_existing and ckpt is not None:
                if epoch == final_epoch:
                    print(f"[skip] train {rec.unique_key}: {ckpt}", flush=True)
                    continue
                print(f"[train] {rec.unique_key}: newest checkpoint {ckpt.name} is not the final epoch "
                      f"{final_epoch}; training again", flush=True)
            entry, args = stage_command(task, None)
        else:
            if ckpt is None:
                raise SystemExit(f"reproduce_rows: no epoch=*-step=*.ckpt under {run_dir}")
            if epoch != final_epoch:
                print(f"[warn] {task.stage} {rec.unique_key}: {ckpt.name} is not the final-epoch checkpoint "
                      f"(epoch {final_epoch}); export_results.py rejects the row", file=sys.stderr, flush=True)
            summary = run_dir / "eval" / SUMMARY[task.stage]
            if skip_existing and summary_matches(summary, ckpt):
                print(f"[skip] {task.stage} {rec.unique_key}: {summary}", flush=True)
                continue
            entry, args = stage_command(task, str(ckpt))
        command = [python, entry, *args]
        print(f"[{task.stage}] {rec.unique_key}: " + " ".join(pr.shell_quote(a) for a in command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--line", nargs="+", choices=sorted(LINE_ALIASES), help="table(s) to read (default: all)")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--probe", nargs="+", help="probe column (C3 rows: c3_from_c1)")
    parser.add_argument("--layer", nargs="+", type=int, help="table layer as in results/*.csv")
    parser.add_argument("--prefix", nargs="+", type=int)
    parser.add_argument("--timestep", nargs="+", type=int)
    parser.add_argument("--key", nargs="+", help="unique_key (or c1_unique_key of C3 rows)")
    parser.add_argument("--steps", default="train,eval,c3", help="comma list of train,eval,c3 (default: all)")
    parser.add_argument("--list", action="store_true", help="list the selected rows and exit")
    parser.add_argument("--run", action="store_true", help="execute instead of printing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="with --run: skip training when the newest checkpoint is the final epoch, and "
                             "evaluations whose summary was written for that checkpoint")
    parser.add_argument("--python", default=sys.executable, help="interpreter for --run")
    args = parser.parse_args(argv)
    stages = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    if not stages or any(s not in STAGES for s in stages):
        parser.error(f"--steps takes a comma list of {','.join(STAGES)}")

    rows = select(load_rows(args.results_dir, args.line), args)
    if not rows:
        print("reproduce_rows: no table row matches the filters", file=sys.stderr)
        return 2
    if args.list:
        for row in rows:
            print("\t".join([row["unique_key"], row["line"], row["scope"], row["split_id"],
                             f"batch={row['train_batch_size_inferred']}",
                             f"{row['primary_metric']}={row['primary_value']}"]))
        return 0
    tasks, refused = build_tasks(rows, stages)
    if refused:
        print(f"reproduce_rows: refused {len(refused)} MiniMax-H3 all-layers row(s): their owner subset split "
              f"is not public; set {SUBSET_VAR} to its manifests to include them", file=sys.stderr)
    if not tasks:
        return 2
    if args.run:
        run(tasks, args.python, args.skip_existing)
    else:
        sys.stdout.write(render(tasks, len(rows) - len(refused)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
