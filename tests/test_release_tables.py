"""Integrity checks for the shipped result tables and a repository-wide privacy scan.

The tables in results/ are static exports of accepted validation-split evaluations.
These tests pin their row counts and schema, re-derive the inferred training batch
size from the checkpoint name, check the caveat codes, and make sure that no file
in the repository carries storage paths, private addresses or e-mail addresses.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

COLUMNS = [
    "unique_key", "line", "model", "probe", "variant", "layer", "prefix", "timestep", "seed",
    "eval_split", "split_id", "scope", "status", "primary_metric", "primary_value",
    "primary_ci_low", "primary_ci_high", "secondary_json", "n_loader", "n_scored",
    "n_nonfinite_pred", "evaluator_version", "eval_revision", "eval_pack", "training_code_tag",
    "ckpt_epoch", "ckpt_step", "train_batch_size_inferred", "checkpoint_replaced",
    "checkpoint_sha256", "summary_file", "summary_sha256", "caveats",
]
TABLES = {
    "canonical_v2": ("canonical_v2_val_seed42.csv", 804, COLUMNS),
    "timestep": ("timestep_val_seed42.csv", 316, COLUMNS),
    "c3_from_c1": ("c3_from_c1_val_seed42.csv", 180, COLUMNS + ["c1_unique_key"]),
}
SUBSET_SPLIT = "canonical_v2_minimax_owner_subset_r2"
N_TRAIN = {"canonical_v2": 2445, SUBSET_SPLIT: 1523}
N_VAL = {"canonical_v2": 276, SUBSET_SPLIT: 171}
EVALUATORS = {"eval_diag_v2_nonfinite_aware", "a1_streaming_eval_v2_nonfinite_aware"}
KEY_RE = re.compile(
    r"^(?P<model>.+?)__scannetpp__t(?P<t>\d+)__l(?P<l>-?\d+)__p(?P<p>\d+)__(?P<probe>.+)__seed(?P<seed>\d+)$"
)
CAVEAT_RE = re.compile(
    r"^(VAL_SPLIT_SEED42_ONLY|TRAINING_AND_EVAL_PACK_COMMITS_NOT_ON_GITHUB"
    r"|REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS|TRAINING_CODE_SNAPSHOT_IDENTITY_20261017"
    r"|HIDDEN_OBJECT_SCORED_(\d+)_OF_(\d+)|TRAIN_BATCH_SIZE_(\d+)_NOT_CANONICAL_2"
    r"|VAL_171_SAMPLES_NOT_COMPARABLE_TO_276|FP32_RETRAIN_REPLACES_NONFINITE_CHECKPOINT"
    r"|C1_TRAINED_AND_EVALUATED_H1_ONLY|C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL"
    r"|H1_TRAINED_NOT_PROTOCOL_C3)$"
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _read(name: str) -> list[dict[str, str]]:
    with (RESULTS / TABLES[name][0]).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def tables() -> dict[str, list[dict[str, str]]]:
    return {name: _read(name) for name in TABLES}


@pytest.mark.parametrize("name", list(TABLES))
def test_schema_counts_and_acceptance(tables, name):
    filename, n_rows, columns = TABLES[name]
    with (RESULTS / filename).open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == columns
    rows = tables[name]
    assert len(rows) == n_rows
    assert len({r["unique_key"] for r in rows}) == n_rows
    for r in rows:
        key = r["unique_key"]
        assert r["line"] == name, key
        assert r["status"] == "ACCEPTED", key
        assert r["eval_split"] == "val" and r["seed"] == "42", key
        assert r["evaluator_version"] in EVALUATORS, key
        assert r["n_nonfinite_pred"] == "0", key
        assert 0 < int(r["n_scored"]) <= int(r["n_loader"]), key
        assert math.isfinite(float(r["primary_value"])), key
        if r["primary_ci_low"]:
            assert float(r["primary_ci_low"]) <= float(r["primary_value"]) <= float(r["primary_ci_high"]), key
        assert isinstance(json.loads(r["secondary_json"]), dict), key
        assert SHA_RE.match(r["checkpoint_sha256"]) and SHA_RE.match(r["summary_sha256"]), key
        assert r["summary_file"] in {"val_summary.json", "val_c3_action_sensitivity_summary.json"}, key
        assert r["checkpoint_replaced"] in {"true", "false"}, key
        assert r["split_id"] in N_TRAIN, key
        assert (r["split_id"] == SUBSET_SPLIT) == (r["scope"] == "canonical_v2_minimax_all_layers_20261011"), key
        m = KEY_RE.match(key)
        assert m, key
        assert (m["model"], m["l"], m["p"], m["t"], m["seed"]) == (
            r["model"], r["layer"], r["prefix"], r["timestep"], r["seed"]), key
        assert m["probe"] == r["probe"], key


def test_unique_keys_are_unique_across_tables(tables):
    keys = [r["unique_key"] for rows in tables.values() for r in rows]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("name", list(TABLES))
def test_train_batch_size_reproduces_from_checkpoint_name(tables, name):
    for r in tables[name]:
        n_train = N_TRAIN[r["split_id"]]
        batch, epoch, step = (int(r[c]) for c in ("train_batch_size_inferred", "ckpt_epoch", "ckpt_step"))
        assert (n_train // batch) * (epoch + 1) == step, r["unique_key"]
        candidates = [b for b in (1, 2, 3, 4, 6, 8) if (n_train // b) * (epoch + 1) == step]
        assert candidates == [batch], r["unique_key"]


def test_batch_size_distribution(tables):
    canonical = Counter(r["train_batch_size_inferred"] for r in tables["canonical_v2"])
    assert canonical == {"2": 656, "1": 92, "4": 56}
    for r in tables["timestep"]:
        expected = "1" if r["probe"] in {"streaming_depth", "streaming_mask"} else "2"
        assert r["train_batch_size_inferred"] == expected, r["unique_key"]


def test_c3_rows_point_at_their_c1_rows(tables):
    c1 = {r["unique_key"]: r for name in ("canonical_v2", "timestep") for r in tables[name]}
    variants = Counter()
    for r in tables["c3_from_c1"]:
        source = c1[r["c1_unique_key"]]
        assert source["probe"] == "action_dynamics"
        assert r["unique_key"] == r["c1_unique_key"].replace("__action_dynamics__", "__c3_from_c1__")
        for col in ("model", "layer", "prefix", "timestep", "scope", "split_id", "checkpoint_sha256",
                    "ckpt_epoch", "ckpt_step", "train_batch_size_inferred", "training_code_tag"):
            assert r[col] == source[col], (r["unique_key"], col)
        assert r["primary_metric"] == "intervention_validity"
        variants[r["variant"]] += 1
    assert variants == {"c3_from_c1": 120, "h1_trained_diagnostic": 60}


@pytest.mark.parametrize("name", list(TABLES))
def test_caveats(tables, name):
    for r in tables[name]:
        key, codes = r["unique_key"], r["caveats"].split(";")
        assert len(codes) == len(set(codes)), key
        for code in codes:
            m = CAVEAT_RE.match(code)
            assert m, (key, code)
            if m[2]:
                assert (m[2], m[3]) == (r["n_scored"], r["n_loader"]), key
            if m[4]:
                assert m[4] == r["train_batch_size_inferred"] != "2", key
        assert "VAL_SPLIT_SEED42_ONLY" in codes, key
        scored_partial = r["probe"] in {"ego_belief", "ego_belief_v2"} and r["n_scored"] != r["n_loader"]
        assert scored_partial == any(c.startswith("HIDDEN_OBJECT_SCORED_") for c in codes), key
        assert ("VAL_171_SAMPLES_NOT_COMPARABLE_TO_276" in codes) == (r["split_id"] == SUBSET_SPLIT), key
        assert ("FP32_RETRAIN_REPLACES_NONFINITE_CHECKPOINT" in codes) == (r["checkpoint_replaced"] == "true"), key
        if name == "canonical_v2":
            batch = r["train_batch_size_inferred"]
            assert (f"TRAIN_BATCH_SIZE_{batch}_NOT_CANONICAL_2" in codes) == (batch != "2"), key
        if name == "c3_from_c1":
            assert "C3_SAME_TRAJECTORY_ENDPOINTS_NOT_COUNTERFACTUAL" in codes, key
            assert ("H1_TRAINED_NOT_PROTOCOL_C3" in codes) == (r["variant"] == "h1_trained_diagnostic"), key
            assert int(r["n_loader"]) == 3 * N_VAL[r["split_id"]], key
            not_applicable = {"C1_TRAINED_AND_EVALUATED_H1_ONLY", "REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS"}
            assert not not_applicable & set(codes), key
        else:
            # C1 evaluated on one horizon per sample (n_loader = val samples, not 3x) was trained on h1 only.
            single_horizon = r["probe"] == "action_dynamics" and int(r["n_loader"]) == N_VAL[r["split_id"]]
            assert ("C1_TRAINED_AND_EVALUATED_H1_ONLY" in codes) == single_horizon, key
            controls = r["probe"] in {"action_dynamics", "path_integration"}
            assert ("REPORT_WITH_NO_ACTION_SHUFFLED_COPY_LAST_CONTROLS" in codes) == controls, key


def test_readme_documents_every_caveat_code(tables):
    readme = (RESULTS / "README.md").read_text(encoding="utf-8")
    codes = {
        re.sub(r"_\d+_OF_\d+$", "_x_OF_y", re.sub(r"^TRAIN_BATCH_SIZE_\d+_", "TRAIN_BATCH_SIZE_n_", c))
        for rows in tables.values() for r in rows for c in r["caveats"].split(";")
    }
    assert codes
    assert not [c for c in codes if c not in readme]


def test_sources_manifest_matches_tables(tables):
    sources = json.loads((RESULTS / "SOURCES.json").read_text(encoding="utf-8"))
    assert set(sources["files"]) == {f"results/{TABLES[n][0]}" for n in TABLES}
    for name, (filename, n_rows, _) in TABLES.items():
        entry = sources["files"][f"results/{filename}"]
        assert entry["rows"] == n_rows == len(tables[name])
        assert entry["sha256"] == hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert {"checkpoint_path", "summary_path", "reject_reason"} <= set(entry["dropped_columns"])


# --------------------------------------------------------------------------- privacy scan

# Absolute mount points used by shared clusters.
MOUNT_RE = re.compile(r"/(?:mnt|nas)/")
# Absolute per-user paths: /home/<name>/, /data/<name>/, /Users/<name>/. Paths that
# start after a word character, '.', '}', '>' and the like are relative to a variable
# or placeholder (e.g. ${ROOT}/data/splits/) and are not flagged.
USER_PATH_RE = re.compile(r"(?<![\w.\-/}>)\]$~*])/(?:home|data|Users)/[A-Za-z_][\w.-]*/")
# Absolute superuser, scratch and parallel-filesystem trees (root, scratch, lustre,
# gpfs, cpfs*), with the same placeholder rule as USER_PATH_RE.
SYSTEM_PATH_RE = re.compile(r"(?<![\w.\-/}>)\]$~*])/(?:root|scratch|lustre|gpfs|cpfs\w*)/")
# Host names inside a cluster or intranet domain, e.g. node01 in an hpc subdomain of a
# university domain. Dotted Python names have no top-level domain and are not flagged.
HOST_RE = re.compile(
    r"\b[A-Za-z0-9-]+\.(?:hpc|cluster|corp|intranet|internal)\."
    r"(?:[A-Za-z0-9-]+\.)*(?:edu|com|org|net|io|ai|cn|ac\.[a-z]{2})\b"
)
# Object-store URIs prefixed with a cluster alias, and bare s3/aoss URIs that name a bucket.
CLUSTER_S3_RE = re.compile(r"([\w${}<>.-]+):s3://")
CLUSTER_ALLOWLIST = {"<cluster>", "{cluster}", "mycluster", "$AOSS_CLUSTER", "${AOSS_CLUSTER}"}
BUCKET_RE = re.compile(r"\b(?:s3|aoss)://([A-Za-z0-9][\w.-]*)")
BUCKET_ALLOWLIST = {"bucket", "example", "example-bucket", "my-bucket"}
# RFC 1918 private IPv4 ranges; version pins such as pkg==10.3.5.147 are not addresses.
PRIVATE_IP_RE = re.compile(
    r"(?<![\w.=~<>!-])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![\w.])"
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# Addresses that a shipped upstream license text requires verbatim. Keep this empty
# unless such a text is added.
EMAIL_ALLOWLIST: frozenset[str] = frozenset()

# Never scanned: VCS metadata, caches, build metadata and virtual environments (any
# directory holding pyvenv.cfg).
ALWAYS_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox"}
# Also pruned by the fallback tree walk (git ignores them through .gitignore).
WALK_SKIP_DIRS = ALWAYS_SKIP_DIRS | {
    "venv", ".venv", "env", "node_modules", "build", "dist",
    "logs", "outputs", "multirun", "lightning_logs", "wandb", "data_local", "InsScene-15K", "manifests_out",
}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".sft",
                   ".safetensors", ".pyc", ".so", ".zip", ".tar", ".whl"}


def find_private_strings(text: str) -> list[tuple[str, str]]:
    """Return (rule, match) pairs for every private-looking string in ``text``."""
    hits = [("mount", m.group(0)) for m in MOUNT_RE.finditer(text)]
    if any(s in text for s in ("/home/", "/data/", "/Users/")):
        hits += [("user_path", m.group(0)) for m in USER_PATH_RE.finditer(text)]
    hits += [("system_path", m.group(0)) for m in SYSTEM_PATH_RE.finditer(text)]
    hits += [("host", m.group(0)) for m in HOST_RE.finditer(text)]
    if ":s3://" in text:
        hits += [("cluster_s3", m.group(0)) for m in CLUSTER_S3_RE.finditer(text)
                 if m.group(1) not in CLUSTER_ALLOWLIST]
    if "s3://" in text or "aoss://" in text:
        hits += [("bucket", m.group(0)) for m in BUCKET_RE.finditer(text) if m.group(1) not in BUCKET_ALLOWLIST]
    hits += [("private_ip", m.group(0)) for m in PRIVATE_IP_RE.finditer(text)]
    if "@" in text:
        hits += [("email", m.group(0)) for m in EMAIL_RE.finditer(text) if m.group(0) not in EMAIL_ALLOWLIST]
    return hits


def _skipped(rel: Path, venv_dirs: dict[Path, bool]) -> bool:
    for i, part in enumerate(rel.parts[:-1]):
        if part in ALWAYS_SKIP_DIRS or part.endswith(".egg-info"):
            return True
        parent = Path(*rel.parts[: i + 1])
        if parent not in venv_dirs:
            venv_dirs[parent] = (ROOT / parent / "pyvenv.cfg").is_file()
        if venv_dirs[parent]:
            return True
    return False


def repository_files() -> list[Path]:
    """Files that would be published: `git ls-files` (tracked and unignored), else a tree walk."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True, capture_output=True, timeout=60,
        ).stdout.decode("utf-8")
        toplevel = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        if Path(toplevel).resolve() != ROOT.resolve():
            raise RuntimeError("checkout is nested inside another repository")
        rels = [Path(p) for p in out.split("\0") if p]
    except (OSError, subprocess.SubprocessError, RuntimeError):
        rels = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in WALK_SKIP_DIRS and not d.startswith("FEAT_")]
            rels += [Path(dirpath, f).relative_to(ROOT) for f in filenames]
    venv_dirs: dict[Path, bool] = {}
    return sorted(p for p in rels if not _skipped(p, venv_dirs) and (ROOT / p).is_file())


def _text(rel: Path) -> str | None:
    data = (ROOT / rel).read_bytes()
    if rel.suffix == ".gz":  # e.g. data/splits/**/*.jsonl.gz
        data = gzip.decompress(data)
    elif rel.suffix.lower() in BINARY_SUFFIXES or b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


# Synthetic positives are assembled at run time ({S} is '/', {AT} is '@', {D} is '.') so
# that this file itself passes the repository scan.
@pytest.mark.parametrize(
    "template",
    [
        "PYTHON={S}mnt{S}shared/envs/x/bin/python",
        "root: {S}nas{S}team/datasets",
        "cache in {S}home{S}alice/.cache/x",
        'FEAT_ROOT="{S}data{S}bob/features"',
        "open('{S}Users{S}carol/Desktop/a.csv')",
        "src = 'corp-a:s3:{S}{S}bucket/features'",
        "aoss:{S}{S}team-private/InsScene-15K/",
        "s3:{S}{S}internal-data/run/epoch=1.ckpt",
        "host 10{D}140{D}2{D}17 port 22",
        "http:{S}{S}172{D}20{D}0{D}5:8080/",
        "192{D}168{D}1{D}10",
        "contact jane.doe{AT}example.org",
        "cfg = '{S}root{S}.cache/tool.conf'",
        "out_dir={S}scratch{S}abc123{S}run",
        "ROOT={S}cpfs01{S}user/x",
        "ln -s {S}lustre{S}proj/feats",
        "{S}gpfs{S}u/x",
        "ssh node01{D}hpc{D}example{D}edu",
        "gpu7{D}cluster{D}example{D}ac{D}uk:8888",
    ],
)
def test_privacy_patterns_flag_synthetic_examples(template):
    text = template.format(S="/", AT="@", D=".")
    assert not find_private_strings(template), template
    assert find_private_strings(text), text


@pytest.mark.parametrize(
    "text",
    [
        "``<cluster>:s3://`` (e.g. ``mycluster:s3://bucket/...``)",
        'prefix = f"{cluster}:s3://" if cluster else "s3://"',
        "``$AOSS_CLUSTER:s3://bucket/key``",
        "aoss://<bucket>/InsScene-15K/ and s3://example/features/sample.sft",
        "${INSCENE_DATA_ROOT}/data/splits/canonical_v2 and <repo>/data/splits/x/",
        "data/splits/canonical_v2/inscene_c_h124_prefix24.public.jsonl.gz",
        "https://github.com/user/home/tree/main/",
        "nvidia-curand-cu12==10.3.5.147",
        "secondary global_R@1 and global_R@5",
        "${SCRATCH}/scratch/x and <repo>/root/cfg and $HOME/lustre/x",
        "features/root/x.sft and data/gpfs/y",
        "from sklearn.cluster import KMeans; torch.distributed.internal.api",
    ],
)
def test_privacy_patterns_allow_placeholders(text):
    assert not find_private_strings(text), text


def test_repository_has_no_private_strings():
    files = repository_files()
    assert any(p.parts[0] == "vidfm3d" for p in files)
    assert any(p.parts[:2] == ("data", "splits") and p.suffix == ".gz" for p in files)
    hits = []
    for rel in files:
        for rule, match in find_private_strings(rel.as_posix()):
            hits.append(f"{rel} (file name): {rule}: {match}")
        text = _text(rel)
        if text is None:
            continue
        for rule, match in find_private_strings(text):
            line = text.count("\n", 0, text.find(match)) + 1
            hits.append(f"{rel}:{line}: {rule}: {match[:80]}")
    assert not hits, "private strings found:\n" + "\n".join(hits[:50])
