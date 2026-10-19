#!/usr/bin/env python3
"""Download and extract the InsScene-15K dataset from the Hugging Face Hub.

The probes in this repository use only the ScanNet++ subset
(``processed_scannetpp_v2``), so by default this script downloads and extracts
that subset. The Infinigen subset is opt-in.

Usage:
    python data/download_inscene15k.py --base-dir <BASE>
    python data/download_inscene15k.py --base-dir <BASE> --step download_scannetpp
    python data/download_inscene15k.py --base-dir <BASE> --step extract_scannetpp
    python data/download_inscene15k.py --base-dir <BASE> --step infinigen   # optional

Layout under <BASE>:
    raw/                       Hugging Face snapshot (zip parts)
    data/processed_scannetpp_v2/processed_scannetpp_v2/<scene_id>/
                               images/, refined_ins_ids/, depth/,
                               scene_iphone_metadata.npz

Afterwards:
    export INSCENE_DATA_ROOT=<BASE>/data
    export INSCENE_LOCAL_SCENE_ROOT=<BASE>/data/processed_scannetpp_v2/processed_scannetpp_v2

The InsScene-15K and ScanNet++ terms apply. If the dataset is gated, log in
with `hf auth login` or set HF_TOKEN first.
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

REPO_ID = "lifuguan/InsScene-15K"
DEFAULT_BASE_DIR = os.environ.get("INSCENE_DIR", "/data/InsScene-15K")
SCANNETPP = "processed_scannetpp_v2"
INFINIGEN = "processed_infinigen"
# Number of ScanNet++ scene directories the canonical_v2 split was drawn from
# (see data/splits/canonical_v2/SPLIT.json).
CANONICAL_SCENE_DIRS = 683

STEPS = {
    "scannetpp": ("download_scannetpp", "extract_scannetpp"),
    "download_scannetpp": ("download_scannetpp",),
    "extract_scannetpp": ("extract_scannetpp",),
    "infinigen": ("download_infinigen", "extract_infinigen"),
    "download_infinigen": ("download_infinigen",),
    "extract_infinigen": ("extract_infinigen",),
    "all": ("download_scannetpp", "extract_scannetpp", "download_infinigen", "extract_infinigen"),
}


def _banner(text: str) -> None:
    print("=" * 60)
    print(text)
    print("=" * 60)


def download(subset: str, raw_dir: Path, revision: str | None) -> None:
    """Download one dataset subset into ``raw_dir``."""
    from huggingface_hub import snapshot_download

    _banner(f"Downloading {subset} (revision: {revision or 'default branch'}) ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(raw_dir),
        allow_patterns=[f"{subset}/**"],
    )
    print(f"{subset} download complete.")


def extract_infinigen(raw_dir: Path, data_dir: Path) -> None:
    """Extract each subscene zip in processed_infinigen."""
    _banner(f"Extracting {INFINIGEN} ...")
    src_root = raw_dir / INFINIGEN
    dst_root = data_dir / INFINIGEN
    if not src_root.is_dir():
        print(f"  {src_root} not found; run --step download_infinigen first. Skipping.")
        return
    dst_root.mkdir(parents=True, exist_ok=True)

    for meta in ["dataset_metadata.json", "README.md"]:
        meta_src = src_root / meta
        meta_dst = dst_root / meta
        if meta_src.exists() and not meta_dst.exists():
            shutil.copy2(meta_src, meta_dst)

    scene_dirs = sorted(d for d in src_root.iterdir() if d.is_dir() and d.name.startswith("scene_"))
    extracted = 0
    for si, scene_dir in enumerate(scene_dirs):
        for zf in sorted(scene_dir.glob("*.zip")):
            out_dir = dst_root / scene_dir.name / zf.stem
            if out_dir.exists() and any(out_dir.iterdir()):
                extracted += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zf, "r") as archive:
                    archive.extractall(out_dir)
                extracted += 1
                if extracted % 50 == 0:
                    print(f"  Extracted {extracted} subscenes ...")
            except (OSError, zipfile.BadZipFile) as exc:
                print(f"  ERROR extracting {zf}: {exc}")
        if (si + 1) % 10 == 0:
            print(f"  Scenes processed: {si + 1}/{len(scene_dirs)}")
    print(f"Infinigen extraction done: {extracted} subscenes total.")


def scannetpp_scene_root(data_dir: Path) -> Path:
    """Directory that directly contains the ScanNet++ ``<scene_id>`` folders.

    The archive extracts to ``<data>/processed_scannetpp_v2/processed_scannetpp_v2``;
    the dataset loader also accepts a flat ``<data>/processed_scannetpp_v2``.
    """
    outer = data_dir / SCANNETPP
    nested = outer / SCANNETPP
    return nested if nested.is_dir() else outer


def extract_scannetpp(base_dir: Path, raw_dir: Path, data_dir: Path) -> None:
    """Reassemble the multi-part zip and extract processed_scannetpp_v2."""
    _banner(f"Extracting {SCANNETPP} ...")
    src_root = raw_dir / SCANNETPP
    dst_root = data_dir / SCANNETPP
    combined_zip = base_dir / f"{SCANNETPP}_combined.zip"

    if not combined_zip.exists():
        parts = sorted(src_root.glob(f"{SCANNETPP}.zip.*"))
        if not parts:
            print(f"  No zip parts found under {src_root}; run --step download_scannetpp first.")
            return
        print(f"  Reassembling {len(parts)} parts into {combined_zip} ...")
        partial = combined_zip.with_name(combined_zip.name + ".partial")
        with partial.open("wb") as out:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, out, length=16 * 1024 * 1024)
        partial.replace(combined_zip)
        print(f"  Combined zip size: {combined_zip.stat().st_size / (1024**3):.1f} GB")
    else:
        print(f"  Combined zip already exists: {combined_zip.stat().st_size / (1024**3):.1f} GB")

    print(f"  Extracting to {dst_root} ...")
    dst_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(combined_zip, "r") as archive:
        archive.extractall(dst_root)
    print("ScanNet++ extraction done.")

    print("  Removing combined zip to save space ...")
    combined_zip.unlink()


def report(data_dir: Path) -> None:
    """Print dataset statistics and the environment variables to export."""
    print("\n" + "=" * 60)
    infinigen_root = data_dir / INFINIGEN
    if infinigen_root.is_dir():
        scenes = [d for d in infinigen_root.iterdir() if d.is_dir() and d.name.startswith("scene_")]
        subscenes = sum(1 for s in scenes for d in s.iterdir() if d.is_dir())
        print(f"{INFINIGEN}: {len(scenes)} scenes, {subscenes} subscenes")
    scene_root = scannetpp_scene_root(data_dir)
    if scene_root.is_dir():
        scene_dirs = [d for d in scene_root.iterdir() if d.is_dir()]
        print(f"{SCANNETPP}: {len(scene_dirs)} scene directories in {scene_root}")
        if len(scene_dirs) != CANONICAL_SCENE_DIRS:
            print(
                f"  NOTE: the canonical_v2 split was drawn from {CANONICAL_SCENE_DIRS} scene "
                "directories; a different count means a different dataset revision or an "
                "incomplete extraction."
            )
    if not scene_root.is_dir():
        scene_root = data_dir / SCANNETPP / SCANNETPP
    print("\nSet these before building manifests, extracting features or training:")
    print(f"  export INSCENE_DATA_ROOT={data_dir}")
    print(f"  export INSCENE_LOCAL_SCENE_ROOT={scene_root}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS),
        default="scannetpp",
        help="what to run (default: scannetpp = download_scannetpp + extract_scannetpp)",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="dataset root that receives raw/ and data/ (default: $INSCENE_DIR or %(default)s)",
    )
    # The release does not pin the dataset revision yet. None downloads the
    # default branch, which may differ from the snapshot behind the results.
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face dataset commit, branch or tag (default: the default branch)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    raw_dir = base_dir / "raw"
    data_dir = base_dir / "data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for step in STEPS[args.step]:
        if step == "download_scannetpp":
            download(SCANNETPP, raw_dir, args.revision)
        elif step == "extract_scannetpp":
            extract_scannetpp(base_dir, raw_dir, data_dir)
        elif step == "download_infinigen":
            download(INFINIGEN, raw_dir, args.revision)
        elif step == "extract_infinigen":
            extract_infinigen(raw_dir, data_dir)

    report(data_dir)


if __name__ == "__main__":
    main()
