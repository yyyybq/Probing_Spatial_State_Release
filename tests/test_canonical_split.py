"""The shipped canonical V2 window universe and its train/val split."""
import gzip
import json
from collections import defaultdict
from pathlib import Path

from scripts.build_tstep_canonical_manifests import (
    EXPECTED_UNIVERSE_SHA256,
    PUBLIC_CANONICAL_SOURCE,
    load_canonical_windows,
    load_split_map,
    universe_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / PUBLIC_CANONICAL_SOURCE


def test_public_canonical_universe_matches_frozen_protocol():
    split_map = load_split_map([SOURCE])
    records = load_canonical_windows(SOURCE, split_map, expected_shared_units=2721)
    assert universe_sha256(records) == EXPECTED_UNIVERSE_SHA256
    units = defaultdict(int)
    scenes = defaultdict(set)
    for row in records:
        units[row["split"]] += 1
        scenes[row["split"]].add(row["scene_id"])
    assert dict(units) == {"train": 2445, "val": 276}
    assert {k: len(v) for k, v in scenes.items()} == {"train": 612, "val": 69}
    assert not scenes["train"] & scenes["val"]


def test_public_canonical_source_has_no_storage_paths():
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert not {"scene_dir", "streaming_feature_root", "target_feature_root"} & row.keys()
