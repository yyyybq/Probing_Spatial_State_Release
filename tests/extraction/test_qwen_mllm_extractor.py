from pathlib import Path
from types import SimpleNamespace

from features.run_inscene15k_mllm import (
    merge_extra_metadata,
    streaming_prefix_records_for_scene,
    streaming_target_indices_for_scene,
)
from features.run_inscene15k import target_selection_for_scene


def _args():
    return SimpleNamespace(
        mode="streaming_prefix",
        temporal_sampling="motion_uniform",
        streaming_window_obs_len=28,
        streaming_motion_step=0.35,
        streaming_rotation_weight=0.5,
        streaming_window_stride=8,
        streaming_max_windows_per_scene=4,
        prefix_lengths=[8, 12, 16, 24],
        prefix_min_len=1,
        prefix_max_len=24,
        prefix_stride=1,
        target_from_streaming_windows=True,
        target_prefix_lengths=[8, 12, 16, 24],
        target_horizons=[1, 2, 4],
    )


def test_qwen_uses_one_obs28_window_for_all_prefixes_and_targets():
    args = _args()
    scene = {
        "source": "scannetpp",
        "scene_dir": "/dataset/scene-a",
        "img_files": [f"{index}.jpg" for index in range(40)],
    }
    records = streaming_prefix_records_for_scene(scene, args)
    assert [record["valid_length"] for record in records] == [8, 12, 16, 24]
    assert all(record["window_indices"] == list(range(28)) for record in records)
    assert all(record["canonical_obs_len"] == 28 for record in records)

    targets = streaming_target_indices_for_scene(scene, args)
    direct_targets, provenance = target_selection_for_scene(scene, args)
    assert targets == direct_targets
    assert {7, 8, 9, 11, 23, 24, 25, 27}.issubset(set(targets))
    assert {item["role"] for item in provenance} == {"anchor", "future"}




def test_qwen_stream_gap_metadata_does_not_drop_num_frames():
    source = Path(merge_extra_metadata.__code__.co_filename).read_text()
    start = source.index("def stream_gap_metadata(")
    end = source.index("def stream_gap_complete(", start)
    block = source[start:end]
    assert 'for key in (\n            "frame_index_schema",\n            "resize",\n            "canonical_protocol",' in block
    assert '"num_frames",' not in block


def test_qwen_stream_gap_extra_keeps_per_prefix_num_frames():
    manifest = {
        "frame_index_schema": "image_mask_intersection_v1",
        "resize": [336, 336],
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "canonical_obs_len": 28,
    }
    extra = {
        "frame_index_schema": "image_mask_intersection_v1",
        "resize": [336, 336],
        "num_frames": 8,
        "input_num_frames": 8,
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "canonical_obs_len": 28,
        "canonical_source_sha256": "a" * 64,
    }
    owned_by_save_feat = {
        "frame_index_schema",
        "resize",
        "canonical_protocol",
        "canonical_protocol_sha256",
        "canonical_obs_len",
    }
    merge_extra_metadata(
        manifest, {key: value for key, value in extra.items() if key not in owned_by_save_feat}
    )
    assert manifest["num_frames"] == 8
    assert manifest["input_num_frames"] == 8


def test_qwen_gap_metadata_merge_allows_identical_reserved_values():
    manifest = {
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "canonical_protocol_sha256": "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f",
        "canonical_obs_len": 28,
        "num_frames": 24,
    }
    extra = {
        "canonical_protocol_sha256": "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f",
        "canonical_source_sha256": "a" * 64,
        "num_frames": 24,
        "produced_by_gap_worklist_file_sha256": "b" * 64,
    }

    merge_extra_metadata(manifest, extra)

    assert manifest["canonical_protocol_sha256"] == extra["canonical_protocol_sha256"]
    assert manifest["canonical_source_sha256"] == "a" * 64
    assert manifest["produced_by_gap_worklist_file_sha256"] == "b" * 64


def test_qwen_gap_metadata_merge_rejects_reserved_value_change():
    manifest = {"canonical_protocol_sha256": "a" * 64}
    try:
        merge_extra_metadata(manifest, {"canonical_protocol_sha256": "b" * 64})
    except RuntimeError as exc:
        assert "canonical_protocol_sha256" in str(exc)
    else:
        raise AssertionError("reserved metadata conflict was accepted")
