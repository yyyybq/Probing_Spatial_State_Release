# Tests

The suite runs on CPU with synthetic inputs and the files shipped in this
repository. It needs no dataset, cached features, checkpoints or model weights.

## Running

Use Python 3.11 and `requirements-dev.txt`:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

`pytest.ini` limits collection to `tests/` and puts the repository root on
`sys.path`. Run a single module with, for example,
`python -m pytest -q tests/test_nonfinite_handling.py`.

Expected result: every test passes. One test is skipped when `pytorch3d` is not
installed (the positive path in `test_optional_dependency_imports.py`; the
point-cloud metrics import `pytorch3d` lazily). For reference, the suite without
`test_release_tables.py`, `test_probe_recipe.py` and `test_export_results.py`
measured 213 passed, 1 skipped on CPU; the full suite measured 282 passed, 1 skipped.

The CI workflow (`.github/workflows/ci.yml`) runs the same suite and then the
probe smoke test (`scripts/smoke_test_probes.py`), builds and audits the
canonical manifests (`scripts/build_tstep_canonical_manifests.py`), checks the
sweep script with `bash -n` and dry-runs it.

## Coverage

| Area | Test files |
| --- | --- |
| Probe contracts A1/A2/B1/B2/C1/C2 | `test_probe_contracts.py`, `test_probe_target_contracts.py`, `test_c1_action_contract.py`, `test_c1_mixed_horizons.py`, `test_a1_relative_movement.py` |
| C3-from-C1 evaluation | `test_c3_from_c1_evaluation.py` |
| Non-finite handling | `test_nonfinite_handling.py`, `test_minimax_bf16_storage_contract.py` |
| Metric accounting | `test_probe_metric_accounting.py` |
| Training regressions | `test_training_regressions.py` |
| Feature-extraction contracts | `extraction/`, `test_canonical_extraction_protocol.py`, `test_deterministic_noise_seed.py`, `test_seeded_local_cache_contract.py`, `test_aether_streaming_temporal_mapping.py`, `test_cogvideox_target_memory.py`, `test_offline_hf_cache_paths.py` |
| Canonical protocol and split | `test_canonical_split.py`, `test_canonical_extraction_protocol.py` |
| Manifest builder | `test_tstep_canonical_manifests.py` |
| Entry-point composition, imports and evaluator switches | `test_entrypoints_compose.py`, `test_optional_dependency_imports.py`, `test_eval_env_aliases.py` |
| Data-loading helpers | `test_owner_overlay_label_cache.py`, `test_aoss_fs_errors.py` |
| Release tables and privacy scan | `test_release_tables.py` |
| Probe recipes | `test_probe_recipe.py` |
| Results export | `test_export_results.py` |

## Files

| File | What it checks |
| --- | --- |
| `conftest.py` | Shared settings (quiet, deterministic Hydra and tokenizers). |
| `test_a1_relative_movement.py` | A1 relative-movement targets, 6D rotation round trip and the movement head. |
| `test_aether_streaming_temporal_mapping.py` | Aether interleaved-clip frame mapping for streaming prefixes. |
| `test_aoss_fs_errors.py` | The remote-storage helper (`vidfm3d/utils/aoss_fs.py`) reports outages and never caches them as missing files. |
| `test_c1_action_contract.py` | C1 output depends on the action; no identifiers or indices reach the probe. |
| `test_c1_mixed_horizons.py` | C1 h1/h2/h4 targets: balanced, seeded training assignment and full expansion at evaluation. |
| `test_c3_from_c1_evaluation.py` | C3 evaluation over a frozen C1 head: one C1 call per candidate action, action-set validation, non-finite candidates. |
| `test_canonical_extraction_protocol.py` | Streaming and target features share the same 28-observation windows; protocol hash. |
| `test_canonical_split.py` | The shipped public source rebuilds 2,721 units (2,445 train / 276 val, 612 / 69 scenes) with the expected universe hash and contains no storage paths. |
| `test_cogvideox_target_memory.py` | CogVideoX target rows are independent CPU tensors. |
| `test_deterministic_noise_seed.py` | Deterministic per-sample noise seeds and strict sidecar metadata. |
| `test_eval_env_aliases.py` | Evaluator environment switches: the `PROBE_EVAL_*` names are read first and the older `Q00_*` aliases still work; require-CUDA and legacy-constants opt-ins. |
| `test_entrypoints_compose.py` | `vidfm3d/train.py`, `vidfm3d/eval_diag.py` (including C3) and `vidfm3d/eval_a1_streaming.py` compose their Hydra configs from any working directory. |
| `test_minimax_bf16_storage_contract.py` | MiniMax-H3 features are stored as finite bf16; other backends keep fp16. |
| `test_nonfinite_handling.py` | NaN/inf features abort training; both evaluators count non-finite predictions instead of scoring them; cache file names. |
| `test_offline_hf_cache_paths.py` | Offline extraction uses the caller's Hugging Face cache. |
| `test_optional_dependency_imports.py` | The evaluators import without the optional point-cloud packages. |
| `test_owner_overlay_label_cache.py` | Owner overlay, C-probe label cache and probe-batch transfer. |
| `test_probe_contracts.py` | B1/B2 conditioning, streaming object selection and other probe input contracts. |
| `test_probe_metric_accounting.py` | Invalid targets never improve losses; metric schemas match across ranks. |
| `test_probe_target_contracts.py` | Strict B2 targets, matched conditioning capacity and the path-integration state update. |
| `test_release_tables.py` | The `results/` tables: row counts, schema, acceptance, unique keys, batch sizes, caveats; and a repository-wide scan for private paths, hosts and e-mail addresses. |
| `test_seeded_local_cache_contract.py` | Seeded feature caches require an exact identity and SHA-256; the runtime protocol hash matches `protocols/inscene_streaming_canonical_v2.json`. |
| `test_training_regressions.py` | Seeding, optimizer logging, autoresume without a logger and post-training test weights. |
| `test_tstep_canonical_manifests.py` | The manifest builder: one shared window universe across prefixes, duplicate and misalignment rejection, feature-namespace labels. |
| `test_probe_recipe.py` | Every experiment and override set that `scripts/probe_recipe.py` produces for the accepted rows exists and composes with Hydra, including the evaluation and C3 override forms (quoted `ckpt_path`, `trainer.devices=[0]`, `action_horizons=[1,2,4]`, `hydra.output_subdir=null`); disk-layer mapping and `streaming_mask` overrides. |
| `test_export_results.py` | `scripts/export_results.py` converts evaluator summaries into the public results schema. |
| `extraction/test_aether_canonical_shape.py` | Aether forward keeps the two-clip axis and requires 81 input frames. |
| `extraction/test_backend_determinism_contracts.py` | Per-backend determinism and noise-provenance contracts. |
| `extraction/test_canonical_gap_worklist.py` | The optional gap-fill worklist hooks (provenance tooling, off by default; see `docs/PROTOCOL.md`, section 4) select exact layers and rows and fail closed. |
| `extraction/test_feature_metadata_merge.py` | Feature metadata merges accept identical values and reject conflicts. |
| `extraction/test_offline_hf.py` | Offline Hub IDs resolve to revision-pinned local snapshots. |
| `extraction/test_qwen_mllm_extractor.py` | The Qwen2.5-VL extractor (`features/run_inscene15k_mllm.py`) uses one 28-observation window for all prefixes and targets. |
