#!/usr/bin/env bash
# Streaming-prefix probe sweep: feature extraction, training, evaluation and C3.
#
# For every model in VFMS, layer in LAYERS, probe in PROBES and prefix in
# PREFIX_LENGTHS the script
#   1. extracts the shared streaming-prefix cache (EXTRACT_STREAMING=1) and, when a
#      C probe is selected, the target-isolated cache (EXTRACT_TARGETS=1);
#   2. trains the probe (TRAIN=1) with the experiment config, overrides,
#      batch size and environment printed by scripts/probe_recipe.py (the recipe
#      of the accepted rows in results/, on the public canonical_v2 split; the
#      recipe's notes are printed as [note] lines, e.g. for the MiniMax-H3 cells
#      whose published rows used the non-public owner subset split, which this
#      sweep does not train on: use scripts/reproduce_rows.py or
#      scripts/probe_recipe.py --split minimax_subset for those);
#   3. evaluates the newest checkpoint (the final epoch after a complete training
#      run; scripts/export_results.py rejects other epochs) on the validation split
#      (EVAL=1): A1 with vidfm3d/eval_a1_streaming.py, every other probe with
#      vidfm3d/eval_diag.py;
#   4. evaluates C3 from every C1 (action_dynamics) checkpoint (EVAL_C3=1).
#
# Required:
#   FEAT_BASE               parent of FEAT_STREAMING_CANONICAL_V2_SEEDED_V1 and
#                           FEAT_TARGET_CANONICAL_V2_SEEDED_V1 (one root per pass,
#                           shared by all models: <root>/<model>/scannetpp/<scene>/...)
#   INSCENE_DATA_ROOT       InsScene-15K data root (DATA_ROOT also accepted); scenes
#                           live in $INSCENE_DATA_ROOT/processed_scannetpp_v2/processed_scannetpp_v2
#                           unless INSCENE_LOCAL_SCENE_ROOT is set
#   CANONICAL_MANIFEST_DIR  output of
#                             python scripts/build_tstep_canonical_manifests.py build --output-dir <dir>
#                           Without it the loader falls back to a different split; set
#                           ALLOW_UNPINNED_SPLIT=1 only if that is what you want.
#
# Optional (defaults in brackets):
#   LINE            canonical | timestep [canonical]
#   VFMS / VFM      models [canonical: dino vjepa2 videomae f3r qwen2_5_vl wan cogvideox
#                   aether minimax_h3 lingbot; timestep: wan cogvideox minimax_h3]
#   PROBES          [streaming_depth streaming_mask view_consistency ego_belief_v2
#                   action_dynamics path_integration]; LINE=timestep adds ego_belief
#   PREFIX_LENGTHS  [8 12 16 24]
#   LAYERS          layer numbers (table numbering, as in results/*.csv: lingbot 48 / minimax_h3 50 are
#                   disk layers 47 / 49) or the tokens default, last, all, accepted [default];
#                   the tokens also resolve to table numbers (lingbot last = 48)
#   T               scheduler timestep: required for LINE=timestep; fixed per model on
#                   the canonical line (749, or 250 for lingbot and minimax_h3)
#   BATCH_SIZE      per-device batch size [the accepted runs' batch size for the cell:
#                   canonical 2, except Wan A1/B2 1, Wan A2/C1/C2 4 and MiniMax-H3 layer 30
#                   A1 1; timestep 1 for A1, 2 otherwise]
#   INSCENE_LOCAL_SCENE_ROOT  ScanNet++ scene folder [see INSCENE_DATA_ROOT]
#   TRAIN EVAL EVAL_C3 EXTRACT_STREAMING EXTRACT_TARGETS  [1]
#   DEV             CUDA device for training and evaluation [0]
#   DRY_RUN         1 prints every command without running anything [0]
#   MODEL_ID        passed to the extractor as --model-id (LingBot and MiniMax-H3 need a
#                   local model path: --model-id, LINGBOT_MODEL_PATH or MINIMAX_H3_MODEL_PATH)
#   EXTRA_EXTRACT EXTRA_TARGET_EXTRACT EXTRA_TRAIN EXTRA_EVAL  extra arguments
#   PYTHON          [python]
#
# Example (what CI runs; no data or GPU needed):
#   DRY_RUN=1 VFMS=dino LINE=canonical FEAT_BASE=/tmp/feat INSCENE_DATA_ROOT=/tmp/data \
#     CANONICAL_MANIFEST_DIR=/tmp/canon EXTRACT_STREAMING=0 EXTRACT_TARGETS=0 \
#     bash scripts/run_streaming_probe_sweep.sh
#
# Runs land in logs/inscene15k_streaming/runs/<split or timestep>/<unique_key>/ with
# eval/val_summary.json (and eval/val_c3_action_sensitivity_summary.json for C1);
# scripts/export_results.py turns them into table rows.
set -euo pipefail

abspath() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *) printf '%s/%s' "$(pwd)" "$1" ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python}
LINE=${LINE:-canonical}
DRY_RUN=${DRY_RUN:-0}
DEV=${DEV:-0}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
EVAL_C3=${EVAL_C3:-1}
EXTRACT_STREAMING=${EXTRACT_STREAMING:-1}
EXTRACT_TARGETS=${EXTRACT_TARGETS:-1}
PREFIX_LENGTHS=${PREFIX_LENGTHS:-"8 12 16 24"}
LAYERS=${LAYERS:-default}
T=${T:-}
BATCH_SIZE=${BATCH_SIZE:-}
MODEL_ID=${MODEL_ID:-}
ALLOW_UNPINNED_SPLIT=${ALLOW_UNPINNED_SPLIT:-0}
# The extractor's seeded-v1 namespaces accept only the canonical temporal protocol,
# so both passes always cache all four prefixes and horizons 1/2/4.
CANONICAL_PREFIXES="8,12,16,24"
CANONICAL_HORIZONS="1,2,4"

case "${LINE}" in
    canonical)
        DEFAULT_VFMS="dino vjepa2 videomae f3r qwen2_5_vl wan cogvideox aether minimax_h3 lingbot"
        DEFAULT_PROBES="streaming_depth streaming_mask view_consistency ego_belief_v2 action_dynamics path_integration"
        ;;
    timestep)
        DEFAULT_VFMS="wan cogvideox minimax_h3"
        DEFAULT_PROBES="streaming_depth streaming_mask view_consistency ego_belief ego_belief_v2 action_dynamics path_integration"
        if [[ -z "${T}" ]]; then
            echo "[err] LINE=timestep needs T (e.g. T=999 for wan/cogvideox, T=1000 for minimax_h3)" >&2
            exit 1
        fi
        ;;
    *)
        echo "[err] LINE must be canonical or timestep, got '${LINE}'" >&2
        exit 1
        ;;
esac
VFMS=${VFMS:-${VFM:-${DEFAULT_VFMS}}}
PROBES=${PROBES:-${DEFAULT_PROBES}}

FEAT_BASE="$(abspath "${FEAT_BASE:?set FEAT_BASE (parent of the FEAT_*_CANONICAL_V2_SEEDED_V1 roots)}")"
DATA_ROOT="$(abspath "${DATA_ROOT:-${INSCENE_DATA_ROOT:?set INSCENE_DATA_ROOT or DATA_ROOT}}")"
# A scene root the caller already set is kept (made absolute before the cd below).
SCENE_ROOT="${INSCENE_LOCAL_SCENE_ROOT:-}"
if [[ -n "${SCENE_ROOT}" ]]; then
    SCENE_ROOT="$(abspath "${SCENE_ROOT}")"
else
    SCENE_ROOT="${DATA_ROOT}/processed_scannetpp_v2/processed_scannetpp_v2"
fi
if [[ -n "${CANONICAL_MANIFEST_DIR:-}" ]]; then
    CANONICAL_MANIFEST_DIR="$(abspath "${CANONICAL_MANIFEST_DIR}")"
elif [[ "${ALLOW_UNPINNED_SPLIT}" != "1" ]]; then
    echo "[err] set CANONICAL_MANIFEST_DIR (python scripts/build_tstep_canonical_manifests.py build --output-dir <dir>)" >&2
    echo "[err] or ALLOW_UNPINNED_SPLIT=1 to train on the loader's fallback split (not the published split)" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=${WANDB_MODE:-offline}
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

STREAM_ROOT="${FEAT_BASE}/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_ROOT="${FEAT_BASE}/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
LOGS_ROOT="${PROJECT_ROOT}/logs/inscene15k_streaming/runs"
# The loader appends /<model>/<source>/<scene> to these roots (as the extractor does),
# and these variables take precedence over the Hydra values.
export INSCENE_DATA_ROOT="${DATA_ROOT}"
export INSCENE_STREAMING_FEAT_ROOT="${STREAM_ROOT}"
export INSCENE_TARGET_FEAT_ROOT="${TARGET_ROOT}"
export INSCENE_LOCAL_SCENE_ROOT="${SCENE_ROOT}"
export INSCENE_TRUST_EXPANDED_MANIFEST=1
export INSCENE_TARGET_LAYOUT=flat

read -r -a extra_extract_args <<< "${EXTRA_EXTRACT:-}"
read -r -a extra_target_extract_args <<< "${EXTRA_TARGET_EXTRACT:-}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN:-}"
read -r -a extra_eval_args <<< "${EXTRA_EVAL:-}"

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# Bash RECIPE_* variables for one cell (scripts/probe_recipe.py --format bash).
load_recipe() {
    local probe="$1" prefix="$2" layer="$3" vars
    local cmd=("${PYTHON}" scripts/probe_recipe.py --line "${LINE}" --model "${VFM}"
               --probe "${probe}" --prefix "${prefix}" --layer "${layer}" --format bash)
    if [[ -n "${T}" ]]; then
        cmd+=(--timestep "${T}")
    fi
    if [[ -n "${BATCH_SIZE}" ]]; then
        cmd+=(--batch-size "${BATCH_SIZE}")
    fi
    vars="$("${cmd[@]}")"
    eval "${vars}"
}

# resolve_feature_layers.py prints disk (0-based) layers; LAYERS and the recipe use
# table numbers (lingbot disk 47 = table 48, minimax_h3 disk 49 = table 50).
to_ledger_layers() {
    "${PYTHON}" -c 'import sys; sys.path.insert(0, "scripts"); import probe_recipe as p; print(*(p.ledger_layer(sys.argv[1], int(t)) for t in sys.stdin.read().split()), sep="\n")' "${VFM}"
}

resolve_layer_token() {
    case "$1" in
        default|last)
            "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --field "$1_layer" | to_ledger_layers
            ;;
        all)
            "${PYTHON}" scripts/resolve_feature_layers.py --vfm "${VFM}" --format list | to_ledger_layers
            ;;
        accepted)
            "${PYTHON}" -c 'import sys; sys.path.insert(0, "scripts"); import probe_recipe as p; print(*p.ACCEPTED_LAYERS[sys.argv[1]].get(sys.argv[2], ()), sep="\n")' "${LINE}" "${VFM}"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

# Newest Lightning checkpoint of a run. No ModelCheckpoint callback is configured
# (callbacks/v3_checkpoint.yaml has no _target_), so Lightning's default
# <run>/VideoProbe3D/<id>/checkpoints/epoch=E-step=S.ckpt is the only one; there is no last.ckpt.
newest_ckpt() {
    local run="$1" best="" f
    for f in "${run}"/VideoProbe3D/*/checkpoints/epoch=*-step=*.ckpt; do
        [[ -f "${f}" ]] || continue
        if [[ -z "${best}" || "${f}" -nt "${best}" ]]; then best="${f}"; fi
    done
    if [[ -z "${best}" ]]; then
        for f in "${run}"/checkpoints/epoch=*-step=*.ckpt; do
            [[ -f "${f}" ]] || continue
            if [[ -z "${best}" || "${f}" -nt "${best}" ]]; then best="${f}"; fi
        done
    fi
    printf '%s' "${best}"
}

probe_needs_targets() {
    [[ "$1" == "action_dynamics" || "$1" == "path_integration" ]]
}

for prefix_len in ${PREFIX_LENGTHS}; do
    case " 8 12 16 24 " in
        *" ${prefix_len} "*) ;;
        *) echo "[err] PREFIX_LENGTHS must be drawn from 8 12 16 24, got ${prefix_len}" >&2; exit 1 ;;
    esac
    if [[ -n "${CANONICAL_MANIFEST_DIR:-}" && ! -f "${CANONICAL_MANIFEST_DIR}/inscene_c_h124_prefix${prefix_len}.jsonl" ]]; then
        echo "[err] missing ${CANONICAL_MANIFEST_DIR}/inscene_c_h124_prefix${prefix_len}.jsonl" >&2
        exit 1
    fi
done
if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${MPLCONFIGDIR}"
fi
needs_targets=0
for probe in ${PROBES}; do
    if probe_needs_targets "${probe}"; then
        needs_targets=1
    fi
done

echo "[info] line=${LINE} vfms=${VFMS}"
echo "[info] probes=${PROBES} prefix_lengths=${PREFIX_LENGTHS}"
echo "[info] streaming_feat_root=${STREAM_ROOT} target_feat_root=${TARGET_ROOT}"
echo "[info] manifests=${CANONICAL_MANIFEST_DIR:-<unpinned>}"

for VFM in ${VFMS}; do
    echo "==== model ${VFM} ===="
    layers=()
    seen=" "
    for token in ${LAYERS}; do
        while read -r layer; do
            if [[ -z "${layer}" || "${seen}" == *" ${layer} "* ]]; then
                continue
            fi
            seen+="${layer} "
            layers+=("${layer}")
        done < <(resolve_layer_token "${token}")
    done
    if (( ${#layers[@]} == 0 )); then
        echo "[err] no layers resolved for ${VFM} from LAYERS='${LAYERS}'" >&2
        exit 1
    fi

    # Disk layers and the extraction timestep come from the recipe.
    disk_layers=()
    for layer in "${layers[@]}"; do
        load_recipe streaming_depth 8 "${layer}"
        disk_layers+=("${RECIPE_DISK_LAYER}")
    done
    extract_t="${RECIPE_TIMESTEP}"
    echo "[info] ${VFM}: layers=${layers[*]} disk_layers=${disk_layers[*]} timestep=${extract_t}"

    if [[ "${VFM}" == "qwen2_5_vl" ]]; then
        extractor=("${PYTHON}" -m features.run_inscene15k_mllm --backend qwen2_5_vl --vfm-name qwen2_5_vl)
    else
        extractor=("${PYTHON}" -m features.run_inscene15k --vfm "${VFM}")
        case "${VFM}" in
            wan|cogvideox|aether|minimax_h3|lingbot) extractor+=(--t "${extract_t}") ;;
        esac
    fi
    if [[ -n "${MODEL_ID}" ]]; then
        extractor+=(--model-id "${MODEL_ID}")
    fi

    if [[ "${EXTRACT_STREAMING}" == "1" ]]; then
        echo "==== extract streaming-prefix cache: ${VFM} ===="
        run_cmd env "FEATURE_NAMESPACE=$(basename "${STREAM_ROOT}")" "${extractor[@]}" \
            --mode streaming_prefix --data-root "${DATA_ROOT}" --out-root "${STREAM_ROOT}" \
            --source scannetpp --output-layers "${disk_layers[@]}" \
            --prefix-lengths "${CANONICAL_PREFIXES}" \
            ${extra_extract_args[@]+"${extra_extract_args[@]}"}
    fi
    if [[ "${EXTRACT_TARGETS}" == "1" && "${needs_targets}" == "1" ]]; then
        echo "==== extract target-isolated cache for C probes: ${VFM} ===="
        run_cmd env "FEATURE_NAMESPACE=$(basename "${TARGET_ROOT}")" "${extractor[@]}" \
            --mode target_isolated --data-root "${DATA_ROOT}" --out-root "${TARGET_ROOT}" \
            --source scannetpp --output-layers "${disk_layers[@]}" \
            --prefix-lengths "${CANONICAL_PREFIXES}" --target-from-streaming-windows \
            --target-prefix-lengths "${CANONICAL_PREFIXES}" --target-horizons "${CANONICAL_HORIZONS}" \
            --num-targets 0 \
            ${extra_target_extract_args[@]+"${extra_target_extract_args[@]}"}
    fi

    for probe in ${PROBES}; do
        for prefix_len in ${PREFIX_LENGTHS}; do
            split_env=()
            if [[ -n "${CANONICAL_MANIFEST_DIR:-}" ]]; then
                manifest="${CANONICAL_MANIFEST_DIR}/inscene_c_h124_prefix${prefix_len}.jsonl"
                split_env=("INSCENE_MANIFEST=${manifest}"
                           "INSCENE_MANIFEST_SHA256=$(sha256sum "${manifest}" | cut -d' ' -f1)")
            fi
            for layer in "${layers[@]}"; do
                load_recipe "${probe}" "${prefix_len}" "${layer}"
                run_dir="${LOGS_ROOT}/${RECIPE_RUN_NAME}"
                names=("job_name=${RECIPE_UNIQUE_KEY}" "paths.run_folder_name=${RECIPE_RUN_NAME}")
                base_env=(env "CUDA_VISIBLE_DEVICES=${DEV}" ${split_env[@]+"${split_env[@]}"} "${RECIPE_ENV[@]}")

                for note in ${RECIPE_NOTES[@]+"${RECIPE_NOTES[@]}"}; do
                    echo "[note] ${RECIPE_UNIQUE_KEY}: ${note}"
                done

                if [[ "${TRAIN}" == "1" ]]; then
                    echo "==== train ${RECIPE_UNIQUE_KEY} (${RECIPE_EXPERIMENT}, batch ${RECIPE_BATCH}) ===="
                    run_cmd "${base_env[@]}" "${PYTHON}" vidfm3d/train.py "${RECIPE_TRAIN_ARGS[@]}" "${names[@]}" \
                        ${extra_train_args[@]+"${extra_train_args[@]}"}
                fi

                do_c3=0
                if [[ "${EVAL_C3}" == "1" && "${probe}" == "action_dynamics" ]]; then
                    do_c3=1
                fi
                if [[ "${EVAL}" != "1" && "${do_c3}" != "1" ]]; then
                    continue
                fi
                ckpt="$(newest_ckpt "${run_dir}")"
                if [[ -z "${ckpt}" ]]; then
                    if [[ "${DRY_RUN}" != "1" ]]; then
                        echo "[warn] no epoch=*-step=*.ckpt under ${run_dir}; skipping evaluation" >&2
                        continue
                    fi
                    ckpt="${run_dir}/VideoProbe3D/<id>/checkpoints/epoch=<E>-step=<S>.ckpt"
                fi
                if [[ "${ckpt}" == *"'"* ]]; then
                    echo "[err] checkpoint path contains a single quote: ${ckpt}" >&2
                    exit 1
                fi
                # Lightning names contain '=', so the path must be a quoted Hydra string.
                ckpt_arg="ckpt_path='${ckpt}'"
                if [[ "${EVAL}" == "1" ]]; then
                    echo "==== eval ${RECIPE_UNIQUE_KEY} (${RECIPE_EVALUATOR}) ===="
                    run_cmd "${base_env[@]}" "${PYTHON}" "${RECIPE_EVALUATOR}" "${RECIPE_EVAL_ARGS[@]}" \
                        "${ckpt_arg}" "${names[@]}" ${extra_eval_args[@]+"${extra_eval_args[@]}"}
                fi
                if [[ "${do_c3}" == "1" ]]; then
                    echo "==== C3 from C1 ${RECIPE_UNIQUE_KEY} ===="
                    run_cmd "${base_env[@]}" "${PYTHON}" vidfm3d/eval_diag.py "${RECIPE_EVAL_ARGS[@]}" \
                        "${RECIPE_C3_ARGS[@]}" "${ckpt_arg}" "${names[@]}" \
                        ${extra_eval_args[@]+"${extra_eval_args[@]}"}
                fi
            done
        done
    done
done
