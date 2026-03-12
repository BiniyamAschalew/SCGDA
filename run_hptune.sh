#!/bin/bash
# Run HP tuning in parallel across GPUs/seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPUS=(0 1 7)
# GPUS=(5 6 7)

SEEDS=(2026 2027 2028)

CONFIG="${CONFIG:-hp_opal}"
SPACE_DIR="${SPACE_DIR:-./__hps__/space/opal.yaml}"
RUN_ID="$(date +%m%d_%H%M%S)"
# Default: start from imported per-pair HPs, then override overlapping keys via safe_space search.
USE_IMPORTED="${USE_IMPORTED:-0}"
FROM_PYGDA="${FROM_PYGDA:-0}"
DRY_RUN_PRECHECK="${DRY_RUN_PRECHECK:-1}"

EXTRA_ARGS=()
if [[ "${USE_IMPORTED}" == "1" || "${USE_IMPORTED}" == "true" || "${USE_IMPORTED}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-imported)
fi
if [[ "${FROM_PYGDA}" == "1" || "${FROM_PYGDA}" == "true" || "${FROM_PYGDA}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--from-pygda)
fi

if [ "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]; then
  echo "GPUS and SEEDS arrays must be the same length." >&2
  exit 1
fi

if [[ "${DRY_RUN_PRECHECK}" == "1" || "${DRY_RUN_PRECHECK}" == "true" || "${DRY_RUN_PRECHECK}" == "TRUE" ]]; then
  echo "Running dry-run precheck..."
  python hp_tune.py --seed "${SEEDS[0]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}" "${EXTRA_ARGS[@]}" --dry-run
fi

# Run sequentially for single GPU, or in parallel for multiple
if [ "${#GPUS[@]}" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} python hp_tune.py --seed "${SEEDS[0]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}" "${EXTRA_ARGS[@]}"
else
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python hp_tune.py --seed "${SEEDS[$i]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}" "${EXTRA_ARGS[@]}" &
    pids+=($!)
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    echo "HP tuning completed with failures in one or more seed jobs." >&2
    exit 1
  fi
fi

echo "HP tuning complete. run_id=${RUN_ID}, config=${CONFIG}, space_dir=${SPACE_DIR}, use_imported=${USE_IMPORTED}, from_pygda=${FROM_PYGDA}"
