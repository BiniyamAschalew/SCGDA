#!/bin/bash
# Run HP tuning in parallel across GPUs/seeds.
set -euo pipefail

GPUS=(1 2 3 4 5)
SEEDS=(2026 2027 2028 2029 2030)

CONFIG="hp9"
SPACE_DIR="./__hps__/space/fan.yaml"
RUN_ID="$(date +%m%d_%H%M%S)"
USE_IMPORTED="0"

EXTRA_ARGS=()
if [[ "${USE_IMPORTED}" == "1" || "${USE_IMPORTED}" == "true" || "${USE_IMPORTED}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-imported)
fi

if [ "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]; then
  echo "GPUS and SEEDS arrays must be the same length." >&2
  exit 1
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

echo "HP tuning complete. run_id=${RUN_ID}, config=${CONFIG}, space_dir=${SPACE_DIR}, use_imported=${USE_IMPORTED}"
