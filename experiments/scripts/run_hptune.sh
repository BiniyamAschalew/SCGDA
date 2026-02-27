#!/bin/bash
# Run HP tuning in parallel across GPUs/seeds.
set -euo pipefail

GPUS=(0 4 5 3 7)
SEEDS=(2026 2027 2028 2029 2030)

CONFIG="${1:-hp0}"
SPACE_DIR="${2:-__hps__/space/default}"
RUN_ID="${3:-$(date +%m%d_%H%M%S)}"

if [ "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]; then
  echo "GPUS and SEEDS arrays must be the same length." >&2
  exit 1
fi

# Run sequentially for single GPU, or in parallel for multiple
if [ "${#GPUS[@]}" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} python hp_tune.py --seed "${SEEDS[0]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}"
else
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python hp_tune.py --seed "${SEEDS[$i]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}" &
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

echo "HP tuning complete. run_id=${RUN_ID}, config=${CONFIG}, space_dir=${SPACE_DIR}"
