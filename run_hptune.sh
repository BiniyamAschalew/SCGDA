#!/bin/bash
# Run hp tuning in parallel across GPUs/seeds.
GPUS=(0 1 2 4 5)
SEEDS=(200 201 202 203 204)

# GPUS=(3)
# SEEDS=(200)

CONFIG="hp0"
SPACE_DIR="__hps__/space/custom"
RUN_ID="${2:-$(date +%m%d_%H%M%S)}"

if [ "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]; then
  echo "GPUS and SEEDS arrays must be the same length." >&2
  exit 1
fi

# Run sequentially for single GPU, or in parallel for multiple
if [ "${#GPUS[@]}" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} python hp_tune.py --seed "${SEEDS[0]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}"
else
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python hp_tune.py --seed "${SEEDS[$i]}" --config "${CONFIG}" --run-id "${RUN_ID}" --space-dir "${SPACE_DIR}" &
  done
  wait
fi
