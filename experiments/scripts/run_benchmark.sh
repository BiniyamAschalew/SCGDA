#!/bin/bash
# running a benchmark script on multiple GPUs in parallel
set -u

GPUS=(0 4 5 6 7)
seeds=(200 201 202 203 204)
config="bench5"

run_timestamp=$(date +"%m%d_%H%M%S")
run_dir="../__saved__/results/benchmark/${config}/run_${run_timestamp}"
mkdir -p "${run_dir}"
echo "Saving benchmark results to: ${run_dir}"

pids=()
for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python benchmark.py --seed ${seeds[$i]} --config ${config} --run_dir "${run_dir}" &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

echo "Per-seed jobs finished. Aggregating tables..."
python benchmark.py --config "${config}" --run_dir "${run_dir}" --aggregate_only

if [ "${failed}" -ne 0 ]; then
    echo "Benchmark completed with failures in one or more seed jobs. Aggregated results were still generated."
    exit 1
fi

echo "All benchmarks complete. Results saved in: ${run_dir}"
