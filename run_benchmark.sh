#!/bin/bash
# running a benchmark script on multiple gpus parallely
GPUS=(5 5 6 6 7)
seeds=(200 201 202 203 204)
config="bench3"

# Create timestamped run directory
run_timestamp=$(date +"%m%d_%H%M%S")
run_dir="__saved__/results/benchmark/run_${run_timestamp}"
mkdir -p "${run_dir}"
echo "Saving benchmark results to: ${run_dir}"

for i in "${!GPUS[@]}"
do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python benchmark.py --seed ${seeds[$i]} --config ${config} --run_dir "${run_dir}" &
done
wait

echo "All benchmarks complete. Results saved in: ${run_dir}"