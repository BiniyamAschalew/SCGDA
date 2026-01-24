#!/bin/bash
# running a benchmark script on multiple gpus parallely
GPUS=(5 5 6 6 7)
seeds=(200 201 202 203 204)
config="bench2"

for i in "${!GPUS[@]}"
do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python benchmark.py --seed ${seeds[$i]} --config ${config} &
done
wait