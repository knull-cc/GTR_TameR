#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gtr_root="$(cd "${script_dir}/../.." && pwd)"
cd "${gtr_root}"

python_command="${PYTHON:-python}"
data_root="${GTR_DATA_ROOT:-./dataset}"
data_path="ETTh1.csv"

if [[ ! -f "${data_root}/${data_path}" ]]; then
    echo "ETTh1 data not found at ${data_root}/${data_path}."
    echo "Set GTR_DATA_ROOT to the directory containing ETTh1.csv and retry."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${1:-0}"

model_name="GTR"
seq_len=96
train_seed=2024
perturb_seed=2024
perturb_ratio=3

for pred_len in 96 192 336 720; do
    "${python_command}" -u run.py \
        --is_training 1 \
        --root_path "${data_root}" \
        --data_path "${data_path}" \
        --model_id "ETTh1_${seq_len}_${pred_len}" \
        --model "${model_name}" \
        --data ETTh1 \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in 7 \
        --cycle 24 \
        --train_epochs 30 \
        --patience 5 \
        --dropout 0.5 \
        --itr 1 \
        --batch_size 256 \
        --learning_rate 0.001 \
        --random_seed "${train_seed}" \
        --perturb_type last \
        --perturb_ratio "${perturb_ratio}" \
        --perturb_seed "${perturb_seed}"
done

"${python_command}" scripts/summarize_perturb.py \
    --results-dir ./results \
    --output-dir ./results/perturbation \
    --dataset ETTh1 \
    --model "${model_name}" \
    --seq-len "${seq_len}" \
    --perturb-type last \
    --perturb-ratio "${perturb_ratio}" \
    --train-seed "${train_seed}" \
    --perturb-seed "${perturb_seed}" \
    --expected-pred-lens 96 192 336 720
