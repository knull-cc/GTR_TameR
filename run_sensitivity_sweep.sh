#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

python_command="${PYTHON:-python}"
if [[ -n "${GTR_DATA_ROOT:-}" ]]; then
    data_root="${GTR_DATA_ROOT}"
else
    dataset_dir="${GTR_DATA_DIR:-./dataset}"
    data_root="${dataset_dir}/ETT-small"
fi
data_path="ETTh1.csv"
gpu="${1:-0}"

if [[ ! -f "${data_root}/${data_path}" ]]; then
    echo "ETTh1 data not found: ${data_root}/${data_path}"
    echo "Set GTR_DATA_DIR to the directory containing ETT-small/, or"
    echo "set GTR_DATA_ROOT directly to the directory containing ETTh1.csv."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu}"

model_name="GTR"
seq_len=96
train_seed=2024
perturb_seed=2024
perturb_ratio=3
pred_lengths=(96 192 336 720)
offsets=(96 72 50 36 24 16 10 8 6 5 4 3 2 1)

common_args=(
    --root_path "${data_root}"
    --data_path "${data_path}"
    --dataset_name ETTh1
    --model "${model_name}"
    --data ETTh1
    --features M
    --seq_len "${seq_len}"
    --enc_in 7
    --cycle 24
    --train_epochs 30
    --patience 5
    --dropout 0.5
    --itr 1
    --batch_size 256
    --learning_rate 0.001
    --random_seed "${train_seed}"
)

for pred_len in "${pred_lengths[@]}"; do
    model_id="ETTh1_${seq_len}_${pred_len}"
    setting="${model_id}_${model_name}_ETTh1_ftM_sl${seq_len}_pl${pred_len}_cycle24_seed${train_seed}"
    checkpoint="./checkpoints/${setting}/checkpoint.pth"

    if [[ -f "${checkpoint}" && "${GTR_FORCE_RETRAIN:-0}" != "1" ]]; then
        echo "Using existing checkpoint for H=${pred_len}: ${checkpoint}"
    else
        echo "Training original GTR on ETTh1 for H=${pred_len}"
        "${python_command}" -u run.py \
            --is_training 1 \
            --model_id "${model_id}" \
            --pred_len "${pred_len}" \
            --perturb_type none \
            "${common_args[@]}"
    fi

    for offset in "${offsets[@]}"; do
        echo "Testing H=${pred_len}, perturbed position=-${offset}"
        "${python_command}" -u run.py \
            --is_training 0 \
            --model_id "${model_id}" \
            --pred_len "${pred_len}" \
            --perturb_type point \
            --perturb_offset "${offset}" \
            --perturb_ratio "${perturb_ratio}" \
            --perturb_seed "${perturb_seed}" \
            "${common_args[@]}"
    done
done

"${python_command}" scripts/summarize_sensitivity_sweep.py \
    --results-dir ./results \
    --output-dir ./results/perturbation_sweep \
    --dataset ETTh1 \
    --model "${model_name}" \
    --seq-len "${seq_len}" \
    --perturb-ratio "${perturb_ratio}" \
    --train-seed "${train_seed}" \
    --perturb-seed "${perturb_seed}" \
    --offsets "${offsets[@]}" \
    --expected-pred-lens "${pred_lengths[@]}"

echo "Sensitivity sweep complete."
echo "Figures and tables: ./results/perturbation_sweep/"
