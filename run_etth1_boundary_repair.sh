#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

python_command="${PYTHON:-python}"
dataset_dir="${GTR_DATA_DIR:-./dataset}"
data_root="${GTR_DATA_ROOT:-${dataset_dir}/ETT-small}"
data_path="ETTh1.csv"
gpu="${1:-0}"

if [[ ! -f "${data_root}/${data_path}" ]]; then
    echo "ETTh1 data not found: ${data_root}/${data_path}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu}"

seq_len=96
train_seed=2024
perturb_ratio="${PERTURB_RATIO:-3.0}"
perturb_seed="${PERTURB_SEED:-2024}"
boundary_threshold="${BOUNDARY_THRESHOLD:-4.0}"
pred_lengths=(96 192 336 720)

for pred_len in "${pred_lengths[@]}"; do
    model_id="ETTh1BoundaryRecon_${seq_len}_${pred_len}"
    setting="${model_id}_GTR_ETTh1_ftM_sl${seq_len}_pl${pred_len}_cycle24_seed${train_seed}"
    checkpoint="./checkpoints/${setting}/checkpoint.pth"
    reconstructor="./checkpoints/${setting}/boundary_reconstructor.pth"

    if [[ ! -f "${checkpoint}" || ! -f "${reconstructor}" ]]; then
        echo "Missing ETTh1 H=${pred_len} checkpoint or boundary reconstructor."
        echo "Run: bash ./run_etth1_boundary_reconstruction.sh ${gpu}"
        exit 1
    fi

    echo "ETTh1 H=${pred_len}: clean vs last-point perturbation vs repaired"
    "${python_command}" -u run.py \
        --is_training 0 \
        --model_id "${model_id}" \
        --model GTR \
        --data ETTh1 \
        --root_path "${data_root}" \
        --data_path "${data_path}" \
        --dataset_name ETTh1 \
        --features M \
        --freq h \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in 7 \
        --cycle 24 \
        --dropout 0.5 \
        --batch_size 256 \
        --random_seed "${train_seed}" \
        --perturb_type last \
        --perturb_ratio "${perturb_ratio}" \
        --perturb_seed "${perturb_seed}" \
        --boundary_reconstruct 1 \
        --boundary_threshold "${boundary_threshold}" \
        --boundary_hidden_dim 64 \
        --boundary_epochs 20 \
        --boundary_patience 5 \
        --boundary_learning_rate 0.001
done

echo "ETTh1 boundary-repair robustness comparison complete."
