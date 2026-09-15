#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

python_command="${PYTHON:-python}"
if [[ -n "${GTR_DATA_ROOT:-}" ]]; then
    data_root="${GTR_DATA_ROOT}"
else
    dataset_dir="${GTR_DATA_DIR:-./dataset}"
    data_root="${dataset_dir}/exchange_rate"
fi
data_path="exchange_rate.csv"
gpu="${1:-0}"

if [[ ! -f "${data_root}/${data_path}" ]]; then
    echo "Exchange data not found: ${data_root}/${data_path}"
    echo "Set GTR_DATA_DIR to the directory containing exchange_rate/, or"
    echo "set GTR_DATA_ROOT directly to the directory containing exchange_rate.csv."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu}"

model_name="GTR"
seq_len=96
train_seed=2024
boundary_threshold="${BOUNDARY_THRESHOLD:-3.0}"
boundary_hidden_dim="${BOUNDARY_HIDDEN_DIM:-64}"
boundary_epochs="${BOUNDARY_EPOCHS:-20}"
pred_lengths=(96 192 336 720)

common_args=(
    --root_path "${data_root}"
    --data_path "${data_path}"
    --dataset_name Exchange
    --model "${model_name}"
    --data custom
    --features M
    --freq d
    --seq_len "${seq_len}"
    --enc_in 8
    --cycle 512
    --train_epochs 30
    --patience 5
    --itr 1
    --batch_size 32
    --learning_rate 0.001
    --random_seed "${train_seed}"
    --boundary_reconstruct 1
    --boundary_threshold "${boundary_threshold}"
    --boundary_hidden_dim "${boundary_hidden_dim}"
    --boundary_epochs "${boundary_epochs}"
    --boundary_patience 5
    --boundary_learning_rate 0.001
)

for pred_len in "${pred_lengths[@]}"; do
    model_id="Exchange_${seq_len}_${pred_len}"

    echo "Training clean original GTR from scratch on Exchange for H=${pred_len}"
    echo "Training the independent prefix-only boundary reconstructor"
    echo "Testing original input versus MAD-filtered reconstructed boundary"
    "${python_command}" -u run.py \
        --is_training 1 \
        --model_id "${model_id}" \
        --pred_len "${pred_len}" \
        --perturb_type none \
        "${common_args[@]}"
done

"${python_command}" scripts/summarize_boundary_reconstruction.py \
    --results-dir ./results \
    --output-dir ./results/boundary_reconstruction \
    --dataset Exchange \
    --model "${model_name}" \
    --seq-len "${seq_len}" \
    --train-seed "${train_seed}" \
    --threshold "${boundary_threshold}" \
    --expected-pred-lens "${pred_lengths[@]}"

echo "Exchange filtered boundary-reconstruction experiment complete."
echo "Summary: ./results/boundary_reconstruction/"
