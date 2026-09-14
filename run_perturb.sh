#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

python_command="${PYTHON:-python}"
dataset_dir="${GTR_DATA_DIR:-./dataset}"
gpu="${1:-0}"

export CUDA_VISIBLE_DEVICES="${gpu}"

model_name="GTR"
seq_len=96
train_seed=2024
perturb_seed=2024
perturb_ratio=3

required_files=(
    "${dataset_dir}/ETT-small/ETTh1.csv"
    "${dataset_dir}/ETT-small/ETTh2.csv"
    "${dataset_dir}/ETT-small/ETTm1.csv"
    "${dataset_dir}/ETT-small/ETTm2.csv"
    "${dataset_dir}/electricity/electricity.csv"
    "${dataset_dir}/traffic/traffic.csv"
    "${dataset_dir}/weather/weather.csv"
)

missing_data=0
for data_file in "${required_files[@]}"; do
    if [[ ! -f "${data_file}" ]]; then
        echo "Dataset file not found: ${data_file}"
        missing_data=1
    fi
done

if [[ "${missing_data}" -ne 0 ]]; then
    echo "Set GTR_DATA_DIR to the directory containing the dataset folders."
    exit 1
fi

run_dataset() {
    local dataset_label="$1"
    local model_id_prefix="$2"
    local data_name="$3"
    local data_root="$4"
    local data_path="$5"
    local enc_in="$6"
    local cycle="$7"
    local batch_size="$8"
    local learning_rate="$9"
    local dropout="${10}"
    local individual="${11}"
    local -a command

    echo "============================================================"
    echo "Running ${dataset_label}: recent single anomalous point"
    echo "============================================================"

    for pred_len in 96 192 336 720; do
        command=(
            "${python_command}" -u run.py
            --is_training 1
            --root_path "${data_root}"
            --data_path "${data_path}"
            --model_id "${model_id_prefix}_${seq_len}_${pred_len}"
            --model "${model_name}"
            --data "${data_name}"
            --features M
            --seq_len "${seq_len}"
            --pred_len "${pred_len}"
            --enc_in "${enc_in}"
            --cycle "${cycle}"
            --train_epochs 30
            --patience 5
            --itr 1
            --batch_size "${batch_size}"
            --learning_rate "${learning_rate}"
            --random_seed "${train_seed}"
            --perturb_type last
            --perturb_ratio "${perturb_ratio}"
            --perturb_seed "${perturb_seed}"
        )

        if [[ -n "${dropout}" ]]; then
            command+=(--dropout "${dropout}")
        fi
        if [[ "${individual}" == "1" ]]; then
            command+=(--individual 1)
        fi

        "${command[@]}"
    done

    "${python_command}" scripts/summarize_perturb.py \
        --results-dir ./results \
        --output-dir ./results/perturbation \
        --dataset "${dataset_label}" \
        --model "${model_name}" \
        --seq-len "${seq_len}" \
        --perturb-type last \
        --perturb-ratio "${perturb_ratio}" \
        --train-seed "${train_seed}" \
        --perturb-seed "${perturb_seed}" \
        --expected-pred-lens 96 192 336 720
}

run_dataset \
    ETTh1 ETTh1 ETTh1 \
    "${dataset_dir}/ETT-small" ETTh1.csv \
    7 24 256 0.001 0.5 0

run_dataset \
    ETTh2 ETTh2 ETTh2 \
    "${dataset_dir}/ETT-small" ETTh2.csv \
    7 24 256 0.001 0.5 0

run_dataset \
    ETTm1 ETTm1 ETTm1 \
    "${dataset_dir}/ETT-small" ETTm1.csv \
    7 96 256 0.001 0.5 0

run_dataset \
    ETTm2 ETTm2 ETTm2 \
    "${dataset_dir}/ETT-small" ETTm2.csv \
    7 96 256 0.001 0.5 0

run_dataset \
    electricity Electricity custom \
    "${dataset_dir}/electricity" electricity.csv \
    321 168 32 0.003 "" 0

run_dataset \
    traffic traffic custom \
    "${dataset_dir}/traffic" traffic.csv \
    862 168 16 0.003 "" 1

run_dataset \
    weather weather custom \
    "${dataset_dir}/weather" weather.csv \
    21 144 64 0.001 0.5 0

echo "============================================================"
echo "All seven perturbation experiments completed."
echo "Summaries are available in ./results/perturbation/."
echo "============================================================"
