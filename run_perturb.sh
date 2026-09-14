#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

python_command="${PYTHON:-python}"
dataset_dir="${GTR_DATA_DIR:-./dataset}"
gpu="${1:-0}"
shift $(( $# > 0 ? 1 : 0 ))

export CUDA_VISIBLE_DEVICES="${gpu}"

model_name="${GTR_MODEL:-GTR}"
seq_len=96
train_seed=2024
perturb_seed=2024
perturb_ratio=3
nte_cutoff_ratio="${NTE_CUTOFF_RATIO:-0.1}"
nte_alpha="${NTE_ALPHA:-1.0}"
nte_gamma_max="${NTE_GAMMA_MAX:-20.0}"
official_datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange Traffic Solar)

if [[ "$#" -eq 0 ]]; then
    requested_datasets=("${official_datasets[@]}")
else
    requested_datasets=("$@")
fi

dataset_file() {
    case "$1" in
        ETTh1) echo "${dataset_dir}/ETT-small/ETTh1.csv" ;;
        ETTh2) echo "${dataset_dir}/ETT-small/ETTh2.csv" ;;
        ETTm1) echo "${dataset_dir}/ETT-small/ETTm1.csv" ;;
        ETTm2) echo "${dataset_dir}/ETT-small/ETTm2.csv" ;;
        Weather) echo "${dataset_dir}/weather/weather.csv" ;;
        Exchange) echo "${dataset_dir}/exchange_rate/exchange_rate.csv" ;;
        Traffic) echo "${dataset_dir}/traffic/traffic.csv" ;;
        Solar) echo "${dataset_dir}/Solar/solar_AL.txt" ;;
        *) return 1 ;;
    esac
}

missing_data=0
for dataset_label in "${requested_datasets[@]}"; do
    if ! data_file="$(dataset_file "${dataset_label}")"; then
        echo "Unknown dataset: ${dataset_label}"
        echo "Available datasets: ${official_datasets[*]}"
        exit 2
    fi
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
    local frequency="${12}"
    local use_revin="${13}"
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
            --dataset_name "${dataset_label}"
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
            --nte_cutoff_ratio "${nte_cutoff_ratio}"
            --nte_alpha "${nte_alpha}"
            --nte_gamma_max "${nte_gamma_max}"
        )

        if [[ -n "${dropout}" ]]; then
            command+=(--dropout "${dropout}")
        fi
        if [[ "${individual}" == "1" ]]; then
            command+=(--individual 1)
        fi
        if [[ -n "${frequency}" ]]; then
            command+=(--freq "${frequency}")
        fi
        if [[ -n "${use_revin}" ]]; then
            command+=(--use_revin "${use_revin}")
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
        --nte-cutoff-ratio "${nte_cutoff_ratio}" \
        --nte-alpha "${nte_alpha}" \
        --nte-gamma-max "${nte_gamma_max}" \
        --expected-pred-lens 96 192 336 720
}

run_named_dataset() {
    case "$1" in
        ETTh1)
            run_dataset ETTh1 ETTh1 ETTh1 \
                "${dataset_dir}/ETT-small" ETTh1.csv \
                7 24 256 0.001 0.5 0 "" ""
            ;;
        ETTh2)
            run_dataset ETTh2 ETTh2 ETTh2 \
                "${dataset_dir}/ETT-small" ETTh2.csv \
                7 24 256 0.001 0.5 0 "" ""
            ;;
        ETTm1)
            run_dataset ETTm1 ETTm1 ETTm1 \
                "${dataset_dir}/ETT-small" ETTm1.csv \
                7 96 256 0.001 0.5 0 "" ""
            ;;
        ETTm2)
            run_dataset ETTm2 ETTm2 ETTm2 \
                "${dataset_dir}/ETT-small" ETTm2.csv \
                7 96 256 0.001 0.5 0 "" ""
            ;;
        Weather)
            run_dataset Weather weather custom \
                "${dataset_dir}/weather" weather.csv \
                21 144 64 0.001 0.5 0 "" ""
            ;;
        Exchange)
            run_dataset Exchange Exchange custom \
                "${dataset_dir}/exchange_rate" exchange_rate.csv \
                8 512 32 0.001 "" 0 d ""
            ;;
        Traffic)
            run_dataset Traffic traffic custom \
                "${dataset_dir}/traffic" traffic.csv \
                862 168 16 0.003 "" 1 "" ""
            ;;
        Solar)
            run_dataset Solar Solar Solar \
                "${dataset_dir}/Solar" solar_AL.txt \
                137 144 64 0.003 "" 0 "" 0
            ;;
    esac
}

echo "Datasets selected: ${requested_datasets[*]}"
for dataset_label in "${requested_datasets[@]}"; do
    run_named_dataset "${dataset_label}"
done

echo "============================================================"
echo "Selected perturbation experiments completed."
echo "Summaries are available in ./results/perturbation/."
echo "============================================================"
