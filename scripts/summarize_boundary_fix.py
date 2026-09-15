#!/usr/bin/env python3
"""Summarize original-versus-boundary-fixed forecasting metrics."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate deterministic last-point boundary-fix results across "
            "prediction lengths."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/boundary_fix"),
    )
    parser.add_argument("--dataset", default="Exchange")
    parser.add_argument("--model", default="GTR")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--train-seed", type=int, default=2024)
    parser.add_argument(
        "--expected-pred-lens",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
    )
    return parser.parse_args()


def improvement_percent(original, fixed):
    if original == 0:
        return None
    return (original - fixed) / original * 100.0


def matches_experiment(result, args):
    boundary_fix = result.get("boundary_fix", {})
    dataset = result.get("dataset", result.get("data"))
    return (
        dataset == args.dataset
        and result.get("model") == args.model
        and result.get("seq_len") == args.seq_len
        and result.get("train_seed") == args.train_seed
        and boundary_fix.get("enabled") is True
        and boundary_fix.get("method") == "last_value"
        and "clean" in result
        and "fixed" in result
    )


def result_to_row(result):
    original_mse = float(result["clean"]["mse"])
    fixed_mse = float(result["fixed"]["mse"])
    original_mae = float(result["clean"]["mae"])
    fixed_mae = float(result["fixed"]["mae"])
    return {
        "pred_len": int(result["pred_len"]),
        "original_mse": original_mse,
        "fixed_mse": fixed_mse,
        "mse_improvement_absolute": original_mse - fixed_mse,
        "mse_improvement_percent": improvement_percent(
            original_mse, fixed_mse
        ),
        "original_mae": original_mae,
        "fixed_mae": fixed_mae,
        "mae_improvement_absolute": original_mae - fixed_mae,
        "mae_improvement_percent": improvement_percent(
            original_mae, fixed_mae
        ),
    }


def average_rows(rows):
    original_mse = mean(row["original_mse"] for row in rows)
    fixed_mse = mean(row["fixed_mse"] for row in rows)
    original_mae = mean(row["original_mae"] for row in rows)
    fixed_mae = mean(row["fixed_mae"] for row in rows)
    return {
        "pred_len": "average",
        "original_mse": original_mse,
        "fixed_mse": fixed_mse,
        "mse_improvement_absolute": original_mse - fixed_mse,
        "mse_improvement_percent": improvement_percent(
            original_mse, fixed_mse
        ),
        "original_mae": original_mae,
        "fixed_mae": fixed_mae,
        "mae_improvement_absolute": original_mae - fixed_mae,
        "mae_improvement_percent": improvement_percent(
            original_mae, fixed_mae
        ),
    }


def main():
    args = parse_args()
    by_horizon = {}
    source_files = {}

    for result_file in args.results_dir.rglob(
        "boundary_fix_last_value.json"
    ):
        with result_file.open("r", encoding="utf-8") as input_file:
            result = json.load(input_file)
        if not matches_experiment(result, args):
            continue
        pred_len = int(result["pred_len"])
        if pred_len in by_horizon:
            raise RuntimeError(
                "Duplicate boundary-fix results for prediction length {}: "
                "{} and {}".format(
                    pred_len, source_files[pred_len], result_file
                )
            )
        by_horizon[pred_len] = result_to_row(result)
        source_files[pred_len] = str(result_file)

    missing = [
        pred_len
        for pred_len in args.expected_pred_lens
        if pred_len not in by_horizon
    ]
    if missing:
        raise RuntimeError(
            "Missing boundary-fix results for prediction lengths: {}".format(
                ", ".join(str(value) for value in missing)
            )
        )

    rows = [by_horizon[pred_len] for pred_len in args.expected_pred_lens]
    average = average_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.dataset}_{args.model}_sl{args.seq_len}_"
        "boundary_fix_last_value"
    )
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    fieldnames = list(average.keys())

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows + [average])

    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "model": args.model,
        "seq_len": args.seq_len,
        "train_seed": args.train_seed,
        "boundary_fix": {
            "method": "last_value",
            "definition": "x[:, -1, :] = x[:, -2, :]",
            "test_time_only": True,
        },
        "prediction_lengths": args.expected_pred_lens,
        "rows": rows,
        "average": average,
        "source_files": [
            source_files[pred_len] for pred_len in args.expected_pred_lens
        ],
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        "Average original MSE/MAE: "
        f"{average['original_mse']:.6f}/{average['original_mae']:.6f}"
    )
    print(
        "Average fixed MSE/MAE: "
        f"{average['fixed_mse']:.6f}/{average['fixed_mae']:.6f}"
    )
    print(
        "Average improvement MSE/MAE: "
        f"{average['mse_improvement_percent']:.2f}%/"
        f"{average['mae_improvement_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
