#!/usr/bin/env python3
"""Summarize filtered context-boundary reconstruction experiments."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/boundary_reconstruction"),
    )
    parser.add_argument("--dataset", default="Exchange")
    parser.add_argument("--model", default="GTR")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--train-seed", type=int, default=2024)
    parser.add_argument("--model-id-prefix", default=None)
    parser.add_argument("--output-tag", default="boundary_reconstruct")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument(
        "--expected-pred-lens",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
    )
    return parser.parse_args()


def numeric_tag(value):
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def improvement_percent(original, fixed):
    if original == 0:
        return None
    return (original - fixed) / original * 100.0


def matches_experiment(result, args):
    reconstruction = result.get("boundary_reconstruction", {})
    dataset = result.get("dataset", result.get("data"))
    try:
        threshold_matches = math.isclose(
            float(reconstruction["threshold"]),
            args.threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        dataset == args.dataset
        and result.get("model") == args.model
        and result.get("seq_len") == args.seq_len
        and result.get("train_seed") == args.train_seed
        and reconstruction.get("enabled") is True
        and reconstruction.get("method") == "context_mlp_delta"
        and reconstruction.get("filter") == "last_increment_mad"
        and threshold_matches
        and "clean" in result
        and "fixed" in result
        and (
            args.model_id_prefix is None
            or str(result.get("model_id", "")).startswith(
                args.model_id_prefix
            )
        )
    )


def result_to_row(result):
    original_mse = float(result["clean"]["mse"])
    fixed_mse = float(result["fixed"]["mse"])
    original_mae = float(result["clean"]["mae"])
    fixed_mae = float(result["fixed"]["mae"])
    reconstruction = result["boundary_reconstruction"]
    return {
        "pred_len": int(result["pred_len"]),
        "original_mse": original_mse,
        "fixed_mse": fixed_mse,
        "mse_improvement_percent": improvement_percent(
            original_mse, fixed_mse
        ),
        "original_mae": original_mae,
        "fixed_mae": fixed_mae,
        "mae_improvement_percent": improvement_percent(
            original_mae, fixed_mae
        ),
        "replacement_rate": float(reconstruction["replacement_rate"]),
        "reconstruction_mse": float(reconstruction["reconstruction_mse"]),
        "persistence_mse": float(reconstruction["persistence_mse"]),
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
        "mse_improvement_percent": improvement_percent(
            original_mse, fixed_mse
        ),
        "original_mae": original_mae,
        "fixed_mae": fixed_mae,
        "mae_improvement_percent": improvement_percent(
            original_mae, fixed_mae
        ),
        "replacement_rate": mean(row["replacement_rate"] for row in rows),
        "reconstruction_mse": mean(
            row["reconstruction_mse"] for row in rows
        ),
        "persistence_mse": mean(row["persistence_mse"] for row in rows),
    }


def main():
    args = parse_args()
    by_horizon = {}
    source_files = {}
    result_name = "boundary_reconstruct_mad{}.json".format(
        numeric_tag(args.threshold)
    )

    for result_file in args.results_dir.rglob(result_name):
        with result_file.open("r", encoding="utf-8") as input_file:
            result = json.load(input_file)
        if not matches_experiment(result, args):
            continue
        pred_len = int(result["pred_len"])
        if pred_len in by_horizon:
            raise RuntimeError(
                "Duplicate boundary reconstruction results for H={}: {} and {}".format(
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
            "Missing boundary reconstruction results for prediction lengths: {}".format(
                ", ".join(str(value) for value in missing)
            )
        )

    rows = [by_horizon[pred_len] for pred_len in args.expected_pred_lens]
    average = average_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "{}_{}_sl{}_{}_mad{}".format(
        args.dataset,
        args.model,
        args.seq_len,
        args.output_tag,
        numeric_tag(args.threshold),
    )
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(average.keys()))
        writer.writeheader()
        writer.writerows(rows + [average])

    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "model": args.model,
        "seq_len": args.seq_len,
        "train_seed": args.train_seed,
        "threshold": args.threshold,
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
        "Average reconstructed MSE/MAE: "
        f"{average['fixed_mse']:.6f}/{average['fixed_mae']:.6f}"
    )
    print(
        "Average improvement MSE/MAE: "
        f"{average['mse_improvement_percent']:.2f}%/"
        f"{average['mae_improvement_percent']:.2f}%"
    )
    print(
        "Average replacement rate: "
        f"{average['replacement_rate'] * 100.0:.2f}%"
    )


if __name__ == "__main__":
    main()
