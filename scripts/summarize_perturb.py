#!/usr/bin/env python3
"""Summarize per-horizon GTR robustness result files."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


METRICS = ("mse", "mae")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect clean and perturbed metrics and average them across "
            "prediction lengths, following the presentation used by TameR."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results/perturbation"))
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--model", default="GTR")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--perturb-type", default="last")
    parser.add_argument("--perturb-ratio", type=float, default=3.0)
    parser.add_argument("--train-seed", type=int, default=2024)
    parser.add_argument("--perturb-seed", type=int, default=2024)
    parser.add_argument("--nte-cutoff-ratio", type=float, default=0.1)
    parser.add_argument("--nte-alpha", type=float, default=1.0)
    parser.add_argument("--nte-gamma-max", type=float, default=20.0)
    parser.add_argument("--nte-guard-sigma", type=float, default=3.0)
    parser.add_argument(
        "--expected-pred-lens",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
    )
    return parser.parse_args()


def ratio_label(value):
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def experiment_stem(args):
    stem = "{}_{}_sl{}_{}_ratio{}_train{}_perturb{}".format(
        args.dataset,
        args.model,
        args.seq_len,
        args.perturb_type,
        ratio_label(args.perturb_ratio),
        args.train_seed,
        args.perturb_seed,
    )
    if args.model == "GTRNTE":
        stem += "_nte_k{}_a{}_g{}_guard{}".format(
            ratio_label(args.nte_cutoff_ratio),
            ratio_label(args.nte_alpha),
            ratio_label(args.nte_gamma_max),
            ratio_label(args.nte_guard_sigma),
        )
    return stem


def matches_experiment(result, args):
    perturbation = result.get("perturbation", {})
    dataset = result.get("dataset", result.get("data"))
    try:
        perturb_ratio = float(perturbation["ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    base_matches = (
        dataset == args.dataset
        and result.get("model") == args.model
        and result.get("seq_len") == args.seq_len
        and result.get("train_seed") == args.train_seed
        and perturbation.get("type") == args.perturb_type
        and math.isclose(
            perturb_ratio,
            args.perturb_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and perturbation.get("seed") == args.perturb_seed
    )
    if not base_matches or args.model != "GTRNTE":
        return base_matches

    plugin = result.get("plugin", {})
    expected_plugin_values = {
        "cutoff_ratio": args.nte_cutoff_ratio,
        "alpha": args.nte_alpha,
        "gamma_max": args.nte_gamma_max,
        "guard_sigma": args.nte_guard_sigma,
    }
    if plugin.get("name") != "NTE":
        return False
    try:
        return all(
            math.isclose(
                float(plugin[name]),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, expected in expected_plugin_values.items()
        )
    except (KeyError, TypeError, ValueError):
        return False


def result_to_row(result):
    row = {"pred_len": int(result["pred_len"])}
    for metric_name in METRICS:
        clean = float(result["clean"][metric_name])
        perturbed = float(result["perturbed"][metric_name])
        delta = perturbed - clean
        row[f"clean_{metric_name}"] = clean
        row[f"perturbed_{metric_name}"] = perturbed
        row[f"delta_{metric_name}"] = delta
        row[f"delta_{metric_name}_percent"] = (
            delta / clean * 100.0 if clean != 0 else None
        )
    return row


def average_rows(rows):
    average = {"pred_len": "average"}
    for metric_name in METRICS:
        clean = mean(row[f"clean_{metric_name}"] for row in rows)
        perturbed = mean(row[f"perturbed_{metric_name}"] for row in rows)
        delta = perturbed - clean
        average[f"clean_{metric_name}"] = clean
        average[f"perturbed_{metric_name}"] = perturbed
        average[f"delta_{metric_name}"] = delta
        average[f"delta_{metric_name}_percent"] = (
            delta / clean * 100.0 if clean != 0 else None
        )
    return average


def main():
    args = parse_args()
    by_horizon = {}
    source_files = {}

    for result_file in args.results_dir.rglob("perturb_*.json"):
        try:
            with result_file.open("r", encoding="utf-8") as input_file:
                result = json.load(input_file)
        except (OSError, json.JSONDecodeError):
            continue
        if not matches_experiment(result, args):
            continue

        pred_len = int(result["pred_len"])
        if pred_len in by_horizon:
            raise RuntimeError(
                "Found duplicate results for prediction length {}: {} and {}".format(
                    pred_len, source_files[pred_len], result_file
                )
            )
        by_horizon[pred_len] = result_to_row(result)
        source_files[pred_len] = str(result_file)

    expected = list(args.expected_pred_lens)
    missing = [pred_len for pred_len in expected if pred_len not in by_horizon]
    if missing:
        raise RuntimeError(
            "Missing perturbation results for prediction lengths: {}".format(
                ", ".join(str(value) for value in missing)
            )
        )

    rows = [by_horizon[pred_len] for pred_len in expected]
    average = average_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = experiment_stem(args)
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(average)

    experiment = {
        "dataset": args.dataset,
        "model": args.model,
        "seq_len": args.seq_len,
        "prediction_lengths": expected,
        "train_seed": args.train_seed,
        "perturbation": {
            "type": args.perturb_type,
            "ratio": args.perturb_ratio,
            "seed": args.perturb_seed,
        },
    }
    if args.model == "GTRNTE":
        experiment["plugin"] = {
            "name": "NTE",
            "parameter_free": True,
            "cutoff_ratio": args.nte_cutoff_ratio,
            "alpha": args.nte_alpha,
            "gamma_max": args.nte_gamma_max,
            "guard_sigma": args.nte_guard_sigma,
        }

    summary = {
        "schema_version": 1,
        "experiment": experiment,
        "source_files": [source_files[pred_len] for pred_len in expected],
        "per_horizon": rows,
        "average": average,
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        "Average clean MSE/MAE: {:.6f}/{:.6f}".format(
            average["clean_mse"], average["clean_mae"]
        )
    )
    print(
        "Average perturbed MSE/MAE: {:.6f}/{:.6f}".format(
            average["perturbed_mse"], average["perturbed_mae"]
        )
    )


if __name__ == "__main__":
    main()
