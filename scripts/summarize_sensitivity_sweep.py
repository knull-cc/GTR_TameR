#!/usr/bin/env python3
"""Aggregate a point-position robustness sweep and draw its sensitivity curve."""

import argparse
import colorsys
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ETTh1/GTR point-perturbation results across positions "
            "and prediction lengths."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/perturbation_sweep"),
    )
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--model", default="GTR")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--perturb-ratio", type=float, default=3.0)
    parser.add_argument("--train-seed", type=int, default=2024)
    parser.add_argument("--perturb-seed", type=int, default=2024)
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=[96, 72, 50, 36, 24, 16, 10, 8, 6, 5, 4, 3, 2, 1],
    )
    parser.add_argument(
        "--expected-pred-lens",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
    )
    return parser.parse_args()


def numeric_label(value):
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def output_stem(args):
    grid_identity = json.dumps(
        {
            "offsets": args.offsets,
            "prediction_lengths": args.expected_pred_lens,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    grid_hash = hashlib.sha256(grid_identity).hexdigest()[:10]
    return (
        f"{args.dataset}_{args.model}_sl{args.seq_len}_point_"
        f"ratio{numeric_label(args.perturb_ratio)}_train{args.train_seed}_"
        f"perturb{args.perturb_seed}_sweep_n{len(args.offsets)}x"
        f"{len(args.expected_pred_lens)}_{grid_hash}"
    )


def series_colors(count):
    if count <= 0:
        return []
    return [
        "#{:02x}{:02x}{:02x}".format(
            *(round(channel * 255) for channel in colorsys.hsv_to_rgb(
                0.58,
                0.35 + 0.45 * index / max(count - 1, 1),
                0.85 - 0.25 * index / max(count - 1, 1),
            ))
        )
        for index in range(count)
    ]


def matches_experiment(result, args):
    perturbation = result.get("perturbation", {})
    dataset = result.get("dataset", result.get("data"))
    try:
        ratio_matches = math.isclose(
            float(perturbation["ratio"]),
            args.perturb_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        offset = int(perturbation["offset"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        dataset == args.dataset
        and result.get("model") == args.model
        and result.get("seq_len") == args.seq_len
        and result.get("train_seed") == args.train_seed
        and perturbation.get("type") == "point"
        and ratio_matches
        and perturbation.get("seed") == args.perturb_seed
        and offset in args.offsets
    )


def drop_percent(clean_value, perturbed_value):
    if clean_value == 0:
        return None
    return (perturbed_value - clean_value) / clean_value * 100.0


def aggregate_position(offset, by_horizon, horizons, seq_len):
    results = [by_horizon[(offset, pred_len)] for pred_len in horizons]
    clean_mse = mean(float(result["clean"]["mse"]) for result in results)
    perturbed_mse = mean(
        float(result["perturbed"]["mse"]) for result in results
    )
    clean_mae = mean(float(result["clean"]["mae"]) for result in results)
    perturbed_mae = mean(
        float(result["perturbed"]["mae"]) for result in results
    )
    per_horizon = {
        str(pred_len): {
            "clean_mse": float(result["clean"]["mse"]),
            "perturbed_mse": float(result["perturbed"]["mse"]),
            "mse_drop_percent": drop_percent(
                float(result["clean"]["mse"]),
                float(result["perturbed"]["mse"]),
            ),
        }
        for pred_len, result in zip(horizons, results)
    }
    return {
        "perturb_offset": offset,
        "relative_position": -offset,
        "input_index": seq_len - offset + 1,
        "average_clean_mse": clean_mse,
        "average_perturbed_mse": perturbed_mse,
        "average_mse_drop_percent": drop_percent(clean_mse, perturbed_mse),
        "average_clean_mae": clean_mae,
        "average_perturbed_mae": perturbed_mae,
        "average_mae_drop_percent": drop_percent(clean_mae, perturbed_mae),
        "per_horizon": per_horizon,
    }


def write_csv(path, points, horizons):
    fieldnames = [
        "perturb_offset",
        "relative_position",
        "input_index",
        "average_clean_mse",
        "average_perturbed_mse",
        "average_mse_drop_percent",
        "average_clean_mae",
        "average_perturbed_mae",
        "average_mae_drop_percent",
    ] + [f"mse_drop_percent_pl{pred_len}" for pred_len in horizons]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row = {name: point[name] for name in fieldnames[:9]}
            for pred_len in horizons:
                row[f"mse_drop_percent_pl{pred_len}"] = point["per_horizon"][
                    str(pred_len)
                ]["mse_drop_percent"]
            writer.writerow(row)


def write_svg_figure(svg_path, points, horizons, title):
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [point["input_index"] for point in points]
    all_y_values = [
        point["per_horizon"][str(pred_len)]["mse_drop_percent"]
        for point in points
        for pred_len in horizons
    ] + [point["average_mse_drop_percent"] for point in points]
    y_min = min(0.0, min(all_y_values))
    y_max = max(0.0, max(all_y_values))
    y_padding = max((y_max - y_min) * 0.08, 1.0)
    y_min -= y_padding
    y_max += y_padding

    def x_pixel(value):
        if len(x_values) == 1:
            return left + plot_width / 2
        return left + (value - min(x_values)) / (
            max(x_values) - min(x_values)
        ) * plot_width

    def y_pixel(value):
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = series_colors(len(horizons))
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" '
        'viewBox="0 0 900 520">',
        '<rect width="900" height="520" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#222">',
        '<text x="450" y="28" text-anchor="middle" font-size="20">'
        f'{html.escape(title)}</text>',
    ]
    for tick_index in range(6):
        value = y_min + (y_max - y_min) * tick_index / 5
        y = y_pixel(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="#e2e2e2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'font-size="12">{value:.1f}</text>'
        )
    for point in points:
        x = x_pixel(point["input_index"])
        lines.append(
            f'<text x="{x:.2f}" y="{height-bottom+25}" text-anchor="end" '
            f'transform="rotate(-45 {x:.2f} {height-bottom+25})" '
            f'font-size="12">{point["relative_position"]}</text>'
        )
    series = []
    for color, pred_len in zip(colors, horizons):
        values = [
            point["per_horizon"][str(pred_len)]["mse_drop_percent"]
            for point in points
        ]
        series.append((f"H={pred_len}", color, 1.4, values))
    series.append(
        (
            "Average",
            "#b2182b",
            3.0,
            [point["average_mse_drop_percent"] for point in points],
        )
    )
    for label, color, stroke_width, values in series:
        coordinates = " ".join(
            f'{x_pixel(x):.2f},{y_pixel(y):.2f}'
            for x, y in zip(x_values, values)
        )
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}"/>'
        )
        for x, y in zip(x_values, values):
            lines.append(
                f'<circle cx="{x_pixel(x):.2f}" cy="{y_pixel(y):.2f}" '
                f'r="{2.2 if label != "Average" else 3.5}" fill="{color}"/>'
            )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
            'stroke="#222"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#222"/>',
            f'<text x="{left+plot_width/2:.2f}" y="505" text-anchor="middle" '
            'font-size="14">Perturbed position relative to forecast origin</text>',
            '<text x="20" y="245" text-anchor="middle" font-size="14" '
            'transform="rotate(-90 20 245)">MSE performance drop (%)</text>',
        ]
    )
    legend_x = left + 12
    for label, color, stroke_width, _ in series:
        lines.append(
            f'<line x1="{legend_x}" y1="48" x2="{legend_x+24}" y2="48" '
            f'stroke="{color}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text x="{legend_x+30}" y="52" font-size="11">{label}</text>'
        )
        legend_x += 105
    lines.extend(["</g>", "</svg>"])
    svg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matplotlib_figure(png_path, pdf_path, points, horizons, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    x_values = [point["input_index"] for point in points]
    x_labels = [str(point["relative_position"]) for point in points]

    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    colors = [
        plt.cm.Blues(0.35 + 0.5 * index / max(len(horizons) - 1, 1))
        for index in range(len(horizons))
    ]
    for color, pred_len in zip(colors, horizons):
        axis.plot(
            x_values,
            [
                point["per_horizon"][str(pred_len)]["mse_drop_percent"]
                for point in points
            ],
            color=color,
            linewidth=1.2,
            alpha=0.72,
            marker="o",
            markersize=3.0,
            label=f"H={pred_len}",
        )
    axis.plot(
        x_values,
        [point["average_mse_drop_percent"] for point in points],
        color="#B2182B",
        linewidth=2.6,
        marker="o",
        markersize=4.5,
        label="Average",
        zorder=5,
    )
    axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    axis.set_xticks(x_values, labels=x_labels, rotation=45, ha="right")
    axis.set_xlabel("Perturbed position relative to forecast origin")
    axis.set_ylabel("MSE performance drop (%)")
    axis.set_title(title)
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=5, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    args = parse_args()
    if len(set(args.offsets)) != len(args.offsets):
        raise ValueError("--offsets must not contain duplicates")
    invalid_offsets = [
        offset for offset in args.offsets if not 1 <= offset <= args.seq_len
    ]
    if invalid_offsets:
        raise ValueError(
            "Offsets must be in [1, seq_len]: "
            + ", ".join(str(offset) for offset in invalid_offsets)
        )
    if len(set(args.expected_pred_lens)) != len(args.expected_pred_lens):
        raise ValueError("--expected-pred-lens must not contain duplicates")
    invalid_horizons = [
        pred_len for pred_len in args.expected_pred_lens if pred_len <= 0
    ]
    if invalid_horizons:
        raise ValueError(
            "Prediction lengths must be positive: "
            + ", ".join(str(pred_len) for pred_len in invalid_horizons)
        )

    by_horizon = {}
    source_files = {}
    for result_file in args.results_dir.rglob(
        "perturb_point_ratio*_offset*.json"
    ):
        try:
            with result_file.open("r", encoding="utf-8") as input_file:
                result = json.load(input_file)
        except (OSError, json.JSONDecodeError):
            continue
        if not matches_experiment(result, args):
            continue
        offset = int(result["perturbation"]["offset"])
        pred_len = int(result["pred_len"])
        key = (offset, pred_len)
        if key in by_horizon:
            raise RuntimeError(
                f"Duplicate result for offset {offset}, horizon {pred_len}: "
                f"{source_files[key]} and {result_file}"
            )
        by_horizon[key] = result
        source_files[key] = str(result_file)

    expected_keys = [
        (offset, pred_len)
        for offset in args.offsets
        for pred_len in args.expected_pred_lens
    ]
    missing = [key for key in expected_keys if key not in by_horizon]
    if missing:
        preview = ", ".join(
            f"offset={offset}/H={pred_len}" for offset, pred_len in missing[:12]
        )
        if len(missing) > 12:
            preview += f", ... ({len(missing)} missing)"
        raise RuntimeError(f"Missing sweep results: {preview}")

    points = [
        aggregate_position(
            offset,
            by_horizon,
            args.expected_pred_lens,
            args.seq_len,
        )
        for offset in args.offsets
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    svg_path = args.output_dir / f"{stem}.svg"

    write_csv(csv_path, points, args.expected_pred_lens)
    summary = {
        "schema_version": 1,
        "experiment": {
            "dataset": args.dataset,
            "model": args.model,
            "seq_len": args.seq_len,
            "prediction_lengths": args.expected_pred_lens,
            "train_seed": args.train_seed,
            "perturbation": {
                "type": "point",
                "ratio": args.perturb_ratio,
                "seed": args.perturb_seed,
                "offsets": args.offsets,
            },
            "aggregation": (
                "Average clean and perturbed metrics across prediction "
                "lengths, then compute relative performance drop."
            ),
        },
        "source_files": [source_files[key] for key in expected_keys],
        "per_position": points,
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    figure_title = (
        f"{args.model} Perturbation Sensitivity on {args.dataset}"
    )
    write_svg_figure(
        svg_path, points, args.expected_pred_lens, figure_title
    )
    wrote_raster = write_matplotlib_figure(
        png_path,
        pdf_path,
        points,
        args.expected_pred_lens,
        figure_title,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote sensitivity figure {svg_path}")
    if wrote_raster:
        print(f"Wrote sensitivity figure {png_path}")
        print(f"Wrote sensitivity figure {pdf_path}")
    else:
        print(
            "matplotlib is unavailable; SVG was written, while PNG/PDF were "
            "skipped. Install requirements.txt to produce all formats."
        )


if __name__ == "__main__":
    main()
