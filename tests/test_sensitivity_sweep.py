import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from scripts.summarize_sensitivity_sweep import output_stem, series_colors


class SensitivitySweepSummaryTest(unittest.TestCase):
    def test_output_identity_separates_scan_grids(self):
        args = SimpleNamespace(
            dataset="ETTh1",
            model="GTR",
            seq_len=96,
            perturb_ratio=3.0,
            train_seed=2024,
            perturb_seed=2024,
            offsets=[96, 50, 1],
            expected_pred_lens=[96, 192],
        )
        first = output_stem(args)
        args.offsets = [96, 10, 1]
        second = output_stem(args)
        args.expected_pred_lens = [96, 720]
        third = output_stem(args)

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

        args.offsets = list(range(96, 0, -1))
        args.expected_pred_lens = [96, 192, 336, 720]
        self.assertLess(len(output_stem(args) + ".json"), 255)

    def test_every_horizon_receives_a_plot_color(self):
        self.assertEqual(len(series_colors(7)), 7)

    def test_cli_rejects_duplicate_prediction_lengths(self):
        script_path = (
            Path(__file__).parents[1]
            / "scripts"
            / "summarize_sensitivity_sweep.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--results-dir",
                    temp_dir,
                    "--offsets",
                    "1",
                    "--expected-pred-lens",
                    "96",
                    "96",
                ],
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must not contain duplicates", completed.stderr)

    def test_cli_aggregates_positions_and_writes_publication_outputs(self):
        script_path = (
            Path(__file__).parents[1]
            / "scripts"
            / "summarize_sensitivity_sweep.py"
        )
        offsets = (4, 2, 1)
        horizons = (96, 192)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "results"
            output_dir = temp_path / "summary"
            clean_by_horizon = {96: 1.0, 192: 3.0}
            drop_by_offset = {4: 0.1, 2: 0.4, 1: 1.0}

            for offset in offsets:
                for pred_len in horizons:
                    clean_mse = clean_by_horizon[pred_len]
                    perturbed_mse = clean_mse + drop_by_offset[offset]
                    result = {
                        "dataset": "ETTh1",
                        "data": "ETTh1",
                        "model": "GTR",
                        "seq_len": 4,
                        "pred_len": pred_len,
                        "train_seed": 2024,
                        "clean": {"mse": clean_mse, "mae": 1.0},
                        "perturbed": {
                            "mse": perturbed_mse,
                            "mae": 1.1,
                        },
                        "perturbation": {
                            "type": "point",
                            "ratio": 3.0,
                            "seed": 2024,
                            "offset": offset,
                        },
                    }
                    result_dir = results_dir / f"run_{pred_len}"
                    result_dir.mkdir(parents=True, exist_ok=True)
                    result_file = result_dir / (
                        "perturb_point_ratio3_seed2024_"
                        f"offset{offset}.json"
                    )
                    with result_file.open("w", encoding="utf-8") as output:
                        json.dump(result, output)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--results-dir",
                    str(results_dir),
                    "--output-dir",
                    str(output_dir),
                    "--seq-len",
                    "4",
                    "--offsets",
                    "4",
                    "2",
                    "1",
                    "--expected-pred-lens",
                    "96",
                    "192",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            stem = output_stem(
                SimpleNamespace(
                    dataset="ETTh1",
                    model="GTR",
                    seq_len=4,
                    perturb_ratio=3.0,
                    train_seed=2024,
                    perturb_seed=2024,
                    offsets=[4, 2, 1],
                    expected_pred_lens=[96, 192],
                )
            )
            json_path = output_dir / f"{stem}.json"
            csv_path = output_dir / f"{stem}.csv"
            png_path = output_dir / f"{stem}.png"
            pdf_path = output_dir / f"{stem}.pdf"
            svg_path = output_dir / f"{stem}.svg"
            for output_path in (json_path, csv_path, svg_path):
                self.assertTrue(output_path.is_file(), output_path)
                self.assertGreater(output_path.stat().st_size, 0)
            try:
                import matplotlib  # noqa: F401
            except ModuleNotFoundError:
                self.assertFalse(png_path.exists())
                self.assertFalse(pdf_path.exists())
            else:
                self.assertTrue(png_path.is_file())
                self.assertTrue(pdf_path.is_file())

            with json_path.open("r", encoding="utf-8") as input_file:
                summary = json.load(input_file)
            with csv_path.open("r", encoding="utf-8") as input_file:
                rows = list(csv.DictReader(input_file))

        self.assertIn("Wrote sensitivity figure", completed.stdout)
        self.assertEqual([row["perturb_offset"] for row in rows], ["4", "2", "1"])
        self.assertEqual(
            [point["relative_position"] for point in summary["per_position"]],
            [-4, -2, -1],
        )
        self.assertAlmostEqual(
            summary["per_position"][0]["average_mse_drop_percent"],
            5.0,
        )
        self.assertAlmostEqual(
            summary["per_position"][-1]["average_mse_drop_percent"],
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
