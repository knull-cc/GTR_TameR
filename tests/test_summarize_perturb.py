import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from scripts.summarize_perturb import (
    average_rows,
    experiment_stem,
    matches_experiment,
)


class SummarizePerturbationTest(unittest.TestCase):
    def test_average_percentage_is_computed_from_average_metrics(self):
        rows = [
            {
                "pred_len": 96,
                "clean_mse": 1.0,
                "perturbed_mse": 2.0,
                "delta_mse": 1.0,
                "delta_mse_percent": 100.0,
                "clean_mae": 2.0,
                "perturbed_mae": 3.0,
                "delta_mae": 1.0,
                "delta_mae_percent": 50.0,
            },
            {
                "pred_len": 192,
                "clean_mse": 3.0,
                "perturbed_mse": 4.0,
                "delta_mse": 1.0,
                "delta_mse_percent": 100.0 / 3.0,
                "clean_mae": 4.0,
                "perturbed_mae": 5.0,
                "delta_mae": 1.0,
                "delta_mae_percent": 25.0,
            },
        ]

        average = average_rows(rows)

        self.assertEqual(average["pred_len"], "average")
        self.assertEqual(average["clean_mse"], 2.0)
        self.assertEqual(average["perturbed_mse"], 3.0)
        self.assertEqual(average["delta_mse"], 1.0)
        self.assertEqual(average["delta_mse_percent"], 50.0)
        self.assertAlmostEqual(average["delta_mae_percent"], 100.0 / 3.0)

    def test_match_requires_the_complete_experiment_identity(self):
        args = SimpleNamespace(
            dataset="ETTh1",
            model="GTR",
            seq_len=96,
            train_seed=2024,
            perturb_type="last",
            perturb_ratio=3.0,
            perturb_seed=2024,
        )
        result = {
            "data": "ETTh1",
            "model": "GTR",
            "seq_len": 96,
            "train_seed": 2024,
            "perturbation": {"type": "last", "ratio": 3.0, "seed": 2024},
        }

        self.assertTrue(matches_experiment(result, args))
        result["perturbation"]["seed"] = 7
        self.assertFalse(matches_experiment(result, args))

    def test_dataset_field_distinguishes_custom_datasets(self):
        args = SimpleNamespace(
            dataset="weather",
            model="GTR",
            seq_len=96,
            train_seed=2024,
            perturb_type="last",
            perturb_ratio=3.0,
            perturb_seed=2024,
        )
        result = {
            "dataset": "weather",
            "data": "custom",
            "model": "GTR",
            "seq_len": 96,
            "train_seed": 2024,
            "perturbation": {"type": "last", "ratio": 3.0, "seed": 2024},
        }

        self.assertTrue(matches_experiment(result, args))
        result["dataset"] = "traffic"
        self.assertFalse(matches_experiment(result, args))

    def test_nte_match_includes_the_plugin_configuration(self):
        args = SimpleNamespace(
            dataset="ETTh1",
            model="GTRNTE",
            seq_len=96,
            train_seed=2024,
            perturb_type="last",
            perturb_ratio=3.0,
            perturb_seed=2024,
            nte_cutoff_ratio=0.1,
            nte_alpha=1.0,
            nte_gamma_max=20.0,
            nte_guard_sigma=3.0,
        )
        result = {
            "dataset": "ETTh1",
            "data": "ETTh1",
            "model": "GTRNTE",
            "seq_len": 96,
            "train_seed": 2024,
            "plugin": {
                "name": "NTE",
                "cutoff_ratio": 0.1,
                "alpha": 1.0,
                "gamma_max": 20.0,
                "guard_sigma": 3.0,
            },
            "perturbation": {"type": "last", "ratio": 3.0, "seed": 2024},
        }

        self.assertTrue(matches_experiment(result, args))
        result["plugin"]["alpha"] = 0.5
        self.assertFalse(matches_experiment(result, args))

    def test_nte_output_name_includes_the_plugin_configuration(self):
        args = SimpleNamespace(
            dataset="ETTh1",
            model="GTRNTE",
            seq_len=96,
            train_seed=2024,
            perturb_type="last",
            perturb_ratio=3.0,
            perturb_seed=2024,
            nte_cutoff_ratio=0.1,
            nte_alpha=1.0,
            nte_gamma_max=20.0,
            nte_guard_sigma=3.0,
        )

        first = experiment_stem(args)
        args.nte_alpha = 0.5
        second = experiment_stem(args)

        self.assertIn("nte_k0p1_a1_g20_guard3", first)
        self.assertIn("nte_k0p1_a0p5_g20_guard3", second)
        self.assertNotEqual(first, second)

    def test_cli_writes_four_horizon_csv_and_json_summary(self):
        script_path = (
            Path(__file__).parents[1] / "scripts" / "summarize_perturb.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "results"
            output_dir = temp_path / "summary"
            for index, pred_len in enumerate((96, 192, 336, 720), start=1):
                clean_mse = float(index)
                clean_mae = float(index + 1)
                perturbed_mse = clean_mse + 0.5
                perturbed_mae = clean_mae + 0.25
                result = {
                    "data": "ETTh1",
                    "model": "GTR",
                    "seq_len": 96,
                    "pred_len": pred_len,
                    "train_seed": 2024,
                    "clean": {"mse": clean_mse, "mae": clean_mae},
                    "perturbed": {
                        "mse": perturbed_mse,
                        "mae": perturbed_mae,
                    },
                    "degradation_absolute": {"mse": 0.5, "mae": 0.25},
                    "degradation_percent": {
                        "mse": 0.5 / clean_mse * 100.0,
                        "mae": 0.25 / clean_mae * 100.0,
                    },
                    "perturbation": {
                        "type": "last",
                        "ratio": 3.0,
                        "seed": 2024,
                    },
                }
                result_dir = results_dir / f"run_{pred_len}"
                result_dir.mkdir(parents=True)
                with (
                    result_dir / "perturb_last_ratio3_seed2024.json"
                ).open("w", encoding="utf-8") as output_file:
                    json.dump(result, output_file)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--results-dir",
                    str(results_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary_path = (
                output_dir
                / "ETTh1_GTR_sl96_last_ratio3_train2024_perturb2024.json"
            )
            csv_path = summary_path.with_suffix(".csv")
            self.assertTrue(summary_path.is_file())
            self.assertTrue(csv_path.is_file())
            with summary_path.open("r", encoding="utf-8") as input_file:
                summary = json.load(input_file)

        self.assertIn("Average clean MSE/MAE", completed.stdout)
        self.assertEqual(len(summary["per_horizon"]), 4)
        self.assertEqual(summary["average"]["clean_mse"], 2.5)
        self.assertEqual(summary["average"]["perturbed_mse"], 3.0)
        self.assertEqual(summary["average"]["delta_mse_percent"], 20.0)

    def test_cli_keeps_two_nte_configurations_separate(self):
        script_path = (
            Path(__file__).parents[1] / "scripts" / "summarize_perturb.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "results"
            output_dir = temp_path / "summary"
            for alpha in (1.0, 0.5):
                for pred_len in (96, 192, 336, 720):
                    result = {
                        "dataset": "ETTh1",
                        "data": "ETTh1",
                        "model": "GTRNTE",
                        "seq_len": 96,
                        "pred_len": pred_len,
                        "train_seed": 2024,
                        "clean": {"mse": 1.0, "mae": 1.0},
                        "perturbed": {"mse": 2.0, "mae": 2.0},
                        "plugin": {
                            "name": "NTE",
                            "parameter_free": True,
                            "cutoff_ratio": 0.1,
                            "alpha": alpha,
                            "gamma_max": 20.0,
                            "guard_sigma": 3.0,
                        },
                        "perturbation": {
                            "type": "last",
                            "ratio": 3.0,
                            "seed": 2024,
                        },
                    }
                    result_dir = results_dir / f"alpha_{alpha}_{pred_len}"
                    result_dir.mkdir(parents=True)
                    with (
                        result_dir / "perturb_last_ratio3_seed2024.json"
                    ).open("w", encoding="utf-8") as output_file:
                        json.dump(result, output_file)

            for alpha in (1.0, 0.5):
                subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--results-dir",
                        str(results_dir),
                        "--output-dir",
                        str(output_dir),
                        "--model",
                        "GTRNTE",
                        "--nte-alpha",
                        str(alpha),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            alpha_one = output_dir / (
                "ETTh1_GTRNTE_sl96_last_ratio3_train2024_perturb2024_"
                "nte_k0p1_a1_g20_guard3.json"
            )
            alpha_half = output_dir / (
                "ETTh1_GTRNTE_sl96_last_ratio3_train2024_perturb2024_"
                "nte_k0p1_a0p5_g20_guard3.json"
            )
            self.assertTrue(alpha_one.is_file())
            self.assertTrue(alpha_half.is_file())
            with alpha_half.open("r", encoding="utf-8") as input_file:
                summary = json.load(input_file)

        self.assertEqual(summary["experiment"]["plugin"]["alpha"], 0.5)


if __name__ == "__main__":
    unittest.main()
