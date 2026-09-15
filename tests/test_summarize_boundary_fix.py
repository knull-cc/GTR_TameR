import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SummarizeBoundaryFixTest(unittest.TestCase):
    def test_cli_writes_four_horizon_summary(self):
        script_path = (
            Path(__file__).parents[1]
            / "scripts"
            / "summarize_boundary_fix.py"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "results"
            output_dir = temp_path / "summary"

            for index, pred_len in enumerate((96, 192, 336, 720), start=1):
                clean_mse = float(index)
                clean_mae = float(index + 1)
                fixed_mse = clean_mse * 0.9
                fixed_mae = clean_mae * 0.95
                result = {
                    "dataset": "Exchange",
                    "model": "GTR",
                    "seq_len": 96,
                    "pred_len": pred_len,
                    "train_seed": 2024,
                    "clean": {"mse": clean_mse, "mae": clean_mae},
                    "fixed": {"mse": fixed_mse, "mae": fixed_mae},
                    "boundary_fix": {
                        "enabled": True,
                        "method": "last_value",
                    },
                }
                run_dir = results_dir / f"run_{pred_len}"
                run_dir.mkdir(parents=True)
                with (run_dir / "boundary_fix_last_value.json").open(
                    "w", encoding="utf-8"
                ) as output_file:
                    json.dump(result, output_file)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--results-dir",
                    str(results_dir),
                    "--output-dir",
                    str(output_dir),
                    "--dataset",
                    "Exchange",
                    "--model",
                    "GTR",
                    "--seq-len",
                    "96",
                    "--train-seed",
                    "2024",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            json_path = output_dir / "Exchange_GTR_sl96_boundary_fix_last_value.json"
            csv_path = output_dir / "Exchange_GTR_sl96_boundary_fix_last_value.csv"
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())

            with json_path.open("r", encoding="utf-8") as input_file:
                summary = json.load(input_file)
            self.assertEqual(len(summary["rows"]), 4)
            self.assertAlmostEqual(
                summary["average"]["mse_improvement_percent"], 10.0
            )
            self.assertAlmostEqual(
                summary["average"]["mae_improvement_percent"], 5.0
            )

            with csv_path.open("r", encoding="utf-8", newline="") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[-1]["pred_len"], "average")
            self.assertIn("Average original MSE/MAE", completed.stdout)
            self.assertIn("Average fixed MSE/MAE", completed.stdout)


if __name__ == "__main__":
    unittest.main()
