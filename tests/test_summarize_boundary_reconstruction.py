import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SummarizeBoundaryReconstructionTest(unittest.TestCase):
    def test_cli_writes_filtered_four_horizon_summary(self):
        script_path = (
            Path(__file__).parents[1]
            / "scripts"
            / "summarize_boundary_reconstruction.py"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "results"
            output_dir = temp_path / "summary"
            for index, pred_len in enumerate((96, 192, 336, 720), start=1):
                clean_mse = float(index)
                clean_mae = float(index + 1)
                result = {
                    "dataset": "Exchange",
                    "model": "GTR",
                    "seq_len": 96,
                    "pred_len": pred_len,
                    "train_seed": 2024,
                    "clean": {"mse": clean_mse, "mae": clean_mae},
                    "fixed": {
                        "mse": clean_mse * 0.9,
                        "mae": clean_mae * 0.95,
                    },
                    "boundary_reconstruction": {
                        "enabled": True,
                        "method": "context_mlp_delta",
                        "filter": "last_increment_mad",
                        "threshold": 3.0,
                        "replacement_rate": 0.1 * index,
                        "reconstruction_mse": 0.2,
                        "persistence_mse": 0.3,
                    },
                }
                run_dir = results_dir / f"run_{pred_len}"
                run_dir.mkdir(parents=True)
                with (run_dir / "boundary_reconstruct_mad3.json").open(
                    "w", encoding="utf-8"
                ) as output_file:
                    json.dump(result, output_file)

            subprocess.run(
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

            stem = "Exchange_GTR_sl96_boundary_reconstruct_mad3"
            with (output_dir / f"{stem}.json").open(
                "r", encoding="utf-8"
            ) as input_file:
                summary = json.load(input_file)
            with (output_dir / f"{stem}.csv").open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                rows = list(csv.DictReader(input_file))

            self.assertEqual(len(summary["rows"]), 4)
            self.assertAlmostEqual(
                summary["average"]["mse_improvement_percent"], 10.0
            )
            self.assertAlmostEqual(
                summary["average"]["replacement_rate"], 0.25
            )
            self.assertEqual(rows[-1]["pred_len"], "average")


if __name__ == "__main__":
    unittest.main()
