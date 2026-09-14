import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


def load_exp_main_for_test():
    """Load exp_main without importing optional model dependencies."""

    data_factory = types.ModuleType("data_provider.data_factory")
    data_factory.data_provider = lambda *args, **kwargs: None

    exp_basic = types.ModuleType("exp.exp_basic")

    class StubExpBasic:
        def __init__(self, args):
            self.args = args

    exp_basic.Exp_Basic = StubExpBasic

    models = types.ModuleType("models")
    for model_name in (
        "Informer",
        "Autoformer",
        "Transformer",
        "DLinear",
        "Linear",
        "NLinear",
        "PatchTST",
        "SegRNN",
        "CycleNet",
        "iTransformer",
        "TimeXer",
        "GTR",
        "GTRNTE",
        "GTRDLinear",
        "GTRPatchTST",
        "GTRiTransformer",
    ):
        setattr(models, model_name, None)

    tools = types.ModuleType("utils.tools")
    tools.EarlyStopping = object
    tools.adjust_learning_rate = lambda *args, **kwargs: None
    tools.visual = lambda *args, **kwargs: None
    tools.test_params_flop = lambda *args, **kwargs: None

    metrics = types.ModuleType("utils.metrics")

    def metric(predictions, targets):
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        return mae, mse, np.sqrt(mse), 0.0, 0.0, 0.0, 0.0

    metrics.metric = metric

    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot

    stubs = {
        "data_provider.data_factory": data_factory,
        "exp.exp_basic": exp_basic,
        "models": models,
        "utils.tools": tools,
        "utils.metrics": metrics,
        "matplotlib": matplotlib,
        "matplotlib.pyplot": pyplot,
    }
    module_path = Path(__file__).parents[1] / "exp" / "exp_main.py"
    spec = importlib.util.spec_from_file_location(
        "robustness_test_exp_main", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.Exp_Main


class RecordingGTR(torch.nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.inputs = []

    def forward(self, batch_x, batch_cycle):
        self.inputs.append(batch_x.detach().clone())
        return batch_x[:, -self.pred_len:, :]


class RobustnessEvaluationTest(unittest.TestCase):
    def test_clean_and_perturbed_metrics_share_one_test_batch(self):
        ExpMain = load_exp_main_for_test()
        args = types.SimpleNamespace(
            model="GTRNTE",
            use_amp=False,
            output_attention=False,
            pred_len=2,
            label_len=0,
            features="M",
            test_flop=False,
            perturb_type="point",
            perturb_ratio=3.0,
            perturb_seed=2024,
            perturb_offset=2,
            model_id="synthetic_4_2",
            data="ETTh1",
            data_path="synthetic.csv",
            dataset_name="SyntheticOfficial",
            seq_len=4,
            cycle=24,
            random_seed=2024,
            nte_cutoff_ratio=0.1,
            nte_alpha=1.0,
            nte_gamma_max=20.0,
            nte_guard_sigma=3.0,
        )
        experiment = ExpMain(args)
        experiment.device = torch.device("cpu")
        experiment.model = RecordingGTR(pred_len=args.pred_len)

        batch_x = torch.tensor(
            [[[1.0], [2.0], [3.0], [4.0]], [[2.0], [4.0], [6.0], [8.0]]]
        )
        original_batch_x = batch_x.clone()
        batch_y = torch.zeros((2, args.pred_len, 1))
        batch_x_mark = torch.zeros((2, 4, 1))
        batch_y_mark = torch.zeros((2, args.pred_len, 1))
        batch_cycle = torch.tensor([0, 1])
        test_loader = [
            (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
        ]
        experiment._get_data = lambda flag: (None, test_loader)

        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                result = experiment.test("synthetic_setting")
                result_file = (
                    Path("results")
                    / "synthetic_setting"
                    / "perturb_point_ratio3_seed2024_offset2.json"
                )
                self.assertTrue(result_file.is_file())
                with result_file.open("r", encoding="utf-8") as input_file:
                    saved_result = json.load(input_file)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(result, saved_result)
        self.assertEqual(len(experiment.model.inputs), 2)
        torch.testing.assert_close(experiment.model.inputs[0], original_batch_x)
        torch.testing.assert_close(
            experiment.model.inputs[1][:, :-2, :], original_batch_x[:, :-2, :]
        )
        torch.testing.assert_close(
            experiment.model.inputs[1][:, -1:, :], original_batch_x[:, -1:, :]
        )
        self.assertFalse(
            torch.equal(
                experiment.model.inputs[1][:, -2:-1, :],
                original_batch_x[:, -2:-1, :],
            )
        )
        torch.testing.assert_close(batch_x, original_batch_x)
        self.assertEqual(result["dataset"], "SyntheticOfficial")
        self.assertEqual(
            result["plugin"],
            {
                "name": "NTE",
                "parameter_free": True,
                "cutoff_ratio": 0.1,
                "alpha": 1.0,
                "gamma_max": 20.0,
                "guard_sigma": 3.0,
            },
        )
        self.assertIn("clean", result)
        self.assertIn("perturbed", result)
        self.assertIn("degradation_percent", result)
        self.assertEqual(result["perturbation"]["offset"], 2)


if __name__ == "__main__":
    unittest.main()
