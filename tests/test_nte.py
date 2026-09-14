import unittest
from types import SimpleNamespace

import torch

from layers.NTE import NTE
from models import GTR, GTRNTE


class NTETest(unittest.TestCase):
    def test_constant_series_is_split_and_reconstructed(self):
        module = NTE(pred_len=4, cutoff_ratio=0.1)
        history = torch.full((2, 16, 3), 5.0)

        residual = module(history, mode="norm")
        forecast = module(torch.zeros(2, 4, 3), mode="denorm")

        torch.testing.assert_close(residual, torch.zeros_like(history))
        torch.testing.assert_close(forecast, torch.full_like(forecast, 5.0))
        self.assertEqual(list(module.parameters()), [])

    def test_future_time_uses_the_same_step_as_the_history(self):
        module = NTE(pred_len=2, cutoff_ratio=1.0, alpha=0.0)
        history = torch.tensor([[[0.0], [1.0], [4.0], [9.0], [16.0]]])

        residual = module(history, mode="norm")
        forecast = module(torch.zeros(1, 2, 1), mode="denorm")

        torch.testing.assert_close(residual, torch.zeros_like(history))
        torch.testing.assert_close(
            forecast,
            torch.tensor([[[25.0], [36.0]]]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_noisier_residual_produces_stronger_damping(self):
        module = NTE(pred_len=8, cutoff_ratio=0.05, alpha=1.0)
        time = torch.arange(64, dtype=torch.float32)
        trend = torch.sin(2.0 * torch.pi * time / 64.0)
        high_frequency = torch.sin(2.0 * torch.pi * 12.0 * time / 64.0)

        module((trend + 0.1 * high_frequency).view(1, 64, 1), mode="norm")
        clean_gamma = module.gamma.clone()
        module((trend + 2.0 * high_frequency).view(1, 64, 1), mode="norm")
        noisy_gamma = module.gamma.clone()

        self.assertTrue(torch.all(noisy_gamma > clean_gamma))

    def test_detached_trend_keeps_identity_gradient_paths(self):
        module = NTE(pred_len=3)
        history = torch.randn(2, 16, 4, requires_grad=True)
        residual = module(history, mode="norm")
        residual.sum().backward()

        torch.testing.assert_close(history.grad, torch.ones_like(history))

        residual_forecast = torch.randn(2, 3, 4, requires_grad=True)
        forecast = module(residual_forecast, mode="denorm")
        forecast.sum().backward()

        torch.testing.assert_close(
            residual_forecast.grad, torch.ones_like(residual_forecast)
        )

    def test_gtr_nte_wraps_gtr_without_adding_trainable_parameters(self):
        config = SimpleNamespace(
            seq_len=16,
            pred_len=4,
            enc_in=3,
            cycle=8,
            d_model=8,
            dropout=0.0,
            use_revin=1,
            individual=0,
            nte_cutoff_ratio=0.1,
            nte_alpha=1.0,
            nte_gamma_max=20.0,
        )
        baseline = GTR.Model(config)
        wrapped = GTRNTE.Model(config)

        output = wrapped(torch.randn(2, 16, 3), torch.tensor([0, 1]))
        output.square().mean().backward()

        self.assertEqual(output.shape, (2, 4, 3))
        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in wrapped.parameters()),
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in wrapped.parameters()
            )
        )

    def test_invalid_mode_or_state_is_rejected(self):
        module = NTE(pred_len=4)

        with self.assertRaisesRegex(RuntimeError, "norm.*before.*denorm"):
            module(torch.zeros(2, 4, 3), mode="denorm")
        with self.assertRaisesRegex(ValueError, "mode"):
            module(torch.zeros(2, 16, 3), mode="invalid")

        module(torch.zeros(2, 16, 3), mode="norm")
        with self.assertRaisesRegex(ValueError, "pred_len"):
            module(torch.zeros(2, 3, 3), mode="denorm")
        with self.assertRaisesRegex(ValueError, "batch and feature"):
            module(torch.zeros(1, 4, 3), mode="denorm")


if __name__ == "__main__":
    unittest.main()
