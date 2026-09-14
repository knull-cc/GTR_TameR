import math

import torch
import torch.nn as nn


class NTE(nn.Module):
    """Parameter-free, noise-aware trend extrapolation.

    ``norm`` separates an input window into a Fourier low-frequency trend and
    a residual. ``denorm`` adds the cached, damped trend extrapolation to a
    residual forecast. The trend path is deliberately detached: gradients
    through the residual forecast remain an identity path to the backbone,
    while NTE itself has no trainable state.
    """

    def __init__(
        self,
        pred_len: int,
        cutoff_ratio: float = 0.1,
        alpha: float = 1.0,
        gamma_max: float = 20.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if not 0.0 < cutoff_ratio <= 1.0:
            raise ValueError("cutoff_ratio must be in (0, 1]")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if gamma_max <= 0.0:
            raise ValueError("gamma_max must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.pred_len = int(pred_len)
        self.cutoff_ratio = float(cutoff_ratio)
        self.alpha = float(alpha)
        self.gamma_max = float(gamma_max)
        self.eps = float(eps)

        # Per-batch state. These are intentionally not buffers, so a checkpoint
        # contains only the backbone and NTE remains parameter/state-dict free.
        self.history_trend = None
        self.history_residual = None
        self.current_level = None
        self.velocity = None
        self.acceleration = None
        self.snr = None
        self.gamma = None
        self.future_trend = None
        self._history_length = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._separate(x)
        if mode == "denorm":
            return self._restore(x)
        raise ValueError("mode must be either 'norm' or 'denorm'")

    def _separate(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x, expected_length=None)
        if x.shape[1] < 3:
            raise ValueError("NTE requires a history length of at least 3")

        with torch.no_grad():
            state_dtype = (
                torch.float32
                if x.dtype in (torch.float16, torch.bfloat16)
                else x.dtype
            )
            state_x = x.detach().to(dtype=state_dtype)
            history_length = state_x.shape[1]

            spectrum = torch.fft.rfft(state_x, dim=1)
            retained_bins = min(
                spectrum.shape[1],
                max(1, math.ceil(history_length * self.cutoff_ratio)),
            )
            low_pass_mask = (
                torch.arange(spectrum.shape[1], device=x.device)
                < retained_bins
            ).view(1, -1, 1)
            history_trend = torch.fft.irfft(
                spectrum * low_pass_mask,
                n=history_length,
                dim=1,
            )
            history_residual = state_x - history_trend

            first = history_trend[:, 0:1, :]
            middle_index = (history_length - 1) // 2
            middle = history_trend[:, middle_index : middle_index + 1, :]
            current_level = history_trend[:, -1:, :]

            # Historical time is normalized to [0, 1]. Fit the unique quadratic
            # through the first, middle and last trend states, then express its
            # velocity at the current (right-hand) boundary.
            middle_time = middle_index / float(history_length - 1)
            displacement = current_level - first
            acceleration = 2.0 * (
                displacement * middle_time - (middle - first)
            ) / (middle_time * (1.0 - middle_time))
            velocity = displacement + 0.5 * acceleration

            trend_variance = torch.var(
                history_trend, dim=1, unbiased=False
            )
            residual_variance = torch.var(
                history_residual, dim=1, unbiased=False
            )
            snr = trend_variance / (residual_variance + self.eps)
            gamma = torch.clamp(
                self.alpha / (snr + self.eps),
                min=0.0,
                max=self.gamma_max,
            )

            # One normalized unit is one complete history span. Therefore one
            # sampling step is 1 / (Seq_Len - 1), and the first forecast point
            # is exactly one sampling step after the last observed point.
            future_time = (
                torch.arange(
                    1,
                    self.pred_len + 1,
                    device=x.device,
                    dtype=state_dtype,
                )
                / float(history_length - 1)
            ).view(1, self.pred_len, 1)
            damping = torch.exp(-gamma.unsqueeze(1) * future_time)
            future_trend = current_level + (
                velocity * future_time
                + 0.5 * acceleration * future_time.square()
            ) * damping

            output_dtype = x.dtype
            self.history_trend = history_trend.to(dtype=output_dtype)
            self.history_residual = history_residual.to(dtype=output_dtype)
            self.current_level = current_level.to(dtype=output_dtype)
            self.velocity = velocity.to(dtype=output_dtype)
            self.acceleration = acceleration.to(dtype=output_dtype)
            self.snr = snr.to(dtype=output_dtype)
            self.gamma = gamma.to(dtype=output_dtype)
            self.future_trend = future_trend.to(dtype=output_dtype)
            self._history_length = history_length

        # The cached trend is detached, so this subtraction preserves an
        # identity gradient from the backbone input to x.
        return x - self.history_trend

    def _restore(self, x: torch.Tensor) -> torch.Tensor:
        if self.future_trend is None:
            raise RuntimeError("call NTE with mode='norm' before mode='denorm'")
        self._validate_input(x, expected_length=self.pred_len)
        if (
            x.shape[0] != self.future_trend.shape[0]
            or x.shape[2] != self.future_trend.shape[2]
        ):
            raise ValueError(
                "denorm input batch and feature dimensions must match the "
                "preceding norm input"
            )
        return x + self.future_trend.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _validate_input(x: torch.Tensor, expected_length):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [Batch, Length, Features]")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype")
        if expected_length is not None and x.shape[1] != expected_length:
            raise ValueError(
                "denorm input length must equal the configured pred_len "
                f"({expected_length})"
            )

    def extra_repr(self) -> str:
        return (
            f"pred_len={self.pred_len}, cutoff_ratio={self.cutoff_ratio}, "
            f"alpha={self.alpha}, gamma_max={self.gamma_max}, eps={self.eps}"
        )
