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

    Like RevIN, this module is stateful: each ``norm`` must be followed by its
    matching ``denorm`` before another input is normalized. A single instance
    is therefore not re-entrant across concurrent forward calls.
    """

    def __init__(
        self,
        pred_len: int,
        cutoff_ratio: float = 0.1,
        alpha: float = 1.0,
        gamma_max: float = 20.0,
        guard_sigma: float = 3.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if not math.isfinite(cutoff_ratio) or not 0.0 < cutoff_ratio <= 1.0:
            raise ValueError("cutoff_ratio must be in (0, 1]")
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if not math.isfinite(gamma_max) or gamma_max <= 0.0:
            raise ValueError("gamma_max must be positive")
        if not math.isfinite(guard_sigma) or guard_sigma <= 0.0:
            raise ValueError("guard_sigma must be positive")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive")

        self.pred_len = int(pred_len)
        self.cutoff_ratio = float(cutoff_ratio)
        self.alpha = float(alpha)
        self.gamma_max = float(gamma_max)
        self.guard_sigma = float(guard_sigma)
        self.eps = float(eps)

        # Per-batch state. These are intentionally not buffers, so a checkpoint
        # contains only the backbone and NTE remains parameter/state-dict free.
        self.history_trend = None
        self.history_residual = None
        self.detrend_velocity = None
        self.detrend_acceleration = None
        self.recent_scale = None
        self.recent_adjustment = None
        self.current_level = None
        self.velocity = None
        self.acceleration = None
        self.snr = None
        self.gamma = None
        self.future_trend = None
        self._history_length = None
        self._detrend_projection = None
        self._detrend_projection_spec = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._separate(x)
        if mode == "denorm":
            return self._restore(x)
        raise ValueError("mode must be either 'norm' or 'denorm'")

    def _separate(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x, expected_length=None)
        if x.shape[1] < 5:
            raise ValueError("NTE requires a history length of at least 5")

        with torch.no_grad():
            state_dtype = (
                torch.float32
                if x.dtype in (torch.float16, torch.bfloat16)
                else x.dtype
            )
            state_x = x.detach().to(dtype=state_dtype)
            history_length = state_x.shape[1]

            # An FFT assumes a periodic boundary. Filtering a non-periodic
            # kinematic trajectory directly would join its last point back to
            # its first and corrupt precisely the boundary we need to forecast.
            # Estimate a quadratic from the prefix (excluding the latest
            # point), remove it before the FFT, and add it back afterwards.
            prefix = state_x[:, :-1, :]
            projection = self._get_detrend_projection(
                history_length, x.device, state_dtype
            )
            coefficients = torch.einsum("ks,bsc->bkc", projection, prefix)
            detrend_intercept = coefficients[:, 0:1, :]
            detrend_velocity = coefficients[:, 1:2, :]
            detrend_acceleration = coefficients[:, 2:3, :]
            history_time = (
                torch.arange(
                    history_length,
                    device=x.device,
                    dtype=state_dtype,
                )
                / float(history_length - 1)
            ).view(1, history_length, 1)
            polynomial_baseline = (
                detrend_intercept
                + detrend_velocity * history_time
                + 0.5 * detrend_acceleration * history_time.square()
            )

            # Bound the influence of the latest observation before either the
            # FFT or the backbone sees it. The center and scale use only prefix
            # increments, so an arbitrarily large final corruption cannot
            # inflate its own acceptance interval. This is a deterministic,
            # per-sample/per-feature Hampel-style guard, not a learned stage.
            prefix_deltas = prefix[:, 1:, :] - prefix[:, :-1, :]
            baseline_prefix = polynomial_baseline[:, :-1, :]
            expected_prefix_deltas = (
                baseline_prefix[:, 1:, :] - baseline_prefix[:, :-1, :]
            )
            innovations = prefix_deltas - expected_prefix_deltas
            innovation_center = torch.median(
                innovations, dim=1, keepdim=True
            ).values
            recent_scale = 1.4826 * torch.median(
                torch.abs(innovations - innovation_center),
                dim=1,
                keepdim=True,
            ).values + self.eps
            expected_last = prefix[:, -1:, :] + (
                polynomial_baseline[:, -1:, :]
                - polynomial_baseline[:, -2:-1, :]
                + innovation_center
            )
            guard_radius = self.guard_sigma * recent_scale
            recent_is_outlier = (
                torch.abs(state_x[:, -1:, :] - expected_last) > guard_radius
            )
            guarded_last = torch.where(
                recent_is_outlier,
                expected_last,
                state_x[:, -1:, :],
            )
            guarded_state_x = torch.cat([prefix, guarded_last], dim=1)
            detrended = guarded_state_x - polynomial_baseline

            spectrum = torch.fft.rfft(detrended, dim=1)
            retained_bins = min(
                spectrum.shape[1],
                max(1, math.ceil(history_length * self.cutoff_ratio)),
            )
            low_pass_mask = (
                torch.arange(spectrum.shape[1], device=x.device)
                < retained_bins
            ).view(1, -1, 1)
            history_trend = polynomial_baseline + torch.fft.irfft(
                spectrum * low_pass_mask,
                n=history_length,
                dim=1,
            )
            history_residual = guarded_state_x - history_trend

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
            self.detrend_velocity = detrend_velocity.to(dtype=output_dtype)
            self.detrend_acceleration = detrend_acceleration.to(
                dtype=output_dtype
            )
            self.recent_scale = recent_scale.to(dtype=output_dtype)
            self.recent_adjustment = (
                guarded_last - state_x[:, -1:, :]
            ).to(dtype=output_dtype)
            self.current_level = current_level.to(dtype=output_dtype)
            self.velocity = velocity.to(dtype=output_dtype)
            self.acceleration = acceleration.to(dtype=output_dtype)
            self.snr = snr.to(dtype=output_dtype)
            self.gamma = gamma.to(dtype=output_dtype)
            self.future_trend = future_trend.to(dtype=output_dtype)
            self._history_length = history_length

        # Straight-through form: forward values equal the guarded residual,
        # while d(output)/d(x) remains the identity for callers that require it.
        return self.history_residual + (x - x.detach())

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

    def _get_detrend_projection(self, history_length, device, dtype):
        spec = (history_length, device, dtype)
        if self._detrend_projection_spec != spec:
            prefix_time = (
                torch.arange(
                    history_length - 1,
                    device=device,
                    dtype=dtype,
                )
                / float(history_length - 1)
            )
            design = torch.stack(
                [
                    torch.ones_like(prefix_time),
                    prefix_time,
                    0.5 * prefix_time.square(),
                ],
                dim=1,
            )
            self._detrend_projection = torch.linalg.pinv(design)
            self._detrend_projection_spec = spec
        return self._detrend_projection

    def extra_repr(self) -> str:
        return (
            f"pred_len={self.pred_len}, cutoff_ratio={self.cutoff_ratio}, "
            f"alpha={self.alpha}, gamma_max={self.gamma_max}, "
            f"guard_sigma={self.guard_sigma}, eps={self.eps}"
        )
