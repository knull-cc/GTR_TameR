"""Context-derived and selectively applied boundary reconstruction."""

import math

import torch
import torch.nn as nn


class BoundaryReconstructor(nn.Module):
    """Predict the final increment from the complete preceding input prefix."""

    def __init__(self, seq_len: int, channels: int, hidden_dim: int = 64):
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least two")
        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear((seq_len - 1) * channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        if batch_x.ndim != 3 or batch_x.shape[1:] != (
            self.seq_len,
            self.channels,
        ):
            raise ValueError(
                "batch_x must have shape [batch, {}, {}], but received {}".format(
                    self.seq_len, self.channels, tuple(batch_x.shape)
                )
            )

        # The true final point is deliberately excluded. This prevents an
        # identity shortcut and makes the reconstruction causal at inference.
        prefix = batch_x[:, :-1, :]
        predicted_increment = self.network(prefix)
        return prefix[:, -1, :] + predicted_increment


def apply_filtered_boundary_reconstruction(
    batch_x: torch.Tensor,
    predicted_last: torch.Tensor,
    threshold: float = 3.0,
    eps: float = 1e-5,
):
    """Replace only channels whose last increment exceeds a robust threshold.

    The reference distribution consists exclusively of prefix increments, so
    neither the future target nor the final increment contributes to the robust
    center and scale. Filtering is performed independently per sample/channel.
    """

    if batch_x.ndim != 3:
        raise ValueError("batch_x must have shape [batch, time, channel]")
    if batch_x.shape[1] < 4:
        raise ValueError("batch_x must contain at least four time steps")
    if predicted_last.shape != (batch_x.shape[0], batch_x.shape[2]):
        raise ValueError(
            "predicted_last must have shape [batch, channel], but received "
            f"{tuple(predicted_last.shape)}"
        )
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    prefix_differences = batch_x[:, 1:-1, :] - batch_x[:, :-2, :]
    center = prefix_differences.median(dim=1).values
    mad = (prefix_differences - center.unsqueeze(1)).abs().median(dim=1).values
    robust_scale = 1.4826 * mad

    final_difference = batch_x[:, -1, :] - batch_x[:, -2, :]
    scores = (final_difference - center).abs() / (robust_scale + eps)
    replacement_mask = scores > float(threshold)

    fixed = batch_x.clone()
    fixed[:, -1, :] = torch.where(
        replacement_mask,
        predicted_last,
        batch_x[:, -1, :],
    )
    return fixed, replacement_mask, scores
