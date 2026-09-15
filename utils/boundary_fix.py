"""Deterministic test-time boundary correction for diagnostic experiments."""

import torch


def apply_boundary_fix(batch_x: torch.Tensor) -> torch.Tensor:
    """Replace the latest observation with the preceding observation.

    Args:
        batch_x: Input tensor with shape ``[batch, time, channel]``.

    Returns:
        An independent copy of ``batch_x`` whose final time step equals the
        preceding time step. The source tensor is never modified in place.
    """

    if batch_x.ndim != 3:
        raise ValueError(
            "batch_x must have shape [batch, time, channel], "
            f"but received {tuple(batch_x.shape)}"
        )
    if batch_x.shape[1] < 2:
        raise ValueError("batch_x must contain at least two time steps")

    fixed = batch_x.clone()
    fixed[:, -1, :] = batch_x[:, -2, :]
    return fixed
