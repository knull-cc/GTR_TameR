"""Test-time input perturbations used by robustness experiments."""

from typing import Optional

import numpy as np
import torch


PERTURBATION_NONE = "none"
PERTURBATION_LAST = "last"
PERTURBATION_TYPES = (PERTURBATION_NONE, PERTURBATION_LAST)


def apply_input_perturbation(
    batch_x: torch.Tensor,
    perturb_type: str,
    perturb_ratio: float,
    rng: Optional[np.random.RandomState] = None,
) -> torch.Tensor:
    """Return a perturbed copy of a ``[batch, time, channel]`` input tensor.

    ``last`` follows TameR's recent-single-anomalous-point protocol. For every
    sample and channel, independent standard Gaussian noise is scaled by the
    channel's population standard deviation within that input window and by
    ``perturb_ratio``. Only the most recent observation is changed.

    The source tensor is never modified in place, which lets callers evaluate
    clean and perturbed inputs from the same test batch.
    """

    if batch_x.ndim != 3:
        raise ValueError(
            "batch_x must have shape [batch, time, channel], "
            f"but received {tuple(batch_x.shape)}"
        )
    if perturb_type not in PERTURBATION_TYPES:
        raise ValueError(
            f"Unknown perturbation type {perturb_type!r}; "
            f"expected one of {PERTURBATION_TYPES}"
        )
    if perturb_ratio < 0:
        raise ValueError("perturb_ratio must be non-negative")

    perturbed = batch_x.clone()
    if perturb_type == PERTURBATION_NONE:
        return perturbed
    if rng is None:
        raise ValueError("rng is required for stochastic perturbations")

    # NumPy's std in the TameR implementation uses ddof=0. ``unbiased=False``
    # is the equivalent PyTorch definition and also works on older releases.
    channel_scale = batch_x.std(dim=1, keepdim=True, unbiased=False)
    noise = rng.standard_normal(
        (batch_x.shape[0], 1, batch_x.shape[2])
    )
    noise = torch.as_tensor(noise, dtype=batch_x.dtype, device=batch_x.device)
    perturbed[:, -1:, :] += noise * channel_scale * float(perturb_ratio)
    return perturbed


def perturbation_tag(
    perturb_type: str, perturb_ratio: float, perturb_seed: int
) -> str:
    """Build a stable, filesystem-safe identifier for a perturbation run."""

    ratio = format(float(perturb_ratio), "g").replace("-", "m").replace(".", "p")
    return f"perturb_{perturb_type}_ratio{ratio}_seed{perturb_seed}"
