"""Frozen stage-three probability geometry and information metrics."""

from __future__ import annotations

import math

import torch

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SPATIAL_DIMS = (-2, -1)


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape H×W or B×H×W, got {tuple(tensor.shape)}.")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64, got {tensor.dtype}.")
    if tensor.shape[-2] == 0 or tensor.shape[-1] == 0:
        raise ValueError(f"{name} must contain at least one pixel state.")
    if tensor.ndim == 3 and tensor.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch sample.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} input contains NaN or Inf.")


def _validate_epsilon(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a finite Python int or float.")
    if not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite.")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")


def distribution_from_image(image: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Convert an image to a strictly positive probability distribution."""

    _validate_tensor(image, "image")
    _validate_epsilon(epsilon)
    positive = image.clamp_min(0.0)
    epsilon_tensor = image.new_tensor(epsilon)
    numerator = positive + epsilon_tensor
    denominator = numerator.sum(dim=_SPATIAL_DIMS, keepdim=True)
    return numerator / denominator


def entropy(distribution: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy ``-sum(rho * log(rho))``."""

    _validate_tensor(distribution, "distribution")
    terms = torch.where(
        distribution > 0.0,
        distribution * torch.log(distribution),
        torch.zeros_like(distribution),
    )
    return -terms.sum(dim=_SPATIAL_DIMS)


def _validate_distribution_pair(rho: torch.Tensor, eta: torch.Tensor) -> None:
    _validate_tensor(rho, "rho")
    _validate_tensor(eta, "eta")
    if rho.shape != eta.shape:
        raise ValueError(f"rho and eta must have identical shapes, got {tuple(rho.shape)} and {tuple(eta.shape)}.")
    if rho.dtype != eta.dtype:
        raise TypeError(f"rho and eta must have identical dtypes, got {rho.dtype} and {eta.dtype}.")
    if rho.device != eta.device:
        raise ValueError(f"rho and eta must have identical devices, got {rho.device} and {eta.device}.")


def kl_divergence(rho: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Compute ``D_KL(rho || eta) = sum(rho * log(rho / eta))``."""

    _validate_distribution_pair(rho, eta)
    terms = rho * (torch.log(rho) - torch.log(eta))
    return terms.sum(dim=_SPATIAL_DIMS)


def hilbert_distance(rho: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Compute the Hilbert projective distance between two distributions."""

    _validate_distribution_pair(rho, eta)
    log_ratios = torch.log(rho) - torch.log(eta)
    return log_ratios.amax(dim=_SPATIAL_DIMS) - log_ratios.amin(dim=_SPATIAL_DIMS)


def total_mass(distribution: torch.Tensor) -> torch.Tensor:
    """Return the total probability mass of each distribution."""

    _validate_tensor(distribution, "distribution")
    return distribution.sum(dim=_SPATIAL_DIMS)


def min_probability(distribution: torch.Tensor) -> torch.Tensor:
    """Return the minimum probability of each distribution."""

    _validate_tensor(distribution, "distribution")
    return distribution.amin(dim=_SPATIAL_DIMS)


def max_probability(distribution: torch.Tensor) -> torch.Tensor:
    """Return the maximum probability of each distribution."""

    _validate_tensor(distribution, "distribution")
    return distribution.amax(dim=_SPATIAL_DIMS)


def effective_state_count(distribution: torch.Tensor) -> torch.Tensor:
    """Return the entropy-exponential effective state count ``exp(H(rho))``."""

    return torch.exp(entropy(distribution))


__all__ = [
    "distribution_from_image",
    "entropy",
    "kl_divergence",
    "hilbert_distance",
    "total_mass",
    "min_probability",
    "max_probability",
    "effective_state_count",
]
