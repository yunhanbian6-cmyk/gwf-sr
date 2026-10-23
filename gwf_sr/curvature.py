"""Frozen stage-four curvature quotient and decay diagnostics.

Implements contract ``gwf_sr/design/graph_curvature_energy_contract.md`` §8
and §10: the nonnegative normalized entropy production, the curvature
quotient ``K_lambda = lambda * I / H`` with ``K_0 = 0``, the curvature
function ``a(R) = R^2 e^R / (e^R - 1 - R)`` with ``a(0) = 2``, and the
exponential decay diagnostic. No API clips to a prescribed theoretical bound
and no dense ``N×N`` object is formed.
"""

from __future__ import annotations

import math

import torch

from .energy import entropy_energy
from .graph import (
    GraphLike,
    _arbitrary_tensor,
    _edge_weight_or_raise,
    _entropy_production,
    _resolve_graph_dtype,
    _validate_graph,
    _validate_lambda,
    _validate_strictly_positive,
)


def entropy_production(distribution: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    """Return the nonnegative normalized squared log-density flux."""

    distribution = _validate_strictly_positive(distribution)
    _validate_graph(graph)
    _resolve_graph_dtype(graph, distribution, "distribution")
    _edge_weight_or_raise(graph)
    return _entropy_production(distribution, graph)


def curvature_quotient(
    distribution: torch.Tensor,
    graph: GraphLike,
    lambda_value: float,
) -> torch.Tensor:
    """Return ``K_lambda = lambda * I(rho) / H(rho)`` with ``K_0 = 0``."""

    distribution = _validate_strictly_positive(distribution)
    _validate_graph(graph)
    _resolve_graph_dtype(graph, distribution, "distribution")
    _validate_lambda(lambda_value)
    _edge_weight_or_raise(graph)

    if lambda_value == 0:
        return torch.zeros(
            () if distribution.ndim == 2 else (distribution.shape[0],),
            dtype=distribution.dtype,
            device=distribution.device,
        )

    production = _entropy_production(distribution, graph)
    denominator = entropy_energy(distribution, graph, lambda_value)
    if torch.any(denominator <= 0.0) or not torch.isfinite(denominator).all():
        raise ValueError("curvature quotient requires a positive finite entropy denominator.")
    return lambda_value * production / denominator


def curvature_acceleration(ratio: torch.Tensor) -> torch.Tensor:
    """Apply ``a(R) = R^2 e^R / (e^R - 1 - R)`` elementwise with ``a(0)=2``."""

    ratio = _arbitrary_tensor(ratio, "ratio")
    if torch.any(ratio < 0.0):
        raise ValueError("ratio must be nonnegative for the stage-four curvature function.")

    small = ratio <= 1e-3
    safe_ratio = torch.where(small, torch.ones_like(ratio), ratio)
    denominator = torch.expm1(safe_ratio) - safe_ratio
    numerator = safe_ratio.square() * torch.exp(safe_ratio)
    general_value = numerator / denominator

    # Series expansion near R=0: a(R) = 2 + (2/3)R + (1/4)R^2 + (1/15)R^3 + ...
    series = 2.0 + ratio * (2.0 / 3.0 + ratio * (0.25 + ratio / 15.0))
    return torch.where(small, series, general_value)


def entropy_decay_rate(
    lambda_value: float,
    initial_ratio: torch.Tensor | float,
    time: torch.Tensor | float,
) -> torch.Tensor:
    """Return ``D_lambda(R_0, t) = lambda + lambda * a(R_0 exp(-lambda t))``."""

    if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)):
        raise TypeError("lambda_value must be a finite Python int or float.")
    if not math.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite.")
    if lambda_value < 0:
        raise ValueError("lambda_value cannot be negative.")

    ratio = _coerce_finite_nonnegative(initial_ratio, "initial_ratio")
    elapsed = _coerce_finite_nonnegative(time, "time")
    if not isinstance(ratio, torch.Tensor):
        ratio = torch.tensor(ratio, dtype=torch.float64)
    if not isinstance(elapsed, torch.Tensor):
        elapsed = torch.tensor(elapsed, dtype=ratio.dtype, device=ratio.device)
    ratio, elapsed = torch.broadcast_tensors(ratio, elapsed)

    if lambda_value == 0:
        return torch.zeros(ratio.shape, dtype=ratio.dtype, device=ratio.device)

    decayed = ratio * torch.exp(-lambda_value * elapsed)
    return lambda_value + lambda_value * curvature_acceleration(decayed)


def _coerce_finite_nonnegative(value: torch.Tensor | float, name: str) -> torch.Tensor | float:
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use torch.float32 or torch.float64, got {value.dtype}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} input contains NaN or Inf.")
        if torch.any(value < 0.0):
            raise ValueError(f"{name} must be nonnegative.")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite Python int or float.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


__all__ = [
    "entropy_production",
    "curvature_quotient",
    "curvature_acceleration",
    "entropy_decay_rate",
]
