"""Frozen stage-four entropy and Dirichlet energy operators.

Implements contract ``gwf_sr/design/graph_curvature_energy_contract.md`` §7
and §9: the undirected Dirichlet energy, the ``lambda``-regularized entropy
that reduces to stage-three entropy at ``lambda = 0``, total variation, and
the log-ratio smoothing diagnostic. All reductions run over the sparse edge
list in ``O(E) = O(N)`` time with ``O(N)`` extra memory.
"""

from __future__ import annotations

import torch

from .geometry import entropy
from .graph import (
    GraphLike,
    _diffusion_energy,
    _edge_weight_or_raise,
    _field_tensor,
    _normalized_log_gradient_energy,
    _resolve_graph_dtype,
    _total_variation,
    _validate_epsilon,
    _validate_graph,
    _validate_graph_values,
    _validate_lambda,
)


def dirichlet_energy(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    """Return ``1/2 sum_{(u,v)} w_uv (f_u - f_v)^2`` for one graph."""

    values = _validate_graph_values(values, graph, "values")
    _validate_graph(graph)
    _resolve_graph_dtype(graph, values, "values")
    _edge_weight_or_raise(graph)
    return _diffusion_energy(values, graph)


def entropy_energy(
    distribution: torch.Tensor,
    graph: GraphLike,
    lambda_value: float,
) -> torch.Tensor:
    """Return stage-three entropy plus ``lambda/2`` normalized log energy."""

    distribution = _validate_distribution_field(distribution)
    _validate_graph(graph)
    _resolve_graph_dtype(graph, distribution, "distribution")
    _validate_lambda(lambda_value)
    _edge_weight_or_raise(graph)

    base = entropy(distribution)
    regularizer = _normalized_log_gradient_energy(distribution, graph)
    return base + lambda_value * regularizer


def total_variation(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    """Return ``1/2 sum_{(u,v)} w_uv |f_u - f_v|`` for one graph."""

    values = _validate_graph_values(values, graph, "values")
    _validate_graph(graph)
    _resolve_graph_dtype(graph, values, "values")
    _edge_weight_or_raise(graph)
    return _total_variation(values, graph)


def _validate_distribution_field(distribution: torch.Tensor, name: str = "distribution") -> torch.Tensor:
    distribution = _field_tensor(distribution, name)
    if torch.any(distribution <= 0.0):
        raise ValueError(f"{name} must be strictly positive for logarithmic graph metrics.")
    return distribution


def total_variation_distribution(
    distribution: torch.Tensor,
    graph: GraphLike,
) -> torch.Tensor:
    """Return the total variation of a strictly positive distribution."""

    distribution = _validate_distribution_field(distribution)
    _validate_graph(graph)
    _resolve_graph_dtype(graph, distribution, "distribution")
    _edge_weight_or_raise(graph)
    return _total_variation(distribution, graph)


def log_ratio_smoothness(
    distribution: torch.Tensor,
    graph: GraphLike,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return ``TV(log rho) / (TV(rho) + epsilon_diag)`` as a diagnostic."""

    distribution = _validate_distribution_field(distribution)
    _validate_graph(graph)
    _resolve_graph_dtype(graph, distribution, "distribution")
    _validate_epsilon(epsilon)
    _edge_weight_or_raise(graph)

    log_numerator = _total_variation(torch.log(distribution), graph)
    denominator = _total_variation(distribution, graph)
    return log_numerator / (denominator + distribution.new_tensor(epsilon))


__all__ = [
    "dirichlet_energy",
    "entropy_energy",
    "total_variation",
    "log_ratio_smoothness",
]
