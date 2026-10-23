"""Frozen stage-four sparse pixel graph construction and normalized operators.

Implements contract ``gwf_sr/design/graph_curvature_energy_contract.md`` §§2-6:
one node per pixel, raster indexing ``index = r * W + c``, uniform positive
weights, explicit undirected edge incidence stored as both directed halves,
and the symmetric degree-normalized adjacency action. No public API forms a
dense ``N×N`` matrix; all reductions are ``O(E) = O(N)`` tensor reductions over
the edge list.

Both directed directions are stored explicitly (``E = 2 * edge_count``).
``edge_count`` always counts undirected edges; every directed formula
therefore halves once so each undirected edge is counted exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SPATIAL_DIMS = (-2, -1)

WEIGHT_MODES = ("uniform",)


def _validate_mode(edge_weight_mode: str) -> None:
    if not isinstance(edge_weight_mode, str):
        raise TypeError("edge_weight_mode must be a string.")
    if edge_weight_mode not in WEIGHT_MODES:
        raise ValueError(
            f"Unknown edge_weight_mode '{edge_weight_mode}'. Choose from: {', '.join(WEIGHT_MODES)}."
        )


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in shape)
    ):
        raise ValueError(f"shape must be a (H, W) tuple of ints, got {shape!r}.")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must contain positive dimensions, got {(height, width)}.")
    return height, width


def _field_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape H×W or B×H×W, got {tuple(values.shape)}.")
    if values.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64, got {values.dtype}.")
    if values.shape[-2] == 0 or values.shape[-1] == 0:
        raise ValueError(f"{name} must contain at least one pixel state.")
    if values.ndim == 3 and values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch sample.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} input contains NaN or Inf.")
    return values


def _arbitrary_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64, got {values.dtype}.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} input contains NaN or Inf.")
    return values


@dataclass(frozen=True)
class PixelGraph:
    """Sparse undirected pixel graph with explicit directed incidence halves."""

    height: int
    width: int
    node_count: int
    edge_count: int
    edge_weight_mode: str
    dtype: torch.dtype
    device: torch.device
    source_index: torch.Tensor
    target_index: torch.Tensor
    edge_weight: torch.Tensor


@dataclass(frozen=True)
class PixelGraphBatch:
    """Concatenated per-sample pixel graphs with explicit batch ownership."""

    batch_size: int
    height: int
    width: int
    node_count: int
    edge_count: int
    edge_weight_mode: str
    dtype: torch.dtype
    device: torch.device
    source_index: torch.Tensor
    target_index: torch.Tensor
    edge_weight: torch.Tensor
    batch_index: torch.Tensor


GraphLike = Union[PixelGraph, PixelGraphBatch]


def _uniform_edge_tables(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)

    horizontal_source = (rows[:, None] * width + cols[None, :-1]).reshape(-1)
    horizontal_target = horizontal_source + 1

    vertical_source = (rows[None, :-1] * width + cols[:, None]).reshape(-1)
    vertical_target = vertical_source + width

    forward_source = torch.cat((horizontal_source, vertical_source))
    forward_target = torch.cat((horizontal_target, vertical_target))
    reverse_source = forward_target
    reverse_target = forward_source

    source = torch.cat((forward_source, reverse_source))
    target = torch.cat((forward_target, reverse_target))
    weight = torch.ones(source.numel(), dtype=dtype, device=device)
    return source, target, weight


def build_pixel_graph(
    shape: tuple[int, int],
    edge_weight_mode: str = "uniform",
) -> PixelGraph:
    """Build a sparse undirected pixel graph for one ``H×W`` image."""

    height, width = _validate_shape(shape)
    _validate_mode(edge_weight_mode)

    node_count = height * width
    edge_count = height * (width - 1) + width * (height - 1)
    device = torch.get_default_device()
    dtype = torch.float32
    source, target, weight = _uniform_edge_tables(height, width, device=device, dtype=dtype)

    return PixelGraph(
        height=height,
        width=width,
        node_count=node_count,
        edge_count=edge_count,
        edge_weight_mode=edge_weight_mode,
        dtype=dtype,
        device=device,
        source_index=source,
        target_index=target,
        edge_weight=weight,
    )


def build_batch_pixel_graphs(
    images: torch.Tensor,
    edge_weight_mode: str = "uniform",
) -> PixelGraphBatch:
    """Build one pixel graph per sample of a ``B×H×W`` tensor."""

    images = _field_tensor(images, "images")
    if images.ndim != 3:
        raise ValueError(f"images must have shape B×H×W for batch graph construction, got {tuple(images.shape)}.")
    _validate_mode(edge_weight_mode)

    batch_size, height, width = images.shape
    node_count = height * width
    edge_count = height * (width - 1) + width * (height - 1)
    source, target, weight = _uniform_edge_tables(height, width, device=images.device, dtype=images.dtype)

    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=images.device, dtype=torch.int64),
        torch.full((batch_size,), source.numel(), dtype=torch.int64, device=images.device),
    )

    offsets = torch.arange(batch_size, device=images.device, dtype=torch.int64) * node_count
    return PixelGraphBatch(
        batch_size=batch_size,
        height=height,
        width=width,
        node_count=node_count,
        edge_count=edge_count,
        edge_weight_mode=edge_weight_mode,
        dtype=images.dtype,
        device=images.device,
        source_index=(source.repeat(batch_size) + offsets.repeat_interleave(source.numel())),
        target_index=(target.repeat(batch_size) + offsets.repeat_interleave(target.numel())),
        edge_weight=weight.repeat(batch_size),
        batch_index=batch_index,
    )


def _validate_graph(graph: GraphLike) -> None:
    if not isinstance(graph, (PixelGraph, PixelGraphBatch)):
        raise TypeError("graph must be a PixelGraph or PixelGraphBatch.")
    if graph.edge_count <= 0 or graph.source_index.numel() == 0:
        raise ValueError("graph must contain at least one undirected edge.")
    if isinstance(graph, PixelGraphBatch):
        expected_edges = 2 * graph.edge_count * graph.batch_size
        if graph.source_index.numel() != expected_edges:
            raise ValueError(
                f"batched directed edge table must contain 2*edge_count*batch_size={expected_edges} entries, "
                f"got {graph.source_index.numel()}."
            )
        if graph.batch_index.numel() != graph.source_index.numel():
            raise ValueError("batch_index must have the same length as the edge tables.")
    else:
        if graph.source_index.numel() != 2 * graph.edge_count:
            raise ValueError("directed edge table must contain exactly 2*edge_count entries.")
    if graph.edge_weight_mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown edge_weight_mode '{graph.edge_weight_mode}'.")
    if graph.source_index.numel() != graph.edge_weight.numel():
        raise ValueError("edge_weight must have the same length as source_index.")
    if graph.target_index.numel() != graph.source_index.numel():
        raise ValueError("target_index must have the same length as source_index.")
    if graph.source_index.min() < 0 or graph.target_index.min() < 0:
        raise ValueError("edge indices must be nonnegative.")
    if isinstance(graph, PixelGraphBatch):
        max_index = graph.node_count * graph.batch_size
    else:
        max_index = graph.node_count
    if graph.source_index.max() >= max_index or graph.target_index.max() >= max_index:
        raise ValueError("edge indices must be less than the total node count.")


def _validate_graph_values(values: torch.Tensor, graph: GraphLike, name: str) -> torch.Tensor:
    values = _field_tensor(values, name)
    if isinstance(graph, PixelGraphBatch):
        if values.ndim != 3:
            raise ValueError(f"{name} must have shape B×H×W for a PixelGraphBatch, got {tuple(values.shape)}.")
        if tuple(values.shape) != (graph.batch_size, graph.height, graph.width):
            raise ValueError(
                f"{name} shape {tuple(values.shape)} does not match graph "
                f"{(graph.batch_size, graph.height, graph.width)}."
            )
    else:
        if values.ndim != 2:
            raise ValueError(f"{name} must have shape H×W for a PixelGraph, got {tuple(values.shape)}.")
        if tuple(values.shape) != (graph.height, graph.width):
            raise ValueError(f"{name} shape {tuple(values.shape)} does not match graph {(graph.height, graph.width)}.")
    return values


def _resolve_graph_dtype(graph: GraphLike, values: torch.Tensor, name: str) -> torch.dtype:
    if values.device != graph.device:
        raise ValueError(f"{name} device {values.device} does not match graph device {graph.device}.")
    return values.dtype


def _node_values(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    return values.reshape(-1)


def _edge_difference(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    flat = _node_values(values, graph)
    return flat[graph.source_index] - flat[graph.target_index]


def _degree(graph: GraphLike) -> torch.Tensor:
    node_count = graph.node_count * (graph.batch_size if isinstance(graph, PixelGraphBatch) else 1)
    degree = torch.zeros(node_count, dtype=graph.dtype, device=graph.device)
    degree.index_add_(0, graph.source_index, graph.edge_weight)
    return degree


def _graph_shape(graph: GraphLike) -> tuple[int, ...]:
    if isinstance(graph, PixelGraphBatch):
        return (graph.batch_size, graph.height, graph.width)
    return (graph.height, graph.width)


def _batch_view(tensor: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    return tensor.reshape(_graph_shape(graph))


def _validate_lambda(lambda_value: float) -> None:
    if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)):
        raise TypeError("lambda_value must be a finite Python int or float.")
    if not math.isfinite(lambda_value):
        raise ValueError("lambda_value must be finite.")
    if lambda_value < 0:
        raise ValueError("lambda_value cannot be negative.")


def _validate_epsilon(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a finite Python int or float.")
    if not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite.")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")


def _validate_strictly_positive(distribution: torch.Tensor, name: str = "distribution") -> torch.Tensor:
    distribution = _field_tensor(distribution, name)
    if torch.any(distribution <= 0.0):
        raise ValueError(f"{name} must be strictly positive for logarithmic graph metrics.")
    return distribution


def graph_degree(graph: GraphLike) -> torch.Tensor:
    """Return the weighted degree of every node."""

    _validate_graph(graph)
    degree = _degree(graph)
    return _batch_view(degree, graph)


def graph_total_weight(graph: GraphLike) -> torch.Tensor:
    """Return the total weight over unique undirected edges."""

    _validate_graph(graph)
    if isinstance(graph, PixelGraphBatch):
        return graph.edge_weight.reshape(graph.batch_size, -1).sum(dim=-1) * 0.5
    return graph.edge_weight.sum() * 0.5


def graph_spectral_radius_bound(graph: GraphLike) -> torch.Tensor:
    """Return the frozen spectral-radius bound of ``A_norm`` (always ``1.0``)."""

    _validate_graph(graph)
    if isinstance(graph, PixelGraphBatch):
        return torch.ones(graph.batch_size, dtype=graph.dtype, device=graph.device)
    return torch.ones((), dtype=graph.dtype, device=graph.device)


def _normalized_weight(graph: GraphLike, values_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    degree = _degree(graph).to(values_dtype)
    normalizer = torch.sqrt(degree[graph.source_index] * degree[graph.target_index])
    if torch.any(normalizer <= 0.0):
        raise ValueError("normalized graph operators require a positive degree for every edge endpoint.")
    return graph.edge_weight.to(values_dtype) / normalizer, degree


def normalized_adjacency_action(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    """Apply the symmetric normalized adjacency without forming ``N×N``."""

    values = _validate_graph_values(values, graph, "values")
    _validate_graph(graph)
    _resolve_graph_dtype(graph, values, "values")

    normalized_weight, degree = _normalized_weight(graph, values.dtype)
    flat = _node_values(values, graph)
    target_value = flat[graph.target_index]
    result = torch.zeros_like(flat)
    result.index_add_(0, graph.source_index, normalized_weight * target_value)
    return result.reshape(values.shape)


def _edge_weight_or_raise(graph: GraphLike) -> torch.Tensor:
    weight = graph.edge_weight
    if torch.any(weight < 0.0):
        raise ValueError("edge weights must be nonnegative.")
    return weight


def _diffusion_energy(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    difference = _edge_difference(values, graph)
    energy = difference.square() * graph.edge_weight.to(values.dtype)
    if isinstance(graph, PixelGraphBatch):
        return energy.reshape(graph.batch_size, -1).sum(dim=-1) * 0.5
    return energy.sum() * 0.5


def _normalized_log_gradient_energy(distribution: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    log_distribution = torch.log(distribution)
    difference = _edge_difference(log_distribution, graph)
    normalized_weight, _ = _normalized_weight(graph, distribution.dtype)
    energy = normalized_weight * difference.square()
    if isinstance(graph, PixelGraphBatch):
        return energy.reshape(graph.batch_size, -1).sum(dim=-1) * 0.5
    return energy.sum() * 0.5


def _entropy_production(distribution: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    flat = _node_values(distribution, graph)
    source_value = flat[graph.source_index]
    target_value = flat[graph.target_index]
    difference = source_value - target_value
    denominator = torch.sqrt(source_value * target_value)
    if torch.any(denominator <= 0.0):
        raise ValueError("entropy production requires a strictly positive distribution.")
    normalized_weight, _ = _normalized_weight(graph, distribution.dtype)
    contribution = normalized_weight * (difference / denominator).square()
    if isinstance(graph, PixelGraphBatch):
        return contribution.reshape(graph.batch_size, -1).sum(dim=-1) * 0.5
    return contribution.sum() * 0.5


def _total_variation(values: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    difference = _edge_difference(values, graph).abs() * graph.edge_weight.to(values.dtype)
    if isinstance(graph, PixelGraphBatch):
        return difference.reshape(graph.batch_size, -1).sum(dim=-1) * 0.5
    return difference.sum() * 0.5


__all__ = [
    "PixelGraph",
    "PixelGraphBatch",
    "GraphLike",
    "WEIGHT_MODES",
    "build_pixel_graph",
    "build_batch_pixel_graphs",
    "normalized_adjacency_action",
    "graph_degree",
    "graph_total_weight",
    "graph_spectral_radius_bound",
    "_validate_graph",
    "_validate_graph_values",
    "_resolve_graph_dtype",
    "_edge_weight_or_raise",
    "_diffusion_energy",
    "_normalized_log_gradient_energy",
    "_entropy_production",
    "_total_variation",
    "_validate_lambda",
    "_validate_epsilon",
    "_validate_strictly_positive",
    "_field_tensor",
    "_arbitrary_tensor",
]
