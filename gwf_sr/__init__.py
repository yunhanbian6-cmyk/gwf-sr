"""GWF-based image super-resolution research package."""

from .config import GWFConfig, load_config, save_config
from .curvature import (
    curvature_acceleration,
    curvature_quotient,
    entropy_decay_rate,
    entropy_production,
)
from .energy import dirichlet_energy, entropy_energy, log_ratio_smoothness, total_variation
from .geometry import (
    distribution_from_image,
    effective_state_count,
    entropy,
    hilbert_distance,
    kl_divergence,
    max_probability,
    min_probability,
    total_mass,
)
from .graph import (
    PixelGraph,
    PixelGraphBatch,
    build_batch_pixel_graphs,
    build_pixel_graph,
    graph_degree,
    graph_spectral_radius_bound,
    graph_total_weight,
    normalized_adjacency_action,
)

__all__ = [
    "GWFConfig",
    "load_config",
    "save_config",
    "distribution_from_image",
    "entropy",
    "kl_divergence",
    "hilbert_distance",
    "total_mass",
    "min_probability",
    "max_probability",
    "effective_state_count",
    "PixelGraph",
    "PixelGraphBatch",
    "build_pixel_graph",
    "build_batch_pixel_graphs",
    "normalized_adjacency_action",
    "graph_degree",
    "graph_total_weight",
    "graph_spectral_radius_bound",
    "dirichlet_energy",
    "entropy_energy",
    "total_variation",
    "log_ratio_smoothness",
    "entropy_production",
    "curvature_quotient",
    "curvature_acceleration",
    "entropy_decay_rate",
]
__version__ = "0.1.0"
