"""GWF-based image super-resolution research package."""

from .config import GWFConfig, load_config, save_config
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
]
__version__ = "0.1.0"
