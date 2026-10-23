"""Configuration loading and validation for GWF-SR experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments.
    raise ImportError("PyYAML is required to load YAML configuration files.") from exc


@dataclass(frozen=True)
class GWFConfig:
    """Single source of truth for an experiment's experimental settings."""

    scale: int = 2
    seed: int = 42
    image_size: int = 128
    sigma: float = 1.0
    noise_std: float = 0.0
    iterations: int = 300
    dt: float = 0.05
    lambda_values: tuple[float, ...] = (0.0, 0.1, 1.0, 5.0, 10.0)
    edge_weight_mode: str = "uniform"
    epsilon: float = 1e-8
    output_dir: str = "results"
    device: str = "auto"
    config_name: str = "default"

    def __post_init__(self) -> None:
        if self.scale < 2:
            raise ValueError("scale must be at least 2 for super-resolution.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.image_size % self.scale != 0:
            raise ValueError("image_size must be divisible by scale.")
        if self.sigma < 0:
            raise ValueError("sigma cannot be negative.")
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative.")
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if not self.lambda_values:
            raise ValueError("lambda_values must contain at least one value.")
        if any(value < 0 for value in self.lambda_values):
            raise ValueError("lambda_values cannot contain negative values.")
        if self.edge_weight_mode not in {"uniform", "gradient"}:
            raise ValueError("edge_weight_mode must be 'uniform' or 'gradient'.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lambda_values"] = list(self.lambda_values)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GWFConfig":
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        unknown_fields = set(data) - known_fields
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown configuration fields: {names}")
        normalized = dict(data)
        if "lambda_values" in normalized:
            normalized["lambda_values"] = tuple(float(value) for value in normalized["lambda_values"])
        return cls(**normalized)


def load_config(path: str | Path) -> GWFConfig:
    """Load and validate a JSON or YAML experiment configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            raise ValueError("Configuration must use a .json, .yaml, or .yml extension.")

    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError("Configuration root must be a mapping.")

    normalized = dict(data)
    normalized.setdefault("config_name", config_path.stem)
    return GWFConfig.from_mapping(normalized)


def save_config(config: GWFConfig, path: str | Path) -> Path:
    """Save a validated configuration in its source format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif suffix in {".yaml", ".yml"}:
        output_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        raise ValueError("Configuration must use a .json, .yaml, or .yml extension.")

    return output_path


def resolve_runtime_dirs(config: GWFConfig, root: str | Path = ".") -> dict[str, Path]:
    """Create and return the directories used by one experiment run."""

    base = Path(root) / config.output_dir / config.config_name
    paths = {
        "root": base,
        "data": base / "data",
        "results": base / "results",
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
