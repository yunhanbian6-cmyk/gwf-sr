"""Reproducible image data generation and persistence utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .config import GWFConfig, resolve_runtime_dirs
from .degradation import degradation_operator_parameters, degrade, bicubic_upsample
from .synthetic import SyntheticKind, make_synthetic_image


@dataclass(frozen=True)
class ImageRecord:
    name: str
    kind: str
    seed: int
    hr_path: str
    lr_path: str
    bicubic_path: str
    hr_size: tuple[int, int]
    lr_size: tuple[int, int]
    scale: int
    sigma: float
    noise_std: float


def _ensure_float_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 2 or not image.is_floating_point():
        raise ValueError("image must be a floating-point HxW tensor.")
    return image.to(torch.float32).clamp(0.0, 1.0)


def save_image_tensor(image: torch.Tensor, path: str | Path) -> Path:
    """Save an HxW image tensor in [0, 1] as an 8-bit grayscale PNG."""

    image = _ensure_float_tensor(image)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(output_path)
    return output_path


def load_image_tensor(path: str | Path, size: int | None = None) -> torch.Tensor:
    """Load a grayscale PNG as an HxW float tensor in [0, 1]."""

    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None:
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def generate_image(
    kind: SyntheticKind = "piecewise",
    size: int = 128,
    seed: int = 42,
    scale: int = 2,
    sigma: float = 1.0,
    noise_std: float = 0.0,
    output_dir: str | Path | None = None,
    name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ImageRecord | None]:
    """Generate one reproducible HR/LR/Bicubic sample, optionally saving all tensors."""

    hr = _ensure_float_tensor(make_synthetic_image(kind=kind, size=size, seed=seed))
    result = degrade(hr, scale=scale, sigma=sigma, noise_std=noise_std, seed=seed)
    low_resolution = _ensure_float_tensor(result.low_resolution)
    bicubic = _ensure_float_tensor(bicubic_upsample(low_resolution, scale=scale))

    if output_dir is None:
        return hr, low_resolution, bicubic, None

    sample_name = name or f"{kind}_{size}_seed{seed}"
    base = Path(output_dir)
    hr_path = save_image_tensor(hr, base / f"{sample_name}_hr.png")
    lr_path = save_image_tensor(low_resolution, base / f"{sample_name}_lr.png")
    bicubic_path = save_image_tensor(bicubic, base / f"{sample_name}_bicubic.png")

    record = ImageRecord(
        name=sample_name,
        kind=kind,
        seed=seed,
        hr_path=hr_path.name,
        lr_path=lr_path.name,
        bicubic_path=bicubic_path.name,
        hr_size=(hr.shape[0], hr.shape[1]),
        lr_size=(low_resolution.shape[0], low_resolution.shape[1]),
        scale=scale,
        sigma=sigma,
        noise_std=noise_std,
    )
    return hr, low_resolution, bicubic, record


def generate_configured_dataset(
    config: GWFConfig,
    kinds: Iterable[SyntheticKind] = ("piecewise", "edges", "texture", "smooth"),
    root: str | Path = ".",
) -> list[ImageRecord]:
    """Generate, save and register all synthetic images for a configuration."""

    directories = resolve_runtime_dirs(config, root=root)
    records: list[ImageRecord] = []
    for index, kind in enumerate(kinds):
        _, _, _, record = generate_image(
            kind=kind,
            size=config.image_size,
            seed=config.seed + index,
            scale=config.scale,
            sigma=config.sigma,
            noise_std=config.noise_std,
            output_dir=directories["data"],
        )
        if record is None:  # pragma: no cover - output_dir is always set above.
            raise RuntimeError("configured dataset generation did not create an image record.")
        records.append(record)

    manifest = {
        "config_name": config.config_name,
        "config": config.to_dict(),
        "degradation": degradation_operator_parameters(config.scale, config.sigma),
        "images": [asdict(record) for record in records],
    }
    manifest_path = directories["data"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return records
