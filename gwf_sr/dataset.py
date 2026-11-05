# -*- coding: utf-8 -*-
"""Real image loading and standard SR dataset support.

Supports local images, grayscale/RGB, and standard benchmarks
(Set5, Set14, BSD100, Urban100) when available on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

STANDARD_DATASETS = {
    "Set5": ["baby", "bird", "butterfly", "head", "woman"],
    "Set14": ["baboon", "barbara", "bridge", "coastguard", "comic",
              "face", "flowers", "foreman", "lenna", "man",
              "monarch", "pepper", "ppt3", "zebra"],
    "BSD100": None,  # 100 images, loaded by glob
    "Urban100": None,  # 100 images, loaded by glob
}


@dataclass(frozen=True)
class ImageRecord:
    """One real image sample."""
    name: str
    path: str
    height: int
    width: int
    channels: int


def load_image_tensor(
    path: str | Path,
    grayscale: bool = True,
    max_size: int | None = None,
) -> torch.Tensor:
    """Load an image as a float32 tensor in [0, 1].

    Args:
        path: Image file path.
        grayscale: If True, convert to single-channel H×W.
                   If False, keep RGB as 3×H×W.
        max_size: If set, resize so the longer side equals max_size.

    Returns:
        H×W (grayscale) or 3×H×W (RGB) float32 tensor.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {p.suffix}")

    with Image.open(p) as img:
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        if max_size is not None and max_size > 0:
            w, h = img.size
            scale = max_size / max(w, h)
            if scale < 1.0:
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        array = np.asarray(img, dtype=np.float32) / 255.0

    if grayscale:
        return torch.from_numpy(array)  # H×W
    else:
        return torch.from_numpy(array).permute(2, 0, 1)  # 3×H×W


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> Path:
    """Save a tensor as PNG. Accepts H×W or 3×H×W in [0,1]."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    t = tensor.detach().cpu().clamp(0, 1)
    if t.ndim == 2:
        array = (t.numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array, mode="L").save(p)
    elif t.ndim == 3 and t.shape[0] == 3:
        array = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(p)
    else:
        raise ValueError(f"Expected H×W or 3×H×W, got {tuple(t.shape)}")
    return p


def load_image_directory(
    directory: str | Path,
    grayscale: bool = True,
    max_size: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Load all supported images from a directory.

    Returns:
        List of (name, tensor) sorted by name.
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    images = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file():
            name = p.stem
            tensor = load_image_tensor(p, grayscale=grayscale, max_size=max_size)
            images.append((name, tensor))

    if not images:
        raise ValueError(f"No supported images found in {d}")
    return images


def load_standard_dataset(
    dataset_name: str,
    root: str | Path,
    grayscale: bool = True,
    max_size: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Load a standard SR benchmark dataset.

    Args:
        dataset_name: One of Set5, Set14, BSD100, Urban100.
        root: Root directory containing the dataset folder.
        grayscale: Convert to grayscale.
        max_size: Resize longer side to this size.

    Returns:
        List of (name, tensor).
    """
    if dataset_name not in STANDARD_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {sorted(STANDARD_DATASETS)}"
        )

    dataset_dir = Path(root) / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Please download {dataset_name} and place it at this location."
        )

    return load_image_directory(dataset_dir, grayscale=grayscale, max_size=max_size)


def load_local_images(
    paths: Iterable[str | Path],
    grayscale: bool = True,
    max_size: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Load specific image files by path.

    Returns:
        List of (name, tensor).
    """
    images = []
    for p in paths:
        p = Path(p)
        name = p.stem
        tensor = load_image_tensor(p, grayscale=grayscale, max_size=max_size)
        images.append((name, tensor))
    return images


def degrade_and_prepare(
    hr: torch.Tensor,
    scale: int = 2,
    sigma: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Degrade HR to LR and produce Bicubic initialization.

    Returns:
        (lr, bicubic_init) where lr is H/scale × W/scale
        and bicubic_init is H × W.
    """
    from gwf_sr.degradation import bicubic_upsample, degrade

    if hr.ndim == 3 and hr.shape[0] == 3:
        # RGB: degrade each channel independently
        lr_channels = []
        bicubic_channels = []
        for c in range(3):
            result = degrade(hr[c], scale=scale, sigma=sigma, noise_std=noise_std, seed=seed)
            lr_channels.append(result.low_resolution)
            bicubic_channels.append(bicubic_upsample(result.low_resolution, scale=scale))
        lr = torch.stack(lr_channels)  # 3×H/scale×W/scale
        bicubic = torch.stack(bicubic_channels)  # 3×H×W
    else:
        result = degrade(hr, scale=scale, sigma=sigma, noise_std=noise_std, seed=seed)
        lr = result.low_resolution
        bicubic = bicubic_upsample(lr, scale=scale)

    return lr, bicubic
