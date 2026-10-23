"""Synthetic grayscale images for reproducible super-resolution experiments."""

from __future__ import annotations

from typing import Literal

import torch

SyntheticKind = Literal["piecewise", "edges", "texture", "smooth"]


def _check_size(size: int) -> None:
    if size <= 1:
        raise ValueError("size must be greater than one.")


def piecewise_constant(size: int, seed: int = 42) -> torch.Tensor:
    """Create a deterministic piecewise-constant grayscale image."""

    _check_size(size)
    generator = torch.Generator().manual_seed(seed)
    column_stride = max(1, size // 4)
    columns = torch.div(torch.arange(size, dtype=torch.int64), column_stride, rounding_mode="floor")
    column_values = torch.rand(columns.max().item() + 1, generator=generator)
    image = column_values[columns].clone()

    row_stride = max(1, size // 3)
    row_blocks = torch.div(torch.arange(size, dtype=torch.int64), row_stride, rounding_mode="floor")
    row_scale = torch.linspace(0.72, 1.0, row_blocks.max().item() + 1)
    return (image[:, None] * row_scale[row_blocks][None, :]).clamp(0.0, 1.0)


def edges(size: int, seed: int = 42) -> torch.Tensor:
    """Create an image containing synthetic straight edges."""

    _check_size(size)
    generator = torch.Generator().manual_seed(seed)
    coordinate = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")

    center = size * 0.52
    half_width = max(size / 24.0, 1.0)
    vertical_edge = (xx >= center - half_width) & (xx <= center + half_width)
    diagonal_edge = torch.abs(xx + 0.72 * yy - size * 1.18) <= half_width

    image = torch.full((size, size), 0.16)
    image[diagonal_edge] = 0.92
    image[vertical_edge] = 0.58

    noise = (torch.rand((size, size), generator=generator) - 0.5) * 0.025
    return (image + noise).clamp(0.0, 1.0)


def texture(size: int, seed: int = 42) -> torch.Tensor:
    """Create a periodic texture with a deterministic phase."""

    _check_size(size)
    generator = torch.Generator().manual_seed(seed)
    coordinate = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    phase = torch.rand((), generator=generator) * 2.0 * torch.pi

    coarse = 0.5 + 0.22 * torch.sin(2.0 * torch.pi * xx / (size / 3.0))
    fine = 0.12 * torch.sin(2.0 * torch.pi * (xx + yy) / (size / 7.0) + phase)
    band = 0.08 * torch.cos(2.0 * torch.pi * (xx - 2.0 * yy) / (size / 11.0))
    return (coarse + fine + band).clamp(0.0, 1.0)


def smooth(size: int, seed: int = 42) -> torch.Tensor:
    """Create a smooth low-frequency grayscale image."""

    _check_size(size)
    generator = torch.Generator().manual_seed(seed)
    low_resolution = max(4, size // 4)
    coarse = torch.rand((low_resolution, low_resolution), generator=generator)
    image = torch.nn.functional.interpolate(
        coarse[None, None],
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0]
    return image.sub(image.min()).div(image.max() - image.min()).clamp(0.0, 1.0)


def make_synthetic_image(
    kind: SyntheticKind = "piecewise",
    size: int = 128,
    seed: int = 42,
) -> torch.Tensor:
    """Create one deterministic grayscale image in the range [0, 1]."""

    generators = {
        "piecewise": piecewise_constant,
        "edges": edges,
        "texture": texture,
        "smooth": smooth,
    }
    try:
        generator = generators[kind]
    except KeyError as exc:
        names = ", ".join(sorted(generators))
        raise ValueError(f"Unknown synthetic image kind '{kind}'. Choose from: {names}.") from exc
    return generator(size=size, seed=seed).to(torch.float32)


def make_synthetic_dataset(
    size: int = 128,
    seed: int = 42,
    kinds: tuple[SyntheticKind, ...] = ("piecewise", "edges", "texture", "smooth"),
) -> dict[str, torch.Tensor]:
    """Create a named collection of deterministic grayscale images."""

    return {kind: make_synthetic_image(kind, size=size, seed=seed + index) for index, kind in enumerate(kinds)}
