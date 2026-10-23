"""Fixed degradation operators used by the GWF-SR experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_image(image: torch.Tensor, expected_ndim: int = 2) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != expected_ndim:
        raise ValueError(f"image must have {expected_ndim} dimensions, got {image.ndim}.")
    if not image.is_floating_point():
        raise TypeError("image must use a floating-point dtype.")


def gaussian_kernel(sigma: float = 1.0, radius: int | None = None) -> torch.Tensor:
    """Create a normalized 1-D Gaussian kernel."""

    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if sigma == 0:
        return torch.ones(1, dtype=torch.float32) / 3.0

    radius = max(1, int(round(3.0 * sigma))) if radius is None else radius
    if radius < 0:
        raise ValueError("radius cannot be negative.")

    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coordinate**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(image: torch.Tensor, sigma: float = 1.0, padding: str = "reflect") -> torch.Tensor:
    """Apply separable 2-D Gaussian blur to a grayscale image."""

    _validate_image(image)
    kernel = gaussian_kernel(sigma=sigma).to(device=image.device, dtype=image.dtype)
    radius = kernel.numel() // 2
    if radius == 0:
        return image.clone()

    if padding == "reflect":
        image = torch.nn.functional.pad(image[None, None], (radius, radius, 0, 0), mode="reflect")
        image = torch.nn.functional.pad(image, (0, 0, radius, radius), mode="reflect")
    elif padding == "replicate":
        image = torch.nn.functional.pad(image[None, None], (radius, radius, 0, 0), mode="replicate")
        image = torch.nn.functional.pad(image, (0, 0, radius, radius), mode="replicate")
    else:
        raise ValueError("padding must be 'reflect' or 'replicate'.")

    horizontal = torch.nn.functional.conv2d(
        image,
        kernel.view(1, 1, 1, -1),
        padding=(0, 0),
    )
    vertical = torch.nn.functional.conv2d(
        horizontal,
        kernel.view(1, 1, -1, 1),
        padding=(0, 0),
    )
    return vertical[0, 0]


def downsample2x(image: torch.Tensor) -> torch.Tensor:
    """Perform deterministic stride-2 average downsampling."""

    _validate_image(image)
    if image.shape[0] % 2 != 0 or image.shape[1] % 2 != 0:
        raise ValueError("both image dimensions must be even for stride-2 downsampling.")
    return image.unfold(0, 2, 2).mean(dim=-1).unfold(1, 2, 2).mean(dim=-1)


def bicubic_upsample(image: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Upsample a grayscale image with antialiased bicubic interpolation."""

    _validate_image(image)
    if scale < 1:
        raise ValueError("scale must be positive.")
    return torch.nn.functional.interpolate(
        image[None, None],
        scale_factor=float(scale),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0]


@dataclass(frozen=True)
class DegradationResult:
    low_resolution: torch.Tensor
    blurred: torch.Tensor
    sigma: float
    scale: int


def degrade(
    image: torch.Tensor,
    scale: int = 2,
    sigma: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> DegradationResult:
    """Apply fixed blur, downsampling and optional seeded Gaussian noise."""

    _validate_image(image)
    if scale < 2:
        raise ValueError("scale must be at least 2.")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")

    blurred = gaussian_blur(image, sigma=sigma)
    low_resolution = downsample2x(blurred)

    if noise_std > 0:
        generator = torch.Generator(device=low_resolution.device).manual_seed(seed)
        noise = torch.randn(
            low_resolution.shape,
            generator=generator,
            dtype=low_resolution.dtype,
            device=low_resolution.device,
        )
        low_resolution = low_resolution + noise_std * noise

    return DegradationResult(
        low_resolution=low_resolution.clamp(0.0, 1.0),
        blurred=blurred,
        sigma=sigma,
        scale=scale,
    )


def degradation_operator_parameters(scale: int = 2, sigma: float = 1.0) -> dict[str, float | int]:
    """Return a serializable description of the fixed degradation operator."""

    return {"scale": scale, "sigma": sigma, "kernel_radius": int(round(3.0 * sigma)) if sigma else 0}
