"""Frozen stage-six image quality metrics.

Implements contract ``gwf_sr/design/metrics_visualization_contract.md``
§§3-8: PSNR, SSIM, optional LPIPS, high-frequency energy, edge preservation
and the frozen combination interfaces. This module only performs numerical
evaluation on tensors: it never reads or writes files, never imports
matplotlib, and never modifies its inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SPATIAL_DIMS = (-2, -1)
_LPIPS_NETS = ("alex", "vgg", "squeeze")
_LPIPS_VERSIONS = ("0.1",)
_PSNR_SENTINEL = 100.0


def _validate_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape H×W or B×H×W, got {tuple(image.shape)}.")
    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64, got {image.dtype}.")
    if image.shape[-2] == 0 or image.shape[-1] == 0:
        raise ValueError(f"{name} must contain at least one pixel.")
    if image.ndim == 3 and image.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} input contains NaN or Inf.")
    return image


def _validate_positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite Python int or float.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return float(value)


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite Python int or float.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return float(value)


def validate_image_pair(reference: torch.Tensor, prediction: torch.Tensor) -> None:
    """Validate a frozen stage-six image pair without broadcasting."""

    _validate_image(reference, "reference")
    _validate_image(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError(
            f"reference and prediction must have identical shapes, "
            f"got {tuple(reference.shape)} and {tuple(prediction.shape)}."
        )
    if reference.dtype != prediction.dtype:
        raise TypeError(
            f"reference and prediction must have identical dtypes, "
            f"got {reference.dtype} and {prediction.dtype}."
        )
    if reference.device != prediction.device:
        raise ValueError(
            f"reference and prediction must be on the same device, "
            f"got {reference.device} and {prediction.device}."
        )


def validate_metric_config(
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    epsilon: float = 1e-12,
) -> None:
    """Validate the frozen scalar parameters of the metric functions."""

    _validate_positive_scalar(data_range, "data_range")
    _validate_positive_scalar(win_sigma, "win_sigma")
    _validate_positive_scalar(epsilon, "epsilon")
    if isinstance(win_size, bool) or not isinstance(win_size, int):
        raise TypeError("win_size must be an integer.")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be an odd integer greater than or equal to 3.")


def psnr(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Return ``10*log10(data_range**2 / MSE)`` with the frozen zero-MSE sentinel."""

    validate_image_pair(reference, prediction)
    _validate_positive_scalar(data_range, "data_range")

    mse = (reference - prediction).square().mean(dim=_SPATIAL_DIMS)
    positive = mse > 0
    safe_mse = torch.where(positive, mse, torch.ones_like(mse))
    value = 20.0 * math.log10(data_range) - 10.0 * torch.log10(safe_mse)
    return torch.where(positive, value, torch.full_like(mse, _PSNR_SENTINEL))


def _gaussian_window_1d(win_size: int, win_sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coordinate = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2.0
    kernel = torch.exp(-(coordinate**2) / (2.0 * win_sigma * win_sigma))
    return kernel / kernel.sum()


def _blur_separable(image_nchw: torch.Tensor, kernel: torch.Tensor, pad: int) -> torch.Tensor:
    horizontal_input = F.pad(image_nchw, (pad, pad, 0, 0), mode="reflect")
    horizontal = F.conv2d(horizontal_input, kernel.view(1, 1, 1, -1))
    vertical_input = F.pad(horizontal, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(vertical_input, kernel.view(1, 1, -1, 1))


def ssim(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """Return the frozen single-channel mean SSIM for each sample."""

    validate_image_pair(reference, prediction)
    validate_metric_config(data_range=data_range, win_size=win_size, win_sigma=win_sigma)

    height, width = reference.shape[-2], reference.shape[-1]
    if height < win_size or width < win_size:
        raise ValueError(
            f"images must be at least {win_size}x{win_size} for win_size={win_size}, "
            f"got {height}x{width}."
        )

    dtype, device = reference.dtype, reference.device
    kernel = _gaussian_window_1d(win_size, win_sigma, dtype, device)
    pad = win_size // 2

    x = reference.reshape(-1, 1, height, width)
    y = prediction.reshape(-1, 1, height, width)

    mu_x = _blur_separable(x, kernel, pad)
    mu_y = _blur_separable(y, kernel, pad)
    sigma_x = _blur_separable(x * x, kernel, pad) - mu_x * mu_x
    sigma_y = _blur_separable(y * y, kernel, pad) - mu_y * mu_y
    sigma_xy = _blur_separable(x * y, kernel, pad) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    local = numerator / denominator
    return local.mean(dim=_SPATIAL_DIMS).reshape(reference.shape[:-2])


def lpips_available() -> bool:
    """Return whether the optional ``lpips`` package imports, without loading weights."""

    try:
        import lpips  # noqa: F401
    except ImportError:
        return False
    return True


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image[None, None].repeat(1, 3, 1, 1)
    return image[:, None].repeat(1, 3, 1, 1)


def lpips_distance(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    net: str = "alex",
    version: str = "0.1",
    require: bool = False,
    cache_dir: str | Path | None = None,
) -> torch.Tensor | None:
    """Return the optional LPIPS distance, or ``None`` when unavailable."""

    validate_image_pair(reference, prediction)
    if net not in _LPIPS_NETS:
        raise ValueError(f"Unknown LPIPS network '{net}'. Choose from: {', '.join(_LPIPS_NETS)}.")
    if version not in _LPIPS_VERSIONS:
        raise ValueError(f"Unknown LPIPS version '{version}'. Choose from: {', '.join(_LPIPS_VERSIONS)}.")
    if not isinstance(require, bool):
        raise TypeError("require must be a bool.")

    try:
        import lpips as lpips_module
    except ImportError as exc:
        if require:
            raise ImportError(
                "LPIPS requires the optional 'lpips' package; install it with: pip install lpips"
            ) from exc
        return None

    try:
        model = lpips_module.LPIPS(net=net, version=version, verbose=False, eval_mode=True)
    except Exception as exc:  # pragma: no cover - depends on the local weight cache.
        cache_hint = str(cache_dir) if cache_dir is not None else "the LPIPS default cache"
        raise RuntimeError(
            f"LPIPS weights for net='{net}' version='{version}' could not be loaded "
            f"(cache: {cache_hint}). Download the weights or reinstall 'lpips'."
        ) from exc

    try:
        model = model.to(reference.device)
    except Exception as exc:  # pragma: no cover - depends on the local device support.
        raise RuntimeError(
            f"LPIPS model could not be moved to device '{reference.device}'; "
            "the model and the inputs must share one device."
        ) from exc
    model.eval()

    reference_rgb = _to_nchw(reference)
    prediction_rgb = _to_nchw(prediction)
    with torch.no_grad():
        output = model(reference_rgb, prediction_rgb)
    values = output.reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("LPIPS returned NaN or Inf.")
    if reference.ndim == 2:
        return values[0]
    return values


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Return the replicate-boundary central-difference gradient magnitude."""

    _validate_image(image, "image")
    nchw = image.reshape(-1, 1, *image.shape[-2:])
    padded = F.pad(nchw, (1, 1, 1, 1), mode="replicate")
    grad_x = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]) * 0.5
    grad_y = (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]) * 0.5
    magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt()
    return magnitude.reshape(image.shape)


def high_frequency_energy(image: torch.Tensor) -> torch.Tensor:
    """Return the mean squared gradient magnitude of an image."""

    _validate_image(image, "image")
    magnitude = gradient_magnitude(image)
    return magnitude.square().mean(dim=_SPATIAL_DIMS)


def high_frequency_energy_ratio(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return ``HFE(prediction) / (HFE(reference) + epsilon)``."""

    validate_image_pair(reference, prediction)
    _validate_positive_scalar(epsilon, "epsilon")
    numerator = high_frequency_energy(prediction)
    denominator = high_frequency_energy(reference) + epsilon
    return numerator / denominator


def edge_preservation_index(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return the cosine similarity of the two gradient-magnitude fields."""

    validate_image_pair(reference, prediction)
    _validate_positive_scalar(epsilon, "epsilon")
    reference_magnitude = gradient_magnitude(reference)
    prediction_magnitude = gradient_magnitude(prediction)
    numerator = (reference_magnitude * prediction_magnitude).sum(dim=_SPATIAL_DIMS)
    reference_energy = reference_magnitude.square().sum(dim=_SPATIAL_DIMS)
    prediction_energy = prediction_magnitude.square().sum(dim=_SPATIAL_DIMS)
    denominator = (reference_energy * prediction_energy).sqrt() + epsilon
    return numerator / denominator


def edge_preservation_ratio(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return ``(EPI + 1) / 2`` for one image pair."""

    validate_image_pair(reference, prediction)
    return (edge_preservation_index(reference, prediction, epsilon=epsilon) + 1.0) * 0.5


@dataclass(frozen=True)
class ImageMetrics:
    """Frozen container for one reference/prediction metric evaluation."""

    psnr: torch.Tensor
    ssim: torch.Tensor
    lpips: torch.Tensor | None
    high_frequency_energy: torch.Tensor
    high_frequency_energy_ratio: torch.Tensor
    edge_preservation_index: torch.Tensor
    edge_preservation_ratio: torch.Tensor


def compare_images(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    lpips_net: str = "alex",
    lpips_version: str = "0.1",
    require_lpips: bool = False,
    lpips_cache_dir: str | Path | None = None,
) -> ImageMetrics:
    """Compute every frozen stage-six metric for one image pair."""

    validate_image_pair(reference, prediction)
    if not isinstance(require_lpips, bool):
        raise TypeError("require_lpips must be a bool.")
    return ImageMetrics(
        psnr=psnr(reference, prediction, data_range=data_range),
        ssim=ssim(reference, prediction, data_range=data_range, win_size=win_size, win_sigma=win_sigma),
        lpips=lpips_distance(
            reference,
            prediction,
            net=lpips_net,
            version=lpips_version,
            require=require_lpips,
            cache_dir=lpips_cache_dir,
        ),
        high_frequency_energy=high_frequency_energy(prediction),
        high_frequency_energy_ratio=high_frequency_energy_ratio(reference, prediction),
        edge_preservation_index=edge_preservation_index(reference, prediction),
        edge_preservation_ratio=edge_preservation_ratio(reference, prediction),
    )


def compute_image_metrics(
    references: torch.Tensor,
    predictions: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    lpips_net: str = "alex",
    lpips_version: str = "0.1",
    require_lpips: bool = False,
    lpips_cache_dir: str | Path | None = None,
) -> ImageMetrics:
    """Compute per-sample metrics for a batch or single image pair."""

    return compare_images(
        references,
        predictions,
        data_range=data_range,
        win_size=win_size,
        win_sigma=win_sigma,
        lpips_net=lpips_net,
        lpips_version=lpips_version,
        require_lpips=require_lpips,
        lpips_cache_dir=lpips_cache_dir,
    )


__all__ = [
    "ImageMetrics",
    "compare_images",
    "compute_image_metrics",
    "edge_preservation_index",
    "edge_preservation_ratio",
    "gradient_magnitude",
    "high_frequency_energy",
    "high_frequency_energy_ratio",
    "lpips_available",
    "lpips_distance",
    "psnr",
    "ssim",
    "validate_image_pair",
    "validate_metric_config",
]
