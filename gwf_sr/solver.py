"""Frozen stage-five super-resolution solver and iteration trajectory.

Implements contract ``gwf_sr/design/solver_contract.md`` §§2-14:
the fixed degradation forward/adjoint operators, data fidelity and its
gradient, curvature-regularized gradient flow via stage-four entropy
energy, Euler/Euler-Maruyama/Strang time discretizations with per-step
``[0,1]`` projection, full trajectory recording, and per-iteration
diagnostics. No stage-two, stage-three, or stage-four module is modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .curvature import curvature_quotient
from .degradation import downsample2x, gaussian_blur
from .energy import entropy_energy
from .geometry import (
    distribution_from_image,
    effective_state_count,
    entropy,
    max_probability,
    min_probability,
)
from .graph import PixelGraph, PixelGraphBatch, build_batch_pixel_graphs, build_pixel_graph

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SPATIAL_DIMS = (-2, -1)
_INTEGRATORS = ("euler", "euler_maruyama", "strang")


def project_image(image: torch.Tensor) -> torch.Tensor:
    """Project an image tensor onto the closed interval ``[0, 1]``."""

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"image must use torch.float32 or torch.float64, got {image.dtype}.")
    if not torch.isfinite(image).all():
        raise ValueError("image input contains NaN or Inf.")
    return image.clamp(0.0, 1.0)


def _validate_solver_tensor(tensor: torch.Tensor, name: str, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}.")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64, got {tensor.dtype}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} input contains NaN or Inf.")


def _degradation_forward(image: torch.Tensor, scale: int, sigma: float) -> torch.Tensor:
    if image.ndim == 3:
        return torch.stack(
            [_degradation_forward(sample, scale=scale, sigma=sigma) for sample in image],
            dim=0,
        )
    kernel = _gaussian_kernel_for(sigma, image.device, image.dtype)
    radius = kernel.numel() // 2

    batched = image[None, None]
    padded = torch.nn.functional.pad(batched, (radius, radius, radius, radius), mode="reflect")
    blurred = torch.nn.functional.conv2d(padded, kernel.view(1, 1, 1, -1))
    blurred = torch.nn.functional.conv2d(blurred, kernel.view(1, 1, -1, 1))
    return downsample2x(blurred[0, 0])


def _degradation_adjoint(z: torch.Tensor, image_shape: tuple[int, ...], scale: int, sigma: float) -> torch.Tensor:
    # Exact adjoint of the reflect-padded forward degradation, computed with autograd.
    # This avoids hand-coding the reflect-padding transpose (a "fold" that adds boundary
    # contributions back into the interior) and guarantees <A(x), z> == <x, A*(z)>.
    differentiable = torch.zeros(
        tuple(image_shape),
        dtype=z.dtype,
        device=z.device,
        requires_grad=True,
    )
    predicted = _degradation_forward(differentiable, scale=scale, sigma=sigma)
    inner = (predicted * z).sum()
    gradient = torch.autograd.grad(inner, differentiable)[0]
    return gradient.detach()


def _gaussian_kernel_for(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma))) if sigma > 0 else 0
    if radius == 0:
        return torch.ones(1, dtype=dtype, device=device) / 3.0
    coordinate = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coordinate**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def degradation_forward(image: torch.Tensor, *, scale: int = 2, sigma: float = 1.0) -> torch.Tensor:
    """Apply the frozen forward degradation ``A(x) = D_s(B_sigma(x))``."""

    _validate_solver_tensor(image, "image", 2 if image.ndim == 2 else 3)
    return _degradation_forward(image, scale=scale, sigma=sigma)


def degradation_adjoint(
    z: torch.Tensor,
    image_shape: tuple[int, ...],
    *,
    scale: int = 2,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply the frozen adjoint degradation ``A*(z) = B_sigma*(D_s*(z))``."""

    _validate_solver_tensor(z, "z", 2 if z.ndim == 2 else 3)
    if z.ndim == 3:
        return torch.stack(
            [
                _degradation_adjoint(sample, image_shape=image_shape, scale=scale, sigma=sigma)
                for sample in z
            ],
            dim=0,
        )
    return _degradation_adjoint(z, image_shape=image_shape, scale=scale, sigma=sigma)


def data_fidelity(
    image: torch.Tensor,
    low_resolution: torch.Tensor,
    *,
    scale: int = 2,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Return ``1/2 ||A(x) - y||_2^2`` per sample."""

    _validate_solver_tensor(image, "image", 2 if image.ndim == 2 else 3)
    _validate_solver_tensor(low_resolution, "low_resolution", 2 if low_resolution.ndim == 2 else 3)
    predicted = _degradation_forward(image, scale=scale, sigma=sigma)
    residual = predicted - low_resolution
    return 0.5 * residual.square().sum(dim=_SPATIAL_DIMS)


def data_fidelity_gradient(
    image: torch.Tensor,
    low_resolution: torch.Tensor,
    *,
    scale: int = 2,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Return ``A*(A(x) - y)`` per pixel."""

    predicted = _degradation_forward(image, scale=scale, sigma=sigma)
    residual = predicted - low_resolution
    return _degradation_adjoint(residual, image.shape, scale=scale, sigma=sigma)


def regularization_gradient(
    image: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
    *,
    lambda_value: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return ``grad_x R_lambda(x)`` via automatic differentiation."""

    if not isinstance(lambda_value, (int, float)) or isinstance(lambda_value, bool):
        raise TypeError("lambda_value must be a finite Python int or float.")
    if not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda_value must be finite and nonnegative.")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a finite Python int or float.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    if lambda_value == 0:
        return torch.zeros_like(project_image(image))

    projected = project_image(image)
    differentiable = projected.detach().clone().requires_grad_(True)
    rho = distribution_from_image(differentiable, epsilon=epsilon)
    energy = entropy_energy(rho, graph, lambda_value=lambda_value)
    gradient = torch.autograd.grad(energy, differentiable)[0]
    return gradient.detach()


@dataclass(frozen=True)
class SolverResult:
    """Frozen result object returned by :meth:`SuperResolutionSolver.solve`."""

    reconstruction: torch.Tensor
    initial_image: torch.Tensor
    low_resolution: torch.Tensor
    trajectory: torch.Tensor
    times: torch.Tensor
    data_fidelity: torch.Tensor
    entropy: torch.Tensor
    regularization_energy: torch.Tensor
    total_energy: torch.Tensor
    curvature: torch.Tensor
    min_probability: torch.Tensor
    max_probability: torch.Tensor
    effective_state_count: torch.Tensor
    update_norm: torch.Tensor
    final_update_norm: torch.Tensor
    iterations: int
    dt: float
    lambda_value: float
    epsilon: float
    integrator: str


class SuperResolutionSolver:
    """Frozen stage-five super-resolution solver."""

    def __init__(
        self,
        low_resolution: torch.Tensor,
        initialization: torch.Tensor,
        *,
        scale: int = 2,
        sigma: float = 1.0,
        iterations: int = 300,
        dt: float = 0.05,
        lambda_value: float = 0.0,
        epsilon: float = 1e-8,
        edge_weight_mode: str = "uniform",
        integrator: str = "euler",
        record_every: int = 1,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        _validate_solver_tensor(low_resolution, "low_resolution", 2 if low_resolution.ndim == 2 else 3)
        _validate_solver_tensor(initialization, "initialization", 2 if initialization.ndim == 2 else 3)

        if low_resolution.ndim != initialization.ndim:
            raise ValueError("low_resolution and initialization must have the same number of dimensions.")
        if low_resolution.ndim == 3 and low_resolution.shape[0] != initialization.shape[0]:
            raise ValueError("batch sizes of low_resolution and initialization must match.")
        if initialization.shape[-2] != scale * low_resolution.shape[-2] or initialization.shape[-1] != scale * low_resolution.shape[-1]:
            raise ValueError("initialization shape must be scale times low_resolution shape.")

        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 2:
            raise ValueError("scale must be an integer at least 2.")
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be a finite nonnegative number.")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
            raise ValueError("iterations must be a positive integer.")
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number.")
        if not isinstance(lambda_value, (int, float)) or isinstance(lambda_value, bool) or not math.isfinite(lambda_value) or lambda_value < 0:
            raise ValueError("lambda_value must be a finite nonnegative number.")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number.")
        if edge_weight_mode not in ("uniform", "gradient"):
            raise ValueError("edge_weight_mode must be 'uniform' or 'gradient'.")
        if integrator not in _INTEGRATORS:
            raise ValueError(f"integrator must be one of {_INTEGRATORS}.")
        if record_every != 1:
            raise ValueError("record_every must be 1 in stage five.")
        if not isinstance(noise_std, (int, float)) or isinstance(noise_std, bool) or not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be a finite nonnegative number.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer.")

        self.low_resolution = low_resolution.detach().clone()
        self.initialization = initialization.detach().clone()
        self.scale = scale
        self.sigma = float(sigma)
        self.iterations = iterations
        self.dt = float(dt)
        self.lambda_value = float(lambda_value)
        self.epsilon = float(epsilon)
        self.edge_weight_mode = edge_weight_mode
        self.integrator = integrator
        self.record_every = record_every
        self.noise_std = float(noise_std)
        self.seed = seed

        self.graph = self._build_graph()

    def _build_graph(self) -> PixelGraph | PixelGraphBatch:
        if self.initialization.ndim == 2:
            return build_pixel_graph(tuple(self.initialization.shape), self.edge_weight_mode)
        return build_batch_pixel_graphs(self.initialization, self.edge_weight_mode)

    def _data_step(self, image: torch.Tensor) -> torch.Tensor:
        gradient = data_fidelity_gradient(
            image,
            self.low_resolution,
            scale=self.scale,
            sigma=self.sigma,
        )
        updated = image - self.dt * gradient
        if self.integrator == "euler_maruyama" and self.noise_std > 0:
            generator = torch.Generator(device=image.device).manual_seed(self.seed)
            noise = torch.randn(
                image.shape,
                generator=generator,
                dtype=image.dtype,
                device=image.device,
            )
            updated = updated + self.noise_std * math.sqrt(self.dt) * noise
        return project_image(updated)

    def _regularization_step(self, image: torch.Tensor, step_fraction: float) -> torch.Tensor:
        if self.lambda_value == 0:
            return project_image(image)
        gradient = regularization_gradient(
            image,
            self.graph,
            lambda_value=self.lambda_value,
            epsilon=self.epsilon,
        )
        return project_image(image - step_fraction * self.dt * gradient)

    def step(self, image: torch.Tensor) -> torch.Tensor:
        """Execute one full solver step and return the projected image."""

        _validate_solver_tensor(image, "image", 2 if image.ndim == 2 else 3)
        projected = project_image(image)

        if self.integrator == "strang":
            half = self._regularization_step(projected, 0.5)
            full = self._data_step(half)
            return self._regularization_step(full, 0.5)
        return self._data_step(projected)

    def _diagnostics(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        rho = distribution_from_image(image, epsilon=self.epsilon)
        fidelity = data_fidelity(
            image,
            self.low_resolution,
            scale=self.scale,
            sigma=self.sigma,
        )
        entropy_value = entropy(rho)
        if self.lambda_value == 0:
            regularization = torch.zeros_like(entropy_value)
        else:
            regularization = entropy_energy(rho, self.graph, lambda_value=self.lambda_value)
        return {
            "data_fidelity": fidelity,
            "entropy": entropy_value,
            "regularization_energy": regularization,
            "total_energy": fidelity + regularization,
            "curvature": curvature_quotient(rho, self.graph, lambda_value=self.lambda_value),
            "min_probability": min_probability(rho),
            "max_probability": max_probability(rho),
            "effective_state_count": effective_state_count(rho),
        }

    def solve(self, x0: torch.Tensor | None = None) -> SolverResult:
        """Run the frozen solver and return the full trajectory and diagnostics."""

        if x0 is None:
            current = project_image(self.initialization)
        else:
            _validate_solver_tensor(x0, "x0", 2 if x0.ndim == 2 else 3)
            if x0.shape != self.initialization.shape:
                raise ValueError("x0 shape must match initialization shape.")
            current = project_image(x0)

        batched = current.ndim == 3
        batch_size = current.shape[0] if batched else None
        sample_count = batch_size if batched else 1
        record_shape = (self.iterations + 1,) + tuple(current.shape)

        trajectory = torch.empty(record_shape, dtype=current.dtype, device=current.device)
        trajectory[0] = current

        diagnostics: dict[str, list[torch.Tensor]] = {
            "data_fidelity": [],
            "entropy": [],
            "regularization_energy": [],
            "total_energy": [],
            "curvature": [],
            "min_probability": [],
            "max_probability": [],
            "effective_state_count": [],
            "update_norm": [],
        }

        first_diagnostics = self._diagnostics(current)
        for key, value in first_diagnostics.items():
            diagnostics[key].append(value)
        diagnostics["update_norm"].append(
            torch.zeros((), dtype=current.dtype, device=current.device)
            if not batched
            else torch.zeros(batch_size, dtype=current.dtype, device=current.device)
        )

        previous = current
        for k in range(1, self.iterations + 1):
            current = self.step(current)
            trajectory[k] = current
            step_diagnostics = self._diagnostics(current)
            for key, value in step_diagnostics.items():
                diagnostics[key].append(value)
            update = (current - previous).square().sum(dim=_SPATIAL_DIMS).sqrt()
            diagnostics["update_norm"].append(update)
            previous = current

        def stack(key: str) -> torch.Tensor:
            return torch.stack(diagnostics[key], dim=0)

        update_norm = stack("update_norm")
        result_kwargs = {
            "reconstruction": current,
            "initial_image": trajectory[0],
            "low_resolution": self.low_resolution,
            "trajectory": trajectory,
            "times": torch.arange(self.iterations + 1, dtype=current.dtype, device=current.device) * self.dt,
            "data_fidelity": stack("data_fidelity"),
            "entropy": stack("entropy"),
            "regularization_energy": stack("regularization_energy"),
            "total_energy": stack("total_energy"),
            "curvature": stack("curvature"),
            "min_probability": stack("min_probability"),
            "max_probability": stack("max_probability"),
            "effective_state_count": stack("effective_state_count"),
            "update_norm": update_norm,
            "final_update_norm": update_norm[-1],
            "iterations": self.iterations,
            "dt": self.dt,
            "lambda_value": self.lambda_value,
            "epsilon": self.epsilon,
            "integrator": self.integrator,
        }
        return SolverResult(**result_kwargs)


__all__ = [
    "project_image",
    "data_fidelity",
    "data_fidelity_gradient",
    "regularization_gradient",
    "step",
    "SuperResolutionSolver",
    "SolverResult",
    "degradation_forward",
    "degradation_adjoint",
]
