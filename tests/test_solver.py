import math

import pytest
import torch

from gwf_sr.solver import (
    SolverResult,
    SuperResolutionSolver,
    data_fidelity,
    data_fidelity_gradient,
    degradation_adjoint,
    degradation_forward,
    project_image,
    regularization_gradient,
)
from gwf_sr.graph import build_pixel_graph


def test_project_image_clamps_to_unit_interval():
    image = torch.tensor([[-0.5, 0.25], [1.5, 0.75]])
    projected = project_image(image)
    assert torch.equal(projected, torch.tensor([[0.0, 0.25], [1.0, 0.75]]))


def test_project_image_preserves_shape_dtype_device():
    for dtype in (torch.float32, torch.float64):
        image = torch.rand(4, 4, dtype=dtype)
        projected = project_image(image)
        assert projected.shape == image.shape
        assert projected.dtype == dtype
        assert projected.device == image.device


def test_degradation_forward_preserves_shape():
    image = torch.rand(8, 8)
    predicted = degradation_forward(image, scale=2, sigma=1.0)
    assert predicted.shape == (4, 4)


def test_degradation_adjoint_matches_inner_product():
    image = torch.rand(8, 8, dtype=torch.float64)
    z = torch.rand(4, 4, dtype=torch.float64)
    lhs = (degradation_forward(image, scale=2, sigma=1.0) * z).sum()
    rhs = (image * degradation_adjoint(z, image.shape, scale=2, sigma=1.0)).sum()
    assert torch.abs(lhs - rhs) <= 1e-12


def test_degradation_forward_and_adjoint_do_not_modify_inputs():
    image = torch.rand(8, 8)
    z = torch.rand(4, 4)
    image_copy = image.clone()
    z_copy = z.clone()
    degradation_forward(image, scale=2, sigma=1.0)
    degradation_adjoint(z, image.shape, scale=2, sigma=1.0)
    assert torch.equal(image, image_copy)
    assert torch.equal(z, z_copy)


def test_data_fidelity_returns_expected_shape():
    image = torch.rand(8, 8)
    y = torch.rand(4, 4)
    value = data_fidelity(image, y, scale=2, sigma=1.0)
    assert value.ndim == 0


def test_data_fidelity_gradient_shape_dtype_device():
    image = torch.rand(8, 8, dtype=torch.float64)
    y = torch.rand(4, 4, dtype=torch.float64)
    gradient = data_fidelity_gradient(image, y, scale=2, sigma=1.0)
    assert gradient.shape == image.shape
    assert gradient.dtype == torch.float64
    assert gradient.device == image.device
    assert torch.all(torch.isfinite(gradient))


def test_data_fidelity_gradient_matches_finite_difference():
    image = torch.rand(8, 8, dtype=torch.float64)
    y = torch.rand(4, 4, dtype=torch.float64)
    gradient = data_fidelity_gradient(image, y, scale=2, sigma=1.0)
    epsilon = 1e-6
    row, col = 3, 5
    perturbed = image.clone()
    perturbed[row, col] += epsilon
    fd = (data_fidelity(perturbed, y, scale=2, sigma=1.0) - data_fidelity(image, y, scale=2, sigma=1.0)) / epsilon
    assert torch.abs(gradient[row, col] - fd) <= 1e-4


def test_data_fidelity_zero_when_observation_matches_prediction():
    image = torch.rand(8, 8)
    y = degradation_forward(image, scale=2, sigma=1.0)
    value = data_fidelity(image, y, scale=2, sigma=1.0)
    assert value.item() == pytest.approx(0.0, abs=1e-7)


def test_regularization_gradient_zero_at_lambda_zero():
    image = torch.rand(4, 4)
    graph = build_pixel_graph((4, 4))
    gradient = regularization_gradient(image, graph, lambda_value=0.0, epsilon=1e-6)
    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-8)


def test_regularization_gradient_shape_dtype_device():
    image = torch.rand(4, 4, dtype=torch.float64)
    graph = build_pixel_graph((4, 4))
    gradient = regularization_gradient(image, graph, lambda_value=1.0, epsilon=1e-6)
    assert gradient.shape == image.shape
    assert gradient.dtype == torch.float64
    assert gradient.device == image.device
    assert torch.all(torch.isfinite(gradient))


def test_solver_rejects_invalid_inputs():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, scale=3)
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, dt=0.0)
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, iterations=0)
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, lambda_value=-1.0)
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, integrator="rk4")
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, record_every=2)
    with pytest.raises(TypeError):
        SuperResolutionSolver(y.to(torch.int64), x0)


def test_solver_rejects_shape_mismatch():
    y = torch.rand(4, 4)
    x0 = torch.rand(7, 7)
    with pytest.raises(ValueError):
        SuperResolutionSolver(y, x0, scale=2)


def test_solver_projects_negative_initialization():
    y = torch.rand(4, 4)
    x0 = -torch.rand(8, 8)
    solver = SuperResolutionSolver(y, x0, iterations=2, dt=0.01)
    result = solver.solve()
    assert result.initial_image.min() >= 0.0
    assert result.initial_image.max() <= 1.0


def test_solver_trajectory_shape_and_bounds():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    iterations = 5
    solver = SuperResolutionSolver(y, x0, iterations=iterations, dt=0.01)
    result = solver.solve()
    assert result.trajectory.shape == (iterations + 1, 8, 8)
    assert result.trajectory[0].min() >= 0.0
    assert result.trajectory[0].max() <= 1.0
    assert torch.equal(result.trajectory[-1], result.reconstruction)


def test_solver_times_are_k_times_dt():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    iterations = 4
    dt = 0.05
    solver = SuperResolutionSolver(y, x0, iterations=iterations, dt=dt)
    result = solver.solve()
    expected = torch.arange(iterations + 1, dtype=result.times.dtype) * dt
    assert torch.allclose(result.times, expected, atol=1e-6)


def test_solver_total_energy_equals_data_plus_regularization():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    solver = SuperResolutionSolver(y, x0, iterations=3, dt=0.01, lambda_value=1.0, epsilon=1e-6)
    result = solver.solve()
    assert torch.allclose(
        result.total_energy,
        result.data_fidelity + result.regularization_energy,
        atol=1e-6,
    )


def test_solver_is_deterministic_for_same_input():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    first = SuperResolutionSolver(y, x0, iterations=3, dt=0.01).solve()
    second = SuperResolutionSolver(y, x0, iterations=3, dt=0.01).solve()
    assert torch.equal(first.reconstruction, second.reconstruction)
    assert torch.equal(first.trajectory, second.trajectory)


def test_strang_lambda_zero_matches_euler():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    euler = SuperResolutionSolver(y, x0, iterations=3, dt=0.01, integrator="euler").solve()
    strang = SuperResolutionSolver(y, x0, iterations=3, dt=0.01, integrator="strang").solve()
    assert torch.allclose(euler.reconstruction, strang.reconstruction, atol=1e-5)


def test_euler_maruyama_zero_noise_matches_euler():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    euler = SuperResolutionSolver(y, x0, iterations=3, dt=0.01, integrator="euler").solve()
    noisy = SuperResolutionSolver(
        y, x0, iterations=3, dt=0.01, integrator="euler_maruyama", noise_std=0.0, seed=7
    ).solve()
    assert torch.equal(euler.reconstruction, noisy.reconstruction)


def test_solver_diagnostic_shapes_for_single_image():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    iterations = 3
    result = SuperResolutionSolver(y, x0, iterations=iterations, dt=0.01).solve()
    assert result.data_fidelity.shape == (iterations + 1,)
    assert result.entropy.shape == (iterations + 1,)
    assert result.regularization_energy.shape == (iterations + 1,)
    assert result.total_energy.shape == (iterations + 1,)
    assert result.curvature.shape == (iterations + 1,)
    assert result.min_probability.shape == (iterations + 1,)
    assert result.max_probability.shape == (iterations + 1,)
    assert result.effective_state_count.shape == (iterations + 1,)
    assert result.update_norm.shape == (iterations + 1,)
    assert result.update_norm[0].item() == pytest.approx(0.0, abs=1e-8)


def test_solver_diagnostic_shapes_for_batch():
    y = torch.rand(2, 4, 4)
    x0 = torch.rand(2, 8, 8)
    iterations = 3
    result = SuperResolutionSolver(y, x0, iterations=iterations, dt=0.01).solve()
    assert result.trajectory.shape == (iterations + 1, 2, 8, 8)
    assert result.data_fidelity.shape == (iterations + 1, 2)
    assert result.entropy.shape == (iterations + 1, 2)
    assert result.curvature.shape == (iterations + 1, 2)
    assert result.update_norm.shape == (iterations + 1, 2)


def test_solver_diagnostics_are_finite():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    result = SuperResolutionSolver(y, x0, iterations=3, dt=0.01, lambda_value=1.0, epsilon=1e-6).solve()
    assert torch.all(torch.isfinite(result.trajectory))
    assert torch.all(torch.isfinite(result.data_fidelity))
    assert torch.all(torch.isfinite(result.entropy))
    assert torch.all(torch.isfinite(result.regularization_energy))
    assert torch.all(torch.isfinite(result.total_energy))
    assert torch.all(torch.isfinite(result.curvature))
    assert torch.all(torch.isfinite(result.update_norm))


def test_solver_trajectory_stays_in_unit_interval():
    y = torch.rand(4, 4)
    x0 = torch.rand(8, 8)
    result = SuperResolutionSolver(y, x0, iterations=5, dt=0.01).solve()
    assert result.reconstruction.min() >= 0.0
    assert result.reconstruction.max() <= 1.0
    assert result.trajectory.min() >= 0.0
    assert result.trajectory.max() <= 1.0


def test_solver_preserves_dtype():
    for dtype in (torch.float32, torch.float64):
        y = torch.rand(4, 4, dtype=dtype)
        x0 = torch.rand(8, 8, dtype=dtype)
        result = SuperResolutionSolver(y, x0, iterations=2, dt=0.01).solve()
        assert result.reconstruction.dtype == dtype
        assert result.trajectory.dtype == dtype
        assert result.times.dtype == dtype
        assert result.data_fidelity.dtype == dtype


def test_solver_does_not_create_dense_n_times_n_matrix():
    y = torch.rand(16, 16)
    x0 = torch.rand(32, 32)
    result = SuperResolutionSolver(y, x0, iterations=2, dt=0.01).solve()
    assert result.trajectory.shape == (3, 32, 32)
    assert not hasattr(result, "adjacency")
