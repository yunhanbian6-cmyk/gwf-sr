import math

import pytest
import torch

from gwf_sr.curvature import (
    curvature_acceleration,
    curvature_quotient,
    entropy_decay_rate,
    entropy_production,
)
from gwf_sr.geometry import distribution_from_image
from gwf_sr.graph import build_batch_pixel_graphs, build_pixel_graph


def test_kappa_zero_is_zero():
    graph = build_pixel_graph((3, 3))
    distribution = distribution_from_image(torch.rand(3, 3, generator=torch.Generator().manual_seed(21)), epsilon=1e-6)
    assert curvature_quotient(distribution, graph, lambda_value=0.0).item() == pytest.approx(0.0, abs=1e-12)


def test_kappa_zero_is_zero_for_batch_inputs():
    images = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(22))
    batch = build_batch_pixel_graphs(images)
    distribution = distribution_from_image(images, epsilon=1e-6)
    values = curvature_quotient(distribution, batch, lambda_value=0.0)
    assert values.shape == (2,)
    assert torch.allclose(values, torch.zeros(2, dtype=values.dtype))


def test_curvature_is_nonnegative_for_valid_inputs():
    graph = build_pixel_graph((4, 4))
    distribution = distribution_from_image(torch.rand(4, 4, generator=torch.Generator().manual_seed(23)), epsilon=1e-6)
    value = curvature_quotient(distribution, graph, lambda_value=1.0)
    assert value.ndim == 0
    assert value >= 0.0
    assert torch.isfinite(value)


def test_production_is_zero_for_constant_distribution():
    graph = build_pixel_graph((3, 3))
    distribution = torch.full((3, 3), 1.0 / 9.0, dtype=torch.float64)
    assert entropy_production(distribution, graph).item() == pytest.approx(0.0, abs=1e-12)


def test_production_is_zero_when_each_sample_is_constant():
    images = torch.stack([torch.full((2, 2), 0.5), torch.full((2, 2), 2.0)])
    batch = build_batch_pixel_graphs(images)
    distribution = distribution_from_image(images, epsilon=1e-6)
    production = entropy_production(distribution, batch)
    assert production.shape == (2,)
    assert torch.allclose(production, torch.zeros(2, dtype=production.dtype), atol=1e-6)


def test_production_is_positive_for_nonconstant_distribution():
    graph = build_pixel_graph((2, 2))
    distribution = distribution_from_image(torch.tensor([[0.1, 0.4], [0.8, 1.6]]), epsilon=1e-6)
    assert entropy_production(distribution, graph).item() > 0.0


def test_curvature_return_shapes_for_single_and_batch_inputs():
    graph = build_pixel_graph((3, 3))
    distribution = distribution_from_image(torch.rand(3, 3, generator=torch.Generator().manual_seed(24)), epsilon=1e-6)
    assert curvature_quotient(distribution, graph, lambda_value=1.0).ndim == 0

    images = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(25))
    batch = build_batch_pixel_graphs(images)
    batch_distribution = distribution_from_image(images, epsilon=1e-6)
    assert curvature_quotient(batch_distribution, batch, lambda_value=1.0).shape == (2,)


@pytest.mark.parametrize("lambda_value", [-1.0, float("inf")])
def test_curvature_rejects_invalid_lambda(lambda_value):
    graph = build_pixel_graph((2, 2))
    distribution = distribution_from_image(torch.rand(2, 2), epsilon=1e-6)
    with pytest.raises((ValueError, TypeError)):
        curvature_quotient(distribution, graph, lambda_value=lambda_value)


def test_curvature_acceleration_at_zero_is_two():
    for dtype in (torch.float32, torch.float64):
        value = curvature_acceleration(torch.tensor(0.0, dtype=dtype))
        tolerance = 1e-5 if dtype == torch.float32 else 1e-12
        assert torch.abs(value - 2.0) <= tolerance


def test_curvature_acceleration_is_strictly_greater_than_two_for_positive_ratios():
    ratios = torch.tensor([0.5, 1.0, 2.0, 10.0], dtype=torch.float64)
    values = curvature_acceleration(ratios)
    assert torch.all(values > 2.0)
    assert torch.all(torch.isfinite(values))


def test_curvature_acceleration_matches_closed_form_for_large_ratios():
    ratio = torch.tensor(2.0, dtype=torch.float64)
    expected = ratio.square() * torch.exp(ratio) / (torch.exp(ratio) - 1.0 - ratio)
    assert torch.abs(curvature_acceleration(ratio) - expected) <= 1e-12


def test_curvature_acceleration_rejects_negative_and_non_finite_input():
    with pytest.raises(ValueError):
        curvature_acceleration(torch.tensor([-0.1, 1.0]))
    with pytest.raises(ValueError):
        curvature_acceleration(torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        curvature_acceleration(torch.tensor([float("inf")]))
    with pytest.raises(TypeError):
        curvature_acceleration(torch.tensor([1.0], dtype=torch.int64))


def test_entropy_decay_rate_is_zero_at_lambda_zero():
    rate = entropy_decay_rate(0.0, initial_ratio=torch.tensor(1.0), time=torch.tensor(1.0))
    assert rate.item() == pytest.approx(0.0, abs=1e-12)


def test_entropy_decay_rate_matches_definition():
    ratio = torch.tensor(1.0, dtype=torch.float64)
    time = torch.tensor(1.0, dtype=torch.float64)
    lambda_value = 2.0
    decayed = ratio * torch.exp(-lambda_value * time)
    expected = lambda_value + lambda_value * curvature_acceleration(decayed)
    rate = entropy_decay_rate(lambda_value, initial_ratio=ratio, time=time)
    assert torch.abs(rate - expected) <= 1e-12


def test_entropy_decay_rate_accepts_python_scalars_and_broadcasts():
    rate = entropy_decay_rate(1.0, initial_ratio=1.0, time=torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert rate.shape == (2,)
    assert torch.all(torch.isfinite(rate))


def test_entropy_decay_rate_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        entropy_decay_rate(-1.0, initial_ratio=1.0, time=1.0)
    with pytest.raises(ValueError):
        entropy_decay_rate(1.0, initial_ratio=-1.0, time=1.0)
    with pytest.raises(ValueError):
        entropy_decay_rate(1.0, initial_ratio=1.0, time=-1.0)
    with pytest.raises(ValueError):
        entropy_decay_rate(1.0, initial_ratio=float("inf"), time=1.0)


def test_curvature_produces_no_nan_or_inf_for_valid_inputs():
    graph = build_pixel_graph((4, 4))
    distribution = distribution_from_image(torch.rand(4, 4, generator=torch.Generator().manual_seed(26)), epsilon=1e-6)
    for lambda_value in (0.0, 0.1, 1.0, 5.0):
        value = curvature_quotient(distribution, graph, lambda_value=lambda_value)
        assert torch.isfinite(value)
    assert torch.all(torch.isfinite(entropy_production(distribution, graph)))
    assert torch.all(torch.isfinite(curvature_acceleration(torch.linspace(0.0, 5.0, 16, dtype=torch.float64))))


def test_metrics_preserve_dtype():
    graph = build_pixel_graph((3, 3))
    distribution = distribution_from_image(torch.rand(3, 3, dtype=torch.float64), epsilon=1e-6)
    assert entropy_production(distribution, graph).dtype == torch.float64
    assert curvature_quotient(distribution, graph, lambda_value=1.0).dtype == torch.float64


def test_uniform_distribution_curvature_is_zero_for_any_lambda():
    graph = build_pixel_graph((3, 3))
    distribution = torch.full((3, 3), 1.0 / 9.0, dtype=torch.float64)
    for lambda_value in (0.0, 1.0, 10.0):
        assert curvature_quotient(distribution, graph, lambda_value=lambda_value).item() == pytest.approx(0.0, abs=1e-10)
    assert math.isfinite(entropy_production(distribution, graph).item())
