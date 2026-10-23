import math

import pytest
import torch

from gwf_sr.energy import dirichlet_energy, entropy_energy, log_ratio_smoothness, total_variation
from gwf_sr.geometry import distribution_from_image, entropy
from gwf_sr.graph import build_batch_pixel_graphs, build_pixel_graph


def test_constant_field_has_zero_dirichlet_energy_and_total_variation():
    graph = build_pixel_graph((4, 4))
    values = torch.full((4, 4), 0.75)
    assert dirichlet_energy(values, graph).item() == pytest.approx(0.0, abs=1e-7)
    assert total_variation(values, graph).item() == pytest.approx(0.0, abs=1e-7)


def test_dirichlet_energy_matches_manual_two_node_formula():
    graph = build_pixel_graph((1, 2))
    values = torch.tensor([[0.0, 1.0]])
    energy = dirichlet_energy(values, graph)
    assert energy.ndim == 0
    assert energy.item() == pytest.approx(1.0, abs=1e-6)


def test_dirichlet_energy_is_nonnegative_and_deterministic():
    graph = build_pixel_graph((4, 4))
    values = torch.rand(4, 4, generator=torch.Generator().manual_seed(2))
    first = dirichlet_energy(values, graph)
    second = dirichlet_energy(values, graph)
    assert first >= 0.0
    assert torch.equal(first, second)


def test_lambda_zero_reproduces_stage_three_entropy():
    graph = build_pixel_graph((4, 4))
    distribution = distribution_from_image(torch.rand(4, 4, generator=torch.Generator().manual_seed(9)), epsilon=1e-6)
    regularized = entropy_energy(distribution, graph, lambda_value=0.0)
    assert regularized.ndim == 0
    assert torch.abs(regularized - entropy(distribution)) <= 1e-6


def test_positive_lambda_increases_energy_for_nonuniform_distribution():
    graph = build_pixel_graph((4, 4))
    distribution = distribution_from_image(torch.rand(4, 4, generator=torch.Generator().manual_seed(6)), epsilon=1e-6)
    unregularized = entropy_energy(distribution, graph, lambda_value=0.0)
    regularized = entropy_energy(distribution, graph, lambda_value=1.0)
    assert regularized > unregularized


def test_energy_is_finite_for_positive_distributions():
    graph = build_pixel_graph((3, 3))
    distribution = distribution_from_image(torch.rand(3, 3, generator=torch.Generator().manual_seed(12)), epsilon=1e-6)
    value = entropy_energy(distribution, graph, lambda_value=1.0)
    assert torch.isfinite(value)


@pytest.mark.parametrize("lambda_value", [-1.0, float("inf"), float("nan")])
def test_energy_rejects_invalid_lambda(lambda_value):
    graph = build_pixel_graph((2, 2))
    distribution = distribution_from_image(torch.rand(2, 2), epsilon=1e-6)
    with pytest.raises((ValueError, TypeError)):
        entropy_energy(distribution, graph, lambda_value=lambda_value)


@pytest.mark.parametrize(
    "distribution",
    [
        torch.zeros(2, 2),
        -torch.ones(2, 2),
        torch.full((2, 2), float("nan")),
        torch.full((2, 2), float("inf")),
    ],
)
def test_energy_rejects_invalid_distributions(distribution):
    graph = build_pixel_graph((2, 2))
    with pytest.raises(ValueError):
        entropy_energy(distribution, graph, lambda_value=1.0)
    with pytest.raises(ValueError):
        log_ratio_smoothness(distribution, graph)


def test_energy_return_shapes_for_single_and_batch_inputs():
    single_graph = build_pixel_graph((3, 3))
    distribution = distribution_from_image(torch.rand(3, 3, generator=torch.Generator().manual_seed(8)), epsilon=1e-6)
    assert entropy_energy(distribution, single_graph, lambda_value=1.0).ndim == 0
    assert dirichlet_energy(distribution, single_graph).ndim == 0
    assert total_variation(distribution, single_graph).ndim == 0

    images = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(10))
    batch = build_batch_pixel_graphs(images)
    batch_distribution = distribution_from_image(images, epsilon=1e-6)
    assert entropy_energy(batch_distribution, batch, lambda_value=1.0).shape == (2,)
    assert dirichlet_energy(batch_distribution, batch).shape == (2,)
    assert total_variation(batch_distribution, batch).shape == (2,)


def test_batch_energy_matches_per_sample_energy():
    images = torch.rand(2, 3, 3, generator=torch.Generator().manual_seed(11))
    batch = build_batch_pixel_graphs(images)
    batch_distribution = distribution_from_image(images, epsilon=1e-6)
    batch_values = entropy_energy(batch_distribution, batch, lambda_value=0.7)

    for index in range(2):
        graph = build_pixel_graph((3, 3))
        expected = entropy_energy(batch_distribution[index], graph, lambda_value=0.7)
        assert torch.allclose(batch_values[index], expected, atol=1e-6)


def test_log_ratio_smoothness_preserves_shapes_and_finiteness():
    graph = build_pixel_graph((4, 4))
    distribution = distribution_from_image(torch.rand(4, 4, generator=torch.Generator().manual_seed(13)), epsilon=1e-6)
    single = log_ratio_smoothness(distribution, graph)
    assert single.ndim == 0
    assert torch.isfinite(single)

    images = torch.rand(2, 4, 4, generator=torch.Generator().manual_seed(14))
    batch = build_batch_pixel_graphs(images)
    batch_distribution = distribution_from_image(images, epsilon=1e-6)
    batch_values = log_ratio_smoothness(batch_distribution, batch)
    assert batch_values.shape == (2,)
    assert torch.all(torch.isfinite(batch_values))


def test_uniform_distribution_smoothness_is_zero():
    graph = build_pixel_graph((3, 3))
    distribution = torch.full((3, 3), 1.0 / 9.0, dtype=torch.float64)
    assert log_ratio_smoothness(distribution, graph).item() == pytest.approx(0.0, abs=1e-9)
    assert entropy_energy(distribution, graph, lambda_value=1.0).item() == pytest.approx(math.log(9), abs=1e-9)


def test_energy_preserves_dtype():
    graph = build_pixel_graph((3, 3))
    distribution = distribution_from_image(torch.rand(3, 3, dtype=torch.float64), epsilon=1e-6)
    assert entropy_energy(distribution, graph, lambda_value=1.0).dtype == torch.float64
    assert dirichlet_energy(distribution, graph).dtype == torch.float64
    assert total_variation(distribution, graph).dtype == torch.float64
