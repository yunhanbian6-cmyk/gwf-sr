import math

import pytest
import torch

from gwf_sr.geometry import (
    distribution_from_image,
    effective_state_count,
    entropy,
    hilbert_distance,
    kl_divergence,
    max_probability,
    min_probability,
    total_mass,
)


def assert_scalar(value: torch.Tensor) -> None:
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0


@pytest.mark.parametrize(
    "function",
    [distribution_from_image, entropy, total_mass, min_probability, max_probability, effective_state_count],
)
def test_geometry_rejects_invalid_shape(function):
    with pytest.raises(ValueError, match="H×W or B×H×W"):
        function(torch.ones(8))


@pytest.mark.parametrize(
    "function",
    [distribution_from_image, entropy, total_mass, min_probability, max_probability, effective_state_count],
)
def test_geometry_rejects_non_tensor(function):
    with pytest.raises(TypeError, match="torch.Tensor"):
        function([[1.0]])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64])
def test_distribution_rejects_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match=r"torch\.float32 or torch\.float64"):
        distribution_from_image(torch.ones(2, 2, dtype=dtype))


@pytest.mark.parametrize(
    "function",
    [distribution_from_image, entropy, total_mass, min_probability, max_probability, effective_state_count],
)
def test_distribution_rejects_non_finite_input(function):
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="NaN or Inf"):
            function(torch.full((2, 2), value))


@pytest.mark.parametrize("epsilon", [0.0, -1.0])
def test_distribution_rejects_non_positive_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        distribution_from_image(torch.ones(2, 2), epsilon=epsilon)


def test_distribution_rejects_non_finite_epsilon():
    with pytest.raises(ValueError, match="finite"):
        distribution_from_image(torch.ones(2, 2), epsilon=float("inf"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distribution_preserves_shape_dtype_and_normalization(dtype):
    image = torch.linspace(-0.5, 1.5, 2 * 3, dtype=dtype).reshape(2, 3)
    distribution = distribution_from_image(image)

    assert distribution.shape == image.shape
    assert distribution.dtype == dtype
    assert distribution.device == image.device
    assert torch.all(distribution > 0.0)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    assert torch.abs(total_mass(distribution) - 1.0) <= tolerance


@pytest.mark.parametrize("shape", [(2, 3), (4, 2, 3)])
def test_distribution_is_deterministic(shape):
    image = torch.linspace(-0.5, 1.5, math.prod(shape), dtype=torch.float64).reshape(shape)
    first = distribution_from_image(image)
    second = distribution_from_image(image)
    assert torch.equal(first, second)


@pytest.mark.parametrize("dtype,expected", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_black_and_negative_images_produce_uniform_distribution(dtype, expected):
    size = 4
    for image in (torch.zeros(size, dtype=dtype), torch.full((size,), -2.0, dtype=dtype)):
        distribution = distribution_from_image(image.reshape(2, 2))

        assert torch.allclose(distribution, torch.full_like(distribution, 0.25), atol=expected, rtol=0.0)
        assert torch.abs(entropy(distribution) - math.log(4)) <= expected
        assert torch.abs(effective_state_count(distribution) - 4.0) <= 1e-4


def test_batch_samples_are_normalized_independently():
    images = torch.stack(
        [
            torch.zeros(2, 2),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.full((2, 2), -1.0),
        ]
    )
    distributions = distribution_from_image(images)

    assert distributions.shape == images.shape
    assert torch.allclose(total_mass(distributions), torch.ones(3), atol=1e-5, rtol=0.0)
    assert torch.allclose(distributions[0], distributions[2], atol=0.0, rtol=0.0)


def test_entropy_returns_expected_shapes_and_values():
    distribution = distribution_from_image(torch.ones(2, 2, dtype=torch.float64))
    assert_scalar(entropy(distribution))
    assert torch.abs(entropy(distribution) - math.log(4)) <= 1e-12

    batch = distribution_from_image(torch.ones(3, 2, 2, dtype=torch.float64))
    batch_entropy = entropy(batch)
    assert batch_entropy.shape == (3,)
    assert torch.allclose(batch_entropy, torch.full((3,), math.log(4), dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_entropy_never_exceeds_uniform_upper_bound():
    cases = (
        (torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32), 4),
        (torch.arange(16, dtype=torch.float32).reshape(4, 4), 16),
        (torch.randn(5, 5, generator=torch.Generator().manual_seed(7)), 25),
    )
    for image, state_count in cases:
        distribution = distribution_from_image(image)
        entropy_value = entropy(distribution)

        assert entropy_value.ndim == 0
        assert entropy_value <= math.log(state_count) + 1e-5
        assert effective_state_count(distribution) <= state_count + 1e-4


def test_kl_and_hilbert_identical_distributions_are_zero():
    distribution = distribution_from_image(
        torch.arange(12, dtype=torch.float64).reshape(3, 4),
        epsilon=1e-6,
    )
    assert abs(float(kl_divergence(distribution, distribution))) <= 1e-12
    assert abs(float(hilbert_distance(distribution, distribution))) <= 1e-12


def test_metrics_return_expected_shapes_for_batch_inputs():
    images = torch.rand(3, 2, 4, generator=torch.Generator().manual_seed(11))
    rho = distribution_from_image(images)
    eta = distribution_from_image(images + 0.125)

    assert kl_divergence(rho, eta).shape == (3,)
    assert hilbert_distance(rho, eta).shape == (3,)
    assert total_mass(rho).shape == (3,)
    assert min_probability(rho).shape == (3,)
    assert max_probability(rho).shape == (3,)
    assert effective_state_count(rho).shape == (3,)


def test_hilbert_distance_is_symmetric_and_nonnegative():
    rho = distribution_from_image(torch.tensor([[0.1, 0.4], [0.8, 1.6]]), epsilon=1e-6)
    eta = distribution_from_image(torch.tensor([[0.9, 0.2], [0.3, 0.7]]), epsilon=1e-6)

    forward = hilbert_distance(rho, eta)
    reverse = hilbert_distance(eta, rho)
    assert_scalar(forward)
    assert forward >= 0.0
    assert torch.abs(forward - reverse) <= 1e-12


def test_metrics_reject_mismatched_distribution_shapes():
    rho = distribution_from_image(torch.ones(2, 2))
    eta = distribution_from_image(torch.ones(2, 3))
    with pytest.raises(ValueError, match="identical shapes"):
        kl_divergence(rho, eta)
    with pytest.raises(ValueError, match="identical shapes"):
        hilbert_distance(rho, eta)


def test_metrics_reject_mismatched_dtypes():
    rho = distribution_from_image(torch.ones(2, 2, dtype=torch.float32))
    eta = distribution_from_image(torch.ones(2, 2, dtype=torch.float64))
    with pytest.raises(TypeError, match="identical dtypes"):
        kl_divergence(rho, eta)
    with pytest.raises(TypeError, match="identical dtypes"):
        hilbert_distance(rho, eta)


def test_diagnostics_match_frozen_definitions():
    distribution = distribution_from_image(
        torch.tensor([[0.2, 0.8], [1.2, 1.8]]),
        epsilon=1e-6,
    )

    assert torch.abs(total_mass(distribution) - 1.0) <= 1e-5
    assert min_probability(distribution) == torch.min(distribution)
    assert max_probability(distribution) == torch.max(distribution)
    assert torch.abs(effective_state_count(distribution) - torch.exp(entropy(distribution))) <= 1e-6
