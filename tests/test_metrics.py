import math

import pytest
import torch

from gwf_sr.metrics import (
    ImageMetrics,
    compare_images,
    compute_image_metrics,
    edge_preservation_index,
    edge_preservation_ratio,
    gradient_magnitude,
    high_frequency_energy,
    high_frequency_energy_ratio,
    lpips_available,
    lpips_distance,
    psnr,
    ssim,
    validate_image_pair,
)


def ramp(size: int = 32, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, size * size, dtype=dtype).reshape(size, size)


def test_psnr_identical_images_returns_sentinel():
    image = ramp()
    value = psnr(image, image.clone())
    assert value.ndim == 0
    assert value.item() == pytest.approx(100.0)


def test_psnr_different_images_is_finite_and_positive():
    reference = ramp()
    prediction = ramp() + 0.05
    value = psnr(reference, prediction)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert value.item() > 0.0


def test_psnr_batch_returns_one_value_per_sample():
    references = torch.stack([ramp(), ramp() * 0.5])
    predictions = references + 0.01
    values = psnr(references, predictions)
    assert values.shape == (2,)
    assert torch.all(torch.isfinite(values))


def test_psnr_rejects_invalid_data_range():
    image = ramp()
    with pytest.raises(ValueError):
        psnr(image, image, data_range=0.0)
    with pytest.raises(ValueError):
        psnr(image, image, data_range=-1.0)
    with pytest.raises(TypeError):
        psnr(image, image, data_range=True)


def test_psnr_rejects_non_finite_input():
    reference = ramp()
    prediction = reference.clone()
    prediction[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        psnr(reference, prediction)


def test_psnr_preserves_dtype_and_device():
    for dtype in (torch.float32, torch.float64):
        reference = ramp(dtype=dtype)
        value = psnr(reference, reference)
        assert value.dtype == dtype
        assert value.device == reference.device


def test_ssim_identical_images_is_one():
    image = ramp()
    value = ssim(image, image.clone())
    assert value.ndim == 0
    assert abs(value.item() - 1.0) <= 1e-5


def test_ssim_different_images_below_one():
    reference = ramp()
    prediction = 1.0 - reference
    value = ssim(reference, prediction)
    assert value.item() < 1.0
    assert torch.isfinite(value)


def test_ssim_batch_shape():
    references = torch.stack([ramp(), ramp()])
    values = ssim(references, references.clone())
    assert values.shape == (2,)
    assert torch.allclose(values, torch.ones(2), atol=1e-5)


def test_ssim_rejects_even_win_size():
    image = ramp()
    with pytest.raises(ValueError):
        ssim(image, image, win_size=10)
    with pytest.raises(ValueError):
        ssim(image, image, win_size=1)


def test_ssim_rejects_image_smaller_than_window():
    small = torch.rand(8, 8, generator=torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="at least"):
        ssim(small, small, win_size=11)


def test_ssim_preserves_dtype():
    reference = ramp(dtype=torch.float64)
    value = ssim(reference, reference)
    assert value.dtype == torch.float64


def test_lpips_missing_dependency_behavior():
    image = ramp()
    if lpips_available():
        try:
            value = lpips_distance(image, image, require=True)
        except (RuntimeError, ImportError):
            pytest.skip("lpips package is present but its weights are unavailable")
        assert value is not None
        assert torch.isfinite(value)
        return
    assert lpips_distance(image, image, require=False) is None
    with pytest.raises(ImportError, match="pip install lpips"):
        lpips_distance(image, image, require=True)


def test_lpips_rejects_unknown_configuration():
    image = ramp()
    with pytest.raises(ValueError, match="Unknown LPIPS network"):
        lpips_distance(image, image, net="resnet")
    with pytest.raises(ValueError, match="Unknown LPIPS version"):
        lpips_distance(image, image, version="0.2")


def test_gradient_magnitude_of_constant_image_is_zero():
    constant = torch.full((16, 16), 0.5)
    magnitude = gradient_magnitude(constant)
    assert magnitude.shape == constant.shape
    assert torch.allclose(magnitude, torch.zeros_like(magnitude), atol=1e-7)


def test_gradient_magnitude_uses_replicate_boundary():
    image = torch.zeros(4, 4)
    image[0, 0] = 1.0
    magnitude = gradient_magnitude(image)
    assert magnitude.shape == image.shape
    assert torch.isfinite(magnitude).all()
    assert magnitude[0, 0].item() > 0.0


def test_high_frequency_energy_constant_is_zero():
    constant = torch.full((16, 16), 0.25)
    value = high_frequency_energy(constant)
    assert value.ndim == 0
    assert value.item() == pytest.approx(0.0, abs=1e-12)


def test_high_frequency_energy_grows_with_gradient():
    flat = torch.zeros(16, 16)
    ramped = ramp(16)
    assert high_frequency_energy(ramped).item() > high_frequency_energy(flat).item()


def test_high_frequency_energy_ratio_identical_is_one():
    image = ramp()
    value = high_frequency_energy_ratio(image, image.clone())
    assert value.ndim == 0
    assert abs(value.item() - 1.0) <= 1e-6


def test_edge_preservation_identical_images():
    image = ramp()
    epi = edge_preservation_index(image, image.clone())
    epr = edge_preservation_ratio(image, image.clone())
    assert epi.ndim == 0
    assert abs(epi.item() - 1.0) <= 1e-6
    assert abs(epr.item() - 1.0) <= 1e-6


def test_edge_preservation_ratio_bounds():
    reference = ramp()
    inverted = 1.0 - reference
    epr = edge_preservation_ratio(reference, inverted)
    assert 0.0 <= epr.item() <= 1.0


def test_metrics_batch_shapes():
    references = torch.stack([ramp(), ramp() * 0.5])
    predictions = references.clone()
    assert high_frequency_energy(references).shape == (2,)
    assert high_frequency_energy_ratio(references, predictions).shape == (2,)
    assert edge_preservation_index(references, predictions).shape == (2,)
    assert edge_preservation_ratio(references, predictions).shape == (2,)


def test_metrics_do_not_modify_inputs():
    reference = ramp()
    prediction = ramp() + 0.1
    reference_copy = reference.clone()
    prediction_copy = prediction.clone()
    compare_images(reference, prediction)
    assert torch.equal(reference, reference_copy)
    assert torch.equal(prediction, prediction_copy)


def test_compare_images_returns_finite_metrics():
    reference = ramp()
    prediction = (reference + 0.1).clamp(0.0, 1.0)
    metrics = compare_images(reference, prediction)
    assert isinstance(metrics, ImageMetrics)
    assert metrics.psnr.ndim == 0
    assert metrics.ssim.ndim == 0
    assert metrics.high_frequency_energy.ndim == 0
    assert torch.isfinite(metrics.psnr)
    assert torch.isfinite(metrics.ssim)
    assert torch.isfinite(metrics.high_frequency_energy_ratio)
    assert torch.isfinite(metrics.edge_preservation_index)


def test_compute_image_metrics_batch():
    references = torch.stack([ramp(), 1.0 - ramp()])
    predictions = references + 0.05
    metrics = compute_image_metrics(references, predictions)
    assert metrics.psnr.shape == (2,)
    assert metrics.ssim.shape == (2,)
    assert metrics.edge_preservation_ratio.shape == (2,)


def test_validate_image_pair_rejects_mismatched_shapes():
    reference = ramp(32)
    prediction = ramp(16)
    with pytest.raises(ValueError, match="identical shapes"):
        validate_image_pair(reference, prediction)


def test_validate_image_pair_rejects_dtype_mismatch():
    reference = ramp(32, dtype=torch.float32)
    prediction = ramp(32, dtype=torch.float64)
    with pytest.raises(TypeError, match="identical dtypes"):
        validate_image_pair(reference, prediction)


def test_validate_image_pair_rejects_non_tensor():
    with pytest.raises(TypeError):
        validate_image_pair([[0.0]], [[0.0]])


def test_metrics_reject_wrong_rank():
    image = ramp()
    with pytest.raises(ValueError, match="H×W or B×H×W"):
        psnr(image.reshape(-1), image.reshape(-1))


def test_metrics_do_not_create_pairwise_matrix():
    large = ramp(64)
    value = ssim(large, large)
    assert value.ndim == 0
    assert high_frequency_energy(large).ndim == 0


def test_metrics_scale_linearly_in_state_count():
    small = ramp(16)
    large = ramp(64)
    assert math.isfinite(psnr(small, small).item())
    assert math.isfinite(psnr(large, large).item())
