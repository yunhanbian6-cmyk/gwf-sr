import torch

from gwf_sr.degradation import bicubic_upsample, degrade, downsample2x, gaussian_blur, gaussian_kernel


def test_gaussian_kernel_is_normalized():
    kernel = gaussian_kernel(sigma=1.0)
    assert kernel.ndim == 1
    assert torch.isclose(kernel.sum(), torch.tensor(1.0))


def test_gaussian_blur_preserves_shape_and_range():
    image = torch.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    blurred = gaussian_blur(image, sigma=1.0)
    assert blurred.shape == image.shape
    assert torch.all(blurred >= 0.0)
    assert torch.all(blurred <= 1.0)


def test_downsample_and_bicubic_round_trip_shape():
    image = torch.rand((64, 64), generator=torch.Generator().manual_seed(7))
    low_resolution = downsample2x(image)
    bicubic = bicubic_upsample(low_resolution, scale=2)
    assert low_resolution.shape == (32, 32)
    assert bicubic.shape == image.shape


def test_degrade_is_reproducible_and_seeded():
    image = torch.rand((64, 64), generator=torch.Generator().manual_seed(11))
    first = degrade(image, scale=2, sigma=1.0, noise_std=0.05, seed=123)
    second = degrade(image, scale=2, sigma=1.0, noise_std=0.05, seed=123)
    third = degrade(image, scale=2, sigma=1.0, noise_std=0.05, seed=124)
    assert torch.equal(first.low_resolution, second.low_resolution)
    assert not torch.equal(first.low_resolution, third.low_resolution)
