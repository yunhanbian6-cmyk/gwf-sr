import torch

from gwf_sr.synthetic import make_synthetic_dataset, make_synthetic_image


def test_make_synthetic_image_is_deterministic():
    first = make_synthetic_image("piecewise", size=32, seed=17)
    second = make_synthetic_image("piecewise", size=32, seed=17)
    different = make_synthetic_image("piecewise", size=32, seed=18)
    assert first.shape == (32, 32)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_all_synthetic_kinds_are_valid_images():
    dataset = make_synthetic_dataset(size=32, seed=3)
    assert set(dataset) == {"piecewise", "edges", "texture", "smooth"}
    for image in dataset.values():
        assert image.shape == (32, 32)
        assert image.dtype == torch.float32
        assert torch.all(image >= 0.0)
        assert torch.all(image <= 1.0)


def test_unknown_synthetic_kind_is_rejected():
    try:
        make_synthetic_image("unknown", size=32)
    except ValueError as error:
        assert "Unknown synthetic image kind" in str(error)
    else:
        raise AssertionError("Expected an unknown image kind to be rejected.")
