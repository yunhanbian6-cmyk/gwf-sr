from pathlib import Path

import torch

from gwf_sr.config import load_config
from gwf_sr.data import generate_configured_dataset, generate_image, load_image_tensor, save_image_tensor


def test_save_and_load_image_tensor_round_trip(tmp_path: Path):
    image = torch.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
    path = save_image_tensor(image, tmp_path / "image.png")
    loaded = load_image_tensor(path)
    assert path.is_file()
    assert loaded.shape == image.shape
    assert torch.max(torch.abs(loaded - image)) < 0.01


def test_generate_image_without_output_dir():
    hr, low_resolution, bicubic, record = generate_image(kind="texture", size=32, seed=5)
    assert record is None
    assert hr.shape == (32, 32)
    assert low_resolution.shape == (16, 16)
    assert bicubic.shape == (32, 32)


def test_generate_configured_dataset(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    records = generate_configured_dataset(config, kinds=("piecewise", "edges"), root=tmp_path)
    assert len(records) == 2
    manifest = tmp_path / "results" / "smoke" / "data" / "manifest.json"
    assert manifest.is_file()
    assert (tmp_path / "results" / "smoke" / "data" / f"{records[0].name}_hr.png").is_file()
