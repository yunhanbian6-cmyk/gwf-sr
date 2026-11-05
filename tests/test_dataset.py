# -*- coding: utf-8 -*-
"""Tests for real image loading and dataset support."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gwf_sr.dataset import (
    degrade_and_prepare,
    load_image_directory,
    load_image_tensor,
    load_local_images,
    load_standard_dataset,
    save_image_tensor,
)


def _make_test_png(path: Path, size: int = 32, grayscale: bool = True) -> Path:
    if grayscale:
        array = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        Image.fromarray(array, mode="L").save(path)
    else:
        array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(path)
    return path


class TestLoadImageTensor:
    def test_load_grayscale(self, tmp_path: Path):
        p = _make_test_png(tmp_path / "test.png", size=32, grayscale=True)
        tensor = load_image_tensor(p, grayscale=True)
        assert tensor.shape == (32, 32)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_load_rgb(self, tmp_path: Path):
        p = _make_test_png(tmp_path / "test.png", size=32, grayscale=False)
        tensor = load_image_tensor(p, grayscale=False)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32

    def test_load_with_max_size(self, tmp_path: Path):
        p = _make_test_png(tmp_path / "test.png", size=64, grayscale=True)
        tensor = load_image_tensor(p, grayscale=True, max_size=32)
        assert max(tensor.shape) == 32

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_image_tensor("nonexistent.png")

    def test_unsupported_format(self, tmp_path: Path):
        p = tmp_path / "test.xyz"
        p.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported"):
            load_image_tensor(p)


class TestSaveImageTensor:
    def test_save_grayscale(self, tmp_path: Path):
        tensor = torch.rand(32, 32)
        p = save_image_tensor(tensor, tmp_path / "out.png")
        assert p.exists()
        loaded = load_image_tensor(p, grayscale=True)
        assert loaded.shape == (32, 32)

    def test_save_rgb(self, tmp_path: Path):
        tensor = torch.rand(3, 32, 32)
        p = save_image_tensor(tensor, tmp_path / "out.png")
        assert p.exists()
        loaded = load_image_tensor(p, grayscale=False)
        assert loaded.shape == (3, 32, 32)

    def test_rejects_bad_shape(self, tmp_path: Path):
        with pytest.raises(ValueError):
            save_image_tensor(torch.rand(2, 32, 32), tmp_path / "bad.png")


class TestLoadImageDirectory:
    def test_load_directory(self, tmp_path: Path):
        for i in range(3):
            _make_test_png(tmp_path / f"img_{i}.png", size=32)
        images = load_image_directory(tmp_path, grayscale=True)
        assert len(images) == 3
        assert all(t.shape == (32, 32) for _, t in images)

    def test_empty_directory(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No supported images"):
            load_image_directory(tmp_path)

    def test_not_a_directory(self):
        with pytest.raises(NotADirectoryError):
            load_image_directory("nonexistent_dir")


class TestStandardDataset:
    def test_unknown_dataset(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_standard_dataset("Unknown", tmp_path)

    def test_missing_directory(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_standard_dataset("Set5", tmp_path)


class TestLoadLocalImages:
    def test_load_multiple(self, tmp_path: Path):
        paths = []
        for i in range(3):
            p = _make_test_png(tmp_path / f"img_{i}.png", size=32)
            paths.append(p)
        images = load_local_images(paths, grayscale=True)
        assert len(images) == 3


class TestDegradeAndPrepare:
    def test_grayscale(self):
        hr = torch.rand(32, 32)
        lr, bicubic = degrade_and_prepare(hr, scale=2, sigma=1.0)
        assert lr.shape == (16, 16)
        assert bicubic.shape == (32, 32)

    def test_rgb(self):
        hr = torch.rand(3, 32, 32)
        lr, bicubic = degrade_and_prepare(hr, scale=2, sigma=1.0)
        assert lr.shape == (3, 16, 16)
        assert bicubic.shape == (3, 32, 32)

    def test_reproducible(self):
        hr = torch.rand(32, 32, generator=torch.Generator().manual_seed(42))
        lr1, _ = degrade_and_prepare(hr, scale=2, sigma=1.0, seed=42)
        lr2, _ = degrade_and_prepare(hr, scale=2, sigma=1.0, seed=42)
        assert torch.equal(lr1, lr2)
