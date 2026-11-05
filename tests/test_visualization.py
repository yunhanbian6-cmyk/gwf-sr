import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from gwf_sr.solver import SolverResult
from gwf_sr.visualization import (
    EvaluationBundle,
    ensure_output_directory,
    plot_convergence_curves,
    plot_method_comparison,
    save_evaluation_bundle,
    save_image_comparison,
    save_metrics_csv,
    save_metrics_json,
    tensor_to_uint8,
    validate_visualization_config,
)


def ramp(size: int = 32, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, size * size, dtype=dtype).reshape(size, size)


def make_images() -> dict[str, torch.Tensor]:
    hr = ramp(32)
    return {
        "LR": hr[::2, ::2].contiguous(),
        "Bicubic": hr,
        "Baseline": (hr * 0.9).clamp(0.0, 1.0),
        "GWF-C": (hr * 0.95).clamp(0.0, 1.0),
        "HR": hr,
    }


def make_solver_result(iterations: int = 4, batch: int | None = None) -> SolverResult:
    width = 1 if batch is None else batch
    shape = (iterations + 1,) if batch is None else (iterations + 1, batch)
    values = {
        name: torch.linspace(1.0, 0.5, (iterations + 1) * width, dtype=torch.float32).reshape(shape)
        for name in (
            "data_fidelity",
            "entropy",
            "regularization_energy",
            "total_energy",
            "curvature",
            "update_norm",
        )
    }
    times = torch.arange(iterations + 1, dtype=torch.float32) * 0.05
    return SolverResult(
        reconstruction=torch.rand(8, 8),
        initial_image=torch.rand(8, 8),
        low_resolution=torch.rand(4, 4),
        trajectory=torch.rand(iterations + 1, 8, 8),
        times=times,
        data_fidelity=values["data_fidelity"],
        entropy=values["entropy"],
        regularization_energy=values["regularization_energy"],
        total_energy=values["total_energy"],
        curvature=values["curvature"],
        min_probability=torch.full(shape, 0.01),
        max_probability=torch.full(shape, 0.5),
        effective_state_count=torch.full(shape, 4.0),
        update_norm=values["update_norm"],
        final_update_norm=values["update_norm"][-1],
        iterations=iterations,
        dt=0.05,
        lambda_value=1.0,
        epsilon=1e-8,
        integrator="euler",
    )


def test_tensor_to_uint8_projects_and_quantizes():
    image = torch.tensor([[-0.5, 0.0], [0.5, 1.5]])
    encoded = tensor_to_uint8(image)
    assert encoded.dtype == torch.uint8
    assert encoded.tolist() == [[0, 0], [128, 255]]


def test_tensor_to_uint8_rejects_non_finite():
    with pytest.raises(ValueError, match="NaN or Inf"):
        tensor_to_uint8(torch.full((2, 2), float("nan")))


def test_ensure_output_directory_creates_parents(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c.txt"
    created = ensure_output_directory(target.parent)
    assert created.is_dir()


def test_validate_visualization_config_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_visualization_config(dpi=0)
    with pytest.raises(TypeError):
        validate_visualization_config(dpi=True)
    with pytest.raises(ValueError):
        validate_visualization_config(batch_mode="median")
    with pytest.raises(ValueError):
        validate_visualization_config(figsize=(0.0, 4.0))


def test_save_metrics_json_round_trip(tmp_path: Path):
    path = save_metrics_json({"schema_version": "1.0", "config_name": "smoke", "psnr": 30.5}, tmp_path / "m.json")
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["schema_version"] == "1.0"
    assert parsed["config_name"] == "smoke"
    assert parsed["extra"]["psnr"] == 30.5
    assert text.endswith("\n")
    assert "NaN" not in text and "Infinity" not in text


def test_save_metrics_json_converts_tensors_and_paths(tmp_path: Path):
    path = save_metrics_json(
        {"value": torch.tensor(1.5), "curve": torch.tensor([1.0, 2.0]), "file": tmp_path / "x.png"},
        tmp_path / "n.json",
    )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["value"] == 1.5
    assert parsed["curve"] == [1.0, 2.0]
    assert parsed["file"] == "x.png"


def test_save_metrics_json_moves_unknown_keys_to_extra(tmp_path: Path):
    path = save_metrics_json(
        {"schema_version": "1.0", "custom_field": 7},
        tmp_path / "e.json",
    )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["extra"] == {"custom_field": 7}
    assert "custom_field" not in parsed


def test_save_metrics_json_supports_unicode(tmp_path: Path):
    path = save_metrics_json({"note": "超分"}, tmp_path / "u.json")
    text = path.read_text(encoding="utf-8")
    assert "超分" in text


def test_save_metrics_csv_header_and_rows(tmp_path: Path):
    path = save_metrics_csv(
        [
            {
                "sample_name": "s1",
                "method": "GWF-C",
                "lambda_value": 1.0,
                "psnr": 30.0,
                "ssim": 0.9,
                "lpips": None,
                "high_frequency_energy": 0.01,
                "high_frequency_energy_ratio": 1.0,
                "edge_preservation_index": 0.95,
                "edge_preservation_ratio": 0.97,
            }
        ],
        tmp_path / "metrics.csv",
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "sample_name,method,lambda_value,psnr,ssim,lpips,"
        "high_frequency_energy,high_frequency_energy_ratio,"
        "edge_preservation_index,edge_preservation_ratio"
    )
    assert len(lines) == 2
    assert lines[1].split(",")[5] == ""
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_save_trajectory_csv_matches_iteration_count(tmp_path: Path):
    from gwf_sr.visualization import save_trajectory_csv

    result = make_solver_result(iterations=4)
    path = save_trajectory_csv(result, tmp_path / "trajectory_metrics.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "time,iteration,data_fidelity,entropy,regularization_energy,"
        "total_energy,curvature,update_norm,lambda_value"
    )
    assert len(lines) == 6
    first = lines[1].split(",")
    assert first[1] == "0"
    assert float(first[0]) == pytest.approx(0.0)
    last = lines[-1].split(",")
    assert last[1] == "4"
    assert float(last[0]) == pytest.approx(0.2)


def test_save_trajectory_csv_aggregates_batch(tmp_path: Path):
    from gwf_sr.visualization import save_trajectory_csv

    result = make_solver_result(iterations=3, batch=2)
    path = save_trajectory_csv(result, tmp_path / "trajectory_metrics.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert all(len(line.split(",")) == 9 for line in lines)


def test_plot_convergence_curves_writes_png(tmp_path: Path):
    output = tmp_path / "convergence.png"
    result = make_solver_result(iterations=4)
    path = plot_convergence_curves({"GWF-C": result}, output)
    assert path.is_file()
    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_plot_convergence_curves_accepts_mapping(tmp_path: Path):
    output = tmp_path / "curves.png"
    payload = {
        "times": torch.arange(3, dtype=torch.float32),
        "data_fidelity": torch.tensor([3.0, 2.0, 1.0]),
        "entropy": torch.tensor([1.0, 1.0, 1.0]),
    }
    path = plot_convergence_curves({"Baseline": payload}, output)
    assert path.is_file()


def test_plot_convergence_curves_skips_missing_fields(tmp_path: Path):
    output = tmp_path / "partial.png"
    payload = {
        "times": torch.arange(3, dtype=torch.float32),
        "update_norm": torch.tensor([1.0, 0.5, 0.25]),
    }
    path = plot_convergence_curves({"GWF-C": payload}, output, fields=["curvature", "update_norm"])
    assert path.is_file()


def test_plot_convergence_curves_rejects_unknown_configuration(tmp_path: Path):
    result = make_solver_result(iterations=2)
    with pytest.raises(ValueError, match="batch_mode"):
        plot_convergence_curves({"GWF-C": result}, tmp_path / "x.png", batch_mode="median")
    with pytest.raises(ValueError, match="Unknown curve field"):
        plot_convergence_curves({"GWF-C": result}, tmp_path / "y.png", fields=["gradient"])
    with pytest.raises(ValueError, match="at least one method"):
        plot_convergence_curves({}, tmp_path / "z.png")


def test_plot_convergence_curves_closes_figures(tmp_path: Path):
    import matplotlib.pyplot as plt

    result = make_solver_result(iterations=2)
    plot_convergence_curves({"GWF-C": result}, tmp_path / "closed.png")
    assert plt.get_fignums() == []


def test_plot_method_comparison_writes_png(tmp_path: Path):
    output = tmp_path / "comparison.png"
    path = plot_method_comparison(make_images(), output)
    assert path.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
    assert plt_figs_closed()


def plt_figs_closed() -> bool:
    import matplotlib.pyplot as plt

    return plt.get_fignums() == []


def test_save_image_comparison_matches_plot(tmp_path: Path):
    path = save_image_comparison(make_images(), tmp_path / "cmp.png")
    assert path.is_file()


def test_plot_method_comparison_rejects_missing_method(tmp_path: Path):
    images = make_images()
    del images["GWF-C"]
    with pytest.raises(ValueError, match="missing required entries"):
        plot_method_comparison(images, tmp_path / "bad.png")


def test_plot_method_comparison_rejects_incomplete_order(tmp_path: Path):
    with pytest.raises(ValueError, match="order must contain"):
        plot_method_comparison(make_images(), tmp_path / "bad2.png", order=("LR", "HR"))


def test_save_evaluation_bundle_writes_frozen_files(tmp_path: Path):
    images = make_images()
    bundle = save_evaluation_bundle(
        tmp_path,
        config_name="smoke",
        sample_name="piecewise_32",
        method="GWF-C",
        lambda_value=1.0,
        iterations=4,
        dt=0.05,
        hr=images["HR"],
        lr=images["LR"],
        bicubic=images["Bicubic"],
        baseline=images["Baseline"],
        gwf_c=images["GWF-C"],
        solver_result=make_solver_result(iterations=4),
        lpips_available=False,
    )
    assert isinstance(bundle, EvaluationBundle)
    method_dir = tmp_path / "smoke" / "metrics" / "piecewise_32" / "GWF-C"
    produced = sorted(path.name for path in method_dir.iterdir())
    assert produced == [
        "comparison.png",
        "convergence.png",
        "metrics.csv",
        "metrics.json",
        "trajectory_metrics.csv",
    ]
    assert bundle.metrics_json.is_file()
    assert bundle.metrics_csv.is_file()
    assert bundle.comparison_png.is_file()
    assert bundle.convergence_png is not None and bundle.convergence_png.is_file()
    assert bundle.metrics_json.is_absolute()


def test_save_evaluation_bundle_without_solver_result(tmp_path: Path):
    images = make_images()
    bundle = save_evaluation_bundle(
        tmp_path,
        config_name="smoke",
        sample_name="piecewise_32",
        method="Baseline",
        lambda_value=0.0,
        iterations=4,
        dt=0.05,
        hr=images["HR"],
        lr=images["LR"],
        bicubic=images["Bicubic"],
        baseline=images["Baseline"],
        gwf_c=images["GWF-C"],
        solver_result=None,
    )
    assert bundle.convergence_png is None
    method_dir = tmp_path / "smoke" / "metrics" / "piecewise_32" / "Baseline"
    produced = sorted(path.name for path in method_dir.iterdir())
    assert produced == ["comparison.png", "metrics.csv", "metrics.json"]


def test_bundle_json_uses_relative_paths_and_null_lpips(tmp_path: Path):
    images = make_images()
    bundle = save_evaluation_bundle(
        tmp_path,
        config_name="smoke",
        sample_name="piecewise_32",
        method="GWF-C",
        lambda_value=1.0,
        iterations=4,
        dt=0.05,
        hr=images["HR"],
        lr=images["LR"],
        bicubic=images["Bicubic"],
        baseline=images["Baseline"],
        gwf_c=images["GWF-C"],
        solver_result=None,
        lpips_available=False,
    )
    record = json.loads(bundle.metrics_json.read_text(encoding="utf-8"))
    assert record["schema_version"] == "1.0"
    assert record["config_name"] == "smoke"
    assert record["sample_name"] == "piecewise_32"
    assert record["method"] == "GWF-C"
    assert record["metrics"]["lpips"] is None
    assert record["dependencies"]["lpips_available"] is False
    assert record["files"]["comparison_png"] == "smoke/metrics/piecewise_32/GWF-C/comparison.png"
    assert not str(record["files"]["comparison_png"]).startswith("/")
    assert "metrics" in record


def test_bundle_rejects_unsafe_names(tmp_path: Path):
    images = make_images()
    with pytest.raises(ValueError):
        save_evaluation_bundle(
            tmp_path,
            config_name="../evil",
            sample_name="s",
            method="GWF-C",
            lambda_value=0.0,
            iterations=1,
            dt=0.1,
            hr=images["HR"],
            lr=images["LR"],
            bicubic=images["Bicubic"],
            baseline=images["Baseline"],
            gwf_c=images["GWF-C"],
        )
    with pytest.raises(ValueError):
        save_evaluation_bundle(
            tmp_path,
            config_name="smoke",
            sample_name="s",
            method="Other",
            lambda_value=0.0,
            iterations=1,
            dt=0.1,
            hr=images["HR"],
            lr=images["LR"],
            bicubic=images["Bicubic"],
            baseline=images["Baseline"],
            gwf_c=images["GWF-C"],
        )


def test_bundle_rejects_invalid_scalars(tmp_path: Path):
    images = make_images()
    with pytest.raises(ValueError):
        save_evaluation_bundle(
            tmp_path,
            config_name="smoke",
            sample_name="s",
            method="GWF-C",
            lambda_value=-1.0,
            iterations=1,
            dt=0.1,
            hr=images["HR"],
            lr=images["LR"],
            bicubic=images["Bicubic"],
            baseline=images["Baseline"],
            gwf_c=images["GWF-C"],
        )
    with pytest.raises(ValueError):
        save_evaluation_bundle(
            tmp_path,
            config_name="smoke",
            sample_name="s",
            method="GWF-C",
            lambda_value=0.0,
            iterations=0,
            dt=0.1,
            hr=images["HR"],
            lr=images["LR"],
            bicubic=images["Bicubic"],
            baseline=images["Baseline"],
            gwf_c=images["GWF-C"],
        )
