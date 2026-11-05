"""Frozen stage-six visualization and persistence utilities.

Implements contract ``gwf_sr/design/metrics_visualization_contract.md``
§§9-13: convergence curves, the five-column method comparison, frozen JSON
and CSV layouts, and the one-shot evaluation bundle. This module only plots
and writes files; it never redefines PSNR/SSIM/LPIPS and never mutates a
solver result.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .degradation import bicubic_upsample  # noqa: E402
from .metrics import ImageMetrics, _validate_image, compare_images  # noqa: E402

CORE_FIELDS = (
    "data_fidelity",
    "entropy",
    "regularization_energy",
    "total_energy",
    "curvature",
    "update_norm",
)
OPTIONAL_FIELDS = ("kl", "hilbert")
ALLOWED_FIELDS = CORE_FIELDS + OPTIONAL_FIELDS
DEFAULT_COMPARISON_ORDER = ("LR", "Bicubic", "Baseline", "GWF-C", "HR")
ALLOWED_METHODS = ("Baseline", "GWF-C")
TRAJECTORY_CSV_HEADER = (
    "time",
    "iteration",
    "data_fidelity",
    "entropy",
    "regularization_energy",
    "total_energy",
    "curvature",
    "update_norm",
    "lambda_value",
)
METHOD_CSV_HEADER = (
    "sample_name",
    "method",
    "lambda_value",
    "psnr",
    "ssim",
    "lpips",
    "high_frequency_energy",
    "high_frequency_energy_ratio",
    "edge_preservation_index",
    "edge_preservation_ratio",
)
JSON_TOP_LEVEL_FIELDS = (
    "schema_version",
    "config_name",
    "sample_name",
    "method",
    "lambda_value",
    "iterations",
    "dt",
    "metrics",
    "files",
    "dependencies",
    "extra",
)


def _validate_positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite Python int or float.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return float(value)


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite Python int or float.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return float(value)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _validate_safe_name(name: str, field: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{field} must be a string.")
    if name == "" or name in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty safe file name.")
    if any(separator in name for separator in ("/", "\\", ":")):
        raise ValueError(f"{field} must not contain path separators or drive letters.")
    if Path(name).is_absolute():
        raise ValueError(f"{field} must not be an absolute path.")
    return name


def validate_visualization_config(
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (20.0, 4.0),
    batch_mode: str = "mean",
    title: str | None = None,
) -> None:
    """Validate the frozen scalar visualization parameters."""

    _validate_positive_int(dpi, "dpi")
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must contain exactly two numbers.")
    for index, value in enumerate(figsize):
        _validate_positive_scalar(value, f"figsize[{index}]")
    if batch_mode not in ("mean", "first"):
        raise ValueError("batch_mode must be 'mean' or 'first'.")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None.")


def tensor_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """Project an image onto ``[0, 1]`` and quantize it to 8-bit grayscale."""

    _validate_image(image, "image")
    projected = image.detach().to("cpu").clamp(0.0, 1.0)
    return (projected * 255.0).round().to(torch.uint8)


def ensure_output_directory(path: str | Path) -> Path:
    """Create the parent directories of ``path`` (or the directory itself)."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output must not contain NaN or Infinity.")
        return value
    if isinstance(value, Path):
        # Contract §11.2.4 forbids absolute paths in the final JSON, so an
        # absolute Path collapses to its file name while a relative Path keeps
        # its POSIX style.
        return value.name if value.is_absolute() else value.as_posix()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            scalar = float(value.detach().cpu().reshape(()))
            if not math.isfinite(scalar):
                raise ValueError("JSON output must not contain NaN or Infinity.")
            return scalar
        return [_to_jsonable(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def save_metrics_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write a metrics record as UTF-8 JSON with two-space indentation."""

    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping.")
    payload = dict(data)
    if "schema_version" in payload:
        extra = dict(payload.get("extra") or {})
        for key in list(payload):
            if key == "extra":
                continue
            if key not in JSON_TOP_LEVEL_FIELDS:
                extra[key] = payload.pop(key)
        payload = {key: value for key, value in payload.items() if key in JSON_TOP_LEVEL_FIELDS}
        if extra:
            payload["extra"] = extra
        payload.setdefault("schema_version", "1.0")

    target = Path(path)
    ensure_output_directory(target.parent)
    text = json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def _write_csv(
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    path: str | Path,
) -> Path:
    target = Path(path)
    ensure_output_directory(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return target


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("CSV cells must hold scalar values, not tensors.")
        value = float(value.detach().cpu().reshape(()))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CSV output must not contain NaN or Infinity.")
        return repr(number)
    return str(value)


def save_metrics_csv(
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write one or more metric rows with the frozen method-metrics header."""

    if isinstance(data, Mapping):
        rows = [data]
    elif isinstance(data, Sequence):
        rows = list(data)
    else:
        raise TypeError("data must be a mapping or a sequence of mappings.")
    if not rows:
        raise ValueError("data must contain at least one row.")
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each row must be a mapping.")

    if fieldnames is None:
        fieldnames = METHOD_CSV_HEADER if all(key in rows[0] for key in METHOD_CSV_HEADER) else list(rows[0])
    header = list(fieldnames)
    table = [[_format_cell(row.get(key)) for key in header] for row in rows]
    return _write_csv(header, table, path)


def _result_field(result: Any, field: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(field)
    return getattr(result, field, None)


def _curve_tensor(result: Any, field: str, label: str) -> torch.Tensor | None:
    value = _result_field(result, field)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"curve '{field}' of method '{label}' must be a torch.Tensor.")
    return value.detach().to("cpu")


def _reduce_batch(curve: torch.Tensor, batch_mode: str, field: str, label: str) -> torch.Tensor:
    if curve.ndim == 1:
        return curve
    if curve.ndim == 2:
        if batch_mode == "mean":
            return curve.mean(dim=1)
        if batch_mode == "first":
            return curve[:, 0]
    raise ValueError(f"curve '{field}' of method '{label}' must be 1-D or 2-D.")


def plot_convergence_curves(
    results: Mapping[str, Any],
    output_path: str | Path,
    *,
    fields: Sequence[str] | None = None,
    batch_mode: str = "mean",
    dpi: int = 150,
    title: str | None = None,
) -> Path:
    """Plot the frozen convergence diagnostics for one or more methods."""

    if not isinstance(results, Mapping) or len(results) == 0:
        raise ValueError("results must contain at least one method.")
    for label in results:
        if not isinstance(label, str) or label == "":
            raise ValueError("every method label must be a non-empty string.")
    validate_visualization_config(dpi=dpi, batch_mode=batch_mode, title=title)

    if fields is None:
        selected = list(CORE_FIELDS)
        selected.extend(field for field in OPTIONAL_FIELDS if any(_result_field(r, field) is not None for r in results.values()))
    else:
        selected = list(fields)
        if not selected:
            raise ValueError("fields must not be empty.")
        for field in selected:
            if field not in ALLOWED_FIELDS:
                raise ValueError(
                    f"Unknown curve field '{field}'. Choose from: {', '.join(ALLOWED_FIELDS)}."
                )

    times: torch.Tensor | None = None
    for label, result in results.items():
        candidate = _curve_tensor(result, "times", label)
        if candidate is None:
            raise ValueError(f"method '{label}' does not provide a 'times' curve.")
        if candidate.ndim != 1:
            raise ValueError(f"'times' of method '{label}' must be one-dimensional.")
        if times is None:
            times = candidate
        elif times.shape != candidate.shape:
            raise ValueError("all methods must share the same 'times' grid.")
    assert times is not None

    usable: list[tuple[str, dict[str, torch.Tensor]]] = []
    for label, result in results.items():
        curves: dict[str, torch.Tensor] = {}
        for field in selected:
            curve = _curve_tensor(result, field, label)
            if curve is None:
                continue
            reduced = _reduce_batch(curve, batch_mode, field, label)
            if reduced.shape[0] != times.shape[0]:
                raise ValueError(
                    f"curve '{field}' of method '{label}' must match the length of 'times'."
                )
            curves[field] = reduced
        if curves:
            usable.append((label, curves))
    if not usable:
        raise ValueError("no plottable curve fields were found in the provided results.")

    plot_fields = [field for field in selected if any(field in curves for _, curves in usable)]
    if not plot_fields:
        raise ValueError("no plottable curve fields were found in the provided results.")
    columns = min(3, len(plot_fields))
    rows = math.ceil(len(plot_fields) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.0 * rows), dpi=dpi, squeeze=False)
    flat_axes = [axis for row in axes for axis in row]

    for index, field in enumerate(plot_fields):
        axis = flat_axes[index]
        for label, curves in usable:
            if field in curves:
                axis.plot(times.numpy(), curves[field].numpy(), label=label)
        axis.set_title(field.replace("_", " ").title())
        axis.set_xlabel("Time")
        axis.set_ylabel(field.replace("_", " ").title())
        axis.grid(True)
        axis.legend()
    for axis in flat_axes[len(plot_fields) :]:
        axis.set_visible(False)

    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()

    target = Path(output_path)
    ensure_output_directory(target.parent)
    figure.savefig(target, format="png", dpi=dpi)
    plt.close(figure)
    return target.resolve()


def _resize_to(image: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape) == target_shape:
        return image
    height, width = image.shape
    target_height, target_width = target_shape
    if target_height % height == 0 and target_width % width == 0:
        scale = target_height // height
        if scale == target_width // width and scale >= 1:
            if scale == 1:
                return image
            return bicubic_upsample(image, scale=scale)

    return F.interpolate(
        image[None, None],
        size=target_shape,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0]


def plot_method_comparison(
    images: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    order: Sequence[str] = DEFAULT_COMPARISON_ORDER,
    dpi: int = 150,
    figsize: tuple[float, float] = (20.0, 4.0),
    title: str | None = None,
) -> Path:
    """Save the frozen five-column LR/Bicubic/Baseline/GWF-C/HR comparison."""

    if not isinstance(images, Mapping):
        raise TypeError("images must be a mapping of name to image tensor.")
    order = tuple(order)
    for required in DEFAULT_COMPARISON_ORDER:
        if required not in order:
            raise ValueError(f"order must contain '{required}'.")
    missing = [key for key in order if key not in images]
    if missing:
        raise ValueError(f"images is missing required entries: {', '.join(missing)}.")
    validate_visualization_config(dpi=dpi, figsize=figsize, title=title)

    for key in order:
        _validate_image(images[key], f"images['{key}']")
        if images[key].ndim != 2:
            raise ValueError(f"images['{key}'] must be a single H×W image.")

    human_reference = images["HR"]
    target_shape = (int(human_reference.shape[-2]), int(human_reference.shape[-1]))

    figure, axes = plt.subplots(1, len(order), figsize=figsize, dpi=dpi, squeeze=False)
    for axis, key in zip(axes[0], order):
        resized = _resize_to(images[key], target_shape)
        axis.imshow(tensor_to_uint8(resized).numpy(), cmap="gray", vmin=0, vmax=255)
        axis.set_title(key)
        axis.axis("off")
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()

    target = Path(output_path)
    ensure_output_directory(target.parent)
    figure.savefig(target, format="png", dpi=dpi)
    plt.close(figure)
    return target.resolve()


def save_image_comparison(
    images: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    order: Sequence[str] = DEFAULT_COMPARISON_ORDER,
    dpi: int = 150,
    figsize: tuple[float, float] = (20.0, 4.0),
    title: str | None = None,
) -> Path:
    """Save the method comparison image (equivalent to ``plot_method_comparison``)."""

    return plot_method_comparison(
        images,
        output_path,
        order=order,
        dpi=dpi,
        figsize=figsize,
        title=title,
    )


def _aggregate(curve: torch.Tensor) -> float:
    value = curve.detach().to("cpu")
    if value.ndim == 2:
        value = value.mean(dim=1)
    elif value.ndim != 1:
        raise ValueError("trajectory curves must be 1-D or 2-D.")
    return value


def save_trajectory_csv(result: Any, path: str | Path, *, batch_mode: str = "mean") -> Path:
    """Write the frozen ``trajectory_metrics.csv`` for one solver result."""

    if batch_mode not in ("mean", "first"):
        raise ValueError("batch_mode must be 'mean' or 'first'.")
    times = _result_field(result, "times")
    if not isinstance(times, torch.Tensor) or times.ndim != 1:
        raise ValueError("the solver result must provide a one-dimensional 'times' curve.")
    times = times.detach().to("cpu")

    lambda_value = float(_result_field(result, "lambda_value") or 0.0)
    _validate_nonnegative_scalar(lambda_value, "lambda_value")

    curves: dict[str, torch.Tensor] = {}
    for field in CORE_FIELDS:
        raw = _result_field(result, field)
        if raw is None:
            raise ValueError(f"the solver result is missing the '{field}' curve.")
        tensor = raw.detach().to("cpu")
        if tensor.ndim == 2:
            tensor = tensor.mean(dim=1) if batch_mode == "mean" else tensor[:, 0]
        elif tensor.ndim != 1:
            raise ValueError(f"curve '{field}' must be 1-D or 2-D.")
        if tensor.shape[0] != times.shape[0]:
            raise ValueError(f"curve '{field}' must match the length of 'times'.")
        curves[field] = tensor

    rows = []
    for index in range(times.shape[0]):
        rows.append(
            [
                repr(float(times[index])),
                str(index),
                repr(float(curves["data_fidelity"][index])),
                repr(float(curves["entropy"][index])),
                repr(float(curves["regularization_energy"][index])),
                repr(float(curves["total_energy"][index])),
                repr(float(curves["curvature"][index])),
                repr(float(curves["update_norm"][index])),
                repr(lambda_value),
            ]
        )
    return _write_csv(TRAJECTORY_CSV_HEADER, rows, path)


@dataclass(frozen=True)
class EvaluationBundle:
    """Frozen output of :func:`save_evaluation_bundle`."""

    metrics: ImageMetrics
    metrics_json: Path
    metrics_csv: Path
    comparison_png: Path
    convergence_png: Path | None


def _relative_posix(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def save_evaluation_bundle(
    output_dir: str | Path,
    *,
    config_name: str,
    sample_name: str,
    method: str,
    lambda_value: float,
    iterations: int,
    dt: float,
    hr: torch.Tensor,
    lr: torch.Tensor,
    bicubic: torch.Tensor,
    baseline: torch.Tensor,
    gwf_c: torch.Tensor,
    solver_result: Any | None = None,
    lpips_available: bool = False,
    require_lpips: bool = False,
) -> EvaluationBundle:
    """Write the complete frozen file set for one sample and method."""

    _validate_safe_name(config_name, "config_name")
    _validate_safe_name(sample_name, "sample_name")
    _validate_safe_name(method, "method")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"method must be one of {', '.join(ALLOWED_METHODS)}.")
    lambda_value = _validate_nonnegative_scalar(lambda_value, "lambda_value")
    iterations = _validate_positive_int(iterations, "iterations")
    dt = _validate_positive_scalar(dt, "dt")
    if not isinstance(require_lpips, bool):
        raise TypeError("require_lpips must be a bool.")
    if not isinstance(lpips_available, bool):
        raise TypeError("lpips_available must be a bool.")

    for name, image in (("hr", hr), ("lr", lr), ("bicubic", bicubic), ("baseline", baseline), ("gwf_c", gwf_c)):
        _validate_image(image, name)

    prediction = baseline if method == "Baseline" else gwf_c
    metrics = compare_images(hr, prediction, require_lpips=require_lpips)

    bundle_dir = Path(output_dir) / config_name / "metrics" / sample_name / method
    ensure_output_directory(bundle_dir)

    comparison_png = save_image_comparison(
        {
            "LR": lr,
            "Bicubic": bicubic,
            "Baseline": baseline,
            "GWF-C": gwf_c,
            "HR": hr,
        },
        bundle_dir / "comparison.png",
    )

    convergence_png: Path | None = None
    if solver_result is not None:
        save_trajectory_csv(solver_result, bundle_dir / "trajectory_metrics.csv")
        convergence_png = plot_convergence_curves(
            {method: solver_result},
            bundle_dir / "convergence.png",
        )

    metrics_csv = save_metrics_csv(
        [
            {
                "sample_name": sample_name,
                "method": method,
                "lambda_value": lambda_value,
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "lpips": metrics.lpips,
                "high_frequency_energy": metrics.high_frequency_energy,
                "high_frequency_energy_ratio": metrics.high_frequency_energy_ratio,
                "edge_preservation_index": metrics.edge_preservation_index,
                "edge_preservation_ratio": metrics.edge_preservation_ratio,
            }
        ],
        bundle_dir / "metrics.csv",
    )

    base = Path(output_dir)
    record = {
        "schema_version": "1.0",
        "config_name": config_name,
        "sample_name": sample_name,
        "method": method,
        "lambda_value": lambda_value,
        "iterations": iterations,
        "dt": dt,
        "metrics": {
            "psnr": metrics.psnr,
            "ssim": metrics.ssim,
            "lpips": metrics.lpips,
            "high_frequency_energy": metrics.high_frequency_energy,
            "high_frequency_energy_ratio": metrics.high_frequency_energy_ratio,
            "edge_preservation_index": metrics.edge_preservation_index,
            "edge_preservation_ratio": metrics.edge_preservation_ratio,
        },
        "files": {
            "hr": "",
            "lr": "",
            "bicubic": "",
            "baseline": "",
            "gwf_c": "",
            "comparison_png": _relative_posix(comparison_png, base),
            "convergence_png": _relative_posix(convergence_png, base) if convergence_png else "",
            "metrics_csv": _relative_posix(metrics_csv, base),
        },
        "dependencies": {"lpips_available": lpips_available},
    }
    metrics_json = save_metrics_json(record, bundle_dir / "metrics.json")

    return EvaluationBundle(
        metrics=metrics,
        metrics_json=metrics_json.resolve(),
        metrics_csv=metrics_csv.resolve(),
        comparison_png=comparison_png,
        convergence_png=convergence_png.resolve() if convergence_png else None,
    )


__all__ = [
    "EvaluationBundle",
    "ensure_output_directory",
    "plot_convergence_curves",
    "plot_method_comparison",
    "save_evaluation_bundle",
    "save_image_comparison",
    "save_metrics_csv",
    "save_metrics_json",
    "save_trajectory_csv",
    "tensor_to_uint8",
    "validate_visualization_config",
]
