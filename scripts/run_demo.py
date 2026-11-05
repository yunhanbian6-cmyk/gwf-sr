# -*- coding: utf-8 -*-
"""End-to-end GWF super-resolution demo.

Generates synthetic images, applies degradation, runs Baseline and GWF-C
solvers, computes metrics, and saves visualizations.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gwf_sr.config import load_config, resolve_runtime_dirs
from gwf_sr.data import generate_image
from gwf_sr.degradation import bicubic_upsample, degrade
from gwf_sr.geometry import distribution_from_image, entropy, hilbert_distance
from gwf_sr.graph import build_pixel_graph
from gwf_sr.metrics import compare_images
from gwf_sr.solver import SuperResolutionSolver
from gwf_sr.visualization import (
    plot_method_comparison,
    save_evaluation_bundle,
    save_metrics_json,
    save_trajectory_csv,
)


def run_one_sample(
    kind: str,
    seed: int,
    size: int,
    scale: int,
    sigma: float,
    noise_std: float,
    iterations: int,
    dt: float,
    lambda_value: float,
    epsilon: float,
    output_root: Path,
) -> dict:
    """Run one end-to-end sample and return summary metrics."""
    print(f"\n{'='*60}")
    print(f"Sample: {kind}  seed={seed}  size={size}  lambda={lambda_value}")
    print(f"{'='*60}")

    # 1. Generate HR and LR
    hr, lr, bicubic, record = generate_image(
        kind=kind,
        size=size,
        seed=seed,
        scale=scale,
        sigma=sigma,
        noise_std=noise_std,
    )
    print(f"  HR shape: {hr.shape}, LR shape: {lr.shape}")

    # 2. Baseline: lambda=0
    print("  Running Baseline (lambda=0)...")
    baseline_solver = SuperResolutionSolver(
        lr, bicubic,
        iterations=iterations,
        dt=dt,
        lambda_value=0.0,
        integrator="strang",
        epsilon=epsilon,
    )
    baseline_result = baseline_solver.solve()
    print(f"  Baseline done. Final data_fidelity: {baseline_result.data_fidelity[-1].item():.6f}")

    # 3. GWF-C: lambda>0
    print(f"  Running GWF-C (lambda={lambda_value})...")
    gwf_solver = SuperResolutionSolver(
        lr, bicubic,
        iterations=iterations,
        dt=dt,
        lambda_value=lambda_value,
        integrator="strang",
        epsilon=epsilon,
    )
    gwf_result = gwf_solver.solve()
    print(f"  GWF-C done. Final data_fidelity: {gwf_result.data_fidelity[-1].item():.6f}")

    # 4. Compute metrics
    print("  Computing metrics...")
    baseline_metrics = compare_images(hr, baseline_result.reconstruction)
    gwf_metrics = compare_images(hr, gwf_result.reconstruction)
    bicubic_metrics = compare_images(hr, bicubic)

    print(f"  Bicubic  PSNR={bicubic_metrics.psnr.item():.4f}  SSIM={bicubic_metrics.ssim.item():.4f}")
    print(f"  Baseline PSNR={baseline_metrics.psnr.item():.4f}  SSIM={baseline_metrics.ssim.item():.4f}")
    print(f"  GWF-C    PSNR={gwf_metrics.psnr.item():.4f}  SSIM={gwf_metrics.ssim.item():.4f}")

    # 5. Save evaluation bundles
    for method_name, result, metrics in [
        ("Baseline", baseline_result, baseline_metrics),
        ("GWF-C", gwf_result, gwf_metrics),
    ]:
        bundle = save_evaluation_bundle(
            output_root,
            config_name="demo",
            sample_name=f"{kind}_{size}_seed{seed}",
            method=method_name,
            lambda_value=0.0 if method_name == "Baseline" else lambda_value,
            iterations=iterations,
            dt=dt,
            hr=hr,
            lr=lr,
            bicubic=bicubic,
            baseline=baseline_result.reconstruction,
            gwf_c=gwf_result.reconstruction,
            solver_result=result,
            lpips_available=False,
        )
        print(f"  Saved {method_name} bundle to {bundle.metrics_json}")

    return {
        "kind": kind,
        "seed": seed,
        "size": size,
        "lambda_value": lambda_value,
        "bicubic_psnr": bicubic_metrics.psnr.item(),
        "bicubic_ssim": bicubic_metrics.ssim.item(),
        "baseline_psnr": baseline_metrics.psnr.item(),
        "baseline_ssim": baseline_metrics.ssim.item(),
        "gwf_psnr": gwf_metrics.psnr.item(),
        "gwf_ssim": gwf_metrics.ssim.item(),
    }


def main() -> None:
    config = load_config("configs/smoke.yaml")
    dirs = resolve_runtime_dirs(config)
    output_root = dirs["results"]

    print("GWF Super-Resolution End-to-End Demo")
    print(f"Output directory: {output_root}")

    samples = [
        ("piecewise", 42),
        ("edges", 43),
        ("texture", 44),
        ("smooth", 45),
    ]

    summaries = []
    for kind, seed in samples:
        summary = run_one_sample(
            kind=kind,
            seed=seed,
            size=config.image_size,
            scale=config.scale,
            sigma=config.sigma,
            noise_std=config.noise_std,
            iterations=config.iterations,
            dt=config.dt,
            lambda_value=1.0,
            epsilon=config.epsilon,
            output_root=output_root,
        )
        summaries.append(summary)

    # Save summary
    summary_path = output_root / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Sample':<20} {'Bicubic':>12} {'Baseline':>12} {'GWF-C':>12}")
    print(f"{'':20} {'PSNR/SSIM':>12} {'PSNR/SSIM':>12} {'PSNR/SSIM':>12}")
    print(f"{'='*80}")
    for s in summaries:
        print(
            f"{s['kind']:<20} "
            f"{s['bicubic_psnr']:>6.3f}/{s['bicubic_ssim']:<6.4f} "
            f"{s['baseline_psnr']:>6.3f}/{s['baseline_ssim']:<6.4f} "
            f"{s['gwf_psnr']:>6.3f}/{s['gwf_ssim']:<6.4f}"
        )
    print(f"{'='*80}")
    print("Demo complete.")


if __name__ == "__main__":
    main()
