# -*- coding: utf-8 -*-
"""Lambda ablation experiment for GWF super-resolution.

Scans lambda values and reports PSNR/SSIM/convergence for each.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gwf_sr.config import load_config, resolve_runtime_dirs
from gwf_sr.data import generate_image
from gwf_sr.metrics import compare_images
from gwf_sr.solver import SuperResolutionSolver


def run_ablation(
    kind: str,
    seed: int,
    size: int,
    scale: int,
    sigma: float,
    iterations: int,
    dt: float,
    epsilon: float,
    lambda_values: list[float],
) -> list[dict]:
    """Run lambda ablation for one sample."""
    hr, lr, bicubic, _ = generate_image(
        kind=kind, size=size, seed=seed, scale=scale, sigma=sigma, noise_std=0.0
    )

    results = []
    for lam in lambda_values:
        print(f"  lambda={lam}...", end=" ", flush=True)
        solver = SuperResolutionSolver(
            lr, bicubic,
            iterations=iterations,
            dt=dt,
            lambda_value=lam,
            integrator="strang",
            epsilon=epsilon,
        )
        result = solver.solve()
        metrics = compare_images(hr, result.reconstruction)
        print(
            f"PSNR={metrics.psnr.item():.4f}  SSIM={metrics.ssim.item():.4f}  "
            f"data_fid={result.data_fidelity[-1].item():.6f}"
        )
        results.append({
            "kind": kind,
            "lambda_value": lam,
            "psnr": metrics.psnr.item(),
            "ssim": metrics.ssim.item(),
            "lpips": None,
            "data_fidelity": result.data_fidelity[-1].item(),
            "entropy": result.entropy[-1].item(),
            "regularization_energy": result.regularization_energy[-1].item(),
            "total_energy": result.total_energy[-1].item(),
            "curvature": result.curvature[-1].item(),
            "final_update_norm": result.final_update_norm.item(),
        })
    return results


def main() -> None:
    config = load_config("configs/smoke.yaml")
    dirs = resolve_runtime_dirs(config)

    lambda_values = list(config.lambda_values)
    print(f"Lambda ablation: {lambda_values}")
    print(f"Image size: {config.image_size}, iterations: {config.iterations}, dt: {config.dt}")

    samples = [
        ("piecewise", 42),
        ("edges", 43),
        ("texture", 44),
        ("smooth", 45),
    ]

    all_results = []
    for kind, seed in samples:
        print(f"\n--- {kind} (seed={seed}) ---")
        results = run_ablation(
            kind=kind,
            seed=seed,
            size=config.image_size,
            scale=config.scale,
            sigma=config.sigma,
            iterations=config.iterations,
            dt=config.dt,
            epsilon=config.epsilon,
            lambda_values=lambda_values,
        )
        all_results.extend(results)

    # Save results
    output_path = dirs["results"] / "ablation_results.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Sample':<15} {'Lambda':>8} {'PSNR':>8} {'SSIM':>8} {'DataFid':>12} {'Energy':>12} {'Curv':>12}")
    print(f"{'='*90}")
    for r in all_results:
        print(
            f"{r['kind']:<15} {r['lambda_value']:>8.1f} "
            f"{r['psnr']:>8.4f} {r['ssim']:>8.4f} "
            f"{r['data_fidelity']:>12.6f} {r['total_energy']:>12.6f} {r['curvature']:>12.6f}"
        )
    print(f"{'='*90}")
    print("Ablation complete.")


if __name__ == "__main__":
    main()
