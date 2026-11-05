# -*- coding: utf-8 -*-
"""Stability experiment for GWF super-resolution.

Tests convergence with different initializations and measures
Hilbert distance contraction between trajectories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gwf_sr.config import load_config, resolve_runtime_dirs
from gwf_sr.data import generate_image
from gwf_sr.geometry import distribution_from_image, hilbert_distance
from gwf_sr.metrics import compare_images
from gwf_sr.solver import SuperResolutionSolver


def run_stability(
    kind: str,
    seed: int,
    size: int,
    scale: int,
    sigma: float,
    iterations: int,
    dt: float,
    lambda_value: float,
    epsilon: float,
    n_inits: int = 3,
) -> dict:
    """Test stability across different initializations."""
    hr, lr, bicubic, _ = generate_image(
        kind=kind, size=size, seed=seed, scale=scale, sigma=sigma, noise_std=0.0
    )

    print(f"  Running {n_inits} different initializations...")

    trajectories = []
    final_images = []
    for i in range(n_inits):
        # Different random initializations
        generator = torch.Generator().manual_seed(seed * 100 + i)
        if i == 0:
            x0 = bicubic.clone()
        else:
            x0 = bicubic + 0.1 * torch.randn(
                bicubic.shape, generator=generator, dtype=bicubic.dtype
            )
            x0 = x0.clamp(0.0, 1.0)

        solver = SuperResolutionSolver(
            lr, x0,
            iterations=iterations,
            dt=dt,
            lambda_value=lambda_value,
            integrator="strang",
            epsilon=epsilon,
        )
        result = solver.solve()
        trajectories.append(result.trajectory)
        final_images.append(result.reconstruction)

        metrics = compare_images(hr, result.reconstruction)
        print(f"    init {i}: PSNR={metrics.psnr.item():.4f}  SSIM={metrics.ssim.item():.4f}")

    # Compute Hilbert distances between final images
    dists = []
    for i in range(n_inits):
        for j in range(i + 1, n_inits):
            rho_i = distribution_from_image(final_images[i], epsilon=epsilon)
            rho_j = distribution_from_image(final_images[j], epsilon=epsilon)
            d = hilbert_distance(rho_i, rho_j).item()
            dists.append({"pair": f"{i}-{j}", "hilbert_distance": d})
            print(f"    Hilbert d({i},{j}) = {d:.6f}")

    # Compute Hilbert distance contraction over time (between init 0 and 1)
    if n_inits >= 2:
        contraction = []
        traj0 = trajectories[0]
        traj1 = trajectories[1]
        n_steps = min(traj0.shape[0], traj1.shape[0])
        for k in range(n_steps):
            rho0 = distribution_from_image(traj0[k], epsilon=epsilon)
            rho1 = distribution_from_image(traj1[k], epsilon=epsilon)
            d = hilbert_distance(rho0, rho1).item()
            contraction.append({"step": k, "time": k * dt, "hilbert_distance": d})
    else:
        contraction = []

    return {
        "kind": kind,
        "seed": seed,
        "n_inits": n_inits,
        "lambda_value": lambda_value,
        "final_distances": dists,
        "contraction": contraction,
    }


def main() -> None:
    config = load_config("configs/smoke.yaml")
    dirs = resolve_runtime_dirs(config)

    print("GWF Super-Resolution Stability Experiment")
    print(f"Image size: {config.image_size}, iterations: {config.iterations}, dt: {config.dt}")

    samples = [
        ("piecewise", 42),
        ("edges", 43),
    ]

    all_results = []
    for kind, seed in samples:
        print(f"\n--- {kind} (seed={seed}) ---")
        result = run_stability(
            kind=kind,
            seed=seed,
            size=config.image_size,
            scale=config.scale,
            sigma=config.sigma,
            iterations=config.iterations,
            dt=config.dt,
            lambda_value=1.0,
            epsilon=config.epsilon,
            n_inits=3,
        )
        all_results.append(result)

    # Save results
    output_path = dirs["results"] / "stability_results.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    for r in all_results:
        print(f"{r['kind']}:")
        for d in r["final_distances"]:
            print(f"  Hilbert d({d['pair']}) = {d['hilbert_distance']:.6f}")
        if r["contraction"]:
            print(f"  Contraction steps: {len(r['contraction'])}")
            print(f"  Initial distance: {r['contraction'][0]['hilbert_distance']:.6f}")
            print(f"  Final distance:   {r['contraction'][-1]['hilbert_distance']:.6f}")
    print(f"{'='*70}")
    print("Stability experiment complete.")


if __name__ == "__main__":
    main()
