# -*- coding: utf-8 -*-
"""Stability experiment for GWF super-resolution.

Runs the solver from different random initializations and records how the
Hilbert projective distance contracts along the iteration trajectory, plus
the pairwise Hilbert distance between the final reconstructions.

Usage:
    python scripts/run_stability.py --config configs/smoke.yaml --root .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gwf_sr.config import GWFConfig, load_config, resolve_runtime_dirs
from gwf_sr.data import generate_image
from gwf_sr.geometry import distribution_from_image, hilbert_distance
from gwf_sr.metrics import compare_images
from gwf_sr.solver import SuperResolutionSolver
from gwf_sr.visualization import plot_convergence_curves, save_metrics_csv

CONTRACTION_CSV_HEADER = (
    "sample_name",
    "init_pair",
    "step",
    "time",
    "hilbert_distance",
)
FINAL_DISTANCES_CSV_HEADER = (
    "sample_name",
    "pair",
    "hilbert_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML experiment configuration.")
    parser.add_argument("--root", default=".", help="Root directory for results.")
    parser.add_argument("--inits", type=int, default=3, help="Number of random initializations.")
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=1.0,
        help="Curvature regularization used by every run.",
    )
    parser.add_argument(
        "--init-noise",
        type=float,
        default=0.1,
        help="Standard deviation of the random perturbation added to the Bicubic start.",
    )
    parser.add_argument(
        "--integrator",
        default="strang",
        choices=("euler", "euler_maruyama", "strang"),
        help="Time integrator used by the solver.",
    )
    return parser.parse_args()


def make_initialization(
    bicubic: torch.Tensor,
    index: int,
    seed: int,
    init_noise: float,
) -> torch.Tensor:
    """Build one deterministic initialization: bicubic, or bicubic + noise."""

    if index == 0 or init_noise == 0.0:
        return bicubic.clone()
    generator = torch.Generator().manual_seed(seed * 100 + index)
    noise = torch.randn(bicubic.shape, generator=generator, dtype=bicubic.dtype)
    return (bicubic + init_noise * noise).clamp(0.0, 1.0)


def run_stability(
    config: GWFConfig,
    kind: str,
    seed: int,
    lambda_value: float,
    integrator: str,
    n_inits: int,
    init_noise: float,
) -> dict:
    """Test stability across different initializations."""

    hr, lr, bicubic, _ = generate_image(
        kind=kind,
        size=config.image_size,
        seed=seed,
        scale=config.scale,
        sigma=config.sigma,
        noise_std=config.noise_std,
    )
    sample_name = f"{kind}_{config.image_size}_seed{seed}"
    print(f"  Running {n_inits} different initializations...")

    trajectories = []
    final_images = []
    for index in range(n_inits):
        x0 = make_initialization(bicubic, index, seed, init_noise)
        solver = SuperResolutionSolver(
            lr,
            x0,
            scale=config.scale,
            sigma=config.sigma,
            iterations=config.iterations,
            dt=config.dt,
            lambda_value=lambda_value,
            epsilon=config.epsilon,
            edge_weight_mode=config.edge_weight_mode,
            integrator=integrator,
            seed=config.seed,
        )
        result = solver.solve()
        trajectories.append(result.trajectory)
        final_images.append(result.reconstruction)

        metrics = compare_images(hr, result.reconstruction, require_lpips=False)
        print(f"    init {index}: PSNR={metrics.psnr.item():.4f}  SSIM={metrics.ssim.item():.4f}")

    times = torch.arange(config.iterations + 1, dtype=torch.float32) * config.dt

    # Pairwise Hilbert distance between the final reconstructions.
    final_distributions = [
        distribution_from_image(image, epsilon=config.epsilon) for image in final_images
    ]
    final_rows = []
    for i in range(n_inits):
        for j in range(i + 1, n_inits):
            distance = hilbert_distance(final_distributions[i], final_distributions[j]).item()
            final_rows.append(
                {
                    "sample_name": sample_name,
                    "pair": f"{i}-{j}",
                    "hilbert_distance": distance,
                }
            )
            print(f"    Hilbert d({i},{j}) = {distance:.6f}")

    # Hilbert distance contraction along the trajectory (initialization 0 vs 1).
    contraction_rows = []
    contraction_curve = torch.zeros(0)
    if n_inits >= 2:
        rho_first = distribution_from_image(trajectories[0], epsilon=config.epsilon)
        rho_second = distribution_from_image(trajectories[1], epsilon=config.epsilon)
        contraction_curve = hilbert_distance(rho_first, rho_second)
        for step in range(contraction_curve.shape[0]):
            contraction_rows.append(
                {
                    "sample_name": sample_name,
                    "init_pair": "0-1",
                    "step": step,
                    "time": float(times[step]),
                    "hilbert_distance": contraction_curve[step].item(),
                }
            )

    return {
        "kind": kind,
        "seed": seed,
        "sample_name": sample_name,
        "n_inits": n_inits,
        "lambda_value": lambda_value,
        "times": times,
        "contraction_curve": contraction_curve,
        "final_distances": final_rows,
        "contraction": contraction_rows,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(args.root)
    dirs = resolve_runtime_dirs(config, root=root)

    print("GWF Super-Resolution Stability Experiment")
    print(f"Image size: {config.image_size}, iterations: {config.iterations}, dt: {config.dt}")

    samples = [
        ("piecewise", config.seed),
        ("edges", config.seed + 1),
    ]

    all_results = []
    for kind, seed in samples:
        print(f"\n--- {kind} (seed={seed}) ---")
        all_results.append(
            run_stability(
                config=config,
                kind=kind,
                seed=seed,
                lambda_value=args.lambda_value,
                integrator=args.integrator,
                n_inits=args.inits,
                init_noise=args.init_noise,
            )
        )

    # Save JSON record plus the frozen Hilbert-contraction tables.
    json_path = dirs["results"] / "stability_results.json"
    jsonable = []
    for result in all_results:
        record = {key: value for key, value in result.items() if key not in {"times", "contraction_curve"}}
        record["contraction"] = result["contraction"]
        record["final_distances"] = result["final_distances"]
        jsonable.append(record)
    json_path.write_text(
        json.dumps(jsonable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    contraction_rows = [row for result in all_results for row in result["contraction"]]
    final_rows = [row for result in all_results for row in result["final_distances"]]
    contraction_csv = None
    final_csv = None
    if contraction_rows:
        contraction_csv = save_metrics_csv(
            contraction_rows,
            dirs["results"] / "hilbert_contraction.csv",
            fieldnames=CONTRACTION_CSV_HEADER,
        )
    if final_rows:
        final_csv = save_metrics_csv(
            final_rows,
            dirs["results"] / "hilbert_final_distances.csv",
            fieldnames=FINAL_DISTANCES_CSV_HEADER,
        )

    # One contraction figure for every sample (optional field, frozen API).
    series = {
        f"{result['sample_name']} init0-init1": {
            "times": result["times"],
            "hilbert": result["contraction_curve"],
        }
        for result in all_results
        if result["contraction_curve"].numel() > 0
    }
    contraction_png = None
    if series:
        contraction_png = plot_convergence_curves(
            series,
            dirs["results"] / "hilbert_contraction.png",
            fields=["hilbert"],
            title="Hilbert distance contraction",
        )

    print(f"\nResults saved to {json_path}")
    if contraction_csv is not None:
        print(f"Contraction table saved to {contraction_csv}")
    if final_csv is not None:
        print(f"Final distance table saved to {final_csv}")
    if contraction_png is not None:
        print(f"Contraction figure saved to {contraction_png}")

    # Print summary.
    print(f"\n{'='*70}")
    for result in all_results:
        print(f"{result['kind']}:")
        for row in result["final_distances"]:
            print(f"  Hilbert d({row['pair']}) = {row['hilbert_distance']:.6f}")
        if result["contraction"]:
            print(f"  Contraction steps: {len(result['contraction'])}")
            print(f"  Initial distance: {result['contraction'][0]['hilbert_distance']:.6f}")
            print(f"  Final distance:   {result['contraction'][-1]['hilbert_distance']:.6f}")
    print(f"{'='*70}")
    print("Stability experiment complete.")


if __name__ == "__main__":
    main()
