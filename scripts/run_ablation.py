# -*- coding: utf-8 -*-
"""Lambda ablation experiment for GWF super-resolution.

Scans lambda values (default: 0, 0.1, 1, 5, 10) and reports a metric
summary table for every sample and regularization strength.

Usage:
    python scripts/run_ablation.py --config configs/smoke.yaml --root .
    python scripts/run_ablation.py --lambdas 0 0.1 1 5 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwf_sr.config import GWFConfig, load_config, resolve_runtime_dirs
from gwf_sr.data import generate_image
from gwf_sr.metrics import compare_images
from gwf_sr.solver import SuperResolutionSolver
from gwf_sr.visualization import save_metrics_csv

# Frozen stage-eight scan: lambda = 0, 0.1, 1, 5, 10.
DEFAULT_LAMBDA_SCAN = (0.0, 0.1, 1.0, 5.0, 10.0)

ABLATION_CSV_HEADER = (
    "sample_name",
    "lambda_value",
    "psnr",
    "ssim",
    "lpips",
    "high_frequency_energy",
    "high_frequency_energy_ratio",
    "edge_preservation_index",
    "edge_preservation_ratio",
    "data_fidelity",
    "total_energy",
    "curvature",
    "final_update_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML experiment configuration.")
    parser.add_argument("--root", default=".", help="Root directory for results.")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_SCAN),
        help="Lambda values to scan (default: 0 0.1 1 5 10).",
    )
    parser.add_argument(
        "--integrator",
        default="strang",
        choices=("euler", "euler_maruyama", "strang"),
        help="Time integrator used by the solver.",
    )
    return parser.parse_args()


def run_ablation(
    config: GWFConfig,
    kind: str,
    seed: int,
    lambda_values: list[float],
    integrator: str,
) -> list[dict]:
    """Run lambda ablation for one sample."""
    hr, lr, bicubic, _ = generate_image(
        kind=kind,
        size=config.image_size,
        seed=seed,
        scale=config.scale,
        sigma=config.sigma,
        noise_std=config.noise_std,
    )

    results = []
    for lam in lambda_values:
        print(f"  lambda={lam}...", end=" ", flush=True)
        solver = SuperResolutionSolver(
            lr,
            bicubic,
            scale=config.scale,
            sigma=config.sigma,
            iterations=config.iterations,
            dt=config.dt,
            lambda_value=lam,
            epsilon=config.epsilon,
            edge_weight_mode=config.edge_weight_mode,
            integrator=integrator,
            seed=config.seed,
        )
        result = solver.solve()
        metrics = compare_images(hr, result.reconstruction, require_lpips=False)
        print(
            f"PSNR={metrics.psnr.item():.4f}  SSIM={metrics.ssim.item():.4f}  "
            f"data_fid={result.data_fidelity[-1].item():.6f}"
        )
        results.append(
            {
                "sample_name": f"{kind}_{config.image_size}_seed{seed}",
                "kind": kind,
                "lambda_value": float(lam),
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "lpips": metrics.lpips,
                "high_frequency_energy": metrics.high_frequency_energy,
                "high_frequency_energy_ratio": metrics.high_frequency_energy_ratio,
                "edge_preservation_index": metrics.edge_preservation_index,
                "edge_preservation_ratio": metrics.edge_preservation_ratio,
                "data_fidelity": result.data_fidelity[-1].item(),
                "total_energy": result.total_energy[-1].item(),
                "curvature": result.curvature[-1].item(),
                "final_update_norm": result.final_update_norm.item(),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(args.root)
    dirs = resolve_runtime_dirs(config, root=root)

    lambda_values = [float(value) for value in args.lambdas]
    print(f"Lambda ablation: {lambda_values}")
    print(f"Image size: {config.image_size}, iterations: {config.iterations}, dt: {config.dt}")

    samples = [
        ("piecewise", config.seed),
        ("edges", config.seed + 1),
        ("texture", config.seed + 2),
        ("smooth", config.seed + 3),
    ]

    all_results: list[dict] = []
    for kind, seed in samples:
        print(f"\n--- {kind} (seed={seed}) ---")
        all_results.extend(
            run_ablation(
                config=config,
                kind=kind,
                seed=seed,
                lambda_values=lambda_values,
                integrator=args.integrator,
            )
        )

    # Save JSON record and the frozen metric summary table.
    json_path = dirs["results"] / "ablation_results.json"
    jsonable = [
        {key: (float(value) if hasattr(value, "item") else value) for key, value in row.items()}
        for row in all_results
    ]
    json_path.write_text(
        json.dumps(jsonable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = save_metrics_csv(
        all_results,
        dirs["results"] / "ablation_metrics.csv",
        fieldnames=ABLATION_CSV_HEADER,
    )
    print(f"\nResults saved to {json_path}")
    print(f"Summary table saved to {csv_path}")

    # Print summary table.
    print(f"\n{'='*90}")
    print(f"{'Sample':<26} {'Lambda':>8} {'PSNR':>8} {'SSIM':>8} {'DataFid':>12} {'Energy':>12} {'Curv':>12}")
    print(f"{'='*90}")
    for row in all_results:
        print(
            f"{row['sample_name']:<26} {row['lambda_value']:>8.1f} "
            f"{float(row['psnr']):>8.4f} {float(row['ssim']):>8.4f} "
            f"{row['data_fidelity']:>12.6f} {row['total_energy']:>12.6f} {row['curvature']:>12.6f}"
        )
    print(f"{'='*90}")
    print("Ablation complete.")


if __name__ == "__main__":
    main()
