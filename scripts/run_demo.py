# -*- coding: utf-8 -*-
"""End-to-end GWF super-resolution demo.

Generates synthetic images, applies degradation, runs Baseline and GWF-C
solvers, computes metrics, and saves visualizations.

Usage:
    python scripts/run_demo.py --config configs/smoke.yaml --root .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwf_sr.config import GWFConfig, load_config, resolve_runtime_dirs
from gwf_sr.data import generate_image
from gwf_sr.metrics import compare_images, lpips_available
from gwf_sr.solver import SuperResolutionSolver
from gwf_sr.visualization import save_evaluation_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML experiment configuration.")
    parser.add_argument("--root", default=".", help="Root directory for results.")
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=1.0,
        help="Curvature regularization used by the GWF-C solver.",
    )
    parser.add_argument(
        "--integrator",
        default="strang",
        choices=("euler", "euler_maruyama", "strang"),
        help="Time integrator used by both solvers.",
    )
    return parser.parse_args()


def run_one_sample(
    config: GWFConfig,
    kind: str,
    seed: int,
    lambda_value: float,
    integrator: str,
    output_root: Path,
) -> dict:
    """Run one end-to-end sample and return summary metrics."""
    print(f"\n{'='*60}")
    print(f"Sample: {kind}  seed={seed}  size={config.image_size}  lambda={lambda_value}")
    print(f"{'='*60}")

    # 1. Generate HR and LR with the frozen stage-two pipeline.
    hr, lr, bicubic, _ = generate_image(
        kind=kind,
        size=config.image_size,
        seed=seed,
        scale=config.scale,
        sigma=config.sigma,
        noise_std=config.noise_std,
    )
    print(f"  HR shape: {tuple(hr.shape)}, LR shape: {tuple(lr.shape)}")

    solver_kwargs = dict(
        scale=config.scale,
        sigma=config.sigma,
        iterations=config.iterations,
        dt=config.dt,
        epsilon=config.epsilon,
        edge_weight_mode=config.edge_weight_mode,
        integrator=integrator,
        seed=config.seed,
    )

    # 2. Baseline: lambda=0 (data term only).
    print("  Running Baseline (lambda=0)...")
    baseline_solver = SuperResolutionSolver(
        lr, bicubic, lambda_value=0.0, **solver_kwargs
    )
    baseline_result = baseline_solver.solve()
    print(f"  Baseline done. Final data_fidelity: {baseline_result.data_fidelity[-1].item():.6f}")

    # 3. GWF-C: lambda>0.
    print(f"  Running GWF-C (lambda={lambda_value})...")
    gwf_solver = SuperResolutionSolver(
        lr, bicubic, lambda_value=lambda_value, **solver_kwargs
    )
    gwf_result = gwf_solver.solve()
    print(f"  GWF-C done. Final data_fidelity: {gwf_result.data_fidelity[-1].item():.6f}")

    # 4. Compute stage-six metrics.
    print("  Computing metrics...")
    baseline_metrics = compare_images(hr, baseline_result.reconstruction, require_lpips=False)
    gwf_metrics = compare_images(hr, gwf_result.reconstruction, require_lpips=False)
    bicubic_metrics = compare_images(hr, bicubic, require_lpips=False)

    print(f"  Bicubic  PSNR={bicubic_metrics.psnr.item():.4f}  SSIM={bicubic_metrics.ssim.item():.4f}")
    print(f"  Baseline PSNR={baseline_metrics.psnr.item():.4f}  SSIM={baseline_metrics.ssim.item():.4f}")
    print(f"  GWF-C    PSNR={gwf_metrics.psnr.item():.4f}  SSIM={gwf_metrics.ssim.item():.4f}")

    # 5. Save the frozen stage-six evaluation bundles.
    metrics_available = lpips_available()
    for method_name, result in (
        ("Baseline", baseline_result),
        ("GWF-C", gwf_result),
    ):
        bundle = save_evaluation_bundle(
            output_root,
            config_name=config.config_name,
            sample_name=f"{kind}_{config.image_size}_seed{seed}",
            method=method_name,
            lambda_value=0.0 if method_name == "Baseline" else lambda_value,
            iterations=result.iterations,
            dt=result.dt,
            hr=hr,
            lr=lr,
            bicubic=bicubic,
            baseline=baseline_result.reconstruction,
            gwf_c=gwf_result.reconstruction,
            solver_result=result,
            lpips_available=metrics_available,
            require_lpips=False,
        )
        print(f"  Saved {method_name} bundle to {bundle.metrics_json}")

    return {
        "kind": kind,
        "seed": seed,
        "size": config.image_size,
        "lambda_value": lambda_value,
        "bicubic_psnr": bicubic_metrics.psnr.item(),
        "bicubic_ssim": bicubic_metrics.ssim.item(),
        "baseline_psnr": baseline_metrics.psnr.item(),
        "baseline_ssim": baseline_metrics.ssim.item(),
        "gwf_psnr": gwf_metrics.psnr.item(),
        "gwf_ssim": gwf_metrics.ssim.item(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(args.root)
    dirs = resolve_runtime_dirs(config, root=root)
    output_root = root / config.output_dir

    print("GWF Super-Resolution End-to-End Demo")
    print(f"Output directory: {output_root}")

    samples = [
        ("piecewise", config.seed),
        ("edges", config.seed + 1),
        ("texture", config.seed + 2),
        ("smooth", config.seed + 3),
    ]

    summaries = []
    for kind, seed in samples:
        summary = run_one_sample(
            config=config,
            kind=kind,
            seed=seed,
            lambda_value=args.lambda_value,
            integrator=args.integrator,
            output_root=output_root,
        )
        summaries.append(summary)

    # Save summary next to the other runtime artifacts.
    summary_path = dirs["results"] / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved to {summary_path}")

    # Print summary table.
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
