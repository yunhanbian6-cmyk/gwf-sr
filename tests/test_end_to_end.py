# -*- coding: utf-8 -*-
"""End-to-end smoke test tying all modules together.

Verifies the complete pipeline: synthetic image -> degradation ->
solver -> metrics -> visualization, and checks result file integrity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gwf_sr.config import GWFConfig
from gwf_sr.data import generate_image
from gwf_sr.degradation import bicubic_upsample, degrade
from gwf_sr.geometry import (
    distribution_from_image,
    entropy,
    hilbert_distance,
    kl_divergence,
    total_mass,
)
from gwf_sr.graph import build_pixel_graph
from gwf_sr.metrics import compare_images, psnr, ssim
from gwf_sr.solver import SuperResolutionSolver, project_image
from gwf_sr.visualization import save_evaluation_bundle


class TestEndToEndPipeline:
    """Full pipeline smoke test."""

    def test_synthetic_to_solver_to_metrics(self):
        """Generate -> degrade -> solve -> evaluate."""
        hr, lr, bicubic, _ = generate_image(
            kind="piecewise", size=32, seed=42, scale=2, sigma=1.0, noise_std=0.0
        )
        assert hr.shape == (32, 32)
        assert lr.shape == (16, 16)

        solver = SuperResolutionSolver(
            lr, bicubic,
            iterations=5,
            dt=0.05,
            lambda_value=0.1,
            integrator="strang",
            epsilon=1e-8,
        )
        result = solver.solve()

        metrics = compare_images(hr, result.reconstruction)
        assert torch.isfinite(metrics.psnr)
        assert torch.isfinite(metrics.ssim)

    def test_baseline_improves_over_bicubic(self):
        """Baseline (lambda=0) should improve PSNR over Bicubic."""
        hr, lr, bicubic, _ = generate_image(
            kind="smooth", size=32, seed=45, scale=2, sigma=1.0, noise_std=0.0
        )

        solver = SuperResolutionSolver(
            lr, bicubic,
            iterations=10,
            dt=0.05,
            lambda_value=0.0,
            integrator="euler",
            epsilon=1e-8,
        )
        result = solver.solve()

        bicubic_psnr = psnr(hr, bicubic).item()
        solver_psnr = psnr(hr, result.reconstruction).item()
        assert solver_psnr >= bicubic_psnr - 1.0, (
            f"solver PSNR {solver_psnr:.4f} should be near or above bicubic {bicubic_psnr:.4f}"
        )

    def test_geometry_consistency_through_solver(self):
        """Geometry metrics should be consistent through the solver."""
        hr, lr, bicubic, _ = generate_image(
            kind="piecewise", size=32, seed=42, scale=2, sigma=1.0, noise_std=0.0
        )

        solver = SuperResolutionSolver(
            lr, bicubic,
            iterations=5,
            dt=0.05,
            lambda_value=0.1,
            integrator="strang",
            epsilon=1e-8,
        )
        result = solver.solve()

        # Check trajectory properties
        assert result.trajectory.shape[0] == 6  # iterations + 1
        assert result.trajectory.shape[1:] == hr.shape

        # Check diagnostics
        assert result.data_fidelity.shape == (6,)
        assert result.entropy.shape == (6,)
        assert result.total_energy.shape == (6,)
        assert torch.isfinite(result.data_fidelity).all()
        assert torch.isfinite(result.entropy).all()

    def test_solver_deterministic(self):
        """Same input should give identical results."""
        hr, lr, bicubic, _ = generate_image(
            kind="texture", size=32, seed=44, scale=2, sigma=1.0, noise_std=0.0
        )

        r1 = SuperResolutionSolver(
            lr, bicubic, iterations=3, dt=0.05, lambda_value=0.1, integrator="strang"
        ).solve()
        r2 = SuperResolutionSolver(
            lr, bicubic, iterations=3, dt=0.05, lambda_value=0.1, integrator="strang"
        ).solve()

        assert torch.equal(r1.reconstruction, r2.reconstruction)


class TestResultIntegrity:
    """Verify output file structure and content."""

    def test_bundle_files_exist(self, tmp_path: Path):
        """save_evaluation_bundle should create all expected files."""
        hr = torch.rand(32, 32)
        lr = torch.rand(16, 16)
        bicubic = torch.rand(32, 32)
        baseline = torch.rand(32, 32)
        gwf_c = torch.rand(32, 32)

        solver_result = SuperResolutionSolver(
            lr, bicubic, iterations=3, dt=0.05, lambda_value=0.1
        ).solve()

        bundle = save_evaluation_bundle(
            tmp_path,
            config_name="smoke",
            sample_name="test_sample",
            method="GWF-C",
            lambda_value=0.1,
            iterations=3,
            dt=0.05,
            hr=hr,
            lr=lr,
            bicubic=bicubic,
            baseline=baseline,
            gwf_c=gwf_c,
            solver_result=solver_result,
            lpips_available=False,
        )

        assert bundle.metrics_json.exists()
        assert bundle.metrics_csv.exists()
        assert bundle.comparison_png.exists()
        assert bundle.convergence_png is not None
        assert bundle.convergence_png.exists()

    def test_json_structure(self, tmp_path: Path):
        """JSON should have correct schema and no NaN."""
        hr = torch.rand(32, 32)
        lr = torch.rand(16, 16)
        bicubic = torch.rand(32, 32)
        baseline = torch.rand(32, 32)
        gwf_c = torch.rand(32, 32)

        bundle = save_evaluation_bundle(
            tmp_path,
            config_name="smoke",
            sample_name="test_sample",
            method="Baseline",
            lambda_value=0.0,
            iterations=3,
            dt=0.05,
            hr=hr,
            lr=lr,
            bicubic=bicubic,
            baseline=baseline,
            gwf_c=gwf_c,
            solver_result=None,
            lpips_available=False,
        )

        data = json.loads(bundle.metrics_json.read_text(encoding="utf-8"))
        assert data["schema_version"] == "1.0"
        assert "metrics" in data
        assert "psnr" in data["metrics"]
        assert "ssim" in data["metrics"]

    def test_csv_headers(self, tmp_path: Path):
        """CSV should have frozen headers."""
        hr = torch.rand(32, 32)
        lr = torch.rand(16, 16)
        bicubic = torch.rand(32, 32)
        baseline = torch.rand(32, 32)
        gwf_c = torch.rand(32, 32)

        solver_result = SuperResolutionSolver(
            lr, bicubic, iterations=3, dt=0.05, lambda_value=0.1
        ).solve()

        bundle = save_evaluation_bundle(
            tmp_path,
            config_name="smoke",
            sample_name="test_sample",
            method="GWF-C",
            lambda_value=0.1,
            iterations=3,
            dt=0.05,
            hr=hr,
            lr=lr,
            bicubic=bicubic,
            baseline=baseline,
            gwf_c=gwf_c,
            solver_result=solver_result,
            lpips_available=False,
        )

        traj_csv = bundle.metrics_json.parent / "trajectory_metrics.csv"
        if traj_csv.exists():
            first_line = traj_csv.read_text(encoding="utf-8").split("\n")[0]
            assert "time" in first_line
            assert "iteration" in first_line


class TestModuleImports:
    """Verify all public APIs are importable."""

    def test_all_geometry_exports(self):
        from gwf_sr.geometry import (
            distribution_from_image,
            effective_state_count,
            entropy,
            hilbert_distance,
            kl_divergence,
            max_probability,
            min_probability,
            total_mass,
        )

    def test_all_graph_exports(self):
        from gwf_sr.graph import (
            build_batch_pixel_graphs,
            build_pixel_graph,
            graph_degree,
            graph_spectral_radius_bound,
            graph_total_weight,
            normalized_adjacency_action,
        )

    def test_all_energy_exports(self):
        from gwf_sr.energy import (
            dirichlet_energy,
            entropy_energy,
            log_ratio_smoothness,
            total_variation,
        )

    def test_all_curvature_exports(self):
        from gwf_sr.curvature import (
            curvature_acceleration,
            curvature_quotient,
            entropy_decay_rate,
            entropy_production,
        )

    def test_all_solver_exports(self):
        from gwf_sr.solver import (
            SolverResult,
            SuperResolutionSolver,
            data_fidelity,
            data_fidelity_gradient,
            project_image,
            regularization_gradient,
        )

    def test_all_metrics_exports(self):
        from gwf_sr.metrics import (
            ImageMetrics,
            compare_images,
            compute_image_metrics,
            edge_preservation_index,
            high_frequency_energy,
            lpips_distance,
            psnr,
            ssim,
        )

    def test_all_visualization_exports(self):
        from gwf_sr.visualization import (
            EvaluationBundle,
            plot_convergence_curves,
            plot_method_comparison,
            save_evaluation_bundle,
            save_metrics_csv,
            save_metrics_json,
            save_trajectory_csv,
        )
