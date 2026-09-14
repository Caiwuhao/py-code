#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect  # new import
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from opln_pso.model import (
    build_geometry,
    cumulative_erf_target,
    domain_table,
    evaluate_jsa,
    paper_erf_initial_duties,
    pmf_from_duties,
)
from opln_pso.optimizer import PMFObjective, PSOConfig, optimize_duty_cycles

def parse_args() -> argparse.Namespace:
    import inspect
    from argparse import ArgumentParser
    from pathlib import Path
    
    from opln_pso.model import build_geometry
    from opln_pso.optimizer import PSOConfig

    # 1. 提取 model.py 中 build_geometry 的默认值
    sig_model = inspect.signature(build_geometry)
    p_model = sig_model.parameters
    
    # 2. 提取 optimizer.py 中 PSOConfig 的默认值
    sig_pso = inspect.signature(PSOConfig)
    p_pso = sig_pso.parameters

    parser = ArgumentParser(
        description="PSO duty-cycle design for a type-II OPLN source"
    )
    
    # ---------------- 物理模型参数 (绑定 model.py) ----------------
    parser.add_argument("--pump-um", type=float, default=p_model["pump_wavelength_um"].default)
    parser.add_argument("--signal-um", type=float, default=p_model["signal_wavelength_um"].default)
    parser.add_argument("--pulse-fs", type=float, default=p_model["pump_intensity_fwhm_fs"].default, help="pump intensity FWHM")
    parser.add_argument(
        "--pump-fwhm-nm",
        type=float,
        default=p_model["pump_intensity_fwhm_nm"].default,
        help="measured pump intensity FWHM; when given, overrides --pulse-fs",
    )
    parser.add_argument("--alpha", type=float, default=p_model["alpha"].default, help="L/sigma_z if length is automatic")
    parser.add_argument("--crystal-length-mm", type=float, default=p_model["crystal_length_mm"].default)
    parser.add_argument("--target-sigma-mm", type=float, default=p_model["target_sigma_z_mm"].default)
    parser.add_argument("--min-domain-um", type=float, default=p_model["min_domain_um"].default)
    
    # ---------------- PSO 优化参数 (绑定 optimizer.py) ----------------
    parser.add_argument("--particles", type=int, default=p_pso["particles"].default)
    parser.add_argument("--iterations", type=int, default=p_pso["iterations"].default)
    parser.add_argument("--seed", type=int, default=p_pso["seed"].default)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=p_pso["device"].default)
    parser.add_argument("--wavelength-chunk", type=int, default=p_pso["wavelength_chunk"].default)
    
    # ---------------- 其他固定/无默认值的参数 ----------------
    parser.add_argument(
        "--objective",
        choices=["legacy_intensity", "complex_amplitude"],
        default="complex_amplitude",
    )
    parser.add_argument("--q-points", type=int, default=801)
    parser.add_argument("--q-span-sigma", type=float, default=15.0)
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="optimize each period independently instead of enforcing A[j]+A[M-1-j]=1",
    )
    parser.add_argument("--jsa-grid", type=int, default=140)
    parser.add_argument(
        "--purity-spans",
        default="4.5,6,8",
        help="comma-separated JSA half-widths in pump sigma for convergence checks",
    )
    parser.add_argument("--skip-jsa", action="store_true")
    
    # ---------------- 动态文件夹命名 ----------------
    signal_val = p_model["signal_wavelength_um"].default
    parser.add_argument("--output", type=Path, default=Path(f"output/run_{int(signal_val * 1000)}nm"))
    
    return parser.parse_args()


'''def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSO duty-cycle design for a 1560 nm -> 3120 nm type-II OPLN source"
    )
    sig = inspect.signature(build_geometry)
    default_pump_um = sig.parameters['pump_wavelength_um'].default
    parser.add_argument("--pump-um", type=float, default=1.560)
    parser.add_argument("--signal-um", type=float, default=3.120)
    parser.add_argument("--pulse-fs", type=float, default=60.0, help="pump intensity FWHM")
    parser.add_argument(
        "--pump-fwhm-nm",
        type=float,
        default=None,
        help="measured pump intensity FWHM; when given, overrides --pulse-fs",
    )
    parser.add_argument("--alpha", type=float, default=5.0, help="L/sigma_z if length is automatic")
    parser.add_argument("--crystal-length-mm", type=float, default=None)
    parser.add_argument("--target-sigma-mm", type=float, default=None)
    parser.add_argument("--min-domain-um", type=float, default=0.500)
    parser.add_argument(
        "--objective",
        choices=["legacy_intensity", "complex_amplitude"],
        default="complex_amplitude",
    )
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--q-points", type=int, default=801)
    parser.add_argument("--q-span-sigma", type=float, default=15.0)
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="optimize each period independently instead of enforcing A[j]+A[M-1-j]=1",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--wavelength-chunk", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jsa-grid", type=int, default=140)
    parser.add_argument(
        "--purity-spans",
        default="4.5,6,8",
        help="comma-separated JSA half-widths in pump sigma for convergence checks",
    )
    parser.add_argument("--skip-jsa", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/run_3120nm"))
    return parser.parse_args()
'''

def save_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = build_geometry(
        pump_wavelength_um=args.pump_um,
        signal_wavelength_um=args.signal_um,
        pump_intensity_fwhm_fs=args.pulse_fs,
        pump_intensity_fwhm_nm=args.pump_fwhm_nm,
        alpha=args.alpha,
        crystal_length_mm=args.crystal_length_mm,
        target_sigma_z_mm=args.target_sigma_mm,
        min_domain_um=args.min_domain_um,
    )
    objective = PMFObjective(
        geometry,
        q_points=args.q_points,
        q_span_sigma=args.q_span_sigma,
        objective=args.objective,
        device=args.device,
        wavelength_chunk=args.wavelength_chunk,
    )
    config = PSOConfig(
        particles=args.particles,
        iterations=args.iterations,
        seed=args.seed,
        symmetric=not args.no_symmetry,
        device=args.device,
        wavelength_chunk=args.wavelength_chunk,
    )
    initial = np.clip(
        paper_erf_initial_duties(geometry.number_periods),
        geometry.duty_min,
        geometry.duty_max,
    )
    periodic = np.full(geometry.number_periods, 0.5)

    print(json.dumps(geometry.to_dict(), indent=2))
    result = optimize_duty_cycles(
        objective,
        config,
        initial_duties=initial,
        callback=lambda iteration, cost: print(f"iteration={iteration:4d} best_cost={cost:.6e}"),
    )
    optimized = result["duties"]

    mismatch = 2.0 * np.pi / geometry.qpm_period_um + objective.q_rad_um
    target_amp = np.exp(-0.5 * (objective.q_rad_um * geometry.target_sigma_z_um) ** 2)
    pmf_periodic = pmf_from_duties(periodic, geometry.qpm_period_um, mismatch)
    pmf_initial = pmf_from_duties(initial, geometry.qpm_period_um, mismatch)
    pmf_optimized = pmf_from_duties(optimized, geometry.qpm_period_um, mismatch)

    center_index = objective.q_rad_um.size // 2
    peak_amplitudes = {
        "periodic": float(abs(pmf_periodic[center_index])),
        "erf_initial": float(abs(pmf_initial[center_index])),
        "optimized": float(abs(pmf_optimized[center_index])),
        "ideal_truncated_gaussian": float(objective.target_peak_amplitude),
    }
    periodic_peak = 2.0 / np.pi

    summary = {
        "geometry": geometry.to_dict(),
        "optimization": {
            "objective": args.objective,
            "particles": args.particles,
            "iterations": args.iterations,
            "symmetric": not args.no_symmetry,
            "device": result["device"],
            "best_cost": result["cost"],
            "seed": args.seed,
        },
        "constraints": {
            "minimum_optimized_domain_um": float(
                np.min(np.minimum(optimized, 1.0 - optimized) * geometry.qpm_period_um)
            ),
            "maximum_optimized_domain_um": float(
                np.max(np.maximum(optimized, 1.0 - optimized) * geometry.qpm_period_um)
            ),
        },
        "peak_amplitude": peak_amplitudes,
        "relative_peak_efficiency": {
            name: float((amplitude / periodic_peak) ** 2)
            for name, amplitude in peak_amplitudes.items()
        },
    }

    jsa_results = {}
    if not args.skip_jsa:
        purity_spans = sorted({float(value) for value in args.purity_spans.split(",")})
        if not purity_spans or purity_spans[0] <= 0:
            raise ValueError("--purity-spans must contain positive numbers")
        plot_span = purity_spans[0]
        for name, duties in {
            "periodic": periodic,
            "erf_initial": initial,
            "optimized": optimized,
        }.items():
            jsa_results[name] = evaluate_jsa(
                duties,
                geometry,
                grid_size=args.jsa_grid,
                span_sigma=plot_span,
            )
            summary.setdefault("purity", {})[name] = jsa_results[name]["purity"]
        summary["jsa_evaluation"] = {
            "plot_span_sigma": plot_span,
            "signal_wavelength_range_nm": [
                float(np.min(jsa_results["optimized"]["signal_wavelengths_nm"])),
                float(np.max(jsa_results["optimized"]["signal_wavelengths_nm"])),
            ],
            "idler_wavelength_range_nm": [
                float(np.min(jsa_results["optimized"]["idler_wavelengths_nm"])),
                float(np.max(jsa_results["optimized"]["idler_wavelengths_nm"])),
            ],
            "optimized_purity_by_span_sigma": {str(plot_span): summary["purity"]["optimized"]},
        }
        for span in purity_spans[1:]:
            extended = evaluate_jsa(
                optimized,
                geometry,
                grid_size=args.jsa_grid,
                span_sigma=span,
            )
            summary["jsa_evaluation"]["optimized_purity_by_span_sigma"][str(span)] = extended[
                "purity"
            ]

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "result_arrays.npz",
        periodic_duties=periodic,
        initial_duties=initial,
        optimized_duties=optimized,
        history=result["history"],
        q_rad_um=objective.q_rad_um,
        target_amplitude=target_amp,
        pmf_periodic=pmf_periodic,
        pmf_initial=pmf_initial,
        pmf_optimized=pmf_optimized,
    )
    save_csv(
        args.output / "duty_cycles.csv",
        ["period_index", "position_mm", "duty_cycle"],
        np.column_stack(
            [
                np.arange(geometry.number_periods),
                np.arange(geometry.number_periods) * geometry.qpm_period_um / 1e3,
                optimized,
            ]
        ),
    )
    save_csv(
        args.output / "domains.csv",
        ["domain_index", "orientation", "z_start_um", "z_end_um", "length_um"],
        domain_table(optimized, geometry.qpm_period_um),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    position_mm = np.arange(geometry.number_periods) * geometry.qpm_period_um / 1e3
    axes[0, 0].plot(position_mm, initial, label="erf initialization", lw=1.2)
    axes[0, 0].plot(position_mm, optimized, label="PSO", lw=1.0)
    axes[0, 0].axhline(geometry.duty_min, color="k", ls=":", lw=0.8)
    axes[0, 0].axhline(geometry.duty_max, color="k", ls=":", lw=0.8)
    axes[0, 0].set(xlabel="Crystal position (mm)", ylabel="Duty cycle", ylim=(0, 1))
    axes[0, 0].legend()

    for pmf, label in [
        (pmf_periodic, "periodic"),
        (pmf_initial, "erf initialization"),
        (pmf_optimized, "PSO"),
    ]:
        intensity = np.abs(pmf) ** 2
        axes[0, 1].plot(
            objective.q_rad_um * 1e3,
            intensity / max(float(intensity.max()), 1e-18),
            label=label,
        )
    axes[0, 1].plot(
        objective.q_rad_um * 1e3,
        target_amp**2,
        "k--",
        label="Gaussian target",
    )
    axes[0, 1].set(xlabel=r"Residual mismatch $\delta k$ (rad/mm)", ylabel="Normalized PMF intensity")
    axes[0, 1].legend()

    axes[1, 0].semilogy(np.maximum(result["history"], 1e-16))
    axes[1, 0].set(xlabel="PSO iteration", ylabel="Best cost")

    z = np.linspace(0.0, geometry.crystal_length_um, 500)
    axes[1, 1].plot(
        z / 1e3,
        cumulative_erf_target(z, geometry.crystal_length_um, geometry.target_sigma_z_um),
    )
    axes[1, 1].set(
        xlabel="Crystal position (mm)",
        ylabel="Normalized cumulative target",
        title="Equivalent erf target for orientation design",
    )
    fig.savefig(args.output / "optimization_overview.png", dpi=180)
    plt.close(fig)

    if jsa_results:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, (name, values) in zip(axes, jsa_results.items()):
            intensity = np.abs(values["jsa"]) ** 2
            # Frequency is uniform but wavelength is not. Use the actual
            # wavelength coordinates instead of a linear imshow extent.
            axis.pcolormesh(
                values["idler_wavelengths_nm"][::-1],
                values["signal_wavelengths_nm"][::-1],
                intensity[::-1, ::-1],
                shading="auto",
                cmap="magma",
            )
            axis.set_title(f"{name}; P={values['purity']:.5f}")
            axis.set(xlabel="Idler wavelength (nm)", ylabel="Signal wavelength (nm)")
        fig.savefig(args.output / "jsa_comparison.png", dpi=180)
        plt.close(fig)

    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
