#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import scipy

from opln_pso.model import (
    JSAConfig,
    build_geometry,
    domain_table,
    evaluate_jsa_from_pmf,
    paper_erf_initial_duties,
    pmf_from_duties,
    scan_optimal_pump_bandwidth,
)
from opln_pso.optimizer import (
    PMFObjective,
    PSOConfig,
    optimize_duty_cycles,
    polish_duty_cycles_lbfgsb,
)


def _default(callable_object: Any, parameter: str) -> Any:
    """Read a child-module default instead of duplicating it here."""
    return inspect.signature(callable_object).parameters[parameter].default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSO duty-cycle design for a type-II OPLN source"
    )
    parser.add_argument(
        "--pump-um", type=float, default=_default(build_geometry, "pump_wavelength_um")
    )
    parser.add_argument(
        "--signal-um", type=float, default=_default(build_geometry, "signal_wavelength_um")
    )
    parser.add_argument(
        "--pulse-fs",
        type=float,
        default=_default(build_geometry, "pump_intensity_fwhm_fs"),
        help="initial pump intensity time FWHM used to estimate sigma_z",
    )
    parser.add_argument(
        "--pump-fwhm-nm",
        type=float,
        default=_default(build_geometry, "pump_intensity_fwhm_nm"),
        help="initial measured pump intensity FWHM; overrides --pulse-fs",
    )
    parser.add_argument("--alpha", type=float, default=_default(build_geometry, "alpha"))
    parser.add_argument(
        "--crystal-length-mm",
        type=float,
        default=_default(build_geometry, "crystal_length_mm"),
    )
    parser.add_argument(
        "--target-sigma-mm",
        type=float,
        default=_default(build_geometry, "target_sigma_z_mm"),
    )
    parser.add_argument(
        "--min-domain-um", type=float, default=_default(build_geometry, "min_domain_um")
    )

    parser.add_argument("--particles", type=int, default=_default(PSOConfig, "particles"))
    parser.add_argument("--iterations", type=int, default=_default(PSOConfig, "iterations"))
    parser.add_argument("--cycles", type=int, default=_default(PSOConfig, "cycles"))
    parser.add_argument(
        "--prior-pso-iterations",
        type=int,
        default=_default(PSOConfig, "prior_iterations"),
        help="iterations already represented by --initial-duties-npy",
    )
    parser.add_argument(
        "--initial-noise", type=float, default=_default(PSOConfig, "initial_noise")
    )
    parser.add_argument(
        "--noise-decay", type=float, default=_default(PSOConfig, "noise_decay")
    )
    parser.add_argument("--seed", type=int, default=_default(PSOConfig, "seed"))
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=_default(PSOConfig, "device"),
    )
    parser.add_argument(
        "--wavelength-chunk",
        type=int,
        default=_default(PSOConfig, "wavelength_chunk"),
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=_default(PSOConfig, "symmetric"),
    )
    parser.add_argument(
        "--polish-iterations",
        type=int,
        default=_default(PSOConfig, "polish_iterations"),
        help="optional CPU regression polish; 0 keeps the optimizer pure PSO",
    )

    parser.add_argument(
        "--objective",
        choices=["legacy_intensity", "complex_amplitude"],
        default=_default(PMFObjective, "objective"),
    )
    parser.add_argument(
        "--sampling",
        choices=["wavelength", "q"],
        default=_default(PMFObjective, "sampling"),
    )
    parser.add_argument("--q-points", type=int, default=_default(PMFObjective, "q_points"))
    parser.add_argument(
        "--q-span-sigma", type=float, default=_default(PMFObjective, "q_span_sigma")
    )
    parser.add_argument(
        "--optimization-span-nm",
        type=float,
        default=_default(PMFObjective, "wavelength_span_nm"),
    )

    parser.add_argument("--jsa-grid", type=int, default=_default(JSAConfig, "grid_size"))
    parser.add_argument(
        "--jsa-span-nm", type=float, default=_default(JSAConfig, "wavelength_span_nm")
    )
    parser.add_argument("--jsa-tick-nm", type=float, default=_default(JSAConfig, "tick_nm"))
    parser.add_argument(
        "--pump-scan-min-nm",
        type=float,
        default=_default(JSAConfig, "pump_scan_min_nm"),
    )
    parser.add_argument(
        "--pump-scan-max-nm",
        type=float,
        default=_default(JSAConfig, "pump_scan_max_nm"),
    )
    parser.add_argument(
        "--pump-coarse-step-nm",
        type=float,
        default=_default(JSAConfig, "pump_coarse_step_nm"),
    )
    parser.add_argument(
        "--pump-fine-halfspan-nm",
        type=float,
        default=_default(JSAConfig, "pump_fine_halfspan_nm"),
    )
    parser.add_argument(
        "--pump-fine-step-nm",
        type=float,
        default=_default(JSAConfig, "pump_fine_step_nm"),
    )
    parser.add_argument("--skip-jsa", action="store_true")
    parser.add_argument("--initial-duties-npy", type=Path)
    parser.add_argument("--prior-history-npy", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    signal_default = _default(build_geometry, "signal_wavelength_um")
    parser.add_argument(
        "--output", type=Path, default=Path(f"output/run_{int(signal_default * 1000)}nm")
    )
    return parser.parse_args()


def save_csv(path: Path, header: list[str], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def load_vector(path: Path | None, expected_size: int | None = None) -> np.ndarray:
    if path is None:
        return np.asarray([], dtype=float)
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    if expected_size is not None and values.size != expected_size:
        raise ValueError(f"{path} has {values.size} values; expected {expected_size}")
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"{path} must contain finite values")
    return values


def create_qute_colormap() -> mcolors.LinearSegmentedColormap:
    first = np.linspace(2.0 / 3.0, 1.0 / 12.0, 256)
    second = np.linspace(1.0 / 12.0, 0.0, 256)
    colors = [colorsys.hsv_to_rgb(value, 1.0, 1.0) for value in first]
    colors += [colorsys.hsv_to_rgb(value, 1.0, 1.0) for value in second]
    return mcolors.LinearSegmentedColormap.from_list("qute_hue_blend", colors)


def cuda_name(device: str) -> str | None:
    if device != "cuda":
        return None
    try:
        import cupy as cp

        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "CUDA device (name unavailable)"


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.prior_pso_iterations < 0 or args.polish_iterations < 0:
        raise ValueError("iteration counts cannot be negative")
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
    print(json.dumps(geometry.to_dict(), indent=2))
    objective = PMFObjective(
        geometry,
        q_points=args.q_points,
        q_span_sigma=args.q_span_sigma,
        objective=args.objective,
        sampling=args.sampling,
        wavelength_span_nm=args.optimization_span_nm,
        device=args.device,
        wavelength_chunk=args.wavelength_chunk,
    )
    gpu = cuda_name(objective.device)
    print(
        f"PSO backend resolved: cuda ({gpu})"
        if gpu
        else "PSO backend resolved: cpu"
    )
    config = PSOConfig(
        particles=args.particles,
        iterations=args.iterations,
        cycles=args.cycles,
        prior_iterations=args.prior_pso_iterations,
        seed=args.seed,
        initial_noise=args.initial_noise,
        noise_decay=args.noise_decay,
        symmetric=args.symmetric,
        device=args.device,
        wavelength_chunk=args.wavelength_chunk,
    )
    periodic = np.full(geometry.number_periods, 0.5)
    erf_initial = np.clip(
        paper_erf_initial_duties(geometry.number_periods),
        geometry.duty_min,
        geometry.duty_max,
    )
    checkpoint = load_vector(args.initial_duties_npy, geometry.number_periods)
    prior_history = load_vector(args.prior_history_npy)

    if args.evaluate_only:
        if checkpoint.size == 0:
            raise ValueError("--evaluate-only requires --initial-duties-npy")
        optimized = np.clip(checkpoint, geometry.duty_min, geometry.duty_max)
        cost = float(objective(optimized)[0])
        result: dict[str, Any] = {
            "duties": optimized,
            "cost": cost,
            "history": prior_history if prior_history.size else np.asarray([cost]),
            "device": objective.device,
            "symmetric": args.symmetric,
        }
    else:
        result = optimize_duty_cycles(
            objective,
            config,
            initial_duties=checkpoint if checkpoint.size else erf_initial,
            callback=None,
        )
        if prior_history.size:
            result["history"] = np.concatenate([prior_history, result["history"][1:]])
        optimized = result["duties"]
        if args.polish_iterations:
            polish = polish_duty_cycles_lbfgsb(
                objective, optimized, max_iterations=args.polish_iterations
            )
            optimized = polish["duties"]
            result["cost"] = polish["cost"]
            result["polish"] = polish

    structures = {
        "periodic": periodic,
        "erf_initial": erf_initial,
        "optimized": optimized,
    }
    target_amplitude = np.exp(
        -0.5 * (objective.q_rad_um * geometry.target_sigma_z_um) ** 2
    )
    pmf_slices = {
        name: pmf_from_duties(
            duties, geometry.qpm_period_um, objective.mismatch_rad_um
        )
        for name, duties in structures.items()
    }
    jsa_results: dict[str, dict[str, Any]] = {}
    if not args.skip_jsa:
        for name, duties in structures.items():
            jsa_results[name] = scan_optimal_pump_bandwidth(
                duties,
                geometry,
                grid_size=args.jsa_grid,
                wavelength_span_nm=args.jsa_span_nm,
                bandwidth_min_nm=args.pump_scan_min_nm,
                bandwidth_max_nm=args.pump_scan_max_nm,
                coarse_step_nm=args.pump_coarse_step_nm,
                fine_halfspan_nm=args.pump_fine_halfspan_nm,
                fine_step_nm=args.pump_fine_step_nm,
                pmf_mode="magnitude",
            )
            values = jsa_results[name]
            # Supplement only: never substitute this for the requested
            # Py_JSI_SPDC magnitude-only paper-comparison metric.
            values["complex_purity_at_same_pump"] = float(evaluate_jsa_from_pmf(
                values["pmf"], values["best_bandwidth_nm"],
                values["signal_grid_nm"], values["idler_grid_nm"],
                pmf_mode="complex",
            )["purity"])

    current_iterations = 0 if args.evaluate_only else args.iterations * args.cycles
    if args.evaluate_only:
        algorithm = (
            "particle_swarm_optimization_checkpoint_evaluation"
            if args.prior_pso_iterations
            else "external_checkpoint_evaluation"
        )
    elif args.polish_iterations:
        algorithm = (
            "particle_swarm_optimization_plus_lbfgsb_polish"
            if current_iterations
            else "lbfgsb_numerical_regression"
        )
    else:
        algorithm = "particle_swarm_optimization"
    minimum_domain = float(
        np.min(np.minimum(optimized, 1.0 - optimized) * geometry.qpm_period_um)
    )
    peak_amplitudes = {
        name: float(np.max(np.abs(values))) for name, values in pmf_slices.items()
    }
    summary: dict[str, Any] = {
        "run_arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "elapsed_before_saving_seconds": time.perf_counter() - started,
        },
        "geometry": geometry.to_dict(),
        "optimization": {
            "algorithm": algorithm,
            "objective": args.objective,
            "sampling": args.sampling,
            "sample_points": args.q_points,
            "optimization_wavelength_span_nm": args.optimization_span_nm,
            "particles": args.particles,
            "iterations_per_cycle": 0 if args.evaluate_only else args.iterations,
            "cycles": 0 if args.evaluate_only else args.cycles,
            "prior_pso_iterations": args.prior_pso_iterations,
            "current_pso_iterations": current_iterations,
            "total_pso_iterations": args.prior_pso_iterations + current_iterations,
            "initial_noise": args.initial_noise,
            "noise_decay": args.noise_decay,
            "symmetric": args.symmetric,
            "device_requested": args.device,
            "device_resolved": result["device"],
            "cuda_device_name": cuda_name(result["device"]),
            "best_cost": float(result["cost"]),
            "seed": args.seed,
            "history_points": int(result["history"].size),
        },
        "constraints": {
            "configured_min_domain_um": geometry.min_domain_um,
            "minimum_optimized_domain_um": minimum_domain,
            "constraint_satisfied": minimum_domain >= geometry.min_domain_um - 1e-10,
        },
        "peak_amplitude": peak_amplitudes,
        "relative_peak_efficiency": {
            name: float((amplitude / (2.0 / np.pi)) ** 2)
            for name, amplitude in peak_amplitudes.items()
        },
        "jsa_convention": {
            "grid": "uniform_wavelength",
            "grid_size": args.jsa_grid,
            "wavelength_span_nm": args.jsa_span_nm,
            "x_axis": "signal",
            "y_axis": "idler",
            "pmf_used_for_schmidt": "magnitude",
            "pump_scan": "Py_JSI_SPDC two-stage coarse/fine scan",
        },
    }
    if "polish" in result:
        summary["optimization"]["polish"] = {
            key: result["polish"][key]
            for key in ("device", "iterations", "evaluations", "success", "message")
        }
    if jsa_results:
        summary["structures"] = {
            name: {
                "optimized_pump_intensity_fwhm_nm": values["best_bandwidth_nm"],
                "purity": values["best_purity"],
                "complex_purity_at_same_pump": values["complex_purity_at_same_pump"],
                "pump_optimum_at_scan_boundary": bool(
                    np.isclose(values["best_bandwidth_nm"], args.pump_scan_min_nm)
                    or np.isclose(values["best_bandwidth_nm"], args.pump_scan_max_nm)
                ),
            }
            for name, values in jsa_results.items()
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.save(args.output / "optimized_duties.npy", optimized)
    np.save(args.output / "pso_history.npy", result["history"])
    np.savez_compressed(
        args.output / "result_arrays.npz",
        periodic_duties=periodic,
        initial_duties=erf_initial,
        optimized_duties=optimized,
        pso_history=result["history"],
        polish_history=(
            result["polish"]["local_history"]
            if "polish" in result
            else np.asarray([], dtype=float)
        ),
        residual_mismatch_rad_um=objective.q_rad_um,
        mismatch_rad_um=objective.mismatch_rad_um,
        target_amplitude=target_amplitude,
        pmf_periodic=pmf_slices["periodic"],
        pmf_initial=pmf_slices["erf_initial"],
        pmf_optimized=pmf_slices["optimized"],
    )
    save_csv(
        args.output / "duty_cycles.csv",
        ["period_index", "position_mm", "duty_cycle"],
        zip(
            np.arange(geometry.number_periods),
            np.arange(geometry.number_periods) * geometry.qpm_period_um / 1e3,
            optimized,
        ),
    )
    save_csv(
        args.output / "domains.csv",
        ["domain_index", "orientation", "z_start_um", "z_end_um", "length_um"],
        domain_table(optimized, geometry.qpm_period_um),
    )
    if jsa_results:
        jsa_arrays = {}
        rows = []
        for name, values in jsa_results.items():
            for key in ("best_jsa", "pmf", "pump", "signal_wavelengths_nm",
                        "idler_wavelengths_nm", "coarse_bandwidths_nm",
                        "coarse_purities", "fine_bandwidths_nm", "fine_purities"):
                jsa_arrays[f"{name}_{key}"] = values[key]
            rows.extend(
                zip(
                    [name] * len(values["fine_bandwidths_nm"]),
                    values["fine_bandwidths_nm"],
                    values["fine_purities"],
                )
            )
        save_csv(
            args.output / "pump_bandwidth_scans.csv",
            ["structure", "pump_intensity_fwhm_nm", "purity"],
            rows,
        )
        np.savez_compressed(args.output / "jsa_arrays.npz", **jsa_arrays)

    plot_overview(
        args, geometry, objective, erf_initial, optimized, pmf_slices,
        target_amplitude, result, jsa_results
    )
    if jsa_results:
        plot_jsas(args, jsa_results)
        for name, values in jsa_results.items():
            print(
                f"{name}: pump FWHM={values['best_bandwidth_nm']:.2f} nm, "
                f"purity={values['best_purity']:.8f}"
            )
    print(f"Saved results to {args.output.resolve()}")


def plot_overview(
    args: argparse.Namespace,
    geometry: Any,
    objective: PMFObjective,
    erf_initial: np.ndarray,
    optimized: np.ndarray,
    pmf_slices: dict[str, np.ndarray],
    target_amplitude: np.ndarray,
    result: dict[str, Any],
    jsa_results: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    position = np.arange(geometry.number_periods) * geometry.qpm_period_um / 1e3
    axes[0, 0].plot(position, erf_initial, label="erf initialization", lw=1.1)
    axes[0, 0].plot(position, optimized, label="optimized", lw=0.8)
    axes[0, 0].axhline(geometry.duty_min, color="k", ls=":", lw=0.8)
    axes[0, 0].axhline(geometry.duty_max, color="k", ls=":", lw=0.8)
    axes[0, 0].set(xlabel="Crystal position (mm)", ylabel="Duty cycle", ylim=(0, 1))
    axes[0, 0].legend()

    horizontal = (
        objective.signal_wavelengths_um * 1e3
        if objective.signal_wavelengths_um is not None
        else objective.q_rad_um * 1e3
    )
    horizontal_label = (
        "Signal wavelength (nm), energy-conserving slice"
        if objective.signal_wavelengths_um is not None
        else r"Residual mismatch $\delta k$ (rad/mm)"
    )
    for name, pmf in pmf_slices.items():
        intensity = np.abs(pmf) ** 2
        axes[0, 1].plot(horizontal, intensity / intensity.max(), label=name)
    axes[0, 1].plot(horizontal, target_amplitude**2, "k--", label="Gaussian target")
    axes[0, 1].set(xlabel=horizontal_label, ylabel="Normalized PMF intensity")
    axes[0, 1].legend()

    if result["history"].size > 1:
        axes[1, 0].semilogy(np.maximum(result["history"], 1e-18), label="PSO")
    if "polish" in result:
        axes[1, 0].semilogy(
            np.maximum(result["polish"]["local_history"], 1e-18),
            label="L-BFGS-B regression",
        )
    if result["history"].size <= 1 and "polish" not in result:
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5, 0.5, "Checkpoint evaluation\n(history loaded separately)",
            ha="center", va="center", transform=axes[1, 0].transAxes
        )
    else:
        axes[1, 0].set(
            xlabel="Local function evaluation" if "polish" in result else "PSO iteration",
            ylabel="Best normalized MSE",
        )
        axes[1, 0].legend()

    if jsa_results:
        for name, values in jsa_results.items():
            axes[1, 1].plot(
                values["fine_bandwidths_nm"], values["fine_purities"], label=name
            )
        axes[1, 1].set(
            xlabel="Pump intensity FWHM (nm)", ylabel="Schmidt purity"
        )
        axes[1, 1].legend()
    else:
        axes[1, 1].axis("off")
    fig.savefig(args.output / "optimization_overview.png", dpi=180)
    plt.close(fig)


def plot_jsas(args: argparse.Namespace, values_by_name: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    image = None
    for axis, (name, values) in zip(axes, values_by_name.items()):
        intensity = np.abs(values["best_jsa"]) ** 2
        intensity /= max(float(intensity.max()), 1e-18)
        signal = values["signal_wavelengths_nm"]
        idler = values["idler_wavelengths_nm"]
        image = axis.imshow(
            intensity + 1e-15,
            extent=[signal[0], signal[-1], idler[0], idler[-1]],
            origin="lower",
            aspect="equal",
            cmap=create_qute_colormap(),
            norm=LogNorm(vmin=1e-4, vmax=1.0),
            interpolation="nearest",
        )
        axis.set_title(
            f"{name}\nPump={values['best_bandwidth_nm']:.2f} nm, "
            f"P={values['best_purity']:.6f}"
        )
        axis.set(xlabel="Signal wavelength (nm)", ylabel="Idler wavelength (nm)")
        axis.xaxis.set_major_locator(MultipleLocator(args.jsa_tick_nm))
        axis.yaxis.set_major_locator(MultipleLocator(args.jsa_tick_nm))
        axis.set_xlim(signal[0], signal[-1])
        axis.set_ylim(idler[0], idler[-1])
    if image is not None:
        fig.colorbar(image, ax=axes, label="Normalized JSI", shrink=0.86)
    fig.savefig(args.output / "jsa_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
