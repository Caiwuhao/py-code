from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from .model import (
    DesignGeometry,
    energy_conserving_idler,
    material_mismatch,
    paper_erf_initial_duties,
)


def _load_backend(device: str):
    normalized = device.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if normalized == "cpu":
        return np, "cpu"
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp, "cuda"
        if normalized == "cuda":
            raise RuntimeError("CuPy is installed, but no CUDA device was found")
    except Exception as exc:
        if normalized == "cuda":
            raise RuntimeError(
                "CUDA was requested, but CuPy or a working CUDA device is unavailable"
            ) from exc
    return np, "cpu"


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        import cupy as cp

        return cp.asnumpy(array)
    except ImportError:
        return np.asarray(array)


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 48
    iterations: int = 200
    cycles: int = 1
    prior_iterations: int = 0
    seed: int = 7
    inertia_start: float = 0.90
    inertia_end: float = 0.40
    cognitive: float = 1.49618
    social: float = 1.49618
    initial_noise: float = 0.06
    noise_decay: float = 0.85
    symmetric: bool = False
    device: str = "auto"
    wavelength_chunk: int = 96
    log_every: int = 0
    polish_iterations: int = 0


class PMFObjective:
    """Batched PMF objective for NumPy or CuPy.

    ``legacy_intensity`` reproduces the central limitation of the MATLAB cost:
    it compares normalized PMF intensity with a Gaussian and ignores phase.
    ``complex_amplitude`` removes the unavoidable linear propagation phase and
    compares the complex PMF and its absolute peak with the Gaussian target.
    """

    def __init__(
        self,
        geometry: DesignGeometry,
        q_points: int = 601,
        q_span_sigma: float = 15.0,
        objective: str = "legacy_intensity",
        sampling: str = "wavelength",
        wavelength_span_nm: float = 60.0,
        device: str = "auto",
        wavelength_chunk: int = 96,
    ):
        if q_points < 31 or q_points % 2 == 0:
            raise ValueError("q_points must be an odd integer >= 31")
        if objective not in {"legacy_intensity", "complex_amplitude"}:
            raise ValueError("unknown objective")
        if sampling not in {"wavelength", "q"}:
            raise ValueError("sampling must be 'wavelength' or 'q'")
        if wavelength_span_nm <= 0:
            raise ValueError("wavelength_span_nm must be positive")
        self.geometry = geometry
        self.objective = objective
        self.sampling = sampling
        self.wavelength_span_nm = wavelength_span_nm
        self.xp, self.device = _load_backend(device)
        self.wavelength_chunk = wavelength_chunk
        center_mismatch = float(
            material_mismatch(
                geometry.pump_wavelength_um,
                geometry.signal_wavelength_um,
                geometry.idler_wavelength_um,
                geometry.pump_polarization,
                geometry.signal_polarization,
                geometry.idler_polarization,
            )
        )
        if sampling == "wavelength":
            half_span_um = wavelength_span_nm * 0.5e-3
            self.signal_wavelengths_um = np.linspace(
                geometry.signal_wavelength_um - half_span_um,
                geometry.signal_wavelength_um + half_span_um,
                q_points,
            )
            idler_um = energy_conserving_idler(
                geometry.pump_wavelength_um, self.signal_wavelengths_um
            )
            mismatch_cpu = material_mismatch(
                geometry.pump_wavelength_um,
                self.signal_wavelengths_um,
                idler_um,
                geometry.pump_polarization,
                geometry.signal_polarization,
                geometry.idler_polarization,
            )
            q_cpu = np.asarray(mismatch_cpu - center_mismatch, dtype=float)
        else:
            self.signal_wavelengths_um = None
            q_cpu = np.linspace(
                -q_span_sigma / geometry.target_sigma_z_um,
                q_span_sigma / geometry.target_sigma_z_um,
                q_points,
            )
            mismatch_cpu = center_mismatch + q_cpu
        self.q_rad_um = q_cpu
        self.center_mismatch_rad_um = center_mismatch
        self.mismatch_rad_um = np.asarray(mismatch_cpu, dtype=float)
        self.mismatch = self.xp.asarray(self.mismatch_rad_um, dtype=self.xp.float64)
        self.target_amplitude = self.xp.exp(
            -0.5 * (self.xp.asarray(q_cpu) * geometry.target_sigma_z_um) ** 2
        )
        alpha = geometry.crystal_length_um / geometry.target_sigma_z_um
        truncation = math.erf(alpha / (2.0 * np.sqrt(2.0)))
        self.target_peak_amplitude = (
            np.sqrt(8.0 / np.pi)
            * geometry.target_sigma_z_um
            / geometry.crystal_length_um
            * truncation
        )
        self.target_intensity = self.target_amplitude**2
        self._period_index = self.xp.arange(geometry.number_periods, dtype=self.xp.float64)
        self._fixed = self._compute_fixed_terms()
        self._linear_dephase = self.xp.exp(
            1j * self.xp.asarray(q_cpu) * geometry.crystal_length_um / 2.0
        )
        self._center_index = int(np.argmin(np.abs(q_cpu)))

    def _compute_fixed_terms(self):
        period = self.geometry.qpm_period_um
        z0 = self._period_index[:, None] * period
        z1 = (self._period_index[:, None] + 1.0) * period
        return (
            self.xp.exp(-1j * z0 * self.mismatch[None, :])
            + self.xp.exp(-1j * z1 * self.mismatch[None, :])
        ).sum(axis=0)

    def pmf_batch(self, duties):
        xp = self.xp
        duty = xp.asarray(duties, dtype=xp.float64)
        if duty.ndim == 1:
            duty = duty[None, :]
        if duty.shape[1] != self.geometry.number_periods:
            raise ValueError("duty array has the wrong number of periods")
        positions = (self._period_index[None, :] + duty) * self.geometry.qpm_period_um
        result = xp.empty((duty.shape[0], self.mismatch.size), dtype=xp.complex128)
        for start in range(0, self.mismatch.size, self.wavelength_chunk):
            stop = min(start + self.wavelength_chunk, self.mismatch.size)
            k_chunk = self.mismatch[start:stop]
            variable = xp.exp(-1j * positions[:, :, None] * k_chunk[None, None, :]).sum(axis=1)
            numerator = self._fixed[start:stop][None, :] - 2.0 * variable
            result[:, start:stop] = numerator / (
                1j * k_chunk[None, :] * self.geometry.crystal_length_um
            )
        return result

    def __call__(self, duties):
        xp = self.xp
        phi = self.pmf_batch(duties)
        if self.objective == "legacy_intensity":
            intensity = xp.abs(phi) ** 2
            peak = xp.maximum(intensity.max(axis=1, keepdims=True), 1e-18)
            normalized = intensity / peak
            residual = normalized - self.target_intensity[None, :]
            return xp.mean(residual**2, axis=1)

        dephased = phi * self._linear_dephase[None, :]
        center = dephased[:, self._center_index]
        safe_magnitude = xp.maximum(xp.abs(center), 1e-12)
        # A constant phase is irrelevant. Rotate it away, but retain absolute
        # amplitude so the optimization also sees the brightness trade-off.
        phase_rotation = xp.conj(center) / safe_magnitude
        aligned = dephased * phase_rotation[:, None]
        target = self.target_peak_amplitude * self.target_amplitude[None, :]
        residual = aligned - target
        return xp.mean(xp.abs(residual) ** 2, axis=1) / xp.mean(xp.abs(target) ** 2)


def _expand_symmetric(xp, variables, number_periods: int):
    reverse_complement = 1.0 - variables[:, ::-1]
    if number_periods % 2 == 0:
        return xp.concatenate([variables, reverse_complement], axis=1)
    middle = xp.full((variables.shape[0], 1), 0.5, dtype=variables.dtype)
    return xp.concatenate([variables, middle, reverse_complement], axis=1)


def optimize_duty_cycles(
    objective: PMFObjective,
    config: PSOConfig,
    initial_duties: Any | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """Run bounded PSO and return the best manufacturable duty-cycle array."""
    xp = objective.xp
    geometry = objective.geometry
    rng = xp.random.RandomState(config.seed)
    if initial_duties is None:
        initial = paper_erf_initial_duties(geometry.number_periods)
    else:
        initial = np.asarray(initial_duties, dtype=float)
    initial = np.clip(initial, geometry.duty_min, geometry.duty_max)

    if config.symmetric:
        variable_count = geometry.number_periods // 2
        initial_variables = initial[:variable_count]
        expand = lambda values: _expand_symmetric(xp, values, geometry.number_periods)
    else:
        variable_count = geometry.number_periods
        initial_variables = initial
        expand = lambda values: values

    lower = geometry.duty_min
    upper = geometry.duty_max
    if config.particles < 1:
        raise ValueError("particles must be at least 1")
    if config.iterations < 0:
        raise ValueError("iterations cannot be negative")
    if config.cycles < 1:
        raise ValueError("cycles must be at least 1")
    if config.initial_noise < 0 or not 0 < config.noise_decay <= 1:
        raise ValueError("initial_noise must be nonnegative and noise_decay in (0, 1]")
    if config.symmetric and variable_count == 0:
        raise ValueError("symmetric optimization needs at least two periods")
    global_position = xp.asarray(initial_variables).copy()
    global_cost = float(_to_numpy(objective(expand(global_position[None, :]))[0]))
    history = [global_cost]
    if callback is not None:
        callback(0, global_cost)

    completed_iterations = 0
    for cycle in range(config.cycles):
        positions = rng.uniform(lower, upper, size=(config.particles, variable_count))
        local_count = max(1, int(0.75 * config.particles))
        local_noise = config.initial_noise * config.noise_decay**cycle
        positions[:local_count] = global_position[None, :] + local_noise * rng.normal(
            size=(local_count, variable_count)
        )
        positions = xp.clip(positions, lower, upper)
        positions[0] = global_position
        velocity_scale = max(local_noise, 1e-6)
        velocities = rng.uniform(
            -velocity_scale, velocity_scale, size=positions.shape
        )
        costs = objective(expand(positions))
        personal_positions = positions.copy()
        personal_costs = costs.copy()
        candidate_index = int(_to_numpy(xp.argmin(costs)))
        candidate_cost = float(_to_numpy(costs[candidate_index]))
        if candidate_cost < global_cost:
            global_cost = candidate_cost
            global_position = positions[candidate_index].copy()

        for iteration in range(1, config.iterations + 1):
            fraction = iteration / max(1, config.iterations)
            inertia = config.inertia_start + fraction * (
                config.inertia_end - config.inertia_start
            )
            r1 = rng.random_sample(positions.shape)
            r2 = rng.random_sample(positions.shape)
            velocities = (
                inertia * velocities
                + config.cognitive * r1 * (personal_positions - positions)
                + config.social * r2 * (global_position[None, :] - positions)
            )
            positions = positions + velocities
            out_of_bounds = (positions < lower) | (positions > upper)
            positions = xp.clip(positions, lower, upper)
            velocities = xp.where(out_of_bounds, 0.0, velocities)

            costs = objective(expand(positions))
            improved = costs < personal_costs
            personal_positions = xp.where(improved[:, None], positions, personal_positions)
            personal_costs = xp.where(improved, costs, personal_costs)
            candidate_index = int(_to_numpy(xp.argmin(personal_costs)))
            candidate_cost = float(_to_numpy(personal_costs[candidate_index]))
            if candidate_cost < global_cost:
                global_cost = candidate_cost
                global_position = personal_positions[candidate_index].copy()
            completed_iterations += 1
            history.append(global_cost)
            if callback is not None and (
                completed_iterations == config.cycles * config.iterations
                or (
                    config.log_every > 0
                    and completed_iterations % config.log_every == 0
                )
            ):
                callback(completed_iterations, global_cost)

    best_duties = expand(global_position[None, :])[0]
    return {
        "duties": _to_numpy(best_duties),
        "cost": global_cost,
        "history": np.asarray(history, dtype=float),
        "device": objective.device,
        "symmetric": config.symmetric,
    }


def polish_duty_cycles_lbfgsb(
    objective: PMFObjective,
    initial_duties: Any,
    max_iterations: int = 1000,
) -> dict[str, Any]:
    """Optional analytic-gradient local regression; pure PSO stays separate."""
    if objective.objective != "legacy_intensity":
        raise ValueError("L-BFGS-B polish currently supports legacy_intensity only")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    geometry = objective.geometry
    duties0 = np.asarray(initial_duties, dtype=float).reshape(-1)
    if duties0.size != geometry.number_periods:
        raise ValueError("initial duty array has the wrong number of periods")
    duties0 = np.clip(duties0, geometry.duty_min, geometry.duty_max)
    mismatch = np.asarray(objective.mismatch_rad_um, dtype=float)
    target = np.asarray(_to_numpy(objective.target_intensity), dtype=float)
    center_index = int(objective._center_index)
    period = geometry.qpm_period_um
    length = geometry.crystal_length_um
    period_index = np.arange(geometry.number_periods, dtype=float)
    z0 = period_index * period
    z1 = (period_index + 1.0) * period
    fixed = (
        np.exp(-1j * z0[:, None] * mismatch[None, :])
        + np.exp(-1j * z1[:, None] * mismatch[None, :])
    ).sum(axis=0)
    integral_scale = 1.0 / (1j * mismatch * length)
    derivative_scale = 2.0 / geometry.number_periods
    history: list[float] = []

    def value_and_gradient(duties: np.ndarray) -> tuple[float, np.ndarray]:
        wall_phase = np.exp(
            -1j
            * (period_index + duties)[:, None]
            * period
            * mismatch[None, :]
        )
        phi = (fixed - 2.0 * wall_phase.sum(axis=0)) * integral_scale
        intensity = np.abs(phi) ** 2
        center_intensity = max(float(intensity[center_index]), 1e-18)
        normalized = intensity / center_intensity
        residual = normalized - target
        d_phi = derivative_scale * wall_phase
        d_intensity = 2.0 * np.real(d_phi * np.conj(phi)[None, :])
        d_center = d_intensity[:, center_index]
        d_normalized = (
            d_intensity / center_intensity
            - intensity[None, :] * d_center[:, None] / center_intensity**2
        )
        value = float(np.mean(residual**2))
        gradient = (2.0 / residual.size) * (d_normalized @ residual)
        history.append(value if not history else min(value, history[-1]))
        return value, gradient

    result = minimize(
        value_and_gradient,
        duties0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(geometry.duty_min, geometry.duty_max)] * geometry.number_periods,
        options={
            "maxiter": max_iterations,
            "ftol": 1e-18,
            "gtol": 1e-10,
            "maxls": 40,
            "maxcor": 30,
        },
    )
    best_duties = np.asarray(result.x, dtype=float)
    return {
        "duties": best_duties,
        "cost": float(_to_numpy(objective(best_duties))[0]),
        "local_history": np.asarray(history, dtype=float),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "device": "cpu",
    }
