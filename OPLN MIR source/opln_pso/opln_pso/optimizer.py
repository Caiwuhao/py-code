from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .model import DesignGeometry, paper_erf_initial_duties


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
    except (ImportError, RuntimeError):
        if normalized == "cuda":
            raise RuntimeError(
                "CUDA was requested, but CuPy or a working CUDA device is unavailable"
            )
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
    seed: int = 7
    inertia_start: float = 0.90
    inertia_end: float = 0.40
    cognitive: float = 1.49618
    social: float = 1.49618
    initial_noise: float = 0.06
    symmetric: bool = True
    device: str = "auto"
    wavelength_chunk: int = 96
    log_every: int = 10


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
        q_points: int = 801,
        q_span_sigma: float = 15.0,
        objective: str = "complex_amplitude",
        device: str = "auto",
        wavelength_chunk: int = 96,
    ):
        if q_points < 31 or q_points % 2 == 0:
            raise ValueError("q_points must be an odd integer >= 31")
        if objective not in {"legacy_intensity", "complex_amplitude"}:
            raise ValueError("unknown objective")
        self.geometry = geometry
        self.objective = objective
        self.xp, self.device = _load_backend(device)
        self.wavelength_chunk = wavelength_chunk
        q_cpu = np.linspace(
            -q_span_sigma / geometry.target_sigma_z_um,
            q_span_sigma / geometry.target_sigma_z_um,
            q_points,
        )
        self.q_rad_um = q_cpu
        self.mismatch = self.xp.asarray(
            2.0 * np.pi / geometry.qpm_period_um + q_cpu, dtype=self.xp.float64
        )
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
        self._center_index = q_points // 2

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
    positions = rng.uniform(lower, upper, size=(config.particles, variable_count))
    local_count = max(1, int(0.75 * config.particles))
    positions[:local_count] = xp.asarray(initial_variables)[None, :] + config.initial_noise * rng.normal(
        size=(local_count, variable_count)
    )
    positions = xp.clip(positions, lower, upper)
    positions[0] = xp.asarray(initial_variables)
    velocities = rng.uniform(-0.05, 0.05, size=positions.shape)

    costs = objective(expand(positions))
    personal_positions = positions.copy()
    personal_costs = costs.copy()
    best_index = int(_to_numpy(xp.argmin(costs)))
    global_position = positions[best_index].copy()
    global_cost = float(_to_numpy(costs[best_index]))
    history = [global_cost]
    if callback is not None:
        callback(0, global_cost)

    for iteration in range(1, config.iterations + 1):
        fraction = iteration / max(1, config.iterations)
        inertia = config.inertia_start + fraction * (config.inertia_end - config.inertia_start)
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
        history.append(global_cost)
        if callback is not None and (
            iteration == config.iterations or iteration % config.log_every == 0
        ):
            callback(iteration, global_cost)

    best_duties = expand(global_position[None, :])[0]
    return {
        "duties": _to_numpy(best_duties),
        "cost": global_cost,
        "history": np.asarray(history, dtype=float),
        "device": objective.device,
        "symmetric": config.symmetric,
    }
