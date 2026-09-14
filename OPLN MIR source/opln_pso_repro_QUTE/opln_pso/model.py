"""LN dispersion, exact binary-domain PMF, and paper-style JSA evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.linalg import svd
from scipy.special import erf


C_M_PER_S = 299_792_458.0  # Speed of light in vacuum.

# Sellmeier coefficients definition; changing after obtaining data from manufacturer.
# Wavelength is in micrometres.
_A = np.array([3.29100, 4.91296, 4.54528], dtype=float)
_B = np.array([0.04140, 0.116275, 0.091649], dtype=float)
_C = np.array([0.03978, 0.048398, 0.046079], dtype=float)
_D = np.array([9.35522, 0.0273, 0.0303], dtype=float)
_POL_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class DesignGeometry:
    pump_wavelength_um: float
    signal_wavelength_um: float
    idler_wavelength_um: float
    pump_polarization: str
    signal_polarization: str
    idler_polarization: str
    qpm_period_um: float # 1st order poling period 
    number_periods: int # number of poling period
    crystal_length_um: float 
    target_sigma_z_um: float # target z in space
    alpha_actual: float
    min_domain_um: float # poling fabrication confinement from manufacturer.
    duty_min: float
    duty_max: float
    pump_intensity_fwhm_fs: float
    pump_intensity_fwhm_nm: float
    pump_sigma_omega_rad_s: float
    group_index_pump: float
    group_index_signal: float
    group_index_idler: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
#save to "summary.json"

#need modify, wavelength_span is defined as 10 times FHWM of the signal mode
@dataclass(frozen=True)
class JSAConfig:
    """Defaults for the paper-style JSA evaluation and pump scan."""

    grid_size: int = 200
    wavelength_span_nm: float = 60.0
    tick_nm: float = 20.0
    pump_scan_min_nm: float = 0.5
    pump_scan_max_nm: float = 8.0
    pump_coarse_step_nm: float = 0.1
    pump_fine_halfspan_nm: float = 1.0
    pump_fine_step_nm: float = 0.01

def refractive_index(wavelength_um: Any, polarization: str) -> np.ndarray:
    """Return n(lambda) using the same LN Sellmeier equation as the MATLAB code."""
    try:
        idx = _POL_INDEX[polarization.lower()]
    except KeyError as exc:
        raise ValueError("polarization must be one of 'x', 'y', or 'z'") from exc
    wl = np.asarray(wavelength_um, dtype=float)
    radicand = _A[idx] + _B[idx] / (wl**2 - _C[idx]) - _D[idx] * wl**2
    if np.any(radicand <= 0):
        raise ValueError("Sellmeier equation was evaluated outside its real-valued range")
    return np.sqrt(radicand)


def group_index(wavelength_um: Any, polarization: str) -> np.ndarray:
    """Return n_g = n - lambda dn/dlambda from the analytic Sellmeier derivative."""
    idx = _POL_INDEX[polarization.lower()]
    wl = np.asarray(wavelength_um, dtype=float)
    n = refractive_index(wl, polarization)
    d_n2_d_wl = -2.0 * _B[idx] * wl / (wl**2 - _C[idx]) ** 2 - 2.0 * _D[idx] * wl
    dn_d_wl = d_n2_d_wl / (2.0 * n)
    return n - wl * dn_d_wl


def energy_conserving_idler(pump_wavelength_um: float, signal_wavelength_um: Any) -> np.ndarray:
    """Return idler wavelength from 1/lambda_p = 1/lambda_s + 1/lambda_i."""
    signal = np.asarray(signal_wavelength_um, dtype=float)
    denominator = signal - pump_wavelength_um
    if np.any(denominator <= 0):
        raise ValueError("signal wavelength must exceed pump wavelength")
    return pump_wavelength_um * signal / denominator


def material_mismatch(
    pump_wavelength_um: Any,
    signal_wavelength_um: Any,
    idler_wavelength_um: Any,
    pump_polarization: str = "y",
    signal_polarization: str = "y",
    idler_polarization: str = "z",
) -> np.ndarray:
    """Return k_p-k_s-k_i in rad/um (the grating vector is not subtracted)."""
    pump = np.asarray(pump_wavelength_um, dtype=float)
    signal = np.asarray(signal_wavelength_um, dtype=float)
    idler = np.asarray(idler_wavelength_um, dtype=float)
    k_p = 2.0 * np.pi * refractive_index(pump, pump_polarization) / pump
    k_s = 2.0 * np.pi * refractive_index(signal, signal_polarization) / signal
    k_i = 2.0 * np.pi * refractive_index(idler, idler_polarization) / idler
    return k_p - k_s - k_i


def qpm_period(
    pump_wavelength_um: float,
    signal_wavelength_um: float,
    idler_wavelength_um: float,
    pump_polarization: str = "y",
    signal_polarization: str = "y",
    idler_polarization: str = "z",
) -> float:
    mismatch = material_mismatch(
        pump_wavelength_um,
        signal_wavelength_um,
        idler_wavelength_um,
        pump_polarization,
        signal_polarization,
        idler_polarization,
    )
    return float(2.0 * np.pi / abs(mismatch))


def gaussian_pump_parameters(
    pump_wavelength_um: float,
    intensity_fwhm_fs: float | None = None,
    intensity_fwhm_nm: float | None = None,
) -> tuple[float, float, float]:
    """Return (amplitude sigma_omega, time FWHM, wavelength FWHM).

    The pump amplitude is exp[-Omega^2/(2 sigma_omega^2)]. Supply either the
    Gaussian temporal intensity FWHM or the measured spectral intensity FWHM.
    A measured spectrum is preferable for a real laser.
    """
    lambda0_m = pump_wavelength_um * 1e-6
    omega0 = 2.0 * np.pi * C_M_PER_S / lambda0_m
    if intensity_fwhm_nm is not None:
        if intensity_fwhm_nm <= 0 or intensity_fwhm_nm >= 2e3 * pump_wavelength_um:
            raise ValueError("pump wavelength FWHM must be positive and below 2*lambda0")
        half_width_m = 0.5 * intensity_fwhm_nm * 1e-9
        omega_high = 2.0 * np.pi * C_M_PER_S / (lambda0_m - half_width_m)
        omega_low = 2.0 * np.pi * C_M_PER_S / (lambda0_m + half_width_m)
        intensity_fwhm_omega = omega_high - omega_low
        sigma_omega = intensity_fwhm_omega / (2.0 * np.sqrt(np.log(2.0)))
        equivalent_fwhm_fs = 2.0 * np.sqrt(np.log(2.0)) / sigma_omega * 1e15
        return float(sigma_omega), float(equivalent_fwhm_fs), float(intensity_fwhm_nm)

    if intensity_fwhm_fs is None or intensity_fwhm_fs <= 0:
        raise ValueError("pump temporal or spectral intensity FWHM must be positive")
    tau_s = intensity_fwhm_fs * 1e-15
    sigma_omega = 2.0 * np.sqrt(np.log(2.0)) / tau_s
    half_width_omega = np.sqrt(np.log(2.0)) * sigma_omega
    if half_width_omega >= omega0:
        raise ValueError("pulse is too short for a positive lower half-maximum frequency")
    lambda_long = 2.0 * np.pi * C_M_PER_S / (omega0 - half_width_omega)
    lambda_short = 2.0 * np.pi * C_M_PER_S / (omega0 + half_width_omega)
    fwhm_nm = (lambda_long - lambda_short) * 1e9
    return float(sigma_omega), float(intensity_fwhm_fs), float(fwhm_nm)


def build_geometry(
    pump_wavelength_um: float = 1.750,
    signal_wavelength_um: float = 3.500,
    idler_wavelength_um: float | None = None,
    pump_intensity_fwhm_fs: float = 1198.96,
    pump_intensity_fwhm_nm: float | None = None,
    alpha: float = 5.0,
    crystal_length_mm: float | None = None,
    target_sigma_z_mm: float | None = None,
    min_domain_um: float = 0.500,
    pump_polarization: str = "y",
    signal_polarization: str = "y",
    idler_polarization: str = "z",
) -> DesignGeometry:
    """Build the QPM geometry and first-order Gaussian factorability target.

    If target_sigma_z_mm is omitted, sigma_z is matched to the supplied pump
    duration using the first-order group-velocity condition. If crystal length
    is also omitted, L = alpha*sigma_z. The QPM period is calculated from
    Sellmeier dispersion, and the number of full periods is rounded down.
    """
    if min(pump_wavelength_um, signal_wavelength_um) <= 0:
        raise ValueError("wavelengths must be positive")
    if alpha <= 0 or (crystal_length_mm is not None and crystal_length_mm <= 0):
        raise ValueError("alpha and crystal length must be positive")
    if idler_wavelength_um is None:
        idler_wavelength_um = float(
            energy_conserving_idler(pump_wavelength_um, signal_wavelength_um)
        )
    sigma_omega, equivalent_pulse_fs, pump_fwhm_nm = gaussian_pump_parameters(
        pump_wavelength_um,
        intensity_fwhm_fs=pump_intensity_fwhm_fs,
        intensity_fwhm_nm=pump_intensity_fwhm_nm,
    )
    ng_p = float(group_index(pump_wavelength_um, pump_polarization))
    ng_s = float(group_index(signal_wavelength_um, signal_polarization))
    ng_i = float(group_index(idler_wavelength_um, idler_polarization))
    a_s_per_m = (ng_p - ng_s) / C_M_PER_S
    b_s_per_m = (ng_p - ng_i) / C_M_PER_S
    if target_sigma_z_mm is None:
        if a_s_per_m * b_s_per_m >= 0:
            raise ValueError(
                "pump group index is not between signal and idler group indices; "
                "a separable double-Gaussian JSA cannot be obtained by bandwidth matching"
            )
        sigma_z_m = 1.0 / (sigma_omega * np.sqrt(-a_s_per_m * b_s_per_m))
        sigma_z_um = sigma_z_m * 1e6
    else:
        if target_sigma_z_mm <= 0:
            raise ValueError("target_sigma_z_mm must be positive")
        sigma_z_um = target_sigma_z_mm * 1e3

    period_um = qpm_period(
        pump_wavelength_um,
        signal_wavelength_um,
        idler_wavelength_um,
        pump_polarization,
        signal_polarization,
        idler_polarization,
    )
    if min_domain_um < 0:
        raise ValueError("min_domain_um cannot be negative")
    if 2.0 * min_domain_um >= period_um:
        raise ValueError("two minimum domains do not fit inside one QPM period")
    requested_length_um = (
        alpha * sigma_z_um
        if crystal_length_mm is None
        else crystal_length_mm * 1e3
    )
    number_periods = max(1, int(np.floor(requested_length_um / period_um)))
    actual_length_um = number_periods * period_um
    duty_min = min_domain_um / period_um

    return DesignGeometry(
        pump_wavelength_um=pump_wavelength_um,
        signal_wavelength_um=signal_wavelength_um,
        idler_wavelength_um=idler_wavelength_um,
        pump_polarization=pump_polarization,
        signal_polarization=signal_polarization,
        idler_polarization=idler_polarization,
        qpm_period_um=period_um,
        number_periods=number_periods,
        crystal_length_um=actual_length_um,
        target_sigma_z_um=sigma_z_um,
        alpha_actual=actual_length_um / sigma_z_um,
        min_domain_um=min_domain_um,
        duty_min=duty_min,
        duty_max=1.0 - duty_min,
        pump_intensity_fwhm_fs=equivalent_pulse_fs,
        pump_intensity_fwhm_nm=pump_fwhm_nm,
        pump_sigma_omega_rad_s=sigma_omega,
        group_index_pump=ng_p,
        group_index_signal=ng_s,
        group_index_idler=ng_i,
    )


def paper_erf_initial_duties(number_periods: int) -> np.ndarray:
    """Reproduce the erf-shaped initial duty array in the MATLAB appendix."""
    x = np.arange(number_periods, dtype=float) / number_periods
    return 0.5 + 0.45 * erf((x - 0.5) / 0.45)


def qute_3500_geometry(
    min_domain_um: float = 0.0,
) -> DesignGeometry:
    """Return the 3.5 um, nominal 30 mm geometry using the Sellmeier period."""
    return build_geometry(
        pump_wavelength_um=1.75,
        signal_wavelength_um=3.5,
        pump_intensity_fwhm_nm=3.8,
        crystal_length_mm=30.0,
        target_sigma_z_mm=6.0,
        min_domain_um=min_domain_um,
    )


def cumulative_erf_target(z_um: Any, length_um: float, sigma_z_um: float) -> np.ndarray:
    """Normalized cumulative spatial target corresponding to a Gaussian PMF."""
    z = np.asarray(z_um, dtype=float)
    root2_sigma = np.sqrt(2.0) * sigma_z_um
    raw = erf(length_um / (2.0 * root2_sigma)) + erf((z - length_um / 2.0) / root2_sigma)
    normalization = 2.0 * erf(length_um / (2.0 * root2_sigma))
    return raw / normalization


def domain_table(duties: Any, period_um: float) -> np.ndarray:
    """Return rows (index, orientation, z_start_um, z_end_um, length_um)."""
    duty = np.asarray(duties, dtype=float)
    rows: list[list[float]] = []
    domain_index = 0
    for period_index, value in enumerate(duty):
        z0 = period_index * period_um
        z1 = z0 + value * period_um
        z2 = z0 + period_um
        rows.append([domain_index, 1, z0, z1, z1 - z0])
        rows.append([domain_index + 1, -1, z1, z2, z2 - z1])
        domain_index += 2
    return np.asarray(rows, dtype=float)


def pmf_from_duties(
    duties: Any,
    period_um: float,
    mismatch_rad_um: Any,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Exact normalized complex PMF L^-1 integral g(z)exp(-i*Delta_k*z)dz."""
    duty = np.asarray(duties, dtype=float)
    if duty.ndim != 1 or duty.size == 0 or np.any(~np.isfinite(duty)):
        raise ValueError("duties must be a nonempty finite one-dimensional array")
    if period_um <= 0 or np.any((duty < 0) | (duty > 1)):
        raise ValueError("period must be positive and duties must lie in [0, 1]")
    mismatch = np.asarray(mismatch_rad_um, dtype=float).reshape(-1)
    if np.any(~np.isfinite(mismatch)):
        raise ValueError("mismatch must be finite")
    periods = np.arange(duty.size, dtype=float)
    z0 = periods * period_um
    zm = (periods + duty) * period_um
    z1 = (periods + 1.0) * period_um
    length_um = duty.size * period_um
    result = np.empty(mismatch.size, dtype=complex)
    for start in range(0, mismatch.size, chunk_size):
        stop = min(start + chunk_size, mismatch.size)
        k = mismatch[start:stop]
        phase_sum = (
            np.exp(-1j * np.outer(z0, k))
            - 2.0 * np.exp(-1j * np.outer(zm, k))
            + np.exp(-1j * np.outer(z1, k))
        ).sum(axis=0)
        regular = np.abs(k * length_um) >= 1e-6
        block = np.empty(k.size, dtype=complex)
        block[regular] = phase_sum[regular] / (1j * k[regular] * length_um)
        # The sinc form is the same exact integral and has a finite k=0 limit.
        if np.any(~regular):
            small_k = k[~regular][None, :]
            positive_length = (zm - z0)[:, None]
            negative_length = (z1 - zm)[:, None]
            positive = positive_length * np.sinc(small_k * positive_length / (2*np.pi))
            negative = negative_length * np.sinc(small_k * negative_length / (2*np.pi))
            positive = positive * np.exp(-0.5j * small_k * (z0 + zm)[:, None])
            negative = negative * np.exp(-0.5j * small_k * (zm + z1)[:, None])
            block[~regular] = (positive - negative).sum(axis=0) / length_um
        result[start:stop] = block
    return result


def jsa_wavelength_grid(
    geometry: DesignGeometry,
    grid_size: int = 200,
    wavelength_span_nm: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a uniform wavelength grid with rows=idler and columns=signal."""
    if grid_size < 20:
        raise ValueError("grid_size must be at least 20")
    if wavelength_span_nm <= 0:
        raise ValueError("wavelength_span_nm must be positive")
    half_span = wavelength_span_nm / 2.0
    signal_nm = np.linspace(
        geometry.signal_wavelength_um * 1e3 - half_span,
        geometry.signal_wavelength_um * 1e3 + half_span,
        grid_size,
    )
    idler_nm = np.linspace(
        geometry.idler_wavelength_um * 1e3 - half_span,
        geometry.idler_wavelength_um * 1e3 + half_span,
        grid_size,
    )
    signal_grid_nm, idler_grid_nm = np.meshgrid(signal_nm, idler_nm, indexing="xy")
    return signal_nm, idler_nm, signal_grid_nm, idler_grid_nm


def pump_amplitude_py_jsi(
    pump_fwhm_nm_intensity: float,
    signal_grid_nm: Any,
    idler_grid_nm: Any,
    pump_center_nm: float | None = None,
    phi2_fs2: float = 0.0,
) -> np.ndarray:
    """Gaussian pump amplitude using the convention in Py_JSI_SPDC.py."""
    if pump_fwhm_nm_intensity <= 0:
        raise ValueError("pump_fwhm_nm_intensity must be positive")
    signal = np.asarray(signal_grid_nm, dtype=float)
    idler = np.asarray(idler_grid_nm, dtype=float)
    if signal.shape != idler.shape:
        raise ValueError("signal_grid_nm and idler_grid_nm must have the same shape")
    c_nm_per_fs = 299.792458
    omega_signal = 2.0 * np.pi * c_nm_per_fs / signal
    omega_idler = 2.0 * np.pi * c_nm_per_fs / idler
    if pump_center_nm is None:
        signal_center = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
        idler_center = 0.5 * (float(np.min(idler)) + float(np.max(idler)))
        pump_center_nm = 1.0 / (1.0 / signal_center + 1.0 / idler_center)
    omega_pump_center = 2.0 * np.pi * c_nm_per_fs / pump_center_nm
    pump_fwhm_omega_intensity = (
        2.0 * np.pi * c_nm_per_fs * pump_fwhm_nm_intensity / pump_center_nm**2
    )
    pump_sigma_amplitude = np.sqrt(2.0) * pump_fwhm_omega_intensity / 2.355
    detuning = omega_signal + omega_idler - omega_pump_center
    return np.exp(-0.5 * (detuning / pump_sigma_amplitude) ** 2) * np.exp(
        0.5j * phi2_fs2 * detuning**2
    )


def pmf_on_jsa_grid(
    duties: Any,
    geometry: DesignGeometry,
    signal_grid_nm: Any,
    idler_grid_nm: Any,
) -> np.ndarray:
    """Evaluate the exact complex PMF on a signal/idler wavelength grid."""
    signal_um = np.asarray(signal_grid_nm, dtype=float) * 1e-3
    idler_um = np.asarray(idler_grid_nm, dtype=float) * 1e-3
    pump_um = signal_um * idler_um / (signal_um + idler_um)
    mismatch = material_mismatch(
        pump_um,
        signal_um,
        idler_um,
        geometry.pump_polarization,
        geometry.signal_polarization,
        geometry.idler_polarization,
    )
    return pmf_from_duties(
        duties, geometry.qpm_period_um, mismatch.reshape(-1)
    ).reshape(mismatch.shape)


def evaluate_jsa_from_pmf(
    pmf_complex: Any,
    pump_fwhm_nm_intensity: float,
    signal_grid_nm: Any,
    idler_grid_nm: Any,
    pump_center_nm: float | None = None,
    pmf_mode: Literal["magnitude", "complex"] = "magnitude",
    phi2_fs2: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Build and Schmidt-decompose a JSA from a precomputed PMF."""
    pmf = np.asarray(pmf_complex, dtype=complex)
    if pmf_mode == "magnitude":
        pmf_for_jsa = np.abs(pmf)
    elif pmf_mode == "complex":
        pmf_for_jsa = pmf
    else:
        raise ValueError("pmf_mode must be 'magnitude' or 'complex'")
    pump = pump_amplitude_py_jsi(
        pump_fwhm_nm_intensity,
        signal_grid_nm,
        idler_grid_nm,
        pump_center_nm=pump_center_nm,
        phi2_fs2=phi2_fs2,
    )
    jsa = pump * pmf_for_jsa
    norm = np.linalg.norm(jsa)
    if norm > 0:
        jsa = jsa / norm
    return {"jsa": jsa, "pump": pump, "purity": schmidt_purity(jsa)}


def evaluate_jsa(
    duties: Any,
    geometry: DesignGeometry,
    pump_fwhm_nm_intensity: float | None = None,
    grid_size: int = 200,
    wavelength_span_nm: float = 60.0,
    pump_center_nm: float | None = None,
    pmf_mode: Literal["magnitude", "complex"] = "magnitude",
) -> dict[str, np.ndarray | float]:
    """Evaluate a paper-style wavelength-grid JSA and its Schmidt purity."""
    if pump_fwhm_nm_intensity is None:
        pump_fwhm_nm_intensity = geometry.pump_intensity_fwhm_nm
    signal_nm, idler_nm, signal_grid_nm, idler_grid_nm = jsa_wavelength_grid(
        geometry, grid_size, wavelength_span_nm
    )
    pmf_complex = pmf_on_jsa_grid(
        duties, geometry, signal_grid_nm, idler_grid_nm
    )
    values = evaluate_jsa_from_pmf(
        pmf_complex,
        pump_fwhm_nm_intensity,
        signal_grid_nm,
        idler_grid_nm,
        pump_center_nm,
        pmf_mode,
    )
    values.update(
        {
            "pmf": pmf_complex,
            "signal_wavelengths_nm": signal_nm,
            "idler_wavelengths_nm": idler_nm,
            "signal_grid_nm": signal_grid_nm,
            "idler_grid_nm": idler_grid_nm,
            "pump_fwhm_nm_intensity": float(pump_fwhm_nm_intensity),
        }
    )
    return values


def scan_optimal_pump_bandwidth(
    duties: Any,
    geometry: DesignGeometry,
    grid_size: int = 200,
    wavelength_span_nm: float = 60.0,
    pump_center_nm: float | None = None,
    bandwidth_min_nm: float = 0.5,
    bandwidth_max_nm: float = 8.0,
    coarse_step_nm: float = 0.1,
    fine_halfspan_nm: float = 1.0,
    fine_step_nm: float = 0.01,
    pmf_mode: Literal["magnitude", "complex"] = "magnitude",
) -> dict[str, Any]:
    """Silently reproduce the two-stage pump scan in Py_JSI_SPDC.py."""
    if bandwidth_min_nm <= 0 or bandwidth_max_nm <= bandwidth_min_nm:
        raise ValueError("invalid pump bandwidth range")
    if min(coarse_step_nm, fine_halfspan_nm, fine_step_nm) <= 0:
        raise ValueError("pump scan steps must be positive")
    signal_nm, idler_nm, signal_grid_nm, idler_grid_nm = jsa_wavelength_grid(
        geometry, grid_size, wavelength_span_nm
    )
    pmf_complex = pmf_on_jsa_grid(
        duties, geometry, signal_grid_nm, idler_grid_nm
    )

    def evaluate_bandwidths(bandwidths: np.ndarray) -> np.ndarray:
        purities = np.empty(bandwidths.size, dtype=float)
        for index, bandwidth in enumerate(bandwidths):
            purities[index] = evaluate_jsa_from_pmf(
                pmf_complex,
                float(bandwidth),
                signal_grid_nm,
                idler_grid_nm,
                pump_center_nm,
                pmf_mode,
            )["purity"]
        return purities

    coarse_bandwidths = np.arange(
        bandwidth_min_nm,
        bandwidth_max_nm + 0.5 * coarse_step_nm,
        coarse_step_nm,
    )
    coarse_purities = evaluate_bandwidths(coarse_bandwidths)
    coarse_best = float(coarse_bandwidths[int(np.nanargmax(coarse_purities))])
    fine_start = max(bandwidth_min_nm, coarse_best - fine_halfspan_nm)
    fine_stop = min(bandwidth_max_nm, coarse_best + fine_halfspan_nm)
    fine_bandwidths = np.arange(
        fine_start, fine_stop + 0.5 * fine_step_nm, fine_step_nm
    )
    fine_purities = evaluate_bandwidths(fine_bandwidths)
    best_index = int(np.nanargmax(fine_purities))
    best_bandwidth = float(fine_bandwidths[best_index])
    best_values = evaluate_jsa_from_pmf(
        pmf_complex,
        best_bandwidth,
        signal_grid_nm,
        idler_grid_nm,
        pump_center_nm,
        pmf_mode,
    )
    return {
        "best_bandwidth_nm": best_bandwidth,
        "best_purity": float(best_values["purity"]),
        "best_jsa": best_values["jsa"],
        "pmf": pmf_complex,
        "pump": best_values["pump"],
        "signal_wavelengths_nm": signal_nm,
        "idler_wavelengths_nm": idler_nm,
        "signal_grid_nm": signal_grid_nm,
        "idler_grid_nm": idler_grid_nm,
        "coarse_bandwidths_nm": coarse_bandwidths,
        "coarse_purities": coarse_purities,
        "fine_bandwidths_nm": fine_bandwidths,
        "fine_purities": fine_purities,
        "pmf_mode": pmf_mode,
    }


def schmidt_purity(jsa: Any) -> float:
    """Return sum(s_j**4) after L2 normalization of the JSA."""
    matrix = np.asarray(jsa, dtype=complex)
    norm = np.linalg.norm(matrix)
    if norm == 0:
        return 0.0
    singular_values = svd(
        matrix / norm, compute_uv=False, check_finite=False, overwrite_a=False
    )
    return float(np.sum(singular_values**4))
