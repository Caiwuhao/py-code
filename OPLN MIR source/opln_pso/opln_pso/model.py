#physics model
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import erf


C_M_PER_S = 299_792_458.0 # Speed of light in vaccum

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
    number_periods: int
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
    is also omitted, L = alpha*sigma_z and is rounded to an integer number of
    QPM periods.
    """
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
    if 2.0 * min_domain_um >= period_um:
        raise ValueError("two minimum domains do not fit inside one QPM period")
    requested_length_um = (
        alpha * sigma_z_um if crystal_length_mm is None else crystal_length_mm * 1e3
    )
    number_periods = max(1, int(np.rint(requested_length_um / period_um)))
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
    mismatch = np.asarray(mismatch_rad_um, dtype=float).reshape(-1)
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
        result[start:stop] = phase_sum / (1j * k * length_um)
    return result


def evaluate_jsa(
    duties: Any,
    geometry: DesignGeometry,
    grid_size: int = 160,
    span_sigma: float = 4.5,
) -> dict[str, np.ndarray | float]:
    """Evaluate the JSA on a uniform angular-frequency grid and its purity."""
    if grid_size < 20:
        raise ValueError("grid_size must be at least 20")
    omega_p0 = 2.0 * np.pi * C_M_PER_S / (geometry.pump_wavelength_um * 1e-6)
    omega_s0 = 2.0 * np.pi * C_M_PER_S / (geometry.signal_wavelength_um * 1e-6)
    omega_i0 = 2.0 * np.pi * C_M_PER_S / (geometry.idler_wavelength_um * 1e-6)
    detuning = np.linspace(
        -span_sigma * geometry.pump_sigma_omega_rad_s,
        span_sigma * geometry.pump_sigma_omega_rad_s,
        grid_size,
    )
    omega_s = omega_s0 + detuning
    omega_i = omega_i0 + detuning
    os_grid, oi_grid = np.meshgrid(omega_s, omega_i, indexing="ij")
    op_grid = os_grid + oi_grid
    lambda_p = 2.0 * np.pi * C_M_PER_S / op_grid * 1e6
    lambda_s = 2.0 * np.pi * C_M_PER_S / os_grid * 1e6
    lambda_i = 2.0 * np.pi * C_M_PER_S / oi_grid * 1e6
    mismatch = material_mismatch(
        lambda_p,
        lambda_s,
        lambda_i,
        geometry.pump_polarization,
        geometry.signal_polarization,
        geometry.idler_polarization,
    )
    pmf = pmf_from_duties(
        duties, geometry.qpm_period_um, mismatch.reshape(-1)
    ).reshape(grid_size, grid_size)
    pump = np.exp(
        -0.5 * ((op_grid - omega_p0) / geometry.pump_sigma_omega_rad_s) ** 2
    )
    jsa = pump * pmf
    purity = schmidt_purity(jsa)
    signal_wavelengths_nm = 2.0 * np.pi * C_M_PER_S / omega_s * 1e9
    idler_wavelengths_nm = 2.0 * np.pi * C_M_PER_S / omega_i * 1e9
    return {
        "jsa": jsa,
        "pmf": pmf,
        "pump": pump,
        "purity": purity,
        "signal_wavelengths_nm": signal_wavelengths_nm,
        "idler_wavelengths_nm": idler_wavelengths_nm,
    }


def schmidt_purity(jsa: Any) -> float:
    matrix = np.asarray(jsa, dtype=complex)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    weights = singular_values**2
    total = weights.sum()
    if total == 0:
        return 0.0
    weights /= total
    return float(np.sum(weights**2))
