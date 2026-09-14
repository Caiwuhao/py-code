# -*- coding: utf-8 -*-
"""
Fixed-idler slice of |f(lambda_s, lambda_i)|^2 for CPKTP.

Convention used here:
- The empirical wavelength correction is applied to the PMF idler coordinate only:
      lambda_i_phys_for_PMF = lambda_i_display - SHIFT_NM
  This follows the shifted-theory convention used in the previous plotting code.
- The pump envelope is calculated on the displayed experimental axes
  (lambda_s, lambda_i_display), so a 1310 nm + 1571 nm pair corresponds to ~714 nm pump.

Output:
- signal_slice_idler1571_f2.png
- signal_slice_idler1571_f2.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ===================== User parameters =====================
FIXED_IDLER_DISPLAY_NM = 1571.0   # fixed idler wavelength on the displayed/experimental axis
SHIFT_NM = 19.20181446486231     # 1555.850 - 1536.6481855351376 nm
SIGNAL_MIN_NM = 1270.0
SIGNAL_MAX_NM = 1345.0
N_SIGNAL = 4001

PUMP_FWHM_NM = 3.08                # change this if needed; |f|^2 depends on pump bandwidth
PUMP_CENTER_MODE = "phasematched" # "phasematched" or "manual"
PUMP_CENTER_MANUAL_NM = 714.0     # used only when PUMP_CENTER_MODE = "manual"

QPM_PERIOD_UM = 37.71066
QPM_PERIOD_NM = QPM_PERIOD_UM * 1000.0
DOMAIN_WIDTH_NM = QPM_PERIOD_NM / 2.0
C_CONST = 299.792458              # same numerical convention as the previous code

OUT_PNG = "signal_slice_idler1571_f2_shift19p2018.png"
OUT_CSV = "signal_slice_idler1571_f2_shift19p2018.csv"

# ===================== Domain sequence =====================
_A_RAW = [
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,
  1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,
  1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,
  1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,
  1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,
  1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
  1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
  1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,
  1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,
  1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,
  1,-1,-1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,
  1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,
  1,-1,-1,-1,-1,-1,1,1,1,1,1,1,
  1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1
]
A_SIGN = np.array(_A_RAW[:265], dtype=float)

# ===================== Sellmeier equations =====================
def _ny_um(u_um):
    return np.sqrt(3.45018 + 0.04341/(u_um*u_um - 0.04597) + 16.98825/(u_um*u_um - 39.43799))

def _nz_um(u_um):
    return np.sqrt(4.59423 + 0.06206/(u_um*u_um - 0.04763) + 110.80672/(u_um*u_um - 86.12171))

def n_p(lambda_nm):
    return _ny_um(np.asarray(lambda_nm, dtype=float) * 1e-3)

def n_s(lambda_nm):
    return _nz_um(np.asarray(lambda_nm, dtype=float) * 1e-3)

def n_i(lambda_nm):
    return _ny_um(np.asarray(lambda_nm, dtype=float) * 1e-3)

def lambda_p_from_si(lambda_s_nm, lambda_i_nm):
    lambda_s_nm = np.asarray(lambda_s_nm, dtype=float)
    lambda_i_nm = np.asarray(lambda_i_nm, dtype=float)
    return lambda_s_nm * lambda_i_nm / (lambda_s_nm + lambda_i_nm)

def k_material_nm(lambda_s_nm, lambda_i_nm):
    """Material phase mismatch without explicit grating vector, in rad/nm."""
    lambda_p_nm = lambda_p_from_si(lambda_s_nm, lambda_i_nm)
    return 2*np.pi * (
        n_p(lambda_p_nm)/lambda_p_nm
        - n_s(lambda_s_nm)/lambda_s_nm
        - n_i(lambda_i_nm)/lambda_i_nm
    )

def delta_k_qpm_nm(lambda_s_nm, lambda_i_nm):
    """First-order QPM condition helper: k_material + 2*pi/Lambda."""
    return k_material_nm(lambda_s_nm, lambda_i_nm) + 2*np.pi/QPM_PERIOD_NM

# ===================== PMF and pump envelope =====================
def pmf_amp_from_k(k_grid, domain_width_nm=DOMAIN_WIDTH_NM, domain_signs=A_SIGN):
    """
    Domain-engineered PMF amplitude. Input k_grid should be k_material, not k_material + G.
    Returns |Phi|.
    """
    k = np.asarray(k_grid, dtype=float) + 1e-15
    phi0 = k * domain_width_nm
    j = np.arange(domain_signs.size, dtype=float)

    # Vectorized sum over domains: sum_j A_j exp(i k d j)
    sum_term = np.exp(1j * np.outer(k.ravel() * domain_width_nm, j)) @ domain_signs
    sum_term = sum_term.reshape(k.shape)

    cell_term = (np.exp(-1j * k * domain_width_nm) - 1.0) / k
    return np.abs(sum_term * cell_term)

def solve_signal_for_fixed_idler(lambda_i_phys_nm, signal_min=1200.0, signal_max=1400.0):
    """Solve k_material(lambda_s, lambda_i_phys) + 2*pi/Lambda = 0."""
    def f(ls):
        return float(delta_k_qpm_nm(ls, lambda_i_phys_nm))

    grid = np.linspace(signal_min, signal_max, 2001)
    vals = np.array([f(x) for x in grid])
    idx = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(idx) > 0:
        i = idx[0]
        return brentq(f, grid[i], grid[i+1], maxiter=1000)

    # Fallback: nearest zero if no sign change is found.
    return float(grid[np.argmin(np.abs(vals))])

def gaussian_pump_amp(lambda_s_nm, lambda_i_display_nm, pump_center_nm, pump_fwhm_nm):
    """Gaussian pump amplitude in angular-frequency-like units."""
    omega_s = 2*np.pi*C_CONST / lambda_s_nm
    omega_i = 2*np.pi*C_CONST / lambda_i_display_nm
    omega_p0 = 2*np.pi*C_CONST / pump_center_nm
    pump_fwhm_omega = 2*np.pi*C_CONST * pump_fwhm_nm / (pump_center_nm**2)
    sigma_omega = pump_fwhm_omega / (2*np.sqrt(2*np.log(2)))
    return np.exp(-0.5 * ((omega_s + omega_i - omega_p0)/sigma_omega)**2)

def fwhm_numeric(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.nanmax(y) <= 0:
        return np.nan, np.nan, np.nan
    half = 0.5 * np.nanmax(y)
    imax = int(np.nanargmax(y))

    left = np.where(y[:imax] < half)[0]
    right = np.where(y[imax:] < half)[0]
    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan, np.nan

    il = left[-1]
    ir = imax + right[0]
    xl = x[il] + (half - y[il]) * (x[il+1] - x[il]) / (y[il+1] - y[il])
    xr = x[ir-1] + (half - y[ir-1]) * (x[ir] - x[ir-1]) / (y[ir] - y[ir-1])
    return float(xr-xl), float(xl), float(xr)

# ===================== Main calculation =====================
def main():
    lambda_i_phys_nm = FIXED_IDLER_DISPLAY_NM - SHIFT_NM
    signal_center_nm = solve_signal_for_fixed_idler(lambda_i_phys_nm)

    if PUMP_CENTER_MODE.lower() == "manual":
        pump_center_nm = PUMP_CENTER_MANUAL_NM
    else:
        # Use the displayed experimental idler axis for energy conservation.
        pump_center_nm = lambda_p_from_si(signal_center_nm, FIXED_IDLER_DISPLAY_NM)

    ls = np.linspace(SIGNAL_MIN_NM, SIGNAL_MAX_NM, N_SIGNAL)

    # PMF uses shifted physical idler coordinate.
    k = k_material_nm(ls, lambda_i_phys_nm)
    phi_amp = pmf_amp_from_k(k)
    phi2 = phi_amp**2
    phi2 /= np.nanmax(phi2)

    # f = alpha * Phi. Pump envelope uses displayed wavelengths.
    alpha_amp = gaussian_pump_amp(ls, FIXED_IDLER_DISPLAY_NM, pump_center_nm, PUMP_FWHM_NM)
    pump2 = alpha_amp**2
    pump2 /= np.nanmax(pump2)

    f2 = (alpha_amp * phi_amp)**2
    f2 /= np.nanmax(f2)

    peak_signal = float(ls[np.nanargmax(f2)])
    fwhm_f2, left_f2, right_f2 = fwhm_numeric(ls, f2)
    fwhm_phi2, _, _ = fwhm_numeric(ls, phi2)

    print("Fixed-idler signal slice")
    print(f"  idler display axis       = {FIXED_IDLER_DISPLAY_NM:.3f} nm")
    print(f"  PMF idler after shift    = {lambda_i_phys_nm:.3f} nm")
    print(f"  shift                    = {SHIFT_NM:.3f} nm")
    print(f"  QPM signal center        = {signal_center_nm:.3f} nm")
    print(f"  pump center              = {pump_center_nm:.3f} nm")
    print(f"  pump FWHM                = {PUMP_FWHM_NM:.3f} nm")
    print(f"  |Phi|^2 FWHM             = {fwhm_phi2:.3f} nm")
    print(f"  |f|^2 peak signal        = {peak_signal:.3f} nm")
    print(f"  |f|^2 FWHM               = {fwhm_f2:.3f} nm")

    # Save CSV.
    arr = np.column_stack([ls, phi2, pump2, f2])
    header = "signal_nm,phi2_fixed_idler_norm,pump2_norm,f2_norm"
    np.savetxt(OUT_CSV, arr, delimiter=",", header=header, comments="")

    # Plot.
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    ax.plot(ls, f2, lw=2.2, label=r"$|f(\lambda_s, \lambda_i=1571\,\mathrm{nm})|^2$")
    ax.plot(ls, phi2, lw=1.8, ls="--", label=r"$|\Phi|^2$ only")
    ax.plot(ls, pump2, lw=1.6, ls=":", label=r"pump envelope $|\alpha|^2$")

    ax.axvline(signal_center_nm, lw=1.2, alpha=0.8)
    ax.text(signal_center_nm, 0.05, f"PM center\n{signal_center_nm:.2f} nm",
            rotation=90, va="bottom", ha="right")

    ax.set_xlim(SIGNAL_MIN_NM, SIGNAL_MAX_NM)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Signal wavelength (nm)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(
        "Fixed-idler signal slice with shifted PMF\n"
        f"idler={FIXED_IDLER_DISPLAY_NM:.1f} nm, shift={SHIFT_NM:.3f} nm, "
        f"pump FWHM={PUMP_FWHM_NM:.1f} nm"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
