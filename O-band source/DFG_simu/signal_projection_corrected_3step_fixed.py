# -*- coding: utf-8 -*-
"""
Corrected three-step calculation for signal projection/slice of |f|^2.

Workflow:
1) Use Sellmeier + QPM to solve the theoretical idler for a fixed signal.
   Here the reference point is signal = 1310 nm. The solution is ~1550 nm,
   and the corresponding pump is ~710 nm.
2) Add the experimentally determined idler shift to the idler axis.
   With the shifted idler and signal = 1310 nm, recompute the pump center.
   This gives pump ~714 nm.
3) Fix this shifted idler and scan signal wavelength to plot |f|^2 vs signal.

PMF uses the unshifted physical idler coordinate: li_phys = li_display - SHIFT_NM.
Pump envelope uses the displayed/experimental wavelengths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
import numpy as np

def make_mathematica_hue_cmap(n=256):
    # Mathematica: Blend[{Hue[2/3], Hue[0]}, x]
    hues = np.linspace(2/3, 0, n)
    rgb = [hsv_to_rgb((h, 1, 1)) for h in hues]
    return LinearSegmentedColormap.from_list("mma_hue_2over3_to_0", rgb, N=n)

# ===================== user parameters =====================
# Shift calibration: from previous SFG theory/experiment comparison
SHIFT_REF_SIGNAL_NM = 1290.56
EXP_IDLER_CENTER_NM = 1555.85     # experimental idler center from SFG fit/screenshot

# Main reference point requested here
REFERENCE_SIGNAL_NM = 1310.0      # fixed signal in step 1 and step 2

# Signal scan range for the final plot
SIGNAL_MIN_NM = 1270.0
SIGNAL_MAX_NM = 1350.0
SIGNAL_STEP_NM = 0.01

# Pump bandwidth used in f = alpha * phi.
# This follows the same angular-frequency Gaussian-amplitude convention as Py_JSI - log.py.
PUMP_FWHM_NM = 3.0

# Output files
OUT_PNG = "signal_projection_corrected_3step_fixed.png"
OUT_CSV = "signal_projection_corrected_3step_fixed.csv"
OUT_JSI_PNG = "signal_projection_corrected_3step_fixed_JSI.png"

# ===================== physical constants =====================
LAMBDA_UM = 37.71066
LAMBDA_NM = LAMBDA_UM * 1000.0
D_HALF_NM = LAMBDA_NM / 2.0
C_NM_PER_PS = 299792.458  # nm / ps; constant scale cancels except for bandwidth conversion

# Domain signs
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

# ===================== Sellmeier and QPM =====================
def _ny_um(u):
    return np.sqrt(3.45018 + 0.04341/(u*u - 0.04597) + 16.98825/(u*u - 39.43799))

def _nz_um(u):
    return np.sqrt(4.59423 + 0.06206/(u*u - 0.04763) + 110.80672/(u*u - 86.12171))

def n_y_nm(lam_nm):
    return _ny_um(np.asarray(lam_nm) * 1e-3)

def n_z_nm(lam_nm):
    return _nz_um(np.asarray(lam_nm) * 1e-3)

def n_p(lam_nm):
    return n_y_nm(lam_nm)

def n_s(lam_nm):
    return n_z_nm(lam_nm)

def n_i(lam_nm):
    return n_y_nm(lam_nm)

def lambda_p(ls_nm, li_nm):
    ls = np.asarray(ls_nm, dtype=float)
    li = np.asarray(li_nm, dtype=float)
    return (ls * li) / (ls + li)

def k_basic_nm(ls_nm, li_nm):
    lp = lambda_p(ls_nm, li_nm)
    return 2*np.pi * (n_p(lp)/lp - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm)

def delta_k_grating_nm(ls_nm, li_nm):
    return k_basic_nm(ls_nm, li_nm) + 2*np.pi/LAMBDA_NM

def solve_idler_qpm(ls_nm, scan_min=1400.0, scan_max=1800.0):
    f = lambda li: float(delta_k_grating_nm(ls_nm, li))
    grid = np.linspace(scan_min, scan_max, 2001)
    vals = np.array([f(x) for x in grid])
    idx = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(idx) > 0:
        j = idx[0]
        return brentq(f, grid[j], grid[j+1], maxiter=1000)
    return float(grid[np.argmin(np.abs(vals))])

def phi_amp_from_k(k):
    """Amplitude PMF |Phi| for the custom domain sequence.

    Works for scalar, 1D, and 2D k arrays.
    The 2D JSI grid is evaluated in chunks to avoid huge memory use.
    """
    k_in = np.asarray(k, dtype=complex)
    k_flat = k_in.ravel()

    j = np.arange(A_SIGN.size, dtype=float)
    out_flat = np.empty_like(k_flat, dtype=complex)

    zero_mask = np.abs(k_flat) < 1e-15
    out_flat[zero_mask] = np.sum(A_SIGN) * D_HALF_NM

    idx = np.where(~zero_mask)[0]
    chunk_size = 20000
    for start in range(0, len(idx), chunk_size):
        ids = idx[start:start + chunk_size]
        kk = k_flat[ids]
        phase = np.exp(1j * kk[:, None] * D_HALF_NM * j[None, :])
        s = phase @ A_SIGN
        out_flat[ids] = 1j * s * (np.exp(-1j * kk * D_HALF_NM) - 1.0) / kk

    return np.abs(out_flat).reshape(k_in.shape)

def pmf_amp_shifted(ls_nm, li_display_nm, shift_nm):
    """Shift-corrected PMF: use li_phys = li_display - shift in Sellmeier/QPM."""
    li_phys = np.asarray(li_display_nm, dtype=float) - shift_nm
    return phi_amp_from_k(k_basic_nm(ls_nm, li_phys))

def pump_amp(ls_nm, li_display_nm, pump_center_nm, pump_fwhm_nm):
    """
    Pump field amplitude alpha(lambda_p).

    Here pump_fwhm_nm is the FWHM of the pump intensity/power spectrum |alpha|^2.
    Therefore:
        |alpha|^2 = exp[-4 ln(2) ((lambda_p - lambda_p0)/FWHM)^2]
    and the field amplitude is:
        alpha = exp[-2 ln(2) ((lambda_p - lambda_p0)/FWHM)^2].
    """
    lambda_p_vec = lambda_p(ls_nm, li_display_nm)

    return np.exp(
        -2.0*np.log(2.0)
        * ((lambda_p_vec - pump_center_nm) / pump_fwhm_nm)**2
    )

def fwhm(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.nanmax(y) <= 0:
        return np.nan
    half = np.nanmax(y) / 2.0
    imax = int(np.nanargmax(y))
    left = np.where(y[:imax] < half)[0]
    right = np.where(y[imax:] < half)[0]
    if len(left) == 0 or len(right) == 0:
        return np.nan
    il = left[-1]
    ir = imax + right[0]
    xl = x[il] + (half - y[il]) * (x[il+1] - x[il]) / (y[il+1] - y[il])
    xr = x[ir-1] + (half - y[ir-1]) * (x[ir] - x[ir-1]) / (y[ir] - y[ir-1])
    return float(xr - xl)

# ===================== three-step calculation =====================
# Shift from the previous SFG reference configuration
li_theory_shift_ref = solve_idler_qpm(SHIFT_REF_SIGNAL_NM)
SHIFT_NM = EXP_IDLER_CENTER_NM - li_theory_shift_ref

# Step 1: fixed signal = 1310 nm, solve theory idler and pump
li_theory_ref = solve_idler_qpm(REFERENCE_SIGNAL_NM)
pump_theory_ref = float(lambda_p(REFERENCE_SIGNAL_NM, li_theory_ref))

# Step 2: shift the idler and recompute the pump center
fixed_idler_display = li_theory_ref + SHIFT_NM
pump_center_shifted = float(lambda_p(REFERENCE_SIGNAL_NM, fixed_idler_display))

# Step 3: fix shifted idler and scan signal
signal = np.arange(SIGNAL_MIN_NM, SIGNAL_MAX_NM + 0.5*SIGNAL_STEP_NM, SIGNAL_STEP_NM)
idler_display = np.full_like(signal, fixed_idler_display, dtype=float)

pmf_amp = pmf_amp_shifted(signal, idler_display, SHIFT_NM)
pmf2 = pmf_amp**2
pmf2 /= np.nanmax(pmf2)

alpha_amp = pump_amp(signal, idler_display, pump_center_shifted, PUMP_FWHM_NM)
f2 = (alpha_amp * pmf_amp)**2
f2 /= np.nanmax(f2)

peak_signal_pmf = float(signal[np.nanargmax(pmf2)])
peak_signal_f2 = float(signal[np.nanargmax(f2)])
fwhm_pmf2 = fwhm(signal, pmf2)
fwhm_f2 = fwhm(signal, f2)

# ===================== extra: JSI-only figure =====================
# Keep the same three-step logic and the same corrected pump FWHM treatment.
# We only add a 2D JSI = |f|^2 plot around the operating point.

JSI_SIGNAL_MIN_NM = SIGNAL_MIN_NM
JSI_SIGNAL_MAX_NM = SIGNAL_MAX_NM
JSI_IDLER_HALFSPAN_NM = 30.0
JSI_STEP_NM = 0.1    # 0.05 is too slow for this direct PMF calculation

signal_jsi = np.arange(JSI_SIGNAL_MIN_NM,
                       JSI_SIGNAL_MAX_NM + 0.5*JSI_STEP_NM,
                       JSI_STEP_NM)

idler_jsi = np.arange(fixed_idler_display - JSI_IDLER_HALFSPAN_NM,
                      fixed_idler_display + JSI_IDLER_HALFSPAN_NM + 0.5*JSI_STEP_NM,
                      JSI_STEP_NM)

LS, LI = np.meshgrid(signal_jsi, idler_jsi)

pmf_amp_2d = pmf_amp_shifted(LS, LI, SHIFT_NM)
alpha_amp_2d = pump_amp(LS, LI, pump_center_shifted, PUMP_FWHM_NM)

jsi2 = (pmf_amp_2d * alpha_amp_2d)**2
jsi2 /= np.nanmax(jsi2)

print("=" * 72)
print("Corrected 3-step calculation")
print(f"Shift reference signal        = {SHIFT_REF_SIGNAL_NM:.6f} nm")
print(f"Theory idler at ref signal    = {li_theory_shift_ref:.9f} nm")
print(f"Experimental idler center     = {EXP_IDLER_CENTER_NM:.9f} nm")
print(f"Applied idler shift           = {SHIFT_NM:.9f} nm")
print("-" * 72)
print(f"Step 1 signal fixed           = {REFERENCE_SIGNAL_NM:.6f} nm")
print(f"Step 1 theory idler           = {li_theory_ref:.9f} nm")
print(f"Step 1 theory pump            = {pump_theory_ref:.9f} nm")
print("-" * 72)
print(f"Step 2 shifted fixed idler    = {fixed_idler_display:.9f} nm")
print(f"Step 2 shifted pump center    = {pump_center_shifted:.9f} nm")
print("-" * 72)
print(f"Step 3 fixed idler            = {fixed_idler_display:.9f} nm")
print(f"PMF^2 peak signal             = {peak_signal_pmf:.6f} nm")
print(f"PMF^2 FWHM                    = {fwhm_pmf2:.6f} nm")
print(f"f^2 peak signal               = {peak_signal_f2:.6f} nm")
print(f"f^2 FWHM                      = {fwhm_f2:.6f} nm")
print("=" * 72)

# save data
pd.DataFrame({
    "signal_nm": signal,
    "fixed_idler_display_nm": idler_display,
    "pump_from_energy_nm": lambda_p(signal, idler_display),
    "pmf2_norm": pmf2,
    "pump_amp_norm": alpha_amp / np.nanmax(alpha_amp),
    "f2_norm": f2,
}).to_csv(OUT_CSV, index=False)

# plot
plt.figure(figsize=(8.8, 4.8), dpi=150)
plt.plot(signal, f2, lw=1.8, label=rf"$|f(\lambda_s,\lambda_i={fixed_idler_display:.2f}\,\mathrm{{nm}})|^2$")
plt.plot(signal, pmf2, lw=1.5, linestyle="--", label=rf"$|\Phi|^2$ only")
plt.axvline(REFERENCE_SIGNAL_NM, lw=1.0, linestyle=":", label=rf"reference signal = {REFERENCE_SIGNAL_NM:.0f} nm")
plt.xlabel("Signal wavelength (nm)")
plt.ylabel("Normalized intensity")
plt.title(
    "Corrected signal projection/slice after idler shift\n"
    rf"shift={SHIFT_NM:.3f} nm, fixed idler={fixed_idler_display:.3f} nm, pump center={pump_center_shifted:.3f} nm"
)
plt.grid(alpha=0.3)
plt.legend(frameon=True, fancybox=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_CSV}")

# ---- JSI-only plot ----
cmap_mma = make_mathematica_hue_cmap()

plt.figure(figsize=(7.2, 7.2), dpi=150)

im = plt.imshow(
    jsi2,
    extent=[signal_jsi.min(), signal_jsi.max(), idler_jsi.min(), idler_jsi.max()],
    origin="lower",
    aspect="auto",
    cmap=cmap_mma,
    interpolation="bilinear"
)

plt.xlabel("Signal wavelength (nm)")
plt.ylabel("Idler wavelength (nm)")
plt.title("JSI")

cbar = plt.colorbar(im)
cbar.set_label(r"Normalized JSI = $|f(\lambda_s,\lambda_i)|^2$")

plt.tight_layout()
plt.savefig(OUT_JSI_PNG, dpi=300)
plt.close()

cbar = plt.colorbar(im)
cbar.set_label(r"Normalized JSI = $|f(\lambda_s,\lambda_i)|^2$")

plt.xlabel("Signal wavelength (nm)")
plt.ylabel("Idler wavelength (nm)")
plt.title(
    "JSI only\n"
    rf"shift={SHIFT_NM:.3f} nm, pump center={pump_center_shifted:.3f} nm, "
    rf"pump FWHM={PUMP_FWHM_NM:.1f} nm"
)

plt.tight_layout()
plt.savefig(OUT_JSI_PNG, dpi=300)
plt.close()

print(f"Saved: {OUT_JSI_PNG}")