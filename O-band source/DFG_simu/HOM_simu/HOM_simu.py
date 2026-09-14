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
# FWHM of the pump intensity/power spectrum |alpha|^2, not field-amplitude FWHM.
PUMP_FWHM_NM = 3.0

# Output files
OUT_CSV = "signal_projection_corrected_3step_fixed.csv"
OUT_HOM_PNG = "signal_projection_corrected_3step_fixed_HOM.png"
OUT_HOM_CSV = "signal_projection_corrected_3step_fixed_HOM.csv"

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

# ===================== HOM interference from the simulated JSA =====================
# JSA amplitude from the same 2D grid used above.
# Matrix convention: rows = idler wavelengths, columns = signal wavelengths.
# The HOM curve is calculated for the heralded signal photons:
#
# rho_s = F^\dagger F
# C(tau) = 1/2 * [1 - sum_{m,n} |rho_s[m,n]|^2 cos((omega_m - omega_n) tau)]
#
# tau is in ps and omega is in rad/ps.

jsa_amp = pmf_amp_2d * alpha_amp_2d
jsa_amp = np.asarray(jsa_amp, dtype=float)

# Remove very small numerical values, following the Mathematica Chop[data, 0.00001].
threshold = 1e-5 * np.nanmax(np.abs(jsa_amp))
jsa_amp[np.abs(jsa_amp) < threshold] = 0.0

# Normalize: Sum |F|^2 = 1.
norm = np.sqrt(np.sum(jsa_amp**2))
if norm <= 0:
    raise RuntimeError("JSA normalization failed: zero norm.")
F = jsa_amp / norm

# Schmidt coefficients and purity.
singular_values = np.linalg.svd(F, compute_uv=False)
schmidt_coeffs = singular_values / np.sqrt(np.sum(singular_values**2))
purity = float(np.sum(schmidt_coeffs**4))

# Reduced density matrix of the signal photon.
rho_s = F.T @ F
rho_abs2 = np.abs(rho_s)**2

omega_s = 2.0 * np.pi * C_NM_PER_PS / signal_jsi  # rad/ps

DELAY_MIN_PS = -5.0
DELAY_MAX_PS = 5.0
DELAY_STEP_PS = 0.01
delay_ps = np.arange(DELAY_MIN_PS, DELAY_MAX_PS + 0.5*DELAY_STEP_PS, DELAY_STEP_PS)

def hom_curve(delays_ps, omega, rho2, batch_size=100):
    out = np.empty_like(delays_ps, dtype=float)
    for start in range(0, len(delays_ps), batch_size):
        tau = delays_ps[start:start + batch_size]
        phase = np.exp(-1j * np.outer(omega, tau))  # [signal_index, delay_index]
        overlap = np.einsum("mt,mn,nt->t", phase, rho2, np.conjugate(phase), optimize=True)
        out[start:start + batch_size] = 0.5 * (1.0 - np.real(overlap))
    return out

coincidence = hom_curve(delay_ps, omega_s, rho_abs2)

# Visibility, following the Mathematica definition V = 1 - S1[0]/S1[5 ps].
idx0 = int(np.argmin(np.abs(delay_ps)))
idx_far = int(np.argmin(np.abs(delay_ps - 5.0)))
visibility = float(1.0 - coincidence[idx0] / coincidence[idx_far])

# HOM dip FWHM.
c_min = float(coincidence[idx0])
c_far = float(0.5 * (coincidence[0] + coincidence[-1]))
half_level = c_min + 0.5 * (c_far - c_min)

left_cross = np.where(coincidence[:idx0] > half_level)[0]
right_cross = np.where(coincidence[idx0:] > half_level)[0]

if len(left_cross) > 0 and len(right_cross) > 0:
    il = left_cross[-1]
    ir = idx0 + right_cross[0]

    x1, y1 = delay_ps[il], coincidence[il]
    x2, y2 = delay_ps[il+1], coincidence[il+1]
    tau_left = x1 + (half_level - y1) * (x2 - x1) / (y2 - y1)

    x1, y1 = delay_ps[ir-1], coincidence[ir-1]
    x2, y2 = delay_ps[ir], coincidence[ir]
    tau_right = x1 + (half_level - y1) * (x2 - x1) / (y2 - y1)

    hom_fwhm_ps = float(tau_right - tau_left)
else:
    tau_left = np.nan
    tau_right = np.nan
    hom_fwhm_ps = np.nan

print("-" * 72)
print("HOM interference result")
print(f"Schmidt purity                 = {purity:.6f}")
print(f"HOM visibility                 = {visibility:.6f}")
print(f"HOM FWHM                       = {hom_fwhm_ps:.6f} ps")
print("-" * 72)

pd.DataFrame({
    "delay_ps": delay_ps,
    "coincidence_norm": coincidence,
}).to_csv(OUT_HOM_CSV, index=False)

# Plot HOM interference only.
fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=150)
ax.plot(delay_ps, coincidence, color="red", lw=1.8)

if np.isfinite(hom_fwhm_ps):
    ax.hlines(half_level, tau_left, tau_right, colors="black", linestyles="--", lw=1.0)
    ax.annotate(
        "",
        xy=(tau_right, half_level),
        xytext=(tau_left, half_level),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2)
    )
    ax.text(
        2.0,
        half_level - 0.01,
        f"FWHM = {hom_fwhm_ps:.3f} ps",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_xlim(DELAY_MIN_PS, DELAY_MAX_PS)
ax.set_ylim(0.0, 0.6)
ax.set_xlabel("Delay (ps)")
ax.set_ylabel("Coincidence count")
ax.set_title(f"HOM interference  (Purity={purity:.4f}, Visibility={visibility:.4f})")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_HOM_PNG, dpi=300)
plt.close(fig)

print(f"Saved: {OUT_HOM_PNG}")
print(f"Saved: {OUT_HOM_CSV}")