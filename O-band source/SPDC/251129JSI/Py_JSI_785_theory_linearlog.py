import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
from scipy.linalg import svd
import colorsys

# =============================
# 0) 常量（保持与原程序体系一致）
# =============================
LAMBDA_UM = 37.71066
LAMBDA_NM = LAMBDA_UM * 1000.0
d_um = LAMBDA_UM / 2.0
d_nm = d_um * 1000.0
c_const = 299.792458  # 与你原脚本一致的常量体系（不要改）

# =============================
# 1) 设计极化序列 A_SIGN（与你原脚本一致）
# =============================
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

# =============================
# 2) Sellmeier（保持与你原脚本一致：pump/idler 用 ny；signal 用 nz）
# =============================
def _ny_um(u): return np.sqrt(3.45018 + 0.04341/(u*u-0.04597) + 16.98825/(u*u-39.43799))
def _nz_um(u): return np.sqrt(4.59423 + 0.06206/(u*u-0.04763) + 110.80672/(u*u-86.12171))

def n_p(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)  # pump: y
def n_s(lam_nm): return _nz_um(np.asarray(lam_nm)*1e-3)  # signal: z
def n_i(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)  # idler: y

def get_k_material(ls_nm, li_nm):
    # lp from energy (physical)
    lp_nm = (ls_nm * li_nm) / (ls_nm + li_nm)
    return 2*np.pi*(n_p(lp_nm)/lp_nm - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm)

def delta_k_qpm(ls_nm, li_nm, deltaK=0.0):
    # Δk_QPM = Δk_material + 2π/Λ + δK
    return get_k_material(ls_nm, li_nm) + 2*np.pi/LAMBDA_NM + deltaK

# =============================
# 3) 数值工具函数（保持原风格）
# =============================
def calculate_pmf_design(k_grid, d, domain_signs):
    eps = 1e-15
    k_safe = k_grid + eps
    term1 = (np.exp(-1j * k_safe * d) - 1.0) / k_safe
    sum_term = np.zeros_like(k_grid, dtype=complex)
    phi_0 = k_grid * d
    for j, s_val in enumerate(domain_signs):
        sum_term += s_val * np.exp(1j * phi_0 * j)
    return np.abs(sum_term * term1)

def calculate_fwhm(x, y):
    x = np.asarray(x); y = np.asarray(y)
    ymax = np.max(y)
    if ymax <= 0: return 0.0
    half = ymax/2.0
    idx = np.where(y >= half)[0]
    if len(idx) < 2: return 0.0
    iL, iR = idx[0], idx[-1]

    if iL == 0:
        xL = x[0]
    else:
        xL = x[iL-1] + (half - y[iL-1]) * (x[iL]-x[iL-1]) / (y[iL]-y[iL-1])

    if iR == len(x)-1:
        xR = x[-1]
    else:
        xR = x[iR] + (half - y[iR]) * (x[iR+1]-x[iR]) / (y[iR+1]-y[iR])

    return xR - xL

def robust_solve_idler(ls_nm, deltaK=0.0):
    # solve Δk_QPM(ls,li)+δK=0 for li
    def f(li): return delta_k_qpm(ls_nm, li, deltaK=deltaK)
    scan = np.linspace(1300.0, 2200.0, 800)
    vals = np.array([f(li) for li in scan])
    sc = np.where(np.diff(np.sign(vals)))[0]
    if len(sc) > 0:
        a, b = scan[sc[0]], scan[sc[0]+1]
        return brentq(f, a, b)
    return scan[np.argmin(np.abs(vals))]

# Mathematica-like Hue Blend colormap（与你原脚本一致）
def create_custom_blend_cmap():
    N_half = 256
    hues1 = np.linspace(2/3, 1/12, N_half)
    colors1 = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues1]
    hues2 = np.linspace(1/12, 0.0, N_half)
    colors2 = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues2]
    return mcolors.LinearSegmentedColormap.from_list("math_hue_blend", colors1 + colors2)

my_cmap = create_custom_blend_cmap()

# =============================
# 4) Step-1: 用制造波段峰位偏差标定 δK
# =============================
LAMBDA_S_CAL = 1290.0
SHIFT_IDLER_CAL = 19.5220  # 你从 SFG 得到的峰位偏差（示例：1555.80 - 1536.28 ≈ 19.522）

li_th_cal = robust_solve_idler(LAMBDA_S_CAL, deltaK=0.0)            # model peak
li_exp_cal = li_th_cal + SHIFT_IDLER_CAL                            # "experimental peak" on idler axis
deltaK = -delta_k_qpm(LAMBDA_S_CAL, li_exp_cal, deltaK=0.0)         # enforce Δk_QPM + δK = 0 at exp peak

print("=== Calibration at design band (SFG) ===")
print(f"Fixed signal (lambda_s)              = {LAMBDA_S_CAL:.3f} nm")
print(f"Model idler peak (lambda_i, theory)  = {li_th_cal:.6f} nm")
print(f"Assumed idler peak shift             = {SHIFT_IDLER_CAL:.6f} nm")
print(f"Target idler peak (lambda_i, exp)    = {li_exp_cal:.6f} nm")
print(f"Calibrated phase-mismatch bias δK    = {deltaK:.6e} (same unit as Δk)\n")

# =============================
# 5) Step-2: 在 pump=785 下求解中心波长（未校正 vs δK 校正）
# =============================
PUMP_NM = 785.0

def li_from_energy(ls_nm, pump_nm):
    # 1/p = 1/s + 1/i  ->  i = p*s/(s-p)
    return pump_nm * ls_nm / (ls_nm - pump_nm)

def solve_centers_for_pump(pump_nm, deltaK_use=0.0, ls_min=1350.0, ls_max=1550.0, n_scan=800):
    def f(ls):
        if ls <= pump_nm + 1e-9:
            return np.nan
        li = li_from_energy(ls, pump_nm)
        return delta_k_qpm(ls, li, deltaK=deltaK_use)

    ls_scan = np.linspace(ls_min, ls_max, n_scan)
    vals = np.array([f(ls) for ls in ls_scan])

    # 找变号区间
    sign = np.sign(vals)
    good = np.isfinite(sign)
    idx = np.where(good[:-1] & good[1:] & (sign[:-1]*sign[1:] < 0))[0]
    if len(idx) == 0:
        # fallback：取绝对值最小点
        j = np.nanargmin(np.abs(vals))
        ls0 = ls_scan[j]
        li0 = li_from_energy(ls0, pump_nm)
        return ls0, li0

    i = idx[0]
    a, b = ls_scan[i], ls_scan[i+1]
    ls0 = brentq(lambda x: f(x), a, b)
    li0 = li_from_energy(ls0, pump_nm)
    return ls0, li0

# baseline (no bias)
ls0, li0 = solve_centers_for_pump(PUMP_NM, deltaK_use=0.0)

# bias-corrected (δK)
ls1, li1 = solve_centers_for_pump(PUMP_NM, deltaK_use=deltaK)

print("=== Center wavelengths at pump = 785 nm ===")
print("[Uncorrected model]  (δK=0)")
print(f"  lambda_s0 = {ls0:.6f} nm")
print(f"  lambda_i0 = {li0:.6f} nm")
print("[δK-corrected prediction]")
print(f"  lambda_s1 = {ls1:.6f} nm")
print(f"  lambda_i1 = {li1:.6f} nm")
print(f"  shifts:  Δlambda_s = {ls1-ls0:+.3f} nm,  Δlambda_i = {li1-li0:+.3f} nm\n")

# =============================
# 6) GVM angle（frequency-domain; pump 固定用 785 nm）
# =============================
def group_index(n_func, lam_nm, h=0.05):
    lam = float(lam_nm)
    n0 = float(n_func(lam))
    n_plus = float(n_func(lam + h))
    n_minus = float(n_func(lam - h))
    dn_dlam = (n_plus - n_minus) / (2*h)
    return n0 - lam * dn_dlam

def gvm_angle_deg(pump_nm, ls_nm, li_nm):
    ngp = group_index(n_p, pump_nm)
    ngs = group_index(n_s, ls_nm)
    ngi = group_index(n_i, li_nm)
    slope = - (ngp - ngs) / (ngp - ngi)  # dωi/dωs
    ang = np.degrees(np.arctan(np.abs(slope)))
    return slope, ang, ngp, ngs, ngi

slope, ang, ngp, ngs, ngi = gvm_angle_deg(PUMP_NM, ls1, li1)

print("=== GVM (using pump = 785 nm) ===")
print(f"ng(p)={ngp:.6f}, ng(s)={ngs:.6f}, ng(i)={ngi:.6f}")
print(f"slope dωi/dωs = {slope:.6f}  ->  GVM angle = {ang:.3f} deg\n")

# =============================
# 7) 构建网格并计算 PMF（用 δK-corrected 中心点）
# =============================
SIGNAL_WINDOW_WIDTH = 70.0
IDLER_WINDOW_WIDTH  = 80.0

ls_vec = np.linspace(ls1 - SIGNAL_WINDOW_WIDTH/2, ls1 + SIGNAL_WINDOW_WIDTH/2, 600)
li_vec = np.linspace(li1 - IDLER_WINDOW_WIDTH/2,  li1 + IDLER_WINDOW_WIDTH/2,  600)
LS, LI = np.meshgrid(ls_vec, li_vec)

# 关键：用 k = Δk_material + δK （因为 QPM 由 poling 序列隐含提供）
K_map = get_k_material(LS, LI) + deltaK
PMF_amp = calculate_pmf_design(K_map, d_nm, A_SIGN)
PMF_amp /= np.max(PMF_amp)

center_col = np.argmin(np.abs(ls_vec - ls1))
pmf_profile = PMF_amp[:, center_col]**2
fwhm_pmf = calculate_fwhm(li_vec, pmf_profile)

# =============================
# 8) 纯度优化（保持与你原脚本一致，只改输出名 purity）
# =============================
def calculate_purity(pmf_amp_matrix, pump_fwhm_nm, ls_grid, li_grid):
    WS = 2 * np.pi * c_const / ls_grid
    WI = 2 * np.pi * c_const / li_grid
    ls_center = np.mean(ls_grid)
    li_center = np.mean(li_grid)
    wp0 = 2 * np.pi * c_const * (1/ls_center + 1/li_center)
    lp0 = 2 * np.pi * c_const / wp0
    pump_bw_w = 2 * np.pi * c_const * pump_fwhm_nm / (lp0**2)
    sigma_p = pump_bw_w / 2.355
    WP = WS + WI
    Pump_Amp = np.exp(- (WP - wp0)**2 / (2 * sigma_p**2))
    JSA = Pump_Amp * pmf_amp_matrix
    energy = np.sum(np.abs(JSA)**2)
    if energy == 0:
        return 0.0, JSA
    JSA_norm = JSA / np.sqrt(energy)
    s = svd(JSA_norm, compute_uv=False)
    purity = np.sum(s**4)
    return purity, JSA_norm

pump_bws = np.linspace(0.2, 8.0, 40)
pur_list = []
for bw in pump_bws:
    p, _ = calculate_purity(PMF_amp, bw, LS, LI)
    pur_list.append(p)

best_idx = int(np.argmax(pur_list))
best_bw = float(pump_bws[best_idx])
best_purity = float(pur_list[best_idx])
_, JSA_opt = calculate_purity(PMF_amp, best_bw, LS, LI)

print("=== Optimization result (δK-corrected) ===")
print(f"PMF FWHM (linear slice) = {fwhm_pmf:.3f} nm")
print(f"Optimal pump (FWHM)     = {best_bw:.2f} nm")
print(f"Purity                  = {best_purity:.6f}\n")

# normalize intensities for plotting
PMF_I = PMF_amp**2
PMF_I /= np.max(PMF_I)
JSA_I = np.abs(JSA_opt)**2
JSA_I /= np.max(JSA_I)

# =============================
# 9) 绘图：4 张图（PMF linear/log；JSA linear/log）
# =============================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
extent = [ls_vec[0], ls_vec[-1], li_vec[0], li_vec[-1]]
norm_log = LogNorm(vmin=1e-4, vmax=1.0)

# 1) PMF Linear
im1 = axes[0, 0].imshow(PMF_I, extent=extent, origin='lower', aspect='auto',
                        cmap=my_cmap, vmin=0.0, vmax=1.0)
axes[0, 0].set_title(f"1. Theoretical PMF (Linear)\nFWHM={fwhm_pmf:.2f}nm")
axes[0, 0].set_ylabel("Idler Wavelength (nm)")
plt.colorbar(im1, ax=axes[0, 0], label='Intensity')

# 2) PMF Log
im2 = axes[0, 1].imshow(PMF_I + 1e-15, extent=extent, origin='lower', aspect='auto',
                        cmap=my_cmap, norm=norm_log)
axes[0, 1].set_title(f"2. Theoretical PMF (Log)\nFWHM={fwhm_pmf:.2f}nm")
plt.colorbar(im2, ax=axes[0, 1], label='Intensity')

# 3) JSA Linear
im3 = axes[1, 0].imshow(JSA_I, extent=extent, origin='lower', aspect='auto',
                        cmap=my_cmap, vmin=0.0, vmax=1.0)
axes[1, 0].set_title(f"3. Optimized JSA (Theory, Linear)\nPump={best_bw:.2f}nm | Purity={best_purity:.4f}")
axes[1, 0].set_xlabel("Signal Wavelength (nm)")
axes[1, 0].set_ylabel("Idler Wavelength (nm)")
plt.colorbar(im3, ax=axes[1, 0], label='Intensity')

# 4) JSA Log
im4 = axes[1, 1].imshow(JSA_I + 1e-15, extent=extent, origin='lower', aspect='auto',
                        cmap=my_cmap, norm=norm_log)
axes[1, 1].set_title(f"4. Optimized JSA (Theory, Log)\nPump={best_bw:.2f}nm | Purity={best_purity:.4f}")
axes[1, 1].set_xlabel("Signal Wavelength (nm)")
plt.colorbar(im4, ax=axes[1, 1], label='Intensity')

plt.tight_layout()
plt.show()
