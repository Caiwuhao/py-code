import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.linalg import svd
import io, re, os
import colorsys
from matplotlib.ticker import MultipleLocator, FixedLocator

# ================= 1. 物理参数 =================
LAMBDA_UM = 37.71066
LAMBDA_NM = LAMBDA_UM * 1000.0
d_um = LAMBDA_UM / 2.0
d_nm = d_um * 1000.0
c_const = 299.792458 

# 极化序列 A_SIGN
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

# 折射率公式
def _ny_um(u): return np.sqrt(3.45018 + 0.04341/(u*u-0.04597) + 16.98825/(u*u-39.43799))
def _nz_um(u): return np.sqrt(4.59423 + 0.06206/(u*u-0.04763) + 110.80672/(u*u-86.12171))
def n_p(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)
def n_s(lam_nm): return _nz_um(np.asarray(lam_nm)*1e-3)
def n_i(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)

def get_k_material(ls_nm, li_nm):
    lp_nm = (ls_nm * li_nm) / (ls_nm + li_nm)
    val = 2 * np.pi * (n_p(lp_nm)/lp_nm - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm)
    return val

def robust_solve_idler(ls_nm):
    def target_func(li):
        return get_k_material(ls_nm, li) + 2*np.pi/LAMBDA_NM

    scan_range = np.linspace(1300, 2000, 200)
    vals = np.array([target_func(li) for li in scan_range])
    
    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        try: return brentq(target_func, scan_range[idx], scan_range[idx+1])
        except: pass
            
    def target_func_neg(li): return get_k_material(ls_nm, li) - 2*np.pi/LAMBDA_NM
    vals_neg = np.array([target_func_neg(li) for li in scan_range])
    sign_changes_neg = np.where(np.diff(np.sign(vals_neg)))[0]
    if len(sign_changes_neg) > 0:
        idx = sign_changes_neg[0]
        try: return brentq(target_func_neg, scan_range[idx], scan_range[idx+1])
        except: pass

    min_idx = np.argmin(np.abs(vals))
    return scan_range[min_idx]

def calculate_pmf_design(k_grid, d, domain_signs):
    epsilon = 1e-15
    k_safe = k_grid + epsilon 
    term1 = (np.exp(-1j * k_safe * d) - 1.0) / k_safe
    sum_term = np.zeros_like(k_grid, dtype=complex)
    phi_0 = k_grid * d
    for j, s_val in enumerate(domain_signs):
        sum_term += s_val * np.exp(1j * phi_0 * j)
    return np.abs(sum_term * term1)

def calculate_fwhm(x, y):
    y = np.array(y); x = np.array(x)
    max_val = np.max(y); half_max = max_val / 2.0
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2: return 0.0
    idx_L = indices[0]
    x_left = x[idx_L] if idx_L==0 else x[idx_L-1] + (half_max - y[idx_L-1]) * (x[idx_L] - x[idx_L-1]) / (y[idx_L] - y[idx_L-1])
    idx_R = indices[-1]
    x_right = x[idx_R] if idx_R==len(x)-1 else x[idx_R] + (half_max - y[idx_R]) * (x[idx_R+1] - x[idx_R]) / (y[idx_R+1] - y[idx_R])
    return x_right - x_left

# ================= 2. 加载实验数据 =================
FILES = ["SFG20251029105128.txt", "SFG20251029180550.txt"]

def load_data_full(files):
    points = []
    print(">>> Loading Data (Full Range)...")
    for fpath in files:
        if not os.path.exists(fpath): continue
        with open(fpath,'r',encoding='utf-8',errors='ignore') as f:
            lines = [l for l in f if re.match(r'^\s*[+-]?\d', l)]
        if not lines: continue
        try: arr = np.loadtxt(io.StringIO(''.join(lines)))
        except: continue 
        if arr.ndim==1: arr=arr.reshape(1,-1)
        df = pd.DataFrame(arr)
        while df.shape[1]<9: df[df.shape[1]]=np.nan
        grouped = df.groupby(0)
        for wl, g in grouped:
            valid = g.iloc[50:]
            if len(valid)<5: continue
            v = valid.values
            num = np.mean(np.nan_to_num(v[:,4])-np.nan_to_num(v[:,7])-np.nan_to_num(v[:,8])+np.nan_to_num(v[:,6]))
            den = np.mean(np.nan_to_num(v[:,1]))*np.mean(np.nan_to_num(v[:,3]))
            if abs(den) < 1e-15: continue
            points.append([wl, num/den])
    
    if not points: return None, None
    data = np.array(points)
    data = data[data[:,0].argsort()]
    y = data[:,1]
    
    # 简单去底噪，防止 Log 报错
    bg = np.mean(np.sort(y)[:20])
    y_clean = y - bg
    y_clean[y_clean<0] = 0
    if np.max(y_clean) > 0: y_clean /= np.max(y_clean)
    
    return data[:,0], y_clean

wl_exp, int_exp = load_data_full(FILES)
if wl_exp is None:
    # 模拟数据，确保代码可运行
    print("Warning: No data loaded. Using dummy data for visualization.")
    wl_exp = np.linspace(1280, 1300, 100)
    int_exp = np.exp(-(wl_exp - 1290)**2 / (2 * 1.0**2))

peak_idx = np.argmax(int_exp)
peak_wl_exp = wl_exp[peak_idx]
fwhm_exp = calculate_fwhm(wl_exp, int_exp)

print(f"Experimental Peak: {peak_wl_exp:.3f} nm")
print(f"Experimental FWHM: {fwhm_exp:.3f} nm")

# ================= 3. 计算 Shift =================
FIXED_SIGNAL = 1290.0
theo_idler_ref = robust_solve_idler(FIXED_SIGNAL)
print(f"Theoretical Idler: {theo_idler_ref:.3f} nm")
SHIFT_VAL = peak_wl_exp - theo_idler_ref
print(f"Applied Shift: {SHIFT_VAL:.3f} nm")

num_points = len(wl_exp)
li_uniform = np.linspace(wl_exp.min(), wl_exp.max(), num_points)
int_exp_uniform = interp1d(wl_exp, int_exp, kind='linear', fill_value=0.0, bounds_error=False)(li_uniform)

# ★★★ 范围修正: 1270-1340 nm ★★★
ls_vec = np.linspace(1270, 1340, 600) 
LS, LI = np.meshgrid(ls_vec, li_uniform)

# ================= 4. 重构 =================
print(">>> Computing Theoretical PMF (Shifted)...")
LI_phys = LI - SHIFT_VAL
K_Map = get_k_material(LS, LI_phys)

PMF_Theo_Amp = calculate_pmf_design(K_Map, d_nm, A_SIGN)
PMF_Theo_Amp /= np.max(PMF_Theo_Amp)

center_col_idx = np.argmin(np.abs(ls_vec - FIXED_SIGNAL))
theo_profile_int = PMF_Theo_Amp[:, center_col_idx]**2 
fwhm_theo = calculate_fwhm(li_uniform, theo_profile_int)
print(f"Theoretical PMF FWHM (@1290nm): {fwhm_theo:.3f} nm")

print(">>> Reconstructing Experimental PMF (Cyclic Rolling)...")
PMF_Recon_Int = np.zeros_like(LS)
step_nm = li_uniform[1] - li_uniform[0]
ridge_curve_shifted = np.array([robust_solve_idler(s) for s in ls_vec]) + SHIFT_VAL

for i, s_val in enumerate(ls_vec):
    target_wl = ridge_curve_shifted[i]
    delta_wl = target_wl - peak_wl_exp
    if np.isnan(delta_wl): shift_idx = 0
    else: shift_idx = int(round(delta_wl / step_nm))
    col_data = np.roll(int_exp_uniform, shift_idx)
    PMF_Recon_Int[:, i] = col_data

PMF_Recon_Amp = np.sqrt(PMF_Recon_Int)

# ================= 5. 纯度寻优 =================
print(">>> Optimizing Pump Bandwidth...")

def calculate_purity(pmf_amp_matrix, pump_fwhm_nm_intensity, ls_grid, li_grid, phi2_fs2=0.0):
    WS = 2 * np.pi * c_const / ls_grid
    WI = 2 * np.pi * c_const / li_grid

    ls_center = np.mean(ls_grid)
    li_center = np.mean(li_grid)

    wp0 = 2 * np.pi * c_const * (1/ls_center + 1/li_center)
    lp0 = 2 * np.pi * c_const / wp0  # nm

    pump_fwhm_w_int = 2 * np.pi * c_const * pump_fwhm_nm_intensity / (lp0**2)
    sigma_p = (np.sqrt(2) * pump_fwhm_w_int) / 2.355

    WP = WS + WI
    domega = (WP - wp0)

    Pump_Amp = np.exp(-(domega**2) / (2 * sigma_p**2)) * np.exp(0.5j * phi2_fs2 * (domega**2))
    JSA = Pump_Amp * pmf_amp_matrix

    energy = np.sum(np.abs(JSA)**2)
    if energy == 0:
        return 0.0, JSA

    JSA_Norm = JSA / np.sqrt(energy)
    s = svd(JSA_Norm, compute_uv=False)
    purity = np.sum(s**4)
    return purity, JSA_Norm

def scan_purity_two_stage(pmf_amp, label, bw_min=0.5, bw_max=8.0, coarse_step=0.1, fine_halfspan=1.0, fine_step=0.01):
    # coarse
    bws_c = np.arange(bw_min, bw_max + 1e-12, coarse_step)
    pur_c = np.zeros_like(bws_c, dtype=float)
    for k, bw in enumerate(bws_c):
        pur_c[k], _ = calculate_purity(pmf_amp, bw, LS, LI, phi2_fs2=0.0)
    idx0 = int(np.nanargmax(pur_c))
    bw0 = float(bws_c[idx0])

    # fine around bw0
    bws_f = np.arange(max(bw_min, bw0 - fine_halfspan), bw0 + fine_halfspan + 1e-12, fine_step)
    pur_f = np.zeros_like(bws_f, dtype=float)
    for k, bw in enumerate(bws_f):
        pur_f[k], _ = calculate_purity(pmf_amp, bw, LS, LI, phi2_fs2=0.0)

    idx = int(np.nanargmax(pur_f))
    best_bw = float(bws_f[idx])
    best_p  = float(pur_f[idx])

    # 输出绘图点 (x,y)：写到文件 + 终端打印最优点
    np.savetxt(f"purity_scan_{label}_fine.txt",
               np.c_[bws_f, pur_f],
               header="pump_bw_nm_intensity_FWHM\tpurity",
               fmt="%.4f\t%.8f")

    print(f"[{label}] coarse best ~ {bw0:.4f} nm")
    print(f"[{label}] fine  best = {best_bw:.4f} nm, purity = {best_p:.8f}  (saved: purity_scan_{label}_fine.txt)")
    return bws_f, pur_f, best_bw, best_p

# --- 扫描（Theory / Exp）---
pump_bws_theo, purity_theo, best_bw_theo, best_purity_theo = scan_purity_two_stage(PMF_Theo_Amp, "THEO")
pump_bws_exp,  purity_exp,  best_bw_exp,  best_purity_exp  = scan_purity_two_stage(PMF_Recon_Amp, "EXP")

# --- 用“fine scan 最优值”重新计算 JSA（关键：确保画图用的就是最优） ---
_, JSA_Theo_Opt = calculate_purity(PMF_Theo_Amp,  best_bw_theo, LS, LI)
_, JSA_Exp_Opt  = calculate_purity(PMF_Recon_Amp, best_bw_exp,  LS, LI)

print(f"\n=== Results (from FINE scan) ===")
print(f"Theoretical FWHM: {fwhm_theo:.3f} nm | Optimal Pump: {best_bw_theo:.4f} nm | purity={best_purity_theo:.6f}")
print(f"Experimental FWHM: {fwhm_exp:.3f} nm | Optimal Pump: {best_bw_exp:.4f} nm | purity={best_purity_exp:.6f}")


# ================= 6. 绘图 (修正配色 + 移除标记) =================
def create_custom_blend_cmap():
    # 按照您的要求重现 Mathematica Blend[{Hue[2/3], Hue[0.0833], Hue[0]}, #]
    # 使用 HSV 空间的插值 (Hue Interpolation)，而不是 RGB 直接混合
    
    N_half = 256
    
    # 1. 蓝色到橙色 (占 0.0 ~ 0.5)
    # Hue: 2/3 (Blue) -> 1/12 (Orange)
    # 这会经过 Cyan, Green, Yellow
    hues1 = np.linspace(2/3, 1/12, N_half)
    colors1 = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues1]
    
    # 2. 橙色到红色 (占 0.5 ~ 1.0)
    # Hue: 1/12 (Orange) -> 0.0 (Red)
    # 变化较慢
    hues2 = np.linspace(1/12, 0.0, N_half)
    colors2 = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues2]
    
    # 合并
    all_colors = colors1 + colors2
    return mcolors.LinearSegmentedColormap.from_list("math_hue_blend", all_colors)

my_cmap = create_custom_blend_cmap()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
extent = [ls_vec[0], ls_vec[-1], li_uniform[0], li_uniform[-1]]

# LogNorm: normalized PMF intensity in panels 1/2; JSA magnitude in panels 3/4.
norm_cfg = LogNorm(vmin=1e-4, vmax=1.0)  # PMF intensity: 40 dB
norm_jsa = LogNorm(vmin=1e-2, vmax=1.0)  # JSA magnitude: the same 40 dB

# 1. PMF Theo
im1 = axes[0,0].imshow(PMF_Theo_Amp**2 + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_cfg)
axes[0,0].set_title(f"1. Theoretical PMF (Log)\nFWHM={fwhm_theo:.2f}nm")
axes[0,0].set_ylabel("Idler Wavelength (nm)")
# [已删除] axes[0,0].plot(FIXED_SIGNAL, peak_wl_exp, 'w+', ms=10)
plt.colorbar(im1, ax=axes[0,0], label='Intensity')

# 2. PMF Exp
im2 = axes[0,1].imshow(PMF_Recon_Amp**2 + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_cfg)
axes[0,1].set_title(f"2. Exp Reconstructed PMF (Log)\nFWHM={fwhm_exp:.2f}nm")
# [已删除] axes[0,1].plot(ls_vec, ridge_curve_shifted, 'w--', alpha=0.5)
plt.colorbar(im2, ax=axes[0,1], label='Intensity')

# 3. JSA magnitude (not squared): display only; purity uses the original JSA.
A_JSA_Theo = np.abs(JSA_Theo_Opt)
A_JSA_Theo_Norm = A_JSA_Theo / np.max(A_JSA_Theo) 

im3 = axes[1,0].imshow(A_JSA_Theo_Norm + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_jsa)
axes[1,0].set_title(f"3. Optimized JSA magnitude (Theory, Log)\nPump={best_bw_theo:.2f}nm | purity={best_purity_theo:.4f}")
axes[1,0].set_xlabel("Signal Wavelength (nm)")
axes[1,0].set_ylabel("Idler Wavelength (nm)")
plt.colorbar(im3, ax=axes[1,0], label='Normalized JSA magnitude')

# 4. Experimental JSA magnitude (not squared), peak normalized for display.
A_JSA_Exp = np.abs(JSA_Exp_Opt)
A_JSA_Exp_Norm = A_JSA_Exp / np.max(A_JSA_Exp)

im4 = axes[1,1].imshow(A_JSA_Exp_Norm + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_jsa)
axes[1,1].set_title(f"4. Optimized JSA magnitude (Exp Reconstruction, Log)\nPump={best_bw_exp:.2f}nm | purity={best_purity_exp:.4f}")
axes[1,1].set_xlabel("Signal Wavelength (nm)")
plt.colorbar(im4, ax=axes[1,1], label='Normalized JSA magnitude')

# Poster display only: every other wavelength tick is labelled.
for panel in axes.flat:
    panel.xaxis.set_major_locator(FixedLocator(np.arange(1270, 1341, 20)))
    panel.yaxis.set_major_locator(FixedLocator(np.arange(1530, 1601, 20)))
    panel.xaxis.set_minor_locator(MultipleLocator(10))
    panel.yaxis.set_minor_locator(MultipleLocator(10))
    panel.tick_params(axis='both', labelsize=18)

# Slightly enlarge color-bar numbers only; retain the wavelength tick size.
for panel_image in (im1, im2, im3, im4):
    panel_image.colorbar.ax.tick_params(labelsize=12)

plt.tight_layout()

def save_bare_panel(figure, output_stem):
    # Render into memory before writing each complete export to disk.
    for extension in ('png', 'pdf'):
        with io.BytesIO() as buffer:
            figure.savefig(buffer, format=extension, dpi=300)
            with open(output_stem + '.' + extension, 'wb') as output_file:
                output_file.write(buffer.getvalue())


# Four independent bare panels for flexible poster placement.
# Copy each existing plotted array/norm/cmap; do not repeat or modify calculations.
for panel_image, output_stem in (
    (im1, "poster_b1_pmf_theory_bare"),
    (im2, "poster_b2_pmf_experiment_bare"),
    (im3, "poster_b3_jsa_theory_bare"),
    (im4, "poster_b4_jsa_experiment_bare"),
):
    fig_bare, ax_bare = plt.subplots(figsize=(7.0, 6.5))
    image_bare = ax_bare.imshow(
        panel_image.get_array(), extent=panel_image.get_extent(),
        origin=panel_image.origin, aspect='auto',
        cmap=panel_image.get_cmap(), norm=panel_image.norm)
    ax_bare.set_box_aspect(1)
    ax_bare.set_xlim(panel_image.axes.get_xlim())
    ax_bare.set_ylim(panel_image.axes.get_ylim())
    ax_bare.xaxis.set_major_locator(FixedLocator(np.arange(1270, 1341, 20)))
    ax_bare.yaxis.set_major_locator(FixedLocator(np.arange(1530, 1601, 20)))
    ax_bare.xaxis.set_minor_locator(MultipleLocator(10))
    ax_bare.yaxis.set_minor_locator(MultipleLocator(10))
    ax_bare.tick_params(axis='both', labelsize=18)
    colorbar_bare = fig_bare.colorbar(
        image_bare, ax=ax_bare, fraction=0.046, pad=0.025)
    colorbar_bare.ax.tick_params(labelsize=12)
    # No axis titles, plot titles, panel letters, or in-plot annotations.
    fig_bare.tight_layout(pad=0.7)
    save_bare_panel(fig_bare, output_stem)
    plt.close(fig_bare)

# Additional single-panel output; reuse the existing experimental JSA unchanged.
fig_poster, ax_poster = plt.subplots(figsize=(8.5, 7.8))
im_poster = ax_poster.imshow(
    A_JSA_Exp_Norm + 1e-15, extent=extent, origin='lower',
    aspect='auto', cmap=my_cmap, norm=norm_jsa)
ax_poster.set_box_aspect(1)
ax_poster.set_xlabel("Signal Wavelength (nm)", fontsize=30)
ax_poster.set_ylabel("Idler Wavelength (nm)", fontsize=30)
ax_poster.xaxis.set_major_locator(FixedLocator(np.arange(1270, 1341, 20)))
ax_poster.yaxis.set_major_locator(FixedLocator(np.arange(1530, 1601, 20)))
ax_poster.xaxis.set_minor_locator(MultipleLocator(10))
ax_poster.yaxis.set_minor_locator(MultipleLocator(10))
ax_poster.tick_params(axis='both', labelsize=22)
cbar_poster = fig_poster.colorbar(im_poster, ax=ax_poster, fraction=0.046, pad=0.025)
cbar_poster.ax.tick_params(labelsize=16)
ax_poster.text(0.10, 0.06,
               rf'$\Delta\lambda_p = {best_bw_exp:.2f}\,\mathrm{{nm}}$' + '\n' +
               rf'$P = {best_purity_exp * 100:.2f}\%$',
               transform=ax_poster.transAxes, fontsize=27, color='white', linespacing=1.5)
fig_poster.tight_layout(pad=0.7)
fig_poster.savefig("poster_b.png", dpi=300)
fig_poster.savefig("poster_b.pdf")
plt.close(fig_poster)
plt.show()

# --- coarse scan ---
pump_bws_coarse = np.arange(0.5, 8.0 + 1e-12, 0.1)
pur_coarse = []
for bw in pump_bws_coarse:
    p, _ = calculate_purity(PMF_Recon_Amp, bw, LS, LI, phi2_fs2=0.0)
    pur_coarse.append(p)
pump_bws_coarse = np.array(pump_bws_coarse)
pur_coarse = np.array(pur_coarse)
bw0 = pump_bws_coarse[np.argmax(pur_coarse)]

# --- fine scan around the coarse optimum (0.01 nm step) ---
pump_bws_fine = np.arange(max(0.2, bw0-1.0), bw0+1.0 + 1e-12, 0.01)
pur_fine = []
for bw in pump_bws_fine:
    p, _ = calculate_purity(PMF_Recon_Amp, bw, LS, LI, phi2_fs2=0.0)
    pur_fine.append(p)
pump_bws_fine = np.array(pump_bws_fine)
pur_fine = np.array(pur_fine)

best_bw_exp = pump_bws_fine[np.argmax(pur_fine)]
best_purity_exp = np.max(pur_fine)

print(f"[EXP] Best pump INTENSITY FWHM = {best_bw_exp:.2f} nm, purity = {best_purity_exp:.4f}")

plt.figure()
plt.plot(pump_bws_fine, pur_fine, linewidth=2)
plt.xlabel("Pump bandwidth (Intensity FWHM, nm)")
plt.ylabel("Purity")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================= 7. Purity vs dispersion (pump chirp / GDD) =================
# 使用“fine scan 后”的 best_bw_exp（注意：本段必须放在 best_bw_exp 已更新之后）
phi2_max = 200  # fs^2
phi2_list = np.arange(0, phi2_max + 1, 1)  # step = 1 fs^2

pur_phi2_exp = np.zeros_like(phi2_list, dtype=float)
for k, phi2 in enumerate(phi2_list):
    p, _ = calculate_purity(PMF_Recon_Amp, best_bw_exp, LS, LI, phi2_fs2=float(phi2))
    pur_phi2_exp[k] = p

# 打印 0/50/100/200 的 purity
for target in [0, 50, 100, 200]:
    if target <= phi2_max:
        print(f"[EXP] phi2 = {target:4d} fs^2  ->  purity = {pur_phi2_exp[target]:.8f}")

# 绘图
import matplotlib.ticker as mticker

plt.figure()
ax = plt.gca()
ax.plot(phi2_list, pur_phi2_exp, linewidth=2)

ax.set_xlabel(r"Pump dispersion (GDD)  $\phi_2$ (fs$^2$)")
ax.set_ylabel("Purity")

# 关键：禁用 offset + 禁用科学计数法，并用固定小数格式
sf = mticker.ScalarFormatter(useOffset=False)
sf.set_scientific(False)
ax.yaxis.set_major_formatter(sf)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.8f'))

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# （可选）保存点数据，便于你自己检查/画图
np.savetxt("purity_vs_phi2_EXP.txt",
           np.c_[phi2_list, pur_phi2_exp],
           header="phi2_fs2\tpurity",
           fmt="%d\t%.10f")
print("Saved: purity_vs_phi2_EXP.txt")


