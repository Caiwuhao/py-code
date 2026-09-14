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

    scan_range = np.linspace(1300, 2200, 300)
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

# ================= 2. 加载实验数据 (705 nm pump 的 SFG) =================
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
    wl_exp = np.linspace(1530, 1580, 200)
    int_exp = np.exp(-(wl_exp - 1556)**2 / (2 * 1.0**2))

peak_idx_705 = np.argmax(int_exp)
peak_wl_exp_705 = wl_exp[peak_idx_705]
fwhm_exp_705 = calculate_fwhm(wl_exp, int_exp)

print(f"Experimental Peak (705 nm pump): {peak_wl_exp_705:.3f} nm")
print(f"Experimental FWHM (705 nm pump): {fwhm_exp_705:.3f} nm")

# ================= 3. 计算 Shift (理论 -> 实验，只作用在 idler 轴) =================
FIXED_SIGNAL_705 = 1290.0
theo_idler_ref_705 = robust_solve_idler(FIXED_SIGNAL_705)
print(f"Theoretical Idler @ {FIXED_SIGNAL_705:.1f} nm (705 nm pump): {theo_idler_ref_705:.3f} nm")

SHIFT_VAL = peak_wl_exp_705 - theo_idler_ref_705
print(f"Applied Shift (idler): {SHIFT_VAL:.3f} nm")

# ================= 4. 785 nm pump 下的中心波长：满足
# (1) 1/(λ_i + SHIFT) + 1/λ_s = 1/λ_pump
# (2) Δk(λ_s, λ_i) = 0  (QPM 条件)
PUMP_TARGET_NM = 785.0

def solve_center_wavelengths_with_shift(pump_nm, shift_nm, s_range=(1200.0, 2000.0), n_scan=4000):
    def li_from_ls(ls):
        # from 1/(li + shift) + 1/ls = 1/pump  ->  li = 1/(1/pump - 1/ls) - shift
        return 1.0/(1.0/pump_nm - 1.0/ls) - shift_nm

    def f(ls):
        li = li_from_ls(ls)
        if (not np.isfinite(li)) or li <= 0:
            return np.nan
        return get_k_material(ls, li) + 2*np.pi/LAMBDA_NM

    ls_scan = np.linspace(s_range[0], s_range[1], n_scan)
    vals = np.array([f(ls) for ls in ls_scan])
    mask = np.isfinite(vals)
    ls2 = ls_scan[mask]
    v2 = vals[mask]

    # 找到符号变号区间并求根
    sign_changes = np.where(np.diff(np.sign(v2)) != 0)[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        a, b = ls2[idx], ls2[idx+1]
        ls_root = brentq(lambda x: f(x), a, b, maxiter=200)
    else:
        # 若无严格变号，用 |f| 最小点
        ls_root = ls2[np.argmin(np.abs(v2))]

    li_root = li_from_ls(ls_root)
    return float(ls_root), float(li_root)

signal_center_th_785, idler_center_th_785 = solve_center_wavelengths_with_shift(PUMP_TARGET_NM, SHIFT_VAL)
idler_center_shifted_785 = idler_center_th_785 + SHIFT_VAL

print("\n=== Center wavelengths for pump = 785.0 nm (with idler shift in energy equation) ===")
print(f"Signal center (theory) = {signal_center_th_785:.6f} nm")
print(f"Idler center (theory, unshifted) = {idler_center_th_785:.6f} nm")
print(f"Idler center (shifted, for plotting/experiment axis) = {idler_center_shifted_785:.6f} nm")

# ================= 5. 构建新的 idler 轴：保持原实验数据的扫描宽度不变，只平移中心 =================
num_points = len(wl_exp)

# 原实验扫描宽度（相对 705 nm pump 的峰值）
left_width = peak_wl_exp_705 - wl_exp.min()
right_width = wl_exp.max() - peak_wl_exp_705

li_uniform = np.linspace(idler_center_shifted_785 - left_width,
                         idler_center_shifted_785 + right_width,
                         num_points)

# 把原 1D 实验分布平移到新的中心（保持形状不变）
delta_center = idler_center_shifted_785 - peak_wl_exp_705
f_int0 = interp1d(wl_exp, int_exp, kind='linear', fill_value=0.0, bounds_error=False)
int_exp_uniform = f_int0(li_uniform - delta_center)

# 更新新的“实验峰值”（理论上应接近 idler_center_shifted_785）
peak_idx = np.argmax(int_exp_uniform)
peak_wl_exp = li_uniform[peak_idx]
fwhm_exp = calculate_fwhm(li_uniform, int_exp_uniform)

print(f"\nExperimental Peak (shifted to 785 nm pump axis): {peak_wl_exp:.3f} nm")
print(f"Experimental FWHM (shifted 1D): {fwhm_exp:.3f} nm")

# ★★★ Signal 轴范围：保持原程序宽度 70 nm，不改绘图样式 ★★★
ls_vec = np.linspace(signal_center_th_785 - 35.0, signal_center_th_785 + 35.0, 600) 
LS, LI = np.meshgrid(ls_vec, li_uniform)

# ================= 6. 理论 PMF（在“实验坐标”的 idler 上计算：LI_phys = LI - SHIFT） =================
print(">>> Computing Theoretical PMF (Shifted)...")
LI_phys = LI - SHIFT_VAL
K_Map = get_k_material(LS, LI_phys)

PMF_Theo_Amp = calculate_pmf_design(K_Map, d_nm, A_SIGN)
PMF_Theo_Amp /= np.max(PMF_Theo_Amp)

center_col_idx = np.argmin(np.abs(ls_vec - signal_center_th_785))
theo_profile_int = PMF_Theo_Amp[:, center_col_idx]**2 
fwhm_theo = calculate_fwhm(li_uniform, theo_profile_int)
print(f"Theoretical PMF FWHM (@{signal_center_th_785:.3f}nm): {fwhm_theo:.3f} nm")

# ================= 7. 实验 PMF 重构（修复“卷绕”问题：用插值平移，禁止 np.roll） =================
print(">>> Reconstructing Experimental PMF (Interpolation Shifting, no wrap-around)...")
PMF_Recon_Int = np.zeros_like(LS)

ridge_curve_shifted = np.array([robust_solve_idler(s) for s in ls_vec]) + SHIFT_VAL
f_base = interp1d(li_uniform, int_exp_uniform, kind='linear', fill_value=0.0, bounds_error=False)

for i, s_val in enumerate(ls_vec):
    target_wl = ridge_curve_shifted[i]
    delta_wl = target_wl - peak_wl_exp
    # 关键：用插值实现“平移”，超出范围的部分填 0，避免底部卷到顶部
    PMF_Recon_Int[:, i] = f_base(li_uniform - delta_wl)

PMF_Recon_Amp = np.sqrt(PMF_Recon_Int)

# ================= 8. 纯度寻优（保持原程序一致） =================
print(">>> Optimizing Pump Bandwidth...")

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
    if energy == 0: return 0, JSA
    JSA_Norm = JSA / np.sqrt(energy)

    s = svd(JSA_Norm, compute_uv=False)
    purity = np.sum(s**4)
    return purity, JSA_Norm

pump_bws = np.linspace(0.2, 8.0, 40)

# Theo Opt
purity_theo_list = []
for bw in pump_bws:
    p, _ = calculate_purity(PMF_Theo_Amp, bw, LS, LI)
    purity_theo_list.append(p)
best_idx_theo = np.argmax(purity_theo_list)
best_bw_theo = pump_bws[best_idx_theo]
best_purity_theo = purity_theo_list[best_idx_theo]
_, JSA_Theo_Opt = calculate_purity(PMF_Theo_Amp, best_bw_theo, LS, LI)

# Exp Opt
purity_exp_list = []
for bw in pump_bws:
    p, _ = calculate_purity(PMF_Recon_Amp, bw, LS, LI)
    purity_exp_list.append(p)
best_idx_exp = np.argmax(purity_exp_list)
best_bw_exp = pump_bws[best_idx_exp]
best_purity_exp = purity_exp_list[best_idx_exp]
_, JSA_Exp_Opt = calculate_purity(PMF_Recon_Amp, best_bw_exp, LS, LI)

print(f"\n=== Results ===")
print(f"Theoretical FWHM: {fwhm_theo:.3f} nm  | Optimal Pump: {best_bw_theo:.2f} nm | Indistinguishability={best_purity_theo:.4f}")
print(f"Experimental FWHM: {fwhm_exp:.3f} nm | Optimal Pump: {best_bw_exp:.2f} nm | Indistinguishability={best_purity_exp:.4f}")

# ================= 9. 绘图（完全沿用原程序样式，不改风格） =================
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

# 使用 LogNorm，动态范围设定为 1e-4 (-40dB)
norm_cfg = LogNorm(vmin=1e-4, vmax=1.0) 

# 1. PMF Theo
im1 = axes[0,0].imshow(PMF_Theo_Amp**2 + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_cfg)
axes[0,0].set_title(f"1. Theoretical PMF (Log)\nFWHM={fwhm_theo:.2f}nm")
axes[0,0].set_ylabel("Idler Wavelength (nm)")
plt.colorbar(im1, ax=axes[0,0], label='Intensity')

# 2. PMF Exp
im2 = axes[0,1].imshow(PMF_Recon_Amp**2 + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_cfg)
axes[0,1].set_title(f"2. Exp Reconstructed PMF (Log)\nFWHM={fwhm_exp:.2f}nm")
plt.colorbar(im2, ax=axes[0,1], label='Intensity')

# 3. JSA Theo
I_JSA_Theo = np.abs(JSA_Theo_Opt)**2
I_JSA_Theo_Norm = I_JSA_Theo / np.max(I_JSA_Theo) 

im3 = axes[1,0].imshow(I_JSA_Theo_Norm + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_cfg)
axes[1,0].set_title(f"3. Optimized JSA (Theory, Log)\nPump={best_bw_theo:.2f}nm | Indistinguishability={best_purity_theo:.4f}")
axes[1,0].set_xlabel("Signal Wavelength (nm)")
axes[1,0].set_ylabel("Idler Wavelength (nm)")
plt.colorbar(im3, ax=axes[1,0], label='Intensity')

# 4. JSA Exp
I_JSA_Exp = np.abs(JSA_Exp_Opt)**2
I_JSA_Exp_Norm = I_JSA_Exp / np.max(I_JSA_Exp)

im4 = axes[1,1].imshow(I_JSA_Exp_Norm + 1e-15, extent=extent, origin='lower', aspect='auto', cmap=my_cmap, norm=norm_cfg)
axes[1,1].set_title(f"4. Optimized JSA (Exp Reconstruction, Log)\nPump={best_bw_exp:.2f}nm | Indistinguishability={best_purity_exp:.4f}")
axes[1,1].set_xlabel("Signal Wavelength (nm)")
plt.colorbar(im4, ax=axes[1,1], label='Intensity')

plt.tight_layout()
plt.show()
