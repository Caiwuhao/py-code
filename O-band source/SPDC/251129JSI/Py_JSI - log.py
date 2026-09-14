import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.linalg import svd
import io, re, os

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
    # 修正符号: + 2pi/Lambda (确保 Shift ~ 20nm)
    def target_func(li):
        return get_k_material(ls_nm, li) + 2*np.pi/LAMBDA_NM

    scan_range = np.linspace(1400, 1800, 200)
    vals = np.array([target_func(li) for li in scan_range])
    
    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        try: return brentq(target_func, scan_range[idx], scan_range[idx+1])
        except: pass
            
    # 备选：反向符号
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

# ================= 2. 加载实验数据 (无裁剪) =================
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
    
    # 简单去底噪 (保留原始数据范围)
    bg = np.mean(np.sort(y)[:20])
    y_clean = y - bg
    y_clean[y_clean<0] = 0
    if np.max(y_clean) > 0: y_clean /= np.max(y_clean)
    
    return data[:,0], y_clean

wl_exp, int_exp = load_data_full(FILES)
if wl_exp is None:
    print("Error: No data loaded.")
    exit()

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

# 网格设置
num_points = len(wl_exp)
li_uniform = np.linspace(wl_exp.min(), wl_exp.max(), num_points)
int_exp_uniform = interp1d(wl_exp, int_exp, kind='linear', fill_value=0.0, bounds_error=False)(li_uniform)

# 设定范围: 1260-1320 nm
ls_vec = np.linspace(1270, 1340, 400) 
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
print(f"Theoretical FWHM: {fwhm_theo:.3f} nm  | Optimal Pump: {best_bw_theo:.2f} nm | Purity: {best_purity_theo:.4f}")
print(f"Experimental FWHM: {fwhm_exp:.3f} nm | Optimal Pump: {best_bw_exp:.2f} nm | Purity: {best_purity_exp:.4f}")

# ================= 6. 绘图 (Log Scale) =================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
extent = [ls_vec[0], ls_vec[-1], li_uniform[0], li_uniform[-1]]

def get_log_db(intensity, dyn_range=40):
    val = intensity.copy()
    val[val < 1e-10] = 1e-10
    log_val = 10 * np.log10(val)
    max_val = np.max(log_val)
    vmin = max_val - dyn_range
    return log_val, vmin, max_val

# 1. PMF Theo
log_theo, vmin, vmax = get_log_db(PMF_Theo_Amp**2)
im1 = axes[0,0].imshow(log_theo, extent=extent, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
axes[0,0].set_title(f"1. Theoretical PMF (Log)\nFWHM={fwhm_theo:.2f}nm")
axes[0,0].set_ylabel("Idler Wavelength (nm)")
axes[0,0].plot(FIXED_SIGNAL, peak_wl_exp, 'w+', ms=10)
plt.colorbar(im1, ax=axes[0,0], label='dB')

# 2. PMF Exp
log_exp, vmin, vmax = get_log_db(PMF_Recon_Amp**2)
im2 = axes[0,1].imshow(log_exp, extent=extent, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
axes[0,1].set_title(f"2. Exp Reconstructed PMF (Log)\nFWHM={fwhm_exp:.2f}nm")
axes[0,1].plot(ls_vec, ridge_curve_shifted, 'w--', alpha=0.5)
plt.colorbar(im2, ax=axes[0,1], label='dB')

# 3. JSA Theo
im3 = axes[1,0].imshow(np.abs(JSA_Theo_Opt)**2, extent=extent, origin='lower', aspect='auto', cmap='viridis')
axes[1,0].set_title(f"3. Optimized JSA (Theory)\nPump={best_bw_theo:.2f}nm | Purity={best_purity_theo:.4f}")
axes[1,0].set_xlabel("Signal Wavelength (nm)")
axes[1,0].set_ylabel("Idler Wavelength (nm)")
plt.colorbar(im3, ax=axes[1,0])

# 4. JSA Exp
im4 = axes[1,1].imshow(np.abs(JSA_Exp_Opt)**2, extent=extent, origin='lower', aspect='auto', cmap='viridis')
axes[1,1].set_title(f"4. Optimized JSA (Exp Reconstruction)\nPump={best_bw_exp:.2f}nm | Purity={best_purity_exp:.4f}")
axes[1,1].set_xlabel("Signal Wavelength (nm)")
plt.colorbar(im4, ax=axes[1,1])

plt.tight_layout()
plt.show()