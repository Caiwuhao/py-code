import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import time

# ================= 1. 物理参数与极化序列 =================
#
LAMBDA_UM = 37.71066
d_um = LAMBDA_UM / 2.0  # 畴长 d = Lc
d_nm = d_um * 1000.0

# 极化序列 (A_SIGN)
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
# 取前 265 个 (与您的文件一致)
A_SIGN = np.array(_A_RAW[:265], dtype=float)

# 折射率公式 (Py_SFG_correct-5.py)
def _ny_um(u): return np.sqrt(3.45018 + 0.04341/(u*u-0.04597) + 16.98825/(u*u-39.43799))
def _nz_um(u): return np.sqrt(4.59423 + 0.06206/(u*u-0.04763) + 110.80672/(u*u-86.12171))

def n_p(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)
def n_s(lam_nm): return _nz_um(np.asarray(lam_nm)*1e-3)
def n_i(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)

# 基础波矢失配 k = kp - ks - ki
def get_k_material(ls_nm, li_nm):
    lp_nm = (ls_nm * li_nm) / (ls_nm + li_nm)
    # k (rad/nm)
    val = 2 * np.pi * (n_p(lp_nm)/lp_nm - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm)
    return val

# ================= 2. PMF 计算核心 (严格复现 Mathematica 公式) =================

def calculate_pmf_large_scale(k_grid, d, domain_signs):
    """
    Sum[ a[j] * Exp[i * k * j * d] * (Exp[-i*k*d]-1)/k ]
    """
    # 1. 形状因子项: (Exp[-i*k*d] - 1) / k
    # 避免 k=0 除零错误
    k_safe = np.where(k_grid == 0, 1e-15, k_grid)
    term_shape = (np.exp(-1j * k_safe * d) - 1.0) / k_safe
    
    # 2. 累加项: Sum[ a[j] * Exp[i * k * j * d] ]
    # 使用循环累加，避免构建超大 3D 矩阵
    print(f"   Calculating Summation over {len(domain_signs)} domains...")
    sum_val = np.zeros_like(k_grid, dtype=complex)
    
    # 预计算相位的基数: phi_0 = k * d
    phi_0 = k_grid * d
    
    for j, sign in enumerate(domain_signs):
        # 相位 = k * j * d = phi_0 * j
        # 使用欧拉公式 exp(ix) = cos(x) + i*sin(x) 或直接用 np.exp
        term_phase = np.exp(1j * phi_0 * j)
        sum_val += sign * term_phase
        
    # 3. 最终结果: | i * sum * term_shape |
    # Mathematica公式中有个 I (虚数单位)，取模后不影响
    pmf = np.abs(sum_val * term_shape)
    return pmf

# ================= 3. 配置宽范围网格 =================
# 设置绘图范围 (根据您上传图片的示意)
LS_MIN, LS_MAX = 1270.0, 1310.0  # Signal 范围
LI_MIN, LI_MAX = 1510.0, 1560.0  # Idler 范围

# 提高分辨率以捕捉细节
GRID_W, GRID_H = 800, 800 

ls_vec = np.linspace(LS_MIN, LS_MAX, GRID_W)
li_vec = np.linspace(LI_MIN, LI_MAX, GRID_H)
LS, LI = np.meshgrid(ls_vec, li_vec)

# ================= 4. 执行计算 =================
print(">>> Step 1: Generating K Map...")
t0 = time.time()
K_Map = get_k_material(LS, LI)
print(f"    Map generated in {time.time()-t0:.2f}s")

print(">>> Step 2: Integrating PMF (Large Range)...")
t0 = time.time()
PMF_Val = calculate_pmf_large_scale(K_Map, d_nm, A_SIGN)
print(f"    Integration done in {time.time()-t0:.2f}s")

# 计算理论中心点（用于标记）
FIXED_SIGNAL = 1290.0
try:
    match_func = lambda li: get_k_material(FIXED_SIGNAL, li) - (2*np.pi/LAMBDA_NM)
    THEO_IDLER = brentq(match_func, 1500, 1600)
except:
    THEO_IDLER = 1536.0

# ================= 5. 绘图 (仿照您的图片风格) =================
plt.figure(figsize=(10, 8))

# 归一化并平方得到强度
Intensity = PMF_Val**2
Intensity /= np.max(Intensity)

# 画图
# 使用 'jet' 或 'inferno' 配色，类似物理仿真图
# extent = [xmin, xmax, ymin, ymax]
extent = [LS_MIN, LS_MAX, LI_MIN, LI_MAX]
plt.imshow(Intensity, extent=extent, origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='Normalized PMF Intensity')

# 标记
plt.plot(FIXED_SIGNAL, THEO_IDLER, 'w+', ms=15, mew=2, label=f'Theoretical Center ({FIXED_SIGNAL:.0f}, {THEO_IDLER:.1f})')
# 标记实验测量位置 (作为参考)
plt.plot(FIXED_SIGNAL, 1556.0, 'rx', ms=12, mew=2, label='Measured Center (~1556)')

plt.xlabel('Signal Wavelength (nm)')
plt.ylabel('Idler Wavelength (nm)')
plt.title(f'Theoretical 2D PMF (Large Range View)\nBased on Custom Design Pattern (265 domains)')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()