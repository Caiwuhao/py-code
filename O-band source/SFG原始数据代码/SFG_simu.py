# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import brentq

# ================== 1) 晶体参数/畴序列 ==================
LAM_S0 = 1290.56                     # nm (给定信号波长)
LAMBDA_UM = 37.71066                 # μm (QPM 周期)
LAMBDA_NM = LAMBDA_UM * 1000.0       # nm
D_HALF_NM = LAMBDA_NM / 2.0          # 单个半畴长度 d = Λ/2

# 你粘贴的原始 266 个 a_j（Mathematica 导出）
_A_RAW = [
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1,
  1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1,
  1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1,
  1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1,
  1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1,
  1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
  1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
  1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
  1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
   1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1,
  1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1,
  1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1,
  1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1,
  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1
]
# 有效长度 5 mm => 265 个半畴；多出来的 1 个丢弃
A_SIGN = np.array(_A_RAW[:265], dtype=float)
assert A_SIGN.size == 265, f"A_SIGN length={A_SIGN.size} (expected 265)"

# ================== 2) 折射率 (Kato & Takaoka; λ: μm) ==================
def _ny_um(lam_um):
    n2 = 3.45018 + 0.04341/(lam_um**2 - 0.04597) + 16.98825/(lam_um**2 - 39.43799)
    return np.sqrt(n2)

def _nz_um(lam_um):
    n2 = 4.59423 + 0.06206/(lam_um**2 - 0.04763) + 110.80672/(lam_um**2 - 86.12171)
    return np.sqrt(n2)

def n_y_nm(lam_nm): return _ny_um(np.asarray(lam_nm, float)*1e-3)   # y-axis index
def n_z_nm(lam_nm): return _nz_um(np.asarray(lam_nm, float)*1e-3)   # z-axis index

# 极化：np=ny, ns=nz, ni=ny
def n_p(lam_nm): return n_y_nm(lam_nm)
def n_s(lam_nm): return n_z_nm(lam_nm)
def n_i(lam_nm): return n_y_nm(lam_nm)

# 三波关系（波长）： 1/λp = 1/λs + 1/λi
def lambda_p(ls_nm, li_nm):
    ls_nm = np.asarray(ls_nm, float); li_nm = np.asarray(li_nm, float)
    return (ls_nm * li_nm) / (ls_nm + li_nm)

# ================== 3) 相位失配（两种定义） ==================
# 基本相位失配（不含 grating 项）—— 用在 φ/φ2
def k_basic_nm(ls_nm, li_nm):
    lp = lambda_p(ls_nm, li_nm)
    return 2*np.pi*( n_p(lp)/lp - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm )  # [1/nm]

# 含 grating 的 Δk（= k_basic - 2π/Λ）—— 用来求 li0
def delta_k_grating_nm(ls_nm, li_nm):
    # 正确：与 Mathematica 的 +1/Λ 保持一致
    return k_basic_nm(ls_nm, li_nm) + 2*np.pi/LAMBDA_NM


# Mathematica 的“点验形式”：Δk = 2π( np/lp - ns/ls - ni/li + 1/Λ )
def delta_k_pointcheck(ls_nm, li_nm):
    lp = lambda_p(ls_nm, li_nm)
    return 2*np.pi*( n_p(lp)/lp - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm + 1.0/LAMBDA_NM )

# ================== 4) ϕ 与 Φ2（严格按你的式子） ==================
def phi_of_k(d_nm, k):   # ϕ(d,k) = | i Σ a_j e^{ik j d} * (e^{-ik d}-1)/k |
    k = np.atleast_1d(k).astype(complex)
    j = np.arange(A_SIGN.size, dtype=float)
    out = np.empty_like(k, dtype=complex)
    e_minus = np.exp(-1j*k*d_nm)
    for t, kv in enumerate(k):
        s = np.exp(1j*kv*d_nm*j) @ A_SIGN
        out[t] = 1j * s * (e_minus[t] - 1.0) / (kv + 0j)
    return np.abs(out)

def phi2_lambda(ls_nm, li_nm):
    return phi_of_k(D_HALF_NM, k_basic_nm(ls_nm, li_nm))

# ================== 5) 在给定 λs 下求 li0（Δk_grating=0） ==================
def solve_lambda_i0(ls_nm, a0=1500.0, b0=1605.0):
    f = lambda li: float(delta_k_grating_nm(ls_nm, li))
    a, b = a0, b0
    # 先尝试找到异号区间
    for _ in range(160):
        fa, fb = f(a), f(b)
        if np.sign(fa) != np.sign(fb):
            return brentq(f, a, b, maxiter=800)
        a -= 0.5; b += 0.5
    # 兜底：网格上找 |Δk| 最小点
    grid = np.linspace(a0-30, b0+30, 60001)
    return float(grid[np.argmin(np.abs([f(v) for v in grid]))])

# ================== 6) 主流程（只打印数值） ==================
if __name__ == "__main__":
    li0_target = 1536.6481669897387  # 你给的中心闲频（nm）
    C_REF = 2.3239992305511396e12    # 归一化常数 = Φ2(ls0, li0)^2

    # (a) 点验 Δk 是否接近 0
    dk_check = float(delta_k_pointcheck(LAM_S0, li0_target))
    print("Delta k at (ls0, li0_target) =", f"{dk_check:.6e}", "1/nm  (should be ~ 0)")

    # (b) 反解 li0（Δk_grating=0）
    li0 = solve_lambda_i0(LAM_S0, a0=1505.0, b0=1565.0)
    dk_gr = float(delta_k_grating_nm(LAM_S0, li0))
    print("Solved lambda_i0             =", f"{li0:.9f}", "nm")
    print("Delta k_grating at li0       =", f"{dk_gr:.3e}", "1/nm  (≈ 0)")

    # (c) 计算 Φ2 并与 C_REF 对比（φ 用 k_basic，d=Λ/2）
    phi2_val = float(phi2_lambda(LAM_S0, li0))
    phi2_sq  = phi2_val**2
    ratio    = phi2_sq / C_REF
    print("Phi2(ls0, li0)^2            =", f"{phi2_sq:.12e}")
    print("Phi2^2 / C_REF               =", f"{ratio:.9f}", "(≈ 1.0)")
    print("A_SIGN length used           =", A_SIGN.size)

import matplotlib.pyplot as plt

# ========= 绘制 φ2(λi)（线性域） =========
lam_i_range = np.linspace(1505, 1565, 601)  # nm
phi2_vals = phi2_lambda(LAM_S0, lam_i_range)
phi2_sq = (phi2_vals**2) / C_REF            # 归一化，使中心峰=1

plt.figure(figsize=(8,4))
plt.plot(lam_i_range, phi2_sq, color='#b22222', lw=2.0)
plt.title("CPKTP PMF (idler projection) — linear scale", fontsize=12)
plt.xlabel("Idler wavelength λᵢ (nm)")
plt.ylabel(r"$|\Phi_2|^2 / C_{\rm ref}$ (normalized)")
plt.xlim(1505, 1565)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ========= 绘制 φ2(λi) 的 dB 图 =========
phi2_db = 10 * np.log10(np.maximum(phi2_sq, 1e-20))  # 避免 log(0)
phi2_db -= np.max(phi2_db)  # 归一化到 0 dB 峰值

plt.figure(figsize=(8,4))
plt.plot(lam_i_range, phi2_db, color='#b22222', lw=2.0)
plt.title("CPKTP PMF (idler projection) — dB scale", fontsize=12)
plt.xlabel("Idler wavelength λᵢ (nm)")
plt.ylabel("Relative power (dB vs peak)")
plt.ylim(-50, 0)
plt.xlim(1505, 1565)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===== 参数 =====
SHIFT_NM = 1555.839022075793 - 1536.6481669897387    # 19.1908550860543 nm
X_MIN, X_MAX = 1528.8, 1608.6                         # 目标显示范围（nm）
STEP = 0.01

# ===== 基准常数 C_ref = Φ2(ls0, li0)^2 =====
li0 = solve_lambda_i0(LAM_S0)                 # 你脚本里已有
C_REF = float(phi2_lambda(LAM_S0, li0)**2)    # = 2.3239992305511396e12

# ===== 生成数据：先在移位前的区间采样，再右移 =====
li_grid = np.arange(X_MIN - SHIFT_NM, X_MAX - SHIFT_NM + 0.5*STEP, STEP)
x_plot  = li_grid + SHIFT_NM
y_norm  = (phi2_lambda(LAM_S0, li_grid)**2) / C_REF
y_db    = 10.0 * np.log10(np.maximum(y_norm, 1e-20))   # 避免 -inf

# ===== 画图到文件（不打印、不显示）=====
import matplotlib.pyplot as plt
plt.figure(figsize=(9,4.5))
plt.plot(x_plot, y_db, color='#b22222', lw=2.2)  # 深红 + 线宽
plt.xlim(X_MIN, X_MAX)
plt.ylim(-50, 0)
plt.xlabel("Idler wavelength λᵢ (nm)")
plt.ylabel("Relative power (dB vs peak)")
plt.title("CPKTP PMF (idler projection, shifted) — dB scale")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("CPKTP_PMF_idler_shifted_to_fig_range_dB.png", dpi=400)
plt.close()

