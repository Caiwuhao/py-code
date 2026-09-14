# -*- coding: utf-8 -*-
# ==== SFG 总脚本（稳健拟合 + 正确 Fig.8 叠加）====
# 用法：
# 1) 把下面两行文件名改成你的 C/L 数据；
# 2) 直接运行，会生成 fig1~fig8（同目录）。

C_FILE = "SFG20251029105128.txt"   # C-band 数据（可改）
L_FILE = "SFG20251029180550.txt"   # L-band 数据（可改）

import io, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, brentq, least_squares

# ---------- 公用小函数 ----------
def to_db(y, ref):
    y = np.asarray(y, float)
    return 10*np.log10(np.maximum(y, 1e-15) / max(ref, 1e-15))

def DR_from_means_db(y):
    y = np.asarray(y, float); y_pos = y[y > 0]
    return np.nan if len(y_pos) < 2 else 10*np.log10(y_pos.max() / y_pos.min())

def DR_minmax_db(y):
    y = np.asarray(y, float); y_pos = y[y>0]
    if len(y_pos) < 2: return np.nan
    return 10*np.log10(y_pos.max()/y_pos.min())

def set_title_two_lines(ax, line1, line2=None):
    ax.set_title(line1 + ("\n"+line2 if line2 else ""))

# ---------- 全局风格 ----------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle":  "butt",
})

# ---------- 颜色/样式（与你约定一致） ----------
COL_DATA  = "#e69f00"   # 数据点 + 误差棒
COL_GAUSS = "#61b9ea"   # 高斯
COL_SINC2 = "#009d72"   # sinc²
COL_PMF   = "#b22222"   # 红线（理论 PMF）

GAUSS_LW  = 1.2
SINC_LW   = 1.2
PMF_LW    = 1.8
DATA_MS   = 2.0
ERR_ELW   = 0.8
ERR_CAP   = 0

# “长划线”虚线（兼容 set_linestyle / set_dashes）
SINC_DASH_TUPLE = (0, (6.0, 6.0))
def apply_sinc_dash(line):
    try:
        line.set_linestyle(SINC_DASH_TUPLE)
    except Exception:
        line.set_dashes([6.0, 6.0])

# ---------- 读表/分箱 ----------
def read_numeric_table(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln for ln in f.readlines() if re.match(r'^\s*[+-]?\d', ln)]
    arr = np.loadtxt(io.StringIO(''.join(lines)))
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    df = pd.DataFrame(arr)
    while df.shape[1] < 9: df[df.shape[1]] = np.nan
    df.columns = ["wl","p2","o_wl","p4","sfg","norm","bg00","bg01","bg10"]
    return df.dropna()

def drop50_stats(df, col):
    out=[]
    for wl,g in df.groupby("wl"):
        v = g[col].to_numpy()[50:]
        if len(v) > 5:
            out.append((wl, v.mean(), v.std(ddof=1)))
    return pd.DataFrame(out, columns=["wl","y","yerr"]).sort_values("wl")

def fourstep(df):
    out=[]
    for wl,g in df.groupby("wl"):
        g  = g.iloc[50:]
        v5 = g["sfg"];  v7 = g["bg00"]; v8 = g["bg01"]; v9 = g["bg10"]
        p2 = g["p2"];   p4 = g["p4"]
        if min(len(v5),len(v7),len(v8),len(v9),len(p2),len(p4)) < 5:
            continue
        A   = v5.mean() - v8.mean() - v9.mean() + v7.mean()
        sA  = math.sqrt(v5.std()**2 + v8.std()**2 + v9.std()**2 + v7.std()**2)
        D   = p2.mean() * p4.mean()
        out.append((wl, A/D, sA/D))
    return pd.DataFrame(out, columns=["wl","y","yerr"]).sort_values("wl")

# ---------- 模型/拟合 ----------
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

def gauss_offset(x, A, mu, sigma, B):
    x = np.asarray(x, float)
    return B + A*np.exp(-(x-mu)**2/(2.0*sigma**2))

_SINC2_T_HALF = 0.443113462726  # 使 (sinc(t_half))^2 = 1/2
def sinc2_curve_core(x, mu, FWHM, A):
    x = np.asarray(x, float)
    if FWHM <= 0:
        return np.full_like(x, float(A))
    a = (FWHM * 0.5) / _SINC2_T_HALF
    return A * (np.sinc((x - mu) / a))**2

def sinc2_curve(x, mu, FWHM, A):
    return sinc2_curve_core(x, mu, FWHM, A)

def initial_guess(x, y):
    i    = int(np.argmax(y))
    mu0  = float(x[i]); A0 = float(y[i]); half = A0/2
    xL, xR = mu0-0.8, mu0+0.8
    if i>0:
        left = np.where(y[:i] <= half)[0]
        if len(left):
            j = left[-1]
            xL = np.interp(half, [y[j], y[j+1]], [x[j], x[j+1]])
    if i < len(y)-1:
        right = np.where(y[i:] <= half)[0]
        if len(right):
            k = right[0] + i
            xR = np.interp(half, [y[k-1], y[k]], [x[k-1], x[k]])
    FWHM0 = max(0.2, min(8.0, xR-xL))
    sigma0 = FWHM0 / (2*np.sqrt(2*np.log(2)))
    return A0, mu0, sigma0

def fit_weighted_gauss(x, y, s):
    s = np.asarray(s, float)
    s[~np.isfinite(s) | (s<=0)] = np.median(s[s>0]) if np.any(s>0) else 1.0
    A0, mu0, sigma0 = initial_guess(x, y)
    bounds = ([0.0, x.min()-1.0, 0.05], [10.0*np.max(y), x.max()+1.0, 20.0])
    try:
        popt,_ = curve_fit(gauss, x, y, p0=[A0, mu0, sigma0],
                           sigma=s, absolute_sigma=True,
                           bounds=bounds, maxfev=100000)
        A, mu, sigma = map(float, popt)
    except Exception:
        A, mu, sigma = A0, mu0, sigma0
    FWHM = 2*np.sqrt(2*np.log(2))*sigma
    return A, mu, FWHM

# 稳健 4 参数高斯（A, mu, FWHM, B）
def fit_gauss4_robust(x, y, s=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)

    # 平滑找峰与初值
    win = max(11, (n//8)*2+1)
    win = min(win, (n//2)*2+1 if n>5 else 5)
    y_s = savgol_filter(y, win, 2, mode="interp")

    ipk  = int(np.argmax(y_s))
    mu0  = float(x[ipk])
    B0   = float(np.percentile(y, 5))
    A0   = max(float(y_s[ipk] - B0), 1e-12)

    # 半高宽初值
    half = B0 + 0.5*A0
    jL = np.where(y_s[:ipk] <= half)[0]
    xL = x[max(jL[-1], 0)] if len(jL) else (mu0 - 8.0)
    jR = np.where(y_s[ipk:] <= half)[0]
    xR = x[min(ipk + (jR[0] if len(jR) else 0), n-1)] if len(jR) else (mu0 + 8.0)
    FWHM0 = float(np.clip(xR - xL, 6.0, 16.0))
    sigma0 = FWHM0/(2*np.sqrt(2*np.log(2)))

    # 拟合窗口：仅 μ±20 nm
    mask = (x >= mu0 - 20.0) & (x <= mu0 + 20.0)
    if not np.any(mask): mask = slice(None)

    # 权重
    if s is None:
        w = np.ones_like(y[mask])
    else:
        s_loc = np.asarray(s, float)[mask]
        s_floor = np.nanmedian(s_loc[s_loc>0]) if np.any(s_loc>0) else 1.0
        w = 1.0/np.maximum(s_loc, s_floor)

    # 参数边界
    Fmin, Fmax = 6.0, 16.0
    sig_lo = Fmin/(2*np.sqrt(2*np.log(2)))
    sig_hi = Fmax/(2*np.sqrt(2*np.log(2)))
    A_hi   = 10.0*A0 if A0>0 else (np.nanmax(y) or 1.0)
    B_span = 0.02*A0
    p0     = np.array([A0, mu0, sigma0, B0], float)
    lo     = np.array([0.0, x.min()+1.0, sig_lo, -B_span], float)
    hi     = np.array([A_hi, x.max()-1.0, sig_hi,  B_span], float)

    def resid(p):
        A, mu, sig, B = p
        yhat = gauss_offset(x[mask], A, mu, sig, B)
        return (yhat - y[mask]) * w

    res = least_squares(resid, p0, bounds=(lo, hi),
                        loss="soft_l1", f_scale=3.0, max_nfev=20000)
    A, mu, sig, B = map(float, res.x)
    FWHM = 2*np.sqrt(2*np.log(2))*sig
    return A, mu, FWHM, B

# ---------- 线性轴绘图（带 B） ----------
def ylim_for_zoom(xd, y_model, xlim, y, yerr, pad=1.15):
    mask = (xd>=xlim[0]) & (xd<=xlim[1])
    ymax = 0.0
    if np.any(mask): ymax = max(ymax, float(np.nanmax(y_model[mask])))
    if len(y):       ymax = max(ymax, float(np.nanmax(y)))
    ymin = 0.0
    if len(y): ymin = min(0.0, float(np.nanmin(y - yerr)) * 1.15)
    return ymin, ymax*pad if ymax>0 else 1.0

def draw_panel(spec, mu, FWHM, A, B, xfull, out_png,
               title1, title2=None, zoom_xlim=None,
               label_data="data (mean ± 1σ)", ylim=None):

    x = spec["wl"].to_numpy()
    y = spec["y"].to_numpy()
    s = spec["yerr"].to_numpy()

    xd    = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    y_g   = gauss_offset(xd, A, mu, sigma, B)
    y_s   = B + sinc2_curve_core(xd, mu, FWHM, A)

    fig, ax = plt.subplots(figsize=(8.8,4.6))
    ax.errorbar(x, y, yerr=s, fmt='o', ms=DATA_MS,
                mfc=COL_DATA, mec=COL_DATA, mew=0,
                ecolor=COL_DATA, elinewidth=ERR_ELW, capsize=ERR_CAP,
                label=label_data)
    ax.plot(xd, y_g, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    line_s, = ax.plot(xd, y_s, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    apply_sinc_dash(line_s)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized SFG (a.u.)")
    set_title_two_lines(ax, title1, title2)

    ax.legend(loc="upper left", bbox_to_anchor=(0.02,0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)
    ax.grid(alpha=0.3)

    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ymin, ymax = ylim_for_zoom(xd, y_g, zoom_xlim, y, s)
            ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(xfull)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            y_all = np.concatenate([y, y_g, y_s])
            ypad  = 0.08*(np.nanmax(y_all)-np.nanmin(y_all) + 1e-15)
            ax.set_ylim(np.nanmin(y_all)-ypad, np.nanmax(y_all)+ypad)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png); plt.close(fig)
    print("Saved:", out_png)

# ---------- dB 轴（仅均值点；线性先 +B 再转 dB） ----------
def draw_panel_db_meanonly(spec, mu, FWHM, A, B, xfull, out_png,
                           title1, title2=None, zoom_xlim=None,
                           label_data="data (mean only)", ylim_db=(-50.0, 0.0)):

    x = spec["wl"].to_numpy()
    y = spec["y"].to_numpy()

    ref = float(np.nanmax(y))  # 以数据峰值为 0 dB
    def _to_db(arr):
        arr = np.asarray(arr, float)
        return 10*np.log10(np.maximum(arr, 1e-15)/max(ref, 1e-15))

    xd    = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))

    y_db   = _to_db(y)
    y_g_db = _to_db(gauss_offset(xd, A, mu, sigma, B))
    y_s_db = _to_db(B + sinc2_curve_core(xd, mu, FWHM, A))

    fig, ax = plt.subplots(figsize=(8.8,4.6))
    ax.plot(x, y_db, linestyle='none', marker='o', ms=DATA_MS,
            mfc=COL_DATA, mec=COL_DATA, mew=0, label=label_data)
    ax.plot(xd, y_g_db, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    line_s, = ax.plot(xd, y_s_db, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    apply_sinc_dash(line_s)

    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)
    else:
        ax.set_xlim(xfull)

    ax.set_ylim(ylim_db)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB vs peak)")
    set_title_two_lines(ax, title1, title2)

    ax.legend(loc="upper left", bbox_to_anchor=(0.02,0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png); plt.close(fig)
    print("Saved (dB, mean only):", out_png)

# ================== Fig.8 理论 PMF 叠加所需函数 ==================
LAM_S0     = 1290.56                  # nm (signal)
LAMBDA_UM  = 37.71066                 # μm (QPM period)
LAMBDA_NM  = LAMBDA_UM * 1000.0       # nm
D_HALF_NM  = LAMBDA_NM / 2.0

_A_RAW = [
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
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
A_SIGN = np.array(_A_RAW[:265], float)

def _ny_um(u): return np.sqrt(3.45018 + 0.04341/(u*u-0.04597) + 16.98825/(u*u-39.43799))
def _nz_um(u): return np.sqrt(4.59423 + 0.06206/(u*u-0.04763) + 110.80672/(u*u-86.12171))
def n_y_nm(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)
def n_z_nm(lam_nm): return _nz_um(np.asarray(lam_nm)*1e-3)
def n_p(lam_nm): return n_y_nm(lam_nm)  # np=ny
def n_s(lam_nm): return n_z_nm(lam_nm)  # ns=nz
def n_i(lam_nm): return n_y_nm(lam_nm)  # ni=ny

def lambda_p(ls_nm, li_nm):
    ls = np.asarray(ls_nm, float); li = np.asarray(li_nm, float)
    return (ls*li)/(ls+li)

# 基本相位失配（不含 grating） —— 用在 φ/φ2
def k_basic_nm(ls_nm, li_nm):
    lp = lambda_p(ls_nm, li_nm)
    return 2*np.pi*( n_p(lp)/lp - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm )

# 含 grating 的 Δk（用来解 λi0）：**注意这里是 + 2π/Λ**（与 Mathematica 一致）
def delta_k_grating_nm(ls_nm, li_nm):
    return k_basic_nm(ls_nm, li_nm) + 2*np.pi/LAMBDA_NM

def _phi_of_k(d_nm, k):
    k = np.atleast_1d(k).astype(complex)
    j = np.arange(A_SIGN.size, dtype=float)
    e_minus = np.exp(-1j*k*d_nm)
    out = np.empty_like(k, dtype=complex)
    for t, kv in enumerate(k):
        s = np.exp(1j*kv*d_nm*j) @ A_SIGN
        out[t] = 1j * s * (e_minus[t]-1.0) / (kv+0j)
    return np.abs(out)

def phi2_lambda(ls_nm, li_nm):
    return _phi_of_k(D_HALF_NM, k_basic_nm(ls_nm, li_nm))

def solve_lambda_i0(ls_nm, a0=1505.0, b0=1565.0):
    f = lambda li: float(delta_k_grating_nm(ls_nm, li))
    a,b = a0,b0
    for _ in range(160):
        fa, fb = f(a), f(b)
        if np.sign(fa) != np.sign(fb):
            return brentq(f, a, b, maxiter=800)
        a -= 0.5; b += 0.5
    grid = np.linspace(a0-30, b0+30, 60001)
    vals = np.array([f(v) for v in grid])
    return float(grid[np.argmin(np.abs(vals))])

def theory_curve_shifted_for_fig7(mu_center, xfull, step_nm=0.01):
    li0   = solve_lambda_i0(LAM_S0)       # ≈ 1536.648...
    shift = float(mu_center) - li0        # 对齐到数据 μ
    li    = np.arange(xfull[0]-shift, xfull[1]-shift+1e-9, step_nm)
    x_plot = li + shift
    y_lin  = (phi2_lambda(LAM_S0, li)**2)
    y_lin /= np.nanmax(y_lin)             # 红线自身最大=1 → 0 dB
    y_db   = 10.0*np.log10(np.maximum(y_lin, 1e-20))
    return x_plot, y_db

def draw_fig8_with_theory(spec_full_4, mu_fourstep, FWHM_fourstep, A_fourstep, B_fourstep,
                          xfull, out_png="fig8_full_4step_dB_withTheory.png"):
    # ——底图：等同 Fig.7 的画法（仅均值，0 dB 基准=数据峰值）——
    x = spec_full_4["wl"].to_numpy()
    y = spec_full_4["y"].to_numpy()
    ref = float(np.nanmax(y))
    to_db_local = lambda arr: 10*np.log10(np.maximum(arr, 1e-15)/ref)

    xd    = np.linspace(xfull[0], xfull[1], 1400)
    sig   = FWHM_fourstep / (2*np.sqrt(2*np.log(2)))
    y_db  = to_db_local(y)
    y_gdb = to_db_local(gauss_offset(xd, A_fourstep, mu_fourstep, sig, B_fourstep))
    y_sdb = to_db_local(B_fourstep + sinc2_curve_core(xd, mu_fourstep, FWHM_fourstep, A_fourstep))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(x, y_db, linestyle='none', marker='o', ms=DATA_MS,
            mfc=COL_DATA, mec=COL_DATA, mew=0, label="data (mean only)")
    ax.plot(xd, y_gdb, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    line_s, = ax.plot(xd, y_sdb, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    apply_sinc_dash(line_s)

    # ——红线：理论 PMF（idler），横轴对齐到 μ；红线自身 peak = 0 dB——
    xs, y_th = theory_curve_shifted_for_fig7(mu_fourstep, xfull, step_nm=0.01)
    ax.plot(xs, y_th, color=COL_PMF, lw=PMF_LW, label="CPKTP PMF (idler theory)")

    ax.set_xlim(xfull)
    ax.set_ylim(-50.0, 0.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.6,
              handletextpad=0.6, borderpad=0.3)

    title1 = "Four-step BG-removed Normalized SFG vs Wavelength (C+L, dB scale)"
    title2 = f"μ={mu_fourstep:.3f} nm, FWHM={FWHM_fourstep:.3f} nm  (CPKTP PMF overlaid)"
    ax.set_title(title1 + "\n" + title2)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

# ================== 主流程 ==================
# 读数据
dfC = read_numeric_table(C_FILE)
specC_norm = drop50_stats(dfC, "norm")
specC_4st  = fourstep(dfC)

dfL = read_numeric_table(L_FILE)
specL_norm = drop50_stats(dfL, "norm")
specL_4st  = fourstep(dfL)

# 取范围
x_min = float(min(specC_norm["wl"].min(), specC_4st["wl"].min(),
                  specL_norm["wl"].min(), specL_4st["wl"].min()))
x_max = float(max(specC_norm["wl"].max(), specC_4st["wl"].max(),
                  specL_norm["wl"].max(), specL_4st["wl"].max()))
xfull = (x_min, x_max)
C_xlim = (float(specC_norm["wl"].min()) - 0.7, float(specC_norm["wl"].max()) + 0.7)
L_xlim = (float(specL_norm["wl"].min()) - 0.7, float(specL_norm["wl"].max()) + 0.7)

# y 轴范围（线性）
def y_span(spec):
    y = spec["y"].to_numpy(); s = spec["yerr"].to_numpy()
    lo = np.nanmin(y - s); hi = np.nanmax(y + s); sp = max(1e-12, hi-lo)
    return (min(lo - 0.08*sp, -0.03*np.nanmax(y)), hi + 0.08*sp)

ylim_C_norm   = y_span(specC_norm)
ylim_C_4step  = y_span(specC_4st)
ylim_L_norm   = y_span(specL_norm)

# 拟合（C-band）
A_Cn, mu_Cn, FWHM_Cn, B_Cn = fit_gauss4_robust(specC_norm["wl"], specC_norm["y"], specC_norm["yerr"])
A_C4, mu_C4, FWHM_C4, B_C4 = fit_gauss4_robust(specC_4st["wl"],  specC_4st["y"],  specC_4st["yerr"])

DR_Cn      = DR_minmax_db(specC_norm["y"])
DR_C4      = DR_minmax_db(specC_4st["y"])
spec_full_norm = pd.concat([specC_norm, specL_norm], ignore_index=True).sort_values("wl")
spec_full_4    = pd.concat([specC_4st,  specL_4st],  ignore_index=True).sort_values("wl")
DR_full_n  = DR_minmax_db(spec_full_norm["y"])
DR_full_4  = DR_minmax_db(spec_full_4["y"])

# ====== 出图 1~6：线性轴 ======
draw_panel(specC_norm, mu_Cn, FWHM_Cn, A_Cn, B_Cn, xfull,
           "fig1_C_band_norm.png",
           "Normalized SFG vs Wavelength - Gaussian Fit",
           f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_Cn:.2f} dB",
           zoom_xlim=C_xlim, ylim=ylim_C_norm)

draw_panel(specC_4st, mu_C4, FWHM_C4, A_C4, B_C4, xfull,
           "fig2_C_band_4step.png",
           "Four-step BG-removed Normalized SFG vs Wavelength - Gaussian Fit",
           f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_C4:.2f} dB",
           zoom_xlim=C_xlim, ylim=ylim_C_4step)

draw_panel(specL_norm, mu_Cn, FWHM_Cn, A_Cn, B_Cn, xfull,
           "fig3_L_band_zoom_norm.png",
           "Normalized SFG vs Wavelength (L-band zoom)\nC-model overlaid",
           zoom_xlim=L_xlim, label_data="L-band data (mean ± 1σ)",
           ylim=ylim_L_norm)

draw_panel(specL_4st,  mu_C4, FWHM_C4, A_C4, B_C4, xfull,
           "fig4_L_band_zoom_4step.png",
           "Four-step BG-removed Normalized SFG (L-band zoom)\nC-model overlaid",
           zoom_xlim=L_xlim, label_data="L-band data (mean ± 1σ)")

draw_panel(spec_full_norm, mu_Cn, FWHM_Cn, A_Cn, B_Cn, xfull,
           "fig5_full_norm.png",
           "Normalized SFG vs Wavelength - Gaussian Fit (C+L)",
           f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_full_n:.2f} dB")

draw_panel(spec_full_4, mu_C4, FWHM_C4, A_C4, B_C4, xfull,
           "fig6_full_4step.png",
           "Four-step BG-removed Normalized SFG vs Wavelength (C+L)\nGaussian Fit",
           f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_full_4:.2f} dB")

# ====== 出图 7：dB 轴（仅均值）======
draw_panel_db_meanonly(spec_full_4, mu_C4, FWHM_C4, A_C4, B_C4, xfull,
                       "fig7_full_4step_dB.png",
                       "Four-step BG-removed Normalized SFG vs Wavelength (C+L, dB scale)",
                       ylim_db=(-50.0, 0.0))

# ====== 出图 8：在 Fig.7 上叠加 CPKTP PMF（idler theory）======
draw_fig8_with_theory(spec_full_4, mu_C4, FWHM_C4, A_C4, B_C4,
                      xfull, out_png="fig8_full_4step_dB_withTheory.png")
