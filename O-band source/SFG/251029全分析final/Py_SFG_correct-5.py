# ================== 替换为你的文件名 ==================
#C_FILE = "SFG20251024173128.txt"
#L_FILE = "SFG20251027120732.txt"
C_FILE = "SFG20251029105128.txt"
L_FILE = "SFG20251029180550.txt"

# -*- coding: utf-8 -*-
import io, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit, brentq

def to_db(y, ref):
    y = np.asarray(y, float)
    return 10*np.log10(np.maximum(y, 1e-15) / max(ref, 1e-15))

def DR_from_means_db(y):
    y = np.asarray(y, float); y_pos = y[y > 0]
    return np.nan if len(y_pos) < 2 else 10*np.log10(y_pos.max() / y_pos.min())

# ========= 全局风格 =========
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle":  "butt",
})

# ========= 颜色/样式 =========
COL_DATA  = "#e69f00"
COL_GAUSS = "#61b9ea"
COL_SINC2 = "#009d72"
COL_PMF   = "#d62728"

GAUSS_LW  = 1.2
SINC_LW   = 1.2
SINC_DASH = [3.0, 3.0]

DATA_MS   = 2.0
ERR_ELW   = 0.8
ERR_CAP   = 0

# ========= 读表 =========
def read_numeric_table(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln for ln in f.readlines() if re.match(r'^\s*[+-]?\d', ln)]
    arr = np.loadtxt(io.StringIO(''.join(lines)))
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    df = pd.DataFrame(arr)
    while df.shape[1] < 9: df[df.shape[1]] = np.nan
    df.columns = ["wl","p2","o_wl","p4","sfg","norm","bg00","bg01","bg10"]
    return df.dropna()

# ========= 每 λ 丢前 50，求均值/标准差 =========
def drop50_stats(df, col):
    out=[]
    for wl,g in df.groupby("wl"):
        v = g[col].to_numpy()[50:]
        if len(v) > 5:
            out.append((wl, v.mean(), v.std(ddof=1)))
    return pd.DataFrame(out, columns=["wl","y","yerr"]).sort_values("wl")

# ========= 四步相减相加（丢前 50） =========
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

# ========= 模型/拟合 =========
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

def sinc2_curve(x, mu, FWHM, A):
    t_half = 0.443
    a = (FWHM/2) / t_half if FWHM>0 else 1.0
    return A * (np.sinc((x - mu)/a))**2

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

def DR_minmax_db(y):
    y = np.asarray(y, float); y_pos = y[y>0]
    if len(y_pos) < 2: return np.nan
    return 10*np.log10(y_pos.max()/y_pos.min())

def set_title_two_lines(ax, line1, line2=None):
    ax.set_title(line1 + ("\n"+line2 if line2 else ""))

def ylim_for_zoom(xd, y_model, xlim, y, yerr, pad=1.15):
    mask = (xd>=xlim[0]) & (xd<=xlim[1])
    ymax = 0.0
    if np.any(mask): ymax = max(ymax, float(np.nanmax(y_model[mask])))
    if len(y):       ymax = max(ymax, float(np.nanmax(y)))
    ymin = 0.0
    if len(y): ymin = min(0.0, float(np.nanmin(y - yerr)) * 1.15)
    return ymin, ymax*pad if ymax>0 else 1.0

# ========= 统一绘图 =========
def draw_panel(spec, mu, FWHM, A, xfull, out_png, title1, title2=None,
               zoom_xlim=None, label_data="data (mean ± 1σ)", ylim=None):
    x = spec["wl"].to_numpy(); y = spec["y"].to_numpy(); s = spec["yerr"].to_numpy()
    xd = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    y_g = gauss(xd, A, mu, sigma)
    y_s = sinc2_curve(xd, mu, FWHM, A)

    fig, ax = plt.subplots(figsize=(8.8,4.6))
    ax.errorbar(x, y, yerr=s, fmt='o', ms=DATA_MS,
                mfc=COL_DATA, mec=COL_DATA, mew=0,
                ecolor=COL_DATA, elinewidth=ERR_ELW, capsize=ERR_CAP,
                label=label_data)
    lg, = ax.plot(xd, y_g, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s, color=COL_SINC2, lw=SINC_LW, label="sinc²"); ls.set_dashes(SINC_DASH)

    set_title_two_lines(ax, title1, title2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized SFG (a.u.)")
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
        if ylim is not None:
            ax.set_ylim(ylim)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png); plt.close(fig)
    print("Saved:", out_png)

def draw_panel_db(spec, mu, FWHM, A, xfull, out_png, title1, title2=None,
                  zoom_xlim=None, label_data="data (mean ± 1σ)"):
    x = spec["wl"].to_numpy(); y = spec["y"].to_numpy(); s = spec["yerr"].to_numpy()
    ref = float(np.nanmax(y))
    y_c_db  = to_db(y, ref)
    y_lo_db = to_db(np.maximum(y - s, 1e-15), ref)
    y_hi_db = to_db(y + s, ref)
    yerr_db = np.vstack([y_c_db - y_lo_db, y_hi_db - y_c_db])

    xd = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    y_g_db  = to_db(gauss(xd, A, mu, sigma), ref)
    y_s_db  = to_db(sinc2_curve(xd, mu, FWHM, A), ref)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.errorbar(x, y_c_db, yerr=yerr_db, fmt='o', ms=DATA_MS,
                mfc=COL_DATA, mec=COL_DATA, mew=0,
                ecolor=COL_DATA, elinewidth=ERR_ELW, capsize=ERR_CAP,
                label=label_data)
    lg, = ax.plot(xd, y_g_db, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s_db, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try: ls.set_linestyle((0, (6, 6)))
    except TypeError: ls.set_dashes([6, 6])

    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)
    db_min = float(np.nanmin(y_lo_db))
    ax.set_ylim(db_min - 2.0, 1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB vs peak)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)
    ax.grid(alpha=0.3)

    dr_db = DR_from_means_db(y)
    title2 = (title2 or f"μ={mu:.3f} nm, FWHM={FWHM:.3f} nm, DR={dr_db:.2f} dB")
    set_title_two_lines(ax, title1, title2)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(out_png); plt.close(fig)
    print(f"Saved (dB): {out_png}  |  DR={dr_db:.2f} dB")

def draw_panel_db_meanonly(spec, mu, FWHM, A, xfull, out_png, title1, title2=None,
                           zoom_xlim=None, label_data="data (mean only)", ylim_db=(-50.0, 0.0)):
    x = spec["wl"].to_numpy(); y = spec["y"].to_numpy()
    ref = float(np.nanmax(y))
    def _to_db(arr, refv): arr = np.asarray(arr, float); return 10*np.log10(np.maximum(arr, 1e-15) / max(refv, 1e-15))
    y_db = _to_db(y, ref)

    xd = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    y_g_db = _to_db(gauss(xd, A, mu, sigma), ref)
    y_s_db = _to_db(sinc2_curve(xd, mu, FWHM, A), ref)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(x, y_db, linestyle='none', marker='o', ms=DATA_MS,
            mfc=COL_DATA, mec=COL_DATA, mew=0, label=label_data)
    lg, = ax.plot(xd, y_g_db, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s_db, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try: ls.set_linestyle((0, (6, 6)))
    except TypeError: ls.set_dashes([6, 6])

    if zoom_xlim is not None: ax.set_xlim(zoom_xlim)
    ax.set_ylim(ylim_db)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB vs peak)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)
    ax.grid(alpha=0.3)

    if title2 is None:
        title2 = f"μ={mu:.3f} nm, FWHM={FWHM:.3f} nm"
    set_title_two_lines(ax, title1, title2)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png); plt.close(fig)
    print("Saved (dB, mean only):", out_png)

# ================== 做 C 模型（加权高斯） ==================
dfC = read_numeric_table(C_FILE)
specC_norm = drop50_stats(dfC, "norm")
specC_4st  = fourstep(dfC)

A_Cn, mu_Cn, FWHM_Cn = fit_weighted_gauss(specC_norm["wl"], specC_norm["y"], specC_norm["yerr"])
A_C4, mu_C4, FWHM_C4 = fit_weighted_gauss(specC_4st["wl"],  specC_4st["y"],  specC_4st["yerr"])

DR_Cn = DR_minmax_db(specC_norm["y"])
DR_C4 = DR_minmax_db(specC_4st["y"])

C_xlim = (float(specC_norm["wl"].min()) - 0.7,
          float(specC_norm["wl"].max()) + 0.7)

dfL = read_numeric_table(L_FILE)
specL_norm = drop50_stats(dfL, "norm")
specL_4st  = fourstep(dfL)

x_min = float(min(specC_norm["wl"].min(), specC_4st["wl"].min(),
                  specL_norm["wl"].min(), specL_4st["wl"].min()))
x_max = float(max(specC_norm["wl"].max(), specC_4st["wl"].max(),
                  specL_norm["wl"].max(), specL_4st["wl"].max()))
xfull = (x_min, x_max)

L_xlim = (float(specL_norm["wl"].min())-0.7, float(specL_norm["wl"].max())+0.7)
C_xlim = (float(specC_norm["wl"].min()) - 0.7,
          float(specC_norm["wl"].max()) + 0.7)

# ---- Fig1/2/3 的 y 轴范围 ----
_y = specC_norm["y"].to_numpy(); _s = specC_norm["yerr"].to_numpy()
_lo = np.nanmin(_y - _s); _hi = np.nanmax(_y + _s); _sp = max(1e-12, _hi - _lo)
ylim_C_norm = (min(_lo - 0.08*_sp, -0.03*np.nanmax(_y)), _hi + 0.08*_sp)

_y = specC_4st["y"].to_numpy(); _s = specC_4st["yerr"].to_numpy()
_lo = np.nanmin(_y - _s); _hi = np.nanmax(_y + _s); _sp = max(1e-12, _hi - _lo)
ylim_C_4step = (min(_lo - 0.08*_sp, -0.03*np.nanmax(_y)), _hi + 0.08*_sp)

_y = specL_norm["y"].to_numpy(); _s = specL_norm["yerr"].to_numpy()
_lo = np.nanmin(_y - _s); _hi = np.nanmax(_y + _s); _sp = max(1e-12, _hi - _lo)
ylim_L_norm = (min(_lo - 0.08*_sp, -0.03*np.nanmax(_y)), _hi + 0.08*_sp)

# ================== 六张图（原有） ==================
draw_panel(specC_norm, mu_Cn, FWHM_Cn, A_Cn, xfull,
           "fig1_C_band_norm.png",
           "Normalized SFG vs Wavelength - Gaussian Fit",
           f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_Cn:.2f} dB",
           zoom_xlim=C_xlim, ylim=ylim_C_norm)

draw_panel(specC_4st, mu_C4, FWHM_C4, A_C4, xfull,
           "fig2_C_band_4step.png",
           "Four-step BG-removed Normalized SFG vs Wavelength - Gaussian Fit",
           f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_C4:.2f} dB",
           zoom_xlim=C_xlim, ylim=ylim_C_4step)

draw_panel(specL_norm, mu_Cn, FWHM_Cn, A_Cn, xfull,
           "fig3_L_band_zoom_norm.png",
           "Normalized SFG vs Wavelength (L-band zoom)\nC-model overlaid",
           zoom_xlim=L_xlim, label_data="L-band data (mean ± 1σ)",
           ylim=ylim_L_norm)

draw_panel(specL_4st,  mu_C4, FWHM_C4, A_C4, xfull,
           "fig4_L_band_zoom_4step.png",
           "Four-step BG-removed Normalized SFG (L-band zoom)\nC-model overlaid",
           zoom_xlim=L_xlim, label_data="L-band data (mean ± 1σ)")

spec_full_norm = pd.concat([specC_norm, specL_norm], ignore_index=True).sort_values("wl")
DR_full_n = DR_minmax_db(spec_full_norm["y"])
draw_panel(spec_full_norm, mu_Cn, FWHM_Cn, A_Cn, xfull,
           "fig5_full_norm.png",
           "Normalized SFG vs Wavelength - Gaussian Fit (C+L)",
           f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_full_n:.2f} dB")

spec_full_4 = pd.concat([specC_4st, specL_4st], ignore_index=True).sort_values("wl")
DR_full_4 = DR_minmax_db(spec_full_4["y"])
draw_panel(spec_full_4, mu_C4, FWHM_C4, A_C4, xfull,
           "fig6_full_4step.png",
           "Four-step BG-removed Normalized SFG vs Wavelength (C+L)\nGaussian Fit",
           f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_full_4:.2f} dB")

# 7) 仅均值、无误差棒；Y 轴到 -50 dB
draw_panel_db_meanonly(spec_full_4, mu_C4, FWHM_C4, A_C4, xfull,
                       "fig7_full_4step_dB.png",
                       "Four-step BG-removed Normalized SFG vs Wavelength (C+L, dB scale)",
                       ylim_db=(-50.0, 0.0))

# ================== CPKTP 理论 PMF（叠加到 Fig.7 形成 Fig.8） ==================
import numpy as np
from scipy.optimize import brentq
import matplotlib.ticker as ticker  # 必须导入 ticker 用于 log 格式化

LAM_S0 = 1290.56
LAMBDA_UM = 37.71066
LAMBDA_NM = LAMBDA_UM * 1000.0
D_HALF_NM = LAMBDA_NM / 2.0

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
A_SIGN = np.array(_A_RAW[:265], float)

def _ny_um(u): return np.sqrt(3.45018 + 0.04341/(u*u-0.04597) + 16.98825/(u*u-39.43799))
def _nz_um(u): return np.sqrt(4.59423 + 0.06206/(u*u-0.04763) + 110.80672/(u*u-86.12171))
def n_y_nm(lam_nm): return _ny_um(np.asarray(lam_nm)*1e-3)
def n_z_nm(lam_nm): return _nz_um(np.asarray(lam_nm)*1e-3)
def n_p(lam_nm): return n_y_nm(lam_nm)
def n_s(lam_nm): return n_z_nm(lam_nm)
def n_i(lam_nm): return n_y_nm(lam_nm)

def lambda_p(ls_nm, li_nm):
    ls = np.asarray(ls_nm, float); li = np.asarray(li_nm, float)
    return (ls*li)/(ls+li)

def k_basic_nm(ls_nm, li_nm):
    lp = lambda_p(ls_nm, li_nm)
    return 2*np.pi*( n_p(lp)/lp - n_s(ls_nm)/ls_nm - n_i(li_nm)/li_nm )

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

# 线性理论曲线（用于 Log Scale 绘图）
def theory_curve_shifted_linear_internal(mu_center, xfull, step_nm=0.01):
    li0   = solve_lambda_i0(LAM_S0)
    shift = float(mu_center) - li0
    li    = np.arange(xfull[0]-shift, xfull[1]-shift+1e-9, step_nm)
    x_plot = li + shift
    y_lin  = (phi2_lambda(LAM_S0, li)**2)
    y_lin /= np.nanmax(y_lin) # 归一化到 1
    return x_plot, y_lin

def draw_fig8_with_theory_log(spec_full_4, mu_fourstep, FWHM_fourstep, A_fourstep, xfull,
                              out_png="fig8_full_4step_log_withTheory.png"):
    x = spec_full_4["wl"].to_numpy()
    y = spec_full_4["y"].to_numpy()
    ref = float(np.nanmax(y))
    
    # === 关键修改：计算线性归一化数据，而非 dB ===
    # 将噪音底部的负值/零值截断为一个极小正值 (1e-10)，防止 Log Plot 报错
    y_norm = y / ref
    y_plot = np.maximum(y_norm, 1e-10) 

    # 拟合曲线（线性）
    xd = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM_fourstep / (2*np.sqrt(2*np.log(2)))
    y_g_norm = gauss(xd, A_fourstep, mu_fourstep, sigma) / ref
    y_s_norm = sinc2_curve(xd, mu_fourstep, FWHM_fourstep, A_fourstep) / ref

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    
    # === 关键修改：使用 semilogy 并设置 LogFormatter ===
    ax.semilogy(x, y_plot, linestyle='none', marker='o', ms=2.0,
                mfc=COL_DATA, mec=COL_DATA, mew=0, label="data (mean only)")
    
    lg, = ax.semilogy(xd, y_g_norm, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.semilogy(xd, y_s_norm, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try: ls.set_linestyle((0,(6,6)))
    except: ls.set_dashes([6,6])

    # 理论曲线（线性）
    xs, y_th_lin = theory_curve_shifted_linear_internal(mu_fourstep, xfull, step_nm=0.01)
    # y_th_lin 峰值为 1，与数据归一化一致
    ax.semilogy(xs, y_th_lin, color="#b22222", lw=1.8, label="CPKTP PMF (idler theory)")

    ax.set_xlim(xfull)
    # y轴范围：从 -50dB (1e-5) 到 略大于 0dB (1.5)
    ax.set_ylim(1e-5, 1.5) 
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized SFG (log scale)") # 修改 Label
    ax.grid(alpha=0.3, which="both") # which="both" 显示次级刻度网格
    
    # 设置 Y 轴刻度显示为 10^-x
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.6,
              handletextpad=0.6, borderpad=0.3)

    title1 = "Four-step BG-removed Normalized SFG vs Wavelength (C+L, Log scale)"
    title2 = f"μ={mu_fourstep:.3f} nm, FWHM={FWHM_fourstep:.3f} nm  (CPKTP PMF overlaid)"
    ax.set_title(title1 + "\n" + title2)
    
    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png, dpi=300); plt.close(fig)
    print("Saved:", out_png)

# 8) Fig.8 —— Log Scale + 理论曲线
draw_fig8_with_theory_log(
    spec_full_4,
    mu_C4, FWHM_C4, A_C4,
    xfull,
    out_png="fig8_full_4step_log_withTheory.png"
)# # ======================================================================
#  新增：线性刻度的 CPKTP PMF 理论曲线，叠加在 Fig.6 的四步线性图上
#  输出：fig16_full_4step_withTheory_linear.png
# ======================================================================

def theory_curve_shifted_linear(mu_center, xfull, step_nm=0.01):
    """
    和 theory_curve_shifted_for_fig7 类似，但返回线性归一化的 PMF^2：
    - x_plot：用于绘图的波长（nm），已经按 mu_center 做过平移
    - y_norm：在该区间内 max=1 的 |Phi|^2
    """
    li0   = solve_lambda_i0(LAM_S0)          # LAM_S0 已在前面定义
    shift = float(mu_center) - li0
    li    = np.arange(xfull[0] - shift,
                      xfull[1] - shift + 1e-9,
                      step_nm)
    x_plot = li + shift
    y_lin  = (phi2_lambda(LAM_S0, li)**2)    # |Phi|^2
    y_lin /= np.nanmax(y_lin)                # 归一化到峰值=1
    return x_plot, y_lin

def calculate_numeric_fwhm(x, y):
    """
    数值计算曲线 y(x) 的 FWHM。
    假设 y 是归一化的或单峰的，寻找 y=max(y)/2 的左右交叉点。
    """
    y = np.asarray(y)
    x = np.asarray(x)
    peak_val = np.nanmax(y)
    half_val = peak_val / 2.0
    idx_max = np.argmax(y)
    
    # 向左找交叉点
    # 找到峰值左侧第一个小于 half_val 的点的索引
    left_indices = np.where(y[:idx_max] < half_val)[0]
    if len(left_indices) > 0:
        i_left = left_indices[-1] # 最靠近峰值的一个
        # 线性插值找精确的 x_left
        x_left = x[i_left] + (half_val - y[i_left]) * (x[i_left+1] - x[i_left]) / (y[i_left+1] - y[i_left])
    else:
        return np.nan # 没找到左边界

    # 向右找交叉点
    right_indices = np.where(y[idx_max:] < half_val)[0]
    if len(right_indices) > 0:
        i_right = idx_max + right_indices[0] # 修正索引偏移
        # 线性插值找精确的 x_right (注意 i_right-1 是大于半高值的)
        x_right = x[i_right-1] + (half_val - y[i_right-1]) * (x[i_right] - x[i_right-1]) / (y[i_right] - y[i_right-1])
    else:
        return np.nan # 没找到右边界
        
    return x_right - x_left

def draw_fig16_with_theory_linear(spec_full_4,
                                  mu_fourstep, FWHM_fourstep, A_fourstep,
                                  xfull,
                                  out_png="fig16_full_4step_withTheory_linear.png"):
    """
    在线性刻度下，将 C+L 四步归一化数据 + 高斯 + sinc^2
    再叠加 CPKTP PMF (idler theory, 线性)。
    """
    # --- 数据 (四步 C+L) ---
    x = spec_full_4["wl"].to_numpy()
    y = spec_full_4["y"].to_numpy()
    s = spec_full_4["yerr"].to_numpy()

    # --- 拟合得到的高斯 / sinc^2（与你 fig6 完全一致） ---
    xd    = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM_fourstep / (2*np.sqrt(2*np.log(2)))
    y_g   = gauss(xd, A_fourstep, mu_fourstep, sigma)
    y_s   = sinc2_curve(xd, mu_fourstep, FWHM_fourstep, A_fourstep)

    # --- 理论 PMF (|Phi|^2)，线性归一化后按数据峰值缩放 ---
    xs_th, y_th_norm = theory_curve_shifted_linear(mu_fourstep, xfull, step_nm=0.005) # step_nm 越小 FWHM 越准
    
    # ★★★ 新增：计算理论曲线的 FWHM ★★★
    fwhm_theory = calculate_numeric_fwhm(xs_th, y_th_norm)
    print("-" * 60)
    print(f"Fig.16 Theory Curve Calculation:")
    print(f"  > CPKTP PMF (Red Curve) FWHM = {fwhm_theory:.4f} nm")
    print("-" * 60)
    
    y_peak = float(np.nanmax(y))          # 四步数据的峰值
    y_th   = y_th_norm * y_peak           # 让理论曲线和数据在峰值处同一高度

    # --- 画图 ---
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    ax.errorbar(x, y, yerr=s, fmt='o', ms=DATA_MS,
                mfc=COL_DATA, mec=COL_DATA, mew=0,
                ecolor=COL_DATA, elinewidth=ERR_ELW, capsize=ERR_CAP,
                label="data (mean ± 1σ)")

    lg, = ax.plot(xd, y_g, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try:
        ls.set_linestyle((0, (6, 6)))
    except TypeError:
        ls.set_dashes([6, 6])

    # 理论 CPKTP PMF 曲线（线性）
    ax.plot(xs_th, y_th, color=COL_PMF, lw=1.8,
            label=f"CPKTP PMF (Theory, FWHM≈{fwhm_theory:.2f}nm)")

    ax.set_xlim(xfull)
    # y 轴用自动范围即可，保持和 fig6 的风格一致
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized SFG (a.u.)")

    title1 = "Four-step BG-removed Normalized SFG vs Wavelength (C+L)"
    # 在标题中也显示理论带宽
    title2 = f"Data FWHM={FWHM_fourstep:.3f} nm | Theory FWHM={fwhm_theory:.3f} nm"
    ax.set_title(title1 + "\n" + title2)

    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("Saved:", out_png)

# 实际生成线性叠加图（用的就是 fig6 同一批数据和拟合参数）
draw_fig16_with_theory_linear(
    spec_full_4,   # C+L 四步 BG-removed 结果
    mu_C4, FWHM_C4, A_C4,
    xfull,
    out_png="fig16_full_4step_withTheory_linear.png"
)


# ================== 9) 新增：C-band zoom (四步) ==================
# 范围：从 C 段最小波长 到 1542 nm；其余绘法与 Fig.4 一致（最小改动）
# —— 9) 新增：C-band zoom (四步) —— 
C_zoom_xlim = (float(specC_4st["wl"].min()), 1542.0)

# ★ 新增：根据 zoom 区间内的数据自动设置 y 轴范围（放大）
_m = (specC_4st["wl"] >= C_zoom_xlim[0]) & (specC_4st["wl"] <= C_zoom_xlim[1])
_ylo = (specC_4st.loc[_m, "y"] - specC_4st.loc[_m, "yerr"]).min()
_yhi = (specC_4st.loc[_m, "y"] + specC_4st.loc[_m, "yerr"]).max()
C_zoom_ylim = (min(0.0, float(_ylo)*1.2), float(_yhi)*1.15)

draw_panel(
    specC_4st,            # 数据：C 段四步结果
    mu_C4, FWHM_C4, A_C4, # 叠加模型：C 段四步的高斯/同心 sinc²
    xfull,
    "fig9_C_band_zoom_4step.png",
    "Four-step BG-removed Normalized SFG (C-band zoom)",
    zoom_xlim=C_zoom_xlim,
    ylim=C_zoom_ylim,     # ★ 新增：传入刚计算的 y 轴范围
    label_data="C-band data (mean ± 1σ)"
)

# ========= Fig10/11/12: BG00 / BG01 / BG10 raw mean vs Wavelength =========
# 说明：仅做均值（drop 50, take the rest），不除输入功率，不画误差棒

# ============ Fig10–14: raw BG & raw four-step (C zoom, same range as Fig.9) ============
# 依赖：dfC, dfL, drop50_stats, COL_DATA，且 Fig.9 已设置 C_zoom_xlim=(min_C, 1542.0)

# --- 简单 MAD 去极值（对 DataFrame 的 y 列） ---
def mad_filter_df(df, col='y', k=6.0):
    y = np.asarray(df[col], float)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if not np.isfinite(mad) or mad == 0.0:
        return df.reset_index(drop=True)
    z = 0.6745 * (y - med) / mad
    return df.loc[np.abs(z) <= k].reset_index(drop=True)

# --- 紧缩 y 轴（只用传入数组的范围） ---
def tight_ylim_from_y(y, pad_frac=0.04):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1e-3, 1e-3)
    y_min, y_max = float(y.min()), float(y.max())
    span = y_max - y_min
    if span <= 0:
        span = max(1e-12, 1e-2*max(abs(y_min), abs(y_max), 1.0))
        y_min, y_max = (y_min - span/2, y_max + span/2)
    pad = span*pad_frac
    return (y_min - pad, y_max + pad)

# --- 画 raw 背景均值 ---
def draw_bg_mean(spec, out_png, title, label="mean (drop 50)"):
    x = spec["wl"].to_numpy()
    y = spec["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(x, y, linestyle="none", marker="o", ms=2.0,
            mfc=COL_DATA, mec=COL_DATA, mew=0, label=label)
    ax.set_xlim(float(spec["wl"].min())-0.7, float(spec["wl"].max())+0.7)
    ax.set_ylim(*tight_ylim_from_y(y, pad_frac=0.04))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Raw background (a.u.)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fancybox=True,
              handlelength=1.4, handletextpad=0.6, borderpad=0.3)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    print("Saved:", out_png)

# --------- 生成三路背景 drop50 均值并剔除极值（Fig10/11/12） ---------
specC_bg00 = drop50_stats(dfC, "bg00")
specL_bg00 = drop50_stats(dfL, "bg00")
spec_bg00  = pd.concat([specC_bg00, specL_bg00], ignore_index=True).sort_values("wl")
spec_bg00  = mad_filter_df(spec_bg00, 'y', k=6.0)

specC_bg01 = drop50_stats(dfC, "bg01")
specL_bg01 = drop50_stats(dfL, "bg01")
spec_bg01  = pd.concat([specC_bg01, specL_bg01], ignore_index=True).sort_values("wl")
spec_bg01  = mad_filter_df(spec_bg01, 'y', k=6.0)

specC_bg10 = drop50_stats(dfC, "bg10")
specL_bg10 = drop50_stats(dfL, "bg10")
spec_bg10  = pd.concat([specC_bg10, specL_bg10], ignore_index=True).sort_values("wl")
spec_bg10  = mad_filter_df(spec_bg10, 'y', k=6.0)

draw_bg_mean(spec_bg00, "fig10_bg00_raw.png", "BG00 vs Wavelength (raw mean, drop first 50)")
draw_bg_mean(spec_bg01, "fig11_bg01_raw.png", "BG01 vs Wavelength (raw mean, drop first 50)")
draw_bg_mean(spec_bg10, "fig12_bg10_raw.png", "BG10 vs Wavelength (raw mean, drop first 50)")

# --------- 组合 (BG01 + BG10 − BG00) → drop50 → 剔除极值 → 图（Fig13） ---------
def drop50_combo_bg(df):
    out = []
    for wl, g in df.groupby("wl"):
        g2 = g.iloc[50:]
        if len(g2) < 5: 
            continue
        combo = (g2["bg01"] + g2["bg10"] - g2["bg00"]).to_numpy()
        out.append((float(wl), float(np.mean(combo))))
    return pd.DataFrame(out, columns=["wl","y"]).sort_values("wl")

spec_combo = pd.concat([drop50_combo_bg(dfC), drop50_combo_bg(dfL)], ignore_index=True).sort_values("wl")
spec_combo = mad_filter_df(spec_combo, 'y', k=6.0)

draw_bg_mean(spec_combo,
             "fig13_bg01pbg10_minus_bg00_raw.png",
             "(BG01 + BG10 − BG00) vs Wavelength (raw mean, drop first 50)",
             label="mean(BG01+BG10−BG00)")

# --------- 四步 RAW（不归一化）C 段 zoom；x 范围与 Fig.9 完全一致（Fig14） ---------
def fourstep_raw_mean(df):
    out = []
    for wl, g in df.groupby("wl"):
        g2 = g.iloc[50:]
        if len(g2) < 5: 
            continue
        val = (g2["sfg"].mean()
               - g2["bg01"].mean()
               - g2["bg10"].mean()
               + g2["bg00"].mean())
        out.append((float(wl), float(val)))
    return pd.DataFrame(out, columns=["wl","y"]).sort_values("wl")

# Fig.9 中已定义 C_zoom_xlim=(min_C, 1542.0)；若意外不存在就按同逻辑兜底
if 'C_zoom_xlim' not in globals():
    from math import inf
    C_zoom_xlim = (float(min(dfC["wl"].min(), dfL["wl"].min())), 1542.0)

specC_raw4 = fourstep_raw_mean(dfC)
specC_raw4 = mad_filter_df(specC_raw4, 'y', k=6.0)   # 先剔除异常点

# 只用窗口内的数据做 y 轴紧缩
_mask = (specC_raw4["wl"] >= C_zoom_xlim[0]) & (specC_raw4["wl"] <= C_zoom_xlim[1])
x_win = specC_raw4.loc[_mask, "wl"].to_numpy()
y_win = specC_raw4.loc[_mask, "y"].to_numpy()

fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.plot(x_win, y_win, linestyle="none", marker="o", ms=2.0,
        mfc=COL_DATA, mec=COL_DATA, mew=0, label="mean (drop 50)")
ax.set_xlim(*C_zoom_xlim)
ax.set_ylim(*tight_ylim_from_y(y_win, pad_frac=0.04))   # y 轴 zoom
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Raw four-step SFG (a.u.)")
ax.set_title("Four-step RAW (11−10−01+00) — C-band zoom")  # Fig14
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=True, fancybox=True,
          handlelength=1.4, handletextpad=0.6, borderpad=0.3)
fig.tight_layout(); fig.savefig("fig14_raw4_C_band_zoom.png"); plt.close(fig)
print("Saved:", "fig14_raw4_C_band_zoom.png")
# ==================== END Fig10–14 ====================
# ---------- Fig15：四步 RAW (11−10−01+00) — L-band zoom（不归一化） ----------
# 选择 L 段的 x 范围：优先 L_zoom_xlim，其次 L_xlim，最后用 dfL 的全范围
if 'L_zoom_xlim' in globals():
    _L_xlim = L_zoom_xlim
elif 'L_xlim' in globals():
    _L_xlim = L_xlim
else:
    _L_xlim = (float(dfL["wl"].min()), float(dfL["wl"].max()))

# 计算 L 段 raw 四步并去极值
specL_raw4 = fourstep_raw_mean(dfL)           # 已在上文定义：丢前50求 mean(SFG-BG01-BG10+BG00)
specL_raw4 = mad_filter_df(specL_raw4, 'y', k=6.0)

# 只用窗口内的数据决定 y 轴 zoom
_maskL = (specL_raw4["wl"] >= _L_xlim[0]) & (specL_raw4["wl"] <= _L_xlim[1])
xL = specL_raw4.loc[_maskL, "wl"].to_numpy()
yL = specL_raw4.loc[_maskL, "y"].to_numpy()

fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.plot(xL, yL, linestyle="none", marker="o", ms=2.0,
        mfc=COL_DATA, mec=COL_DATA, mew=0, label="mean (drop 50)")
ax.set_xlim(*_L_xlim)
ax.set_ylim(*tight_ylim_from_y(yL, pad_frac=0.04))  # y 轴紧缩
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Raw four-step SFG (a.u.)")
ax.set_title("Four-step RAW (11−10−01+00) — L-band zoom")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=True, fancybox=True,
          handlelength=1.4, handletextpad=0.6, borderpad=0.3)
fig.tight_layout()
fig.savefig("fig15_raw4_L_band_zoom.png"); plt.close(fig)
print("Saved:", "fig15_raw4_L_band_zoom.png")

# ===================== L-band 统计（BG 与 RAW4 + 高斯阈值裁切） =====================

# ---- 参数区（按需改）----
USE_MAD = True       # 是否做一次简单 MAD 去极值
MAD_K    = 6.0
# 用哪个高斯来判阈值：默认用 C-band 拟合得到的 (mu, FWHM)
MU_GAUSS   = mu_C4 if 'mu_C4' in globals() else 1555.834
FWHM_GAUSS = FWHM_C4 if 'FWHM_C4' in globals() else 10.604
# 阈值：用“相对峰值”的归一化高斯（峰值=1），当 G(λ) ≤ THR_REL 时触发裁切
THR_REL = 2e-3        # 例：当高斯低于峰值的 0.2% 时，把右侧作为“纯背景区”
OUT_CSV = "L_band_bg_raw_stats.csv"

# ---- 小工具 ----
def _mad_filter_df(spec_df, col='y', k=6.0):
    y = np.asarray(spec_df[col].to_numpy(), float)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if not np.isfinite(mad) or mad == 0.0:
        return spec_df.reset_index(drop=True)
    z = 0.6745 * (y - med) / mad
    mask = np.abs(z) <= k
    return spec_df.loc[mask].reset_index(drop=True)

def _band_stats(spec_df, col='y', label=''):
    y = np.asarray(spec_df[col].to_numpy(), float)
    y = y[np.isfinite(y)]
    N = int(y.size)
    mean = float(np.mean(y)) if N else np.nan
    varN = float(np.var(y, ddof=0)) if N else np.nan     # 按 N
    varN1= float(np.var(y, ddof=1)) if N>1 else np.nan   # 按 N-1（无偏）
    stdN = float(np.sqrt(varN))  if np.isfinite(varN)  else np.nan
    stdN1= float(np.sqrt(varN1)) if np.isfinite(varN1) else np.nan
    return dict(label=label, N=N, mean=mean, varN=varN, varN1=varN1, stdN=stdN, stdN1=stdN1,
                ymin=float(np.min(y)) if N else np.nan,
                ymax=float(np.max(y)) if N else np.nan)

def _gauss_norm(x, mu, fwhm):
    # 归一化高斯（峰值=1）
    return np.exp(-4.0*np.log(2.0)*((x-mu)**2)/(fwhm**2))

def _raw4_mean(df):
    # 与你之前的一致：每个波长丢 50 个样本后，再做 mean(SFG - BG01 - BG10 + BG00)
    out = []
    for wl, g in df.groupby("wl"):
        g2 = g.iloc[50:]
        if len(g2) < 5: 
            continue
        val = (g2["sfg"].mean() - g2["bg01"].mean() - g2["bg10"].mean() + g2["bg00"].mean())
        out.append((float(wl), float(val)))
    return pd.DataFrame(out, columns=["wl","y"]).sort_values("wl")

# ---- (1) L-band：三路 BG 的“每波长均值” -> 波长维度上的整体均值/方差 ----
bg00_L = drop50_stats(dfL, "bg00")
bg01_L = drop50_stats(dfL, "bg01")
bg10_L = drop50_stats(dfL, "bg10")

if USE_MAD:
    bg00_L = _mad_filter_df(bg00_L, 'y', MAD_K)
    bg01_L = _mad_filter_df(bg01_L, 'y', MAD_K)
    bg10_L = _mad_filter_df(bg10_L, 'y', MAD_K)

stat_bg00 = _band_stats(bg00_L, 'y', 'BG00_L')
stat_bg01 = _band_stats(bg01_L, 'y', 'BG01_L')
stat_bg10 = _band_stats(bg10_L, 'y', 'BG10_L')

print("[L-band BG] BG00:", stat_bg00)
print("[L-band BG] BG01:", stat_bg01)
print("[L-band BG] BG10:", stat_bg10)

# ---- (2) L-band：RAW4 的整体均值/方差 ----
raw4_L = _raw4_mean(dfL)
if USE_MAD:
    raw4_L = _mad_filter_df(raw4_L, 'y', MAD_K)

stat_raw4_all = _band_stats(raw4_L, 'y', 'RAW4_L (all)')
print("[L-band RAW4] all:", stat_raw4_all)

# ---- (3) 用 C-band 高斯判阈值，确定 λ_cut，并“向右取值”再算 RAW4 的均值/方差 ----
xL = raw4_L["wl"].to_numpy()
gL = _gauss_norm(xL, MU_GAUSS, FWHM_GAUSS)  # 仅形状，峰值=1
mask_thr = gL <= THR_REL

if np.any(mask_thr):
    # “当 G(λ) <= 阈值”的最左点，对应的 λ_cut（含该点及其右侧）
    idx0 = int(np.argmax(mask_thr))         # 第一个 True 的索引
    lambda_cut = float(xL[idx0])
    mask_right = xL >= lambda_cut
    raw4_L_right = raw4_L.loc[mask_right].reset_index(drop=True)
    stat_raw4_right = _band_stats(raw4_L_right, 'y', f'RAW4_L (λ≥{lambda_cut:.3f} nm, thr={THR_REL:g})')
else:
    lambda_cut = np.nan
    raw4_L_right = raw4_L.copy()
    stat_raw4_right = _band_stats(raw4_L_right, 'y', f'RAW4_L (no-threshold)')
    print("[WARN] 阈值过小，L-band 内未出现 G(λ) ≤ 阈值 的点；已退化为全段统计。")

print("[L-band RAW4] right-of-threshold:", stat_raw4_right)

# ---- (4) 保存到 CSV 便于留档 ----
rows = []
rows += [stat_bg00, stat_bg01, stat_bg10, stat_raw4_all, stat_raw4_right]
df_stats = pd.DataFrame(rows, columns=[
    "label","N","mean","varN","varN1","stdN","stdN1","ymin","ymax"
])
df_stats.to_csv(OUT_CSV, index=False)
print("Saved stats ->", OUT_CSV)
print(f"lambda_cut (Gaussian ≤ {THR_REL:g} of peak) = {lambda_cut:.6f} nm")




