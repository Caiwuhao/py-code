# ================== 替换为你的文件名 ==================
#C_FILE = "SFG20251024173128.txt"
#L_FILE = "SFG20251027120732.txt"
#C_FILE = "SFG20251029105128.txt"

C_FILE = "SFG20251104190142.txt"
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

def theory_curve_shifted_for_fig7(mu_center, xfull, step_nm=0.01):
    li0   = solve_lambda_i0(LAM_S0)
    shift = float(mu_center) - li0
    li    = np.arange(xfull[0]-shift, xfull[1]-shift+1e-9, step_nm)
    x_plot = li + shift
    y_lin  = (phi2_lambda(LAM_S0, li)**2)
    y_lin /= np.nanmax(y_lin)
    y_db   = 10.0*np.log10(np.maximum(y_lin, 1e-20))
    return x_plot, y_db

def draw_fig8_with_theory(spec_full_4, mu_fourstep, FWHM_fourstep, A_fourstep, xfull,
                          out_png="fig8_full_4step_dB_withTheory.png"):
    x = spec_full_4["wl"].to_numpy()
    y = spec_full_4["y"].to_numpy()
    ref = float(np.nanmax(y))
    to_db = lambda arr: 10*np.log10(np.maximum(np.asarray(arr, float), 1e-15)/ref)

    y_db = to_db(y)

    xd = np.linspace(xfull[0], xfull[1], 1400)
    sig = FWHM_fourstep / (2*np.sqrt(2*np.log(2)))
    y_g_db = to_db(gauss(xd, A_fourstep, mu_fourstep, sig))
    y_s_db = to_db(sinc2_curve(xd, mu_fourstep, FWHM_fourstep, A_fourstep))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(x, y_db, linestyle='none', marker='o', ms=2.0,
            mfc=COL_DATA, mec=COL_DATA, mew=0, label="data (mean only)")
    lg, = ax.plot(xd, y_g_db, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s_db, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try: ls.set_linestyle((0,(6,6)))
    except: ls.set_dashes([6,6])

    xs, y_th = theory_curve_shifted_for_fig7(mu_fourstep, xfull, step_nm=0.01)
    ax.plot(xs, y_th, color="#b22222", lw=1.8, label="CPKTP PMF (idler theory)")

    ax.set_xlim(xfull)
    ax.set_ylim(-50.0, 0.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB vs peak)")
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

# 8) Fig.8 —— 在 Fig.7 的基础上叠加红色理论曲线
draw_fig8_with_theory(
    spec_full_4,
    mu_C4, FWHM_C4, A_C4,
    xfull,
    out_png="fig8_full_4step_dB_withTheory.png"
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



