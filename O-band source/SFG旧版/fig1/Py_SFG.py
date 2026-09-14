# -*- coding: utf-8 -*-
import io, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

def to_db(y, ref):
    """把线性功率 y 转成 dB（相对 ref 的 10log10），并做极小值保护避免 -inf。"""
    y = np.asarray(y, float)
    return 10*np.log10(np.maximum(y, 1e-15) / max(ref, 1e-15))

def DR_from_means_db(y):
    """用所有“均值点”的最大/最小正值计算 DR（dB）。"""
    y = np.asarray(y, float)
    y_pos = y[y > 0]
    if len(y_pos) < 2:
        return np.nan
    return 10*np.log10(y_pos.max() / y_pos.min())


# ========= 全局风格 =========
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle":  "butt",
})

# ========= 颜色/样式 =========
COL_DATA  = "#e69f00"       # 数据点 + 误差棒
COL_GAUSS = "#61b9ea"       # 高斯
COL_SINC2 = "#009d72"       # sinc^2

GAUSS_LW  = 1.2
SINC_LW   = 1.2
SINC_DASH = [3.0, 3.0]      # 均匀短划线（传奇中“三段短划线”观感）

DATA_MS   = 2.0
ERR_ELW   = 0.8
ERR_CAP   = 0               # 无端帽，只画竖线

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
    # numpy.sinc(u) = sin(pi u)/(pi u)
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

# ========= DR（所有均值点中 >0 的 max/min） =========
def DR_minmax_db(y):
    y = np.asarray(y, float)
    y_pos = y[y>0]
    if len(y_pos) < 2: return np.nan
    return 10*np.log10(y_pos.max()/y_pos.min())

# ========= 标题换行 =========
def set_title_two_lines(ax, line1, line2=None):
    if line2:
        ax.set_title(line1 + "\n" + line2)
    else:
        ax.set_title(line1)

# ========= y 轴范围（L-zoom 要照顾负误差棒） =========
def ylim_for_zoom(xd, y_model, xlim, y, yerr, pad=1.15):
    mask = (xd>=xlim[0]) & (xd<=xlim[1])
    ymax = 0.0
    if np.any(mask):
        ymax = max(ymax, float(np.nanmax(y_model[mask])))
    if len(y):
        ymax = max(ymax, float(np.nanmax(y)))
    ymin = 0.0
    if len(y):
        ymin = min(0.0, float(np.nanmin(y - yerr)) * 1.15)
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
    ls, = ax.plot(xd, y_s, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    ls.set_dashes(SINC_DASH)    # 均匀短划线

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
            # 你外面传进来的 (lo, hi) 直接生效
            ax.set_ylim(ylim)
        else:
            ymin, ymax = ylim_for_zoom(xd, y_g, zoom_xlim, y, s)
            ax.set_ylim(ymin, ymax)
    else:
        # 没有缩放范围时，如果传了 ylim 也要生效
        if ylim is not None:
            ax.set_ylim(ylim)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png); plt.close(fig)
    print("Saved:", out_png)

def draw_panel_db(spec, mu, FWHM, A, xfull, out_png, title1, title2=None,
                  zoom_xlim=None, label_data="data (mean ± 1σ)"):
    """dB 坐标：把均值/误差棒与高斯/sinc²都换算到 dB（以数据峰值=0 dB 为基准）"""
    x = spec["wl"].to_numpy()
    y = spec["y"].to_numpy()
    s = spec["yerr"].to_numpy()

    # 参考值：数据“均值”的最大值（对应 0 dB）
    ref = float(np.nanmax(y))

    # 数据点：中心/上下界都在“线性”上计算，然后再转 dB（保持你之前的规则）
    y_c_db  = to_db(y, ref)
    y_lo_db = to_db(np.maximum(y - s, 1e-15), ref)
    y_hi_db = to_db(y + s, ref)
    yerr_db = np.vstack([y_c_db - y_lo_db, y_hi_db - y_c_db])  # 非对称误差棒

    # 拟合模型（在“线性”上算，再整体转 dB；基准仍为“数据峰值 ref”）
    xd = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    y_g_lin = gauss(xd, A, mu, sigma)
    y_s_lin = sinc2_curve(xd, mu, FWHM, A)
    y_g_db  = to_db(y_g_lin, ref)
    y_s_db  = to_db(y_s_lin, ref)

    # 画图（样式与线宽/虚线沿用你的设置）
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.errorbar(x, y_c_db, yerr=yerr_db, fmt='o', ms=DATA_MS,
                mfc=COL_DATA, mec=COL_DATA, mew=0,
                ecolor=COL_DATA, elinewidth=ERR_ELW, capsize=ERR_CAP,
                label=label_data)
    lg, = ax.plot(xd, y_g_db, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s_db, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try:
        ls.set_linestyle((0, (6, 6)))  # 均匀短划线
    except TypeError:
        ls.set_dashes([6, 6])

    # 轴/标题/图例
    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)
    db_min = float(np.nanmin(y_lo_db))
    ax.set_ylim(db_min - 2.0, 1.0)    # 顶部给 1 dB 空隙（0 dB 在顶部附近）
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB vs peak)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)
    ax.grid(alpha=0.3)

    # DR（用“均值”的 max/min>0 计算，与线性图的定义一致）
    dr_db = DR_from_means_db(y)
    title2 = (title2 or f"μ={mu:.3f} nm, FWHM={FWHM:.3f} nm, DR={dr_db:.2f} dB")
    set_title_two_lines(ax, title1, title2)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(out_png); plt.close(fig)
    print(f"Saved (dB): {out_png}  |  DR={dr_db:.2f} dB")

def draw_panel_db_meanonly(spec, mu, FWHM, A, xfull, out_png, title1, title2=None,
                           zoom_xlim=None, label_data="data (mean only)", ylim_db=(-50.0, 0.0)):
    """dB 坐标：仅绘制均值圆点（无误差棒），Gaussian/sinc² 同基准转 dB；Y 轴固定到 ylim_db。"""
    x = spec["wl"].to_numpy()
    y = spec["y"].to_numpy()

    # 以数据均值的最大值作为 0 dB 参考
    ref = float(np.nanmax(y))

    # 数据均值 → dB
    def to_db(arr, refv):
        arr = np.asarray(arr, float)
        return 10*np.log10(np.maximum(arr, 1e-15) / max(refv, 1e-15))

    y_db = to_db(y, ref)

    # 拟合模型（线性→dB），基准同 ref
    xd = np.linspace(xfull[0], xfull[1], 1400)
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    y_g_db = to_db(gauss(xd, A, mu, sigma), ref)
    y_s_db = to_db(sinc2_curve(xd, mu, FWHM, A), ref)

    # 画图
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(x, y_db, linestyle='none', marker='o', ms=DATA_MS,
            mfc=COL_DATA, mec=COL_DATA, mew=0, label=label_data)
    lg, = ax.plot(xd, y_g_db, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")
    ls, = ax.plot(xd, y_s_db, color=COL_SINC2, lw=SINC_LW, label="sinc²")
    try:
        ls.set_linestyle((0, (6, 6)))  # 均匀短划线
    except TypeError:
        ls.set_dashes([6, 6])

    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)
    ax.set_ylim(ylim_db)  # 固定到 [-50 dB, 0 dB]

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative power (dB vs peak)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, handlelength=1.4,
              handletextpad=0.6, borderpad=0.3)
    ax.grid(alpha=0.3)

    # 标题（第二行可写 μ/FWHM/DR，如需）
    if title2 is None:
        title2 = f"μ={mu:.3f} nm, FWHM={FWHM:.3f} nm"
    set_title_two_lines(ax, title1, title2)

    fig.tight_layout(rect=[0,0,0.98,0.95])
    fig.savefig(out_png); plt.close(fig)
    print("Saved (dB, mean only):", out_png)

# ================== 替换为你的文件名 ==================
C_FILE = "SFG20251013200948.txt"   # C-band 数据
L_FILE = "SFG20251014101522.txt"   # L-band 数据（新）

# ================== 做 C 模型（加权高斯） ==================
dfC = read_numeric_table(C_FILE)
specC_norm = drop50_stats(dfC, "norm")
specC_4st  = fourstep(dfC)

A_Cn, mu_Cn, FWHM_Cn = fit_weighted_gauss(specC_norm["wl"], specC_norm["y"], specC_norm["yerr"])
A_C4, mu_C4, FWHM_C4 = fit_weighted_gauss(specC_4st["wl"],  specC_4st["y"],  specC_4st["yerr"])

DR_Cn = DR_minmax_db(specC_norm["y"])
DR_C4 = DR_minmax_db(specC_4st["y"])

# ---- 关键新增：C-band 的横轴范围（只给 fig1 / fig2 用） ----
C_xlim = (float(specC_norm["wl"].min()) - 0.7,
          float(specC_norm["wl"].max()) + 0.7)

# x 轴范围（全域）& L zoom 范围（供其它图使用，不变）
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

# ---- Fig1/2/3 的 y 轴范围（按数据±σ并加少量留白；不从 0 起）----
# Fig1: C-band norm
_y = specC_norm["y"].to_numpy(); _s = specC_norm["yerr"].to_numpy()
_lo = np.nanmin(_y - _s); _hi = np.nanmax(_y + _s); _sp = max(1e-12, _hi - _lo)
ylim_C_norm = (min(_lo - 0.08*_sp, -0.03*np.nanmax(_y)), _hi + 0.08*_sp)

# Fig2: C-band 4-step
_y = specC_4st["y"].to_numpy(); _s = specC_4st["yerr"].to_numpy()
_lo = np.nanmin(_y - _s); _hi = np.nanmax(_y + _s); _sp = max(1e-12, _hi - _lo)
ylim_C_4step = (min(_lo - 0.08*_sp, -0.03*np.nanmax(_y)), _hi + 0.08*_sp)

# Fig3: L-band zoom (norm)
_y = specL_norm["y"].to_numpy(); _s = specL_norm["yerr"].to_numpy()
_lo = np.nanmin(_y - _s); _hi = np.nanmax(_y + _s); _sp = max(1e-12, _hi - _lo)
ylim_L_norm = (min(_lo - 0.08*_sp, -0.03*np.nanmax(_y)), _hi + 0.08*_sp)


# ================== 六张图 ==================
# 1) C-band 第六列（未去噪）—— 仅画到 C-band 范围
t1 = "Normalized SFG vs Wavelength - Gaussian Fit"
t2 = f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_Cn:.2f} dB"
draw_panel(specC_norm, mu_Cn, FWHM_Cn, A_Cn, xfull,
           "fig1_C_band_norm.png",
           "Normalized SFG vs Wavelength - Gaussian Fit",
           f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_Cn:.2f} dB",
           zoom_xlim=C_xlim, ylim=ylim_C_norm)

# 2) C-band 四步—— 仅画到 C-band 范围
t1 = "Four-step BG-removed Normalized SFG vs Wavelength - Gaussian Fit"
t2 = f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_C4:.2f} dB"
draw_panel(specC_4st, mu_C4, FWHM_C4, A_C4, xfull,
           "fig2_C_band_4step.png",
           "Four-step BG-removed Normalized SFG vs Wavelength - Gaussian Fit",
           f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_C4:.2f} dB",
           zoom_xlim=C_xlim, ylim=ylim_C_4step)

# 3) L-band zoom（未去噪，沿用 C 模型）
t1 = "Normalized SFG vs Wavelength (L-band zoom)\nC-model overlaid"
draw_panel(specL_norm, mu_Cn, FWHM_Cn, A_Cn, xfull,
           "fig3_L_band_zoom_norm.png",
           "Normalized SFG vs Wavelength (L-band zoom)\nC-model overlaid",
           zoom_xlim=L_xlim, label_data="L-band data (mean ± 1σ)",
           ylim=ylim_L_norm)

# 4) L-band zoom（四步，沿用 C 模型）
t1 = "Four-step BG-removed Normalized SFG (L-band zoom)\nC-model overlaid"
draw_panel(specL_4st,  mu_C4, FWHM_C4, A_C4, xfull,
           "fig4_L_band_zoom_4step.png", t1,
           zoom_xlim=L_xlim, label_data="L-band data (mean ± 1σ)")

# 5) C+L 全域（未去噪）
spec_full_norm = pd.concat([specC_norm, specL_norm], ignore_index=True).sort_values("wl")
DR_full_n = DR_minmax_db(spec_full_norm["y"])
t1 = "Normalized SFG vs Wavelength - Gaussian Fit (C+L)"
t2 = f"μ={mu_Cn:.3f} nm, FWHM={FWHM_Cn:.3f} nm, DR={DR_full_n:.2f} dB"
draw_panel(spec_full_norm, mu_Cn, FWHM_Cn, A_Cn, xfull,
           "fig5_full_norm.png", t1, t2)

# 6) C+L 全域（四步）
spec_full_4 = pd.concat([specC_4st, specL_4st], ignore_index=True).sort_values("wl")
DR_full_4 = DR_minmax_db(spec_full_4["y"])
t1 = "Four-step BG-removed Normalized SFG vs Wavelength (C+L)\nGaussian Fit"
t2 = f"μ={mu_C4:.3f} nm, FWHM={FWHM_C4:.3f} nm, DR={DR_full_4:.2f} dB"
draw_panel(spec_full_4, mu_C4, FWHM_C4, A_C4, xfull,
           "fig6_full_4step.png", t1, t2)

# 7) C+L 全域（四步，dB 坐标；以“数据峰值=0 dB”为基准）
#t1 = "Four-step BG-removed Normalized SFG vs Wavelength (C+L, dB scale)"
#t2 = None  # 第二行由函数里自动补成 μ/FWHM/DR
#draw_panel_db(spec_full_4, mu_C4, FWHM_C4, A_C4, xfull,
#              "fig7_full_4step_dB.png", t1, t2)

# 7) C+L 全域（四步，dB 坐标；仅均值、无误差棒；Y 轴到 -50 dB）
t1 = "Four-step BG-removed Normalized SFG vs Wavelength (C+L, dB scale)"
draw_panel_db_meanonly(spec_full_4, mu_C4, FWHM_C4, A_C4, xfull,
                       "fig7_full_4step_dB.png", t1, ylim_db=(-50.0, 0.0))
