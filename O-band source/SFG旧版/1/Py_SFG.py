import numpy as np, pandas as pd, matplotlib.pyplot as plt, io, re, math
from scipy.optimize import curve_fit

COL_DATA  = "#e69f00"
COL_GAUSS = "#61b9ea"
COL_SINC2 = "#009d72"

GAUSS_LW  = 1.2      # 细线且清晰（1.0~1.6 都可）
SINC_LW   = 1.2
SINC_DASH = (0, (8, 6))

DATA_MS   = 2.0      # 数据点
ERR_ELW   = 0.8      # 误差棒线宽
ERR_CAP   = 2        # 误差棒端帽

import matplotlib as mpl
mpl.rcParams.update({
    "lines.solid_capstyle": "butt",
    "savefig.dpi": 300,         # 导出更细腻
    "figure.dpi": 150,          # 屏幕显示
})


def read_numeric_table(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    lines = [ln for ln in lines if re.match(r'^\s*[+-]?\d', ln)]
    arr = np.loadtxt(io.StringIO(''.join(lines)))
    if arr.ndim == 1: arr = arr.reshape(1,-1)
    df = pd.DataFrame(arr)
    while df.shape[1] < 9: df[df.shape[1]] = np.nan
    df.columns = ["wl","p2","o_wl","p4","sfg","norm","bg00","bg01","bg10"]
    return df.dropna()

def drop50_stats(df, col):
    out=[]
    for wl,g in df.groupby("wl"):
        v=g[col].to_numpy()[50:]
        if len(v)>5: out.append((wl,v.mean(),v.std(ddof=1)))
    return pd.DataFrame(out,columns=["wl","y","yerr"]).sort_values("wl")

def fourstep(df):
    out=[]
    for wl,g in df.groupby("wl"):
        g=g.iloc[50:]
        v5,v7,v8,v9=g["sfg"],g["bg00"],g["bg01"],g["bg10"]
        p2,p4=g["p2"],g["p4"]
        if min(len(v5),len(v7),len(v8),len(v9),len(p2),len(p4))<5: continue
        A=(v5.mean()-v8.mean()-v9.mean()+v7.mean())
        sA=math.sqrt(v5.std()**2+v8.std()**2+v9.std()**2+v7.std()**2)
        D=p2.mean()*p4.mean()
        out.append((wl,A/D,sA/D))
    return pd.DataFrame(out,columns=["wl","y","yerr"]).sort_values("wl")

# ---- 高斯拟合函数 ----
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def sinc2_curve(x, mu, FWHM, A):
    # numpy.sinc(u) = sin(pi u)/(pi u)
    t_half = 0.443  # 数值解得到的 sinc^2 半高宽/2 的无量纲比例
    a = (FWHM/2) / t_half if FWHM > 0 else 1.0
    return A * (np.sinc((x - mu)/a))**2

def _initial_guess(x, y):
    i = int(np.argmax(y))
    mu0 = float(x[i]); A0 = float(y[i]); half = A0/2.0
    # 左右半高点线性插值（若找不到就给个安全值）
    xL = mu0 - 0.8
    xR = mu0 + 0.8
    if i > 0:
        left = np.where(y[:i] <= half)[0]
        if len(left):
            j = left[-1]
            xL = np.interp(half, [y[j], y[j+1]], [x[j], x[j+1]])
    if i < len(y)-1:
        right = np.where(y[i:] <= half)[0]
        if len(right):
            k = right[0] + i
            xR = np.interp(half, [y[k-1], y[k]], [x[k-1], x[k]])
    FWHM0 = max(0.2, min(8.0, xR - xL))
    sigma0 = FWHM0 / (2*np.sqrt(2*np.log(2)))
    return A0, mu0, sigma0

def _DR_db(y):
    y = np.asarray(y)
    n = len(y)
    if n < 6: return np.nan
    noise = np.std(np.r_[y[:max(3, n//10)], y[-max(3, n//10):]], ddof=1) + 1e-15
    return 10*np.log10((np.max(y)+1e-15)/noise)

def plot_weighted(spec, title, out_png):
    import matplotlib.pyplot as plt
    x = spec["wl"].to_numpy()
    y = spec["y"].to_numpy()
    s = spec["yerr"].to_numpy()

    # 保护 σ（避免 0/NaN 破坏权重）
    s_fit = s.copy()
    s_fit[~np.isfinite(s_fit) | (s_fit <= 0)] = np.median(s[s>0]) if np.any(s>0) else 1.0

    # 初值 + 约束
    A0, mu0, sigma0 = _initial_guess(x, y)
    bounds = ([0.0, x.min()-1.0, 0.05], [10.0*np.max(y), x.max()+1.0, 20.0])

    try:
        popt, _ = curve_fit(
            gauss, x, y, p0=[A0, mu0, sigma0],
            sigma=s_fit, absolute_sigma=True,
            bounds=bounds, maxfev=100000
        )
        A_fit, mu_fit, sigma_fit = map(float, popt)
    except Exception:
        A_fit, mu_fit, sigma_fit = A0, mu0, sigma0

    FWHM = 2*np.sqrt(2*np.log(2))*sigma_fit
    xd = np.linspace(x.min(), x.max(), 600)

    # 注意：这里用的是 y_gauss / y_sinc2（不是 fit_gauss/fit_sinc2）
    y_gauss = gauss(xd, A_fit, mu_fit, sigma_fit)
    y_sinc2 = sinc2_curve(xd, mu_fit, FWHM, A_fit)

    # —— 绘图（小点 + 误差棒 + 细线）——
    fig, ax = plt.subplots(figsize=(7,4))
    # —— Data（小圆点 + 仅竖向误差棒；无横向端帽）——
    eb = ax.errorbar(
        x, y, yerr=s,
        fmt='o',  # 只画点，不连线
        ms=DATA_MS,  # 小圆点
        mfc=COL_DATA, mec=COL_DATA, mew=0,
        ecolor=COL_DATA,  # 误差棒颜色
        elinewidth=ERR_ELW,  # 误差棒线宽
        capsize=0,  # 端帽=0，避免“横向太宽”
        label="data"
    )

    # Gaussian（细线）
    ax.plot(xd, y_gauss, color=COL_GAUSS, lw=GAUSS_LW, label="Gaussian")

    # sinc²（细线 + 圆端帽的长虚线）
    line_sinc, = ax.plot(xd, y_sinc2, color=COL_SINC2, lw=SINC_LW, label="sinc² (same FWHM)")
    line_sinc.set_dashes((0, (7, 5)))  # (offset, (dash, gap))，可调 7/5
    line_sinc.set_dash_capstyle('round')  # 端帽圆润（配合 rcParams 更统一）

    ax.legend(
        loc='upper left', bbox_to_anchor=(0.02, 0.98),
        frameon=True, fancybox=True,
        handlelength=1.6, handletextpad=0.6, borderpad=0.3
    )
    dr = _DR_db(y)
    ax.set_title(f"{title} (DR = {dr:.2f} dB)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized SFG (a.u.)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    print(f"{title}\n  μ = {mu_fit:.3f} nm,  FWHM = {FWHM:.3f} nm,  DR = {dr:.2f} dB  →  {out_png}")

# ---- 使用 ----
file = "SFG20251013200948.txt"   # 替换为当前数据文件名
df = read_numeric_table(file)
spec6 = drop50_stats(df,"norm")
spec4 = fourstep(df)
plot_weighted(spec6,"Normalized SFG vs Wavelength - Gaussian Fit","norm_weighted.png")
plot_weighted(spec4,"Four-step BG-removed Normalized SFG vs Wavelength - Gaussian Fit","fourstep_weighted.png")
