# -*- coding: utf-8 -*-
import io, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ========= 全局风格 =========
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle":  "butt",
})

# ========= 绘图配色 =========
COL_DATA  = "#e69f00"   # 数据点 + 误差棒
DATA_MS   = 2.0         # 数据点尺寸
ERR_ELW   = 0.8         # 误差棒线宽
ERR_CAP   = 0           # 无端帽

# ========= 读取数据 =========
def read_numeric_table(path):
    """读取纯数字数据文件（自动跳过非数行）"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln for ln in f.readlines() if re.match(r'^\s*[+-]?\d', ln)]
    arr = np.loadtxt(io.StringIO(''.join(lines)))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    df = pd.DataFrame(arr)
    # 假定文件第1列是波长、第2列是SFG强度
    df.columns = ["wl", "sfg"] + [f"col{i}" for i in range(3, df.shape[1]+1)]
    return df[["wl", "sfg"]].dropna()

# ========= 每 λ 丢前 50，保留后200求均值/标准差 =========
def drop50_take200_stats(df):
    out = []
    for wl, g in df.groupby("wl"):
        vals = g["sfg"].to_numpy()
        if len(vals) <= 50:
            continue
        v = vals[50:250]   # 丢前50，取最多200个
        mean = v.mean()
        std  = v.std(ddof=1) if len(v) > 1 else 0.0
        out.append((wl, mean, std))
    return pd.DataFrame(out, columns=["wl","mean","std"]).sort_values("wl")

# ========= 主程序 =========
if __name__ == "__main__":
    FILE = "20251022151821.txt"
    df = read_numeric_table(FILE)
    spec = drop50_take200_stats(df)

    # 绘图
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.errorbar(spec["wl"], spec["mean"], yerr=spec["std"],
                fmt='o', ms=DATA_MS,
                mfc=COL_DATA, mec=COL_DATA, mew=0,
                ecolor=COL_DATA, elinewidth=ERR_ELW, capsize=ERR_CAP,
                label="raw SFG (mean ± 1σ)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Raw SFG intensity (a.u.)")
    ax.set_title("Raw SFG signal vs wavelength (drop 50, take 200)")
    ax.legend(loc="upper left", frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("raw_SFG_mean_std.png", dpi=300)
    plt.close(fig)

    print("Saved: raw_SFG_mean_std.png")
