# -*- coding: utf-8 -*-
import io, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SFG_FILE = "SFG20251014101522.txt"        # 第1列=波长, 第5列=SFG
SPF_XLSX = "FESH0850_Transmission.xlsx"   # FESH0850透射率文件
OUT_PNG  = "SFG_vs_FESH0850_dual_axis.png"

DROP_FIRST = 50
TAKE_NEXT  = 200

# 颜色定义
COL_SFG = "#e69f00"   # 橙色点
COL_SPF = "#b22222"   # 深红线

# ---------- 读 SFG ----------
def read_sfg_raw_mean(path_txt, drop_first=50, take_next=200):
    with open(path_txt, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln for ln in f.readlines() if re.match(r'^\s*[+-]?\d', ln)]
    arr = np.loadtxt(io.StringIO(''.join(lines)))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    wl  = arr[:, 0].astype(float)
    sfg = arr[:, 8].astype(float)
    df = pd.DataFrame({"wl": wl, "sfg": sfg})
    rows = []
    for w, g in df.groupby("wl"):
        v = g["sfg"].to_numpy()
        v = v[drop_first: drop_first + take_next] if take_next is not None else v[drop_first:]
        if v.size > 0:
            rows.append((float(w), float(np.mean(v)), float(np.std(v, ddof=1)) if v.size>1 else 0.0))
    return pd.DataFrame(rows, columns=["wl","mean","std"]).sort_values("wl")

# ---------- 读 FESH0850 ----------
def read_fesh_transmission(path_xlsx):
    xls = pd.read_excel(path_xlsx, sheet_name=0, header=None)
    num = xls.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1)
    df = num.iloc[:, :2].dropna().copy()
    df.columns = ["wl","T"]
    if df["wl"].max() < 10:  # μm → nm
        df["wl"] = df["wl"] * 1000.0
    if df["T"].max() > 1.5:  # % → 0–1
        df["T"] = df["T"] / 100.0
    return df.sort_values("wl")

# ---------- 主程序 ----------
sfg = read_sfg_raw_mean(SFG_FILE)
spf = read_fesh_transmission(SPF_XLSX)

# 只取1624–1667 nm区间
spf = spf.query("1624 <= wl <= 1662").copy()

# 归一化
sfg["y"] = sfg["mean"] / sfg["mean"].max()
spf["y"] = spf["T"] / spf["T"].max()

# ---------- 绘图 ----------
fig, ax_sfg = plt.subplots(figsize=(9.5,4.6))

# 底轴：SFG（橙色）
ax_sfg.plot(sfg["wl"].to_numpy(), sfg["y"].to_numpy(),
            'o', color=COL_SFG, ms=3, label="Raw SFG (normalized)")
ax_sfg.set_xlabel("Wavelength (nm) — SFG band (1570–1608)", color=COL_SFG)
ax_sfg.tick_params(axis='x', colors=COL_SFG)
ax_sfg.spines['bottom'].set_color(COL_SFG)
ax_sfg.xaxis.label.set_color(COL_SFG)
ax_sfg.set_ylabel("Normalized amplitude")
ax_sfg.grid(alpha=0.3)
ax_sfg.set_ylim(-0.05, 1.05)

# 顶轴：SPF（红色）
ax_spf = ax_sfg.twiny()
ax_spf.plot(spf["wl"].to_numpy(), spf["y"].to_numpy(),
            color=COL_SPF, lw=1.6, label="FESH0850 Transmission (normalized)")
ax_spf.set_xlabel("Wavelength (nm) — FESH0850 transmission band (1624–1667)", color=COL_SPF)
ax_spf.tick_params(axis='x', colors=COL_SPF)
ax_spf.spines['top'].set_color(COL_SPF)
ax_spf.xaxis.label.set_color(COL_SPF)

# 图例
lines1, labels1 = ax_sfg.get_legend_handles_labels()
lines2, labels2 = ax_spf.get_legend_handles_labels()
ax_sfg.legend(lines1+lines2, labels1+labels2, loc="best")

fig.tight_layout()
plt.title("Dual-axis comparison: SFG vs FESH0850 transmission")
plt.savefig(OUT_PNG, dpi=300)
plt.show()

print("Saved:", OUT_PNG)
