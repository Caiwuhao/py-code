import glob, re, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== 用户可调参数 =====================
DATA_GLOB = "Data1225-*_#001.txt"   # 或者写成 r"D:\...\Data1225-*_#001.txt"
DROP_FIRST = 8

OUTLIER_POS_MM = 70.0              # 只对这个位置做 outlier removal
OUTLIER_K_MAD = 5.0                # 阈值 = median ± K*MAD（单位：µm）

# 如果你想用“固定阈值”而不是 MAD，改成下面这种：
USE_FIXED_THRESHOLD = False
FIXED_LO_UM = 2950.0
FIXED_HI_UM = 3050.0
# ======================================================

def read_thorlabs_txt(path: str) -> pd.DataFrame:
    """读取 Thorlabs Beam 导出的 txt（含前置 header），并返回表格 DataFrame。"""
    lines = Path(path).read_text(encoding="latin1").splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.startswith("Measurement") and "Time" in l and "Peak Position" in l:
            start = i
            break
    if start is None:
        raise ValueError(f"Could not locate table header in {path}")
    return pd.read_csv(io.StringIO("\n".join(lines[start:])), sep="\t")

def pos_mm_from_name(path: str) -> float:
    m = re.search(r"Data1225-(\d+)_", Path(path).name)
    if not m:
        raise ValueError(f"Could not parse position from filename: {path}")
    return float(m.group(1))

def find_col(df: pd.DataFrame, prefix: str) -> str:
    hits = [c for c in df.columns if c.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"Expected 1 col starting with '{prefix}', got: {hits}")
    return hits[0]

def mad_mask(y: np.ndarray, k: float = 5.0):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        s = np.std(y)
        lo, hi = med - k*s, med + k*s
    else:
        lo, hi = med - k*mad, med + k*mad
    mask = (y >= lo) & (y <= hi)
    return mask, lo, hi, med, mad

def fit_divergence(z_mm: np.ndarray, D_um: np.ndarray):
    """
    线性拟合 D(z) = intercept + slope*z
    slope 单位：µm/mm，数值上 = mrad（full-angle, 对直径 D）
    half-angle = slope/2
    """
    slope_um_per_mm, intercept = np.polyfit(z_mm, D_um, 1)
    pred = intercept + slope_um_per_mm*z_mm
    ss_res = np.sum((D_um - pred)**2)
    ss_tot = np.sum((D_um - np.mean(D_um))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    full_mrad = float(slope_um_per_mm)
    half_mrad = float(slope_um_per_mm)/2.0
    return float(slope_um_per_mm), float(intercept), full_mrad, half_mrad, float(r2)

# -------- 主流程 --------
paths = sorted(glob.glob(DATA_GLOB))
if not paths:
    raise FileNotFoundError(f"No files matched: {DATA_GLOB}")

rows = []
outlier_report = None

for p in paths:
    pos = pos_mm_from_name(p)
    df = read_thorlabs_txt(p).iloc[DROP_FIRST:].copy()

    bwx = find_col(df, "Beam Width Clip X")
    bwy = find_col(df, "Beam Width Clip Y")
    gdx = find_col(df, "Gaussian Diameter X")
    gdy = find_col(df, "Gaussian Diameter Y")

    # 只对 70 mm 的 GDY 做 outlier removal
    if pos == OUTLIER_POS_MM:
        y = df[gdy].astype(float).to_numpy()

        if USE_FIXED_THRESHOLD:
            mask = (y >= FIXED_LO_UM) & (y <= FIXED_HI_UM)
            lo, hi, med, mad = FIXED_LO_UM, FIXED_HI_UM, float(np.median(y)), float(np.median(np.abs(y-np.median(y))))
        else:
            mask, lo, hi, med, mad = mad_mask(y, k=OUTLIER_K_MAD)

        outlier_report = {
            "pos_mm": pos,
            "kept": int(mask.sum()),
            "total": int(len(mask)),
            "removed": int((~mask).sum()),
            "threshold_lo_um": float(lo),
            "threshold_hi_um": float(hi),
            "median_um": float(med),
            "mad_um": float(mad),
        }
        df = df.loc[mask].copy()

    def mean_std(col):
        arr = df[col].astype(float).to_numpy()
        return float(arr.mean()), float(arr.std(ddof=1)), int(len(arr))

    row = {"pos_mm": pos}
    for name, col in [
        ("BeamWidth_X", bwx), ("BeamWidth_Y", bwy),
        ("GaussDiam_X", gdx), ("GaussDiam_Y", gdy)
    ]:
        mu, sd, n = mean_std(col)
        row[f"{name}_mean_um"] = mu
        row[f"{name}_std_um"] = sd
        row[f"{name}_N"] = n
    rows.append(row)

stats = pd.DataFrame(rows).sort_values("pos_mm").reset_index(drop=True)
print("\n=== Outlier report (only for 70 mm GDY) ===")
print(outlier_report)

# 输出 CSV
stats.to_csv("stats_summary_40_100_cleaned.csv", index=False)
print("\nSaved: stats_summary_40_100_cleaned.csv")

# divergence
z = stats["pos_mm"].to_numpy()
div_rows = []
for metric in ["BeamWidth", "GaussDiam"]:
    for axis in ["X", "Y"]:
        D = stats[f"{metric}_{axis}_mean_um"].to_numpy()
        slope, intercept, full_mrad, half_mrad, r2 = fit_divergence(z, D)
        div_rows.append({
            "metric": metric,
            "axis": axis,
            "slope_um_per_mm": slope,
            "div_full_mrad": full_mrad,
            "div_half_mrad": half_mrad,
            "r2": r2
        })
div = pd.DataFrame(div_rows).sort_values(["metric", "axis"])
div.to_csv("divergence_linear_fit.csv", index=False)
print("\n=== Divergence (linear fit of mean diameter vs position) ===")
print(div)
print("\nSaved: divergence_linear_fit.csv")

# -------- 画图：GaussDiam X/Y vs z --------
def plot_metric(axis: str):
    D = stats[f"GaussDiam_{axis}_mean_um"].to_numpy()
    E = stats[f"GaussDiam_{axis}_std_um"].to_numpy()
    slope, intercept, full_mrad, half_mrad, r2 = fit_divergence(z, D)

    zz = np.linspace(z.min(), z.max(), 200)
    plt.figure(figsize=(6,4))
    plt.errorbar(z, D, yerr=E, fmt='o', capsize=3)
    plt.plot(zz, intercept + slope*zz)
    plt.xlabel("Position z (mm)")
    plt.ylabel(f"Gaussian Diameter {axis} (µm)")
    plt.title(f"Gauss Diam {axis} vs z | full={full_mrad:.3f} mrad, half={half_mrad:.3f} mrad, R²={r2:.4f}")
    plt.tight_layout()
    plt.savefig(f"gauss_diam_{axis}.png", dpi=200)
    plt.show()

plot_metric("X")
plot_metric("Y")

# -------- 画图：70 mm GDY outlier 可视化 --------
# 重新加载 70 mm 原始序列（丢前8后）并展示阈值
p70 = [p for p in paths if abs(pos_mm_from_name(p) - OUTLIER_POS_MM) < 1e-9][0]
df70 = read_thorlabs_txt(p70).iloc[DROP_FIRST:].copy()
gdy70 = find_col(df70, "Gaussian Diameter Y")
y_raw = df70[gdy70].astype(float).to_numpy()

if USE_FIXED_THRESHOLD:
    mask = (y_raw >= FIXED_LO_UM) & (y_raw <= FIXED_HI_UM)
    lo, hi = FIXED_LO_UM, FIXED_HI_UM
else:
    mask, lo, hi, med, mad = mad_mask(y_raw, k=OUTLIER_K_MAD)

plt.figure(figsize=(6,4))
plt.plot(y_raw, marker='o', linestyle='None', markersize=3, label="raw")
plt.plot(np.where(mask)[0], y_raw[mask], marker='o', linestyle='None', markersize=3, label="kept")
plt.axhline(lo, linestyle='--')
plt.axhline(hi, linestyle='--')
plt.xlabel("Sample index (after dropping first 8)")
plt.ylabel("Gaussian Diameter Y at 70 mm (µm)")
plt.title(f"70 mm outlier removal: kept {mask.sum()}/{len(mask)} | [{lo:.1f}, {hi:.1f}] µm")
plt.legend()
plt.tight_layout()
plt.savefig("outlier_70mm_GDY.png", dpi=200)
plt.show()
