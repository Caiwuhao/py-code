import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DT_NS = 0.2  # 0.2 ns/bin
FONT_SIZE = 20

# --- VI-like parameters (for g2 processing) ---
THRESHOLD = 1000          # Peak Detector threshold (for peak finding)
WIDTH_MIN_DISTANCE = 40   # rough min distance in bins (comb spacing ~66 bins)
INTEGRATION_WIDTH = 3     # VI “Integration width” knob (N), integrated length = N+1

def load_xy(path):
    df = pd.read_csv(path, sep=r"\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    return x, y

def center_time_axis(x, y):
    i0 = int(np.argmax(y))
    # FIX: center by x[i0], not by index i0
    t = (x - x[i0]) * DT_NS
    return t, i0

def find_peaks_simple(y, min_height, min_distance):
    y = np.asarray(y)
    peaks = []
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= min_height:
            peaks.append(i)
    peaks_sorted = sorted(peaks, key=lambda i: y[i], reverse=True)
    selected = []
    for i in peaks_sorted:
        if all(abs(i-j) >= min_distance for j in selected):
            selected.append(i)
    return sorted(selected)

def integrate_vi(y, peak_pos_float, integration_width):
    """
    Replicate VI logic:
      k = round(peak_pos)
      left = floor(N/2)
      length = N+1
      start = k - left
      area = sum(y[start : start+length])
    """
    N = int(integration_width)
    k = int(np.rint(peak_pos_float))
    left = N // 2
    L = N + 1
    start = k - left
    start = max(0, start)
    end = min(len(y), start + L)
    return float(np.sum(y[start:end]))

def estimate_noise_area_vi(y, peak_bins, integration_width):
    """
    Estimate "true noise" (baseline) in VI-integrated units:
    - mask out the same integration windows around each detected peak
    - compute mean baseline per bin from remaining bins
    - convert to area by multiplying (N+1)
    """
    N = int(integration_width)
    left = N // 2
    L = N + 1

    mask = np.ones(len(y), dtype=bool)
    for k in peak_bins:
        start = max(0, k - left)
        end = min(len(y), start + L)
        mask[start:end] = False

    baseline_per_bin = float(np.mean(y[mask]))
    noise_area = baseline_per_bin * L
    return baseline_per_bin, noise_area

# ---------- 1) unheralded g(2)-i plot (corr2 only) ----------
x2, y2 = load_xy("data260219-g2-s-2.txt")
t2, _ = center_time_axis(x2, y2)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
#ax.plot(t2, y2, color="#E69F00")  # orange, no label
ax.plot(t2, y2)  # default
ax.set_xlabel("Delay relative to main peak (ns)", fontsize=FONT_SIZE)
ax.set_ylabel("Coincidence counts", fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)

# remove whitespace on x
ax.margins(x=0)
ax.set_xlim(t2.min(), t2.max())

fig.tight_layout()
fig.savefig("g2_hist_clean_s.pdf", dpi=200)
plt.close(fig)

# ---------- 2) unheralded g(2) plot + purity (VI-like) ----------
xg, yg = load_xy("data260219-g2-s-2.txt")
tg, i0g = center_time_axis(xg, yg)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(tg, yg)  # default blue
ax.set_xlabel("Delay relative to main peak (ns)", fontsize=FONT_SIZE)
ax.set_ylabel("Coincidence counts", fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.set_ylim(0, 1.03 * yg.max())

# remove whitespace on x
ax.margins(x=0)
ax.set_xlim(tg.min(), tg.max())

fig.tight_layout()
fig.savefig("g2_hist_clean_s2.pdf", dpi=200)
plt.close(fig)

# ---- VI-like peak finding + integration ----
# Peak bins from raw histogram (integer bins)
peaks = find_peaks_simple(yg, min_height=float(THRESHOLD), min_distance=int(WIDTH_MIN_DISTANCE))

# If peak detector returns float positions, you'd use those. Here we only have integer bins;
# we treat them as float positions at the bin center.
peak_pos = [float(p) for p in peaks]

# integrate each peak with VI rule
areas = [integrate_vi(yg, p, INTEGRATION_WIDTH) for p in peak_pos]

# identify central peak (max area)
imax = int(np.argmax(areas))
CC = areas[imax]

# ACC = mean of side peaks (all except central), matches your screenshot behavior
side_areas = [a for idx, a in enumerate(areas) if idx != imax]
ACC = float(np.mean(side_areas))

# estimate noise (baseline) in the same integrated unit
baseline_per_bin, noise_area = estimate_noise_area_vi(yg, peak_bins=peaks, integration_width=INTEGRATION_WIDTH)

# background-corrected g2 and purity
g2 = (CC - noise_area) / (ACC - noise_area)
purity = g2 - 1.0

print("peaks (bins) =", peaks)
print("areas =", [round(a, 2) for a in areas])
print(f"CC={CC:.2f}, ACC={ACC:.2f}")
print(f"baseline={baseline_per_bin:.2f} cnt/bin, noise_area={noise_area:.2f} (for N={INTEGRATION_WIDTH}, L={INTEGRATION_WIDTH+1} bins)")
print(f"g2(0)={g2:.6f}, purity≈g2-1={purity:.6f}")
