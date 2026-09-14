import numpy as np
import matplotlib.pyplot as plt

# =========================
# Input data
# =========================
# Raw 3-bin integrated peak areas
areas_raw = np.array([4142, 4110, 4304, 7940, 4067, 4060, 4092], dtype=float)

# x positions
x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

# =========================
# Analysis
# =========================
center_idx = len(areas_raw) // 2
acc_raw = np.mean(np.delete(areas_raw, center_idx))
areas_norm = areas_raw / acc_raw

# Final result to display
purity = 99.4
purity_err = 3.0

# =========================
# Style
# =========================
acc_color = "#556F9A"   # darker bars for ACC
cc_color  = "#9BBFE0"   # lighter bar for CC
edge_color = "#364A63"

plt.rcParams.update({
    "font.size": 20,
    "axes.linewidth": 1.2,
})

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(8.5, 6.5))

colors = [acc_color] * len(areas_norm)
colors[center_idx] = cc_color

ax.bar(
    x,
    areas_norm,
    width=0.78,
    color=colors,
    edgecolor=edge_color,
    linewidth=1.2
)

# Labels
ax.set_xlabel(r'$\Delta T$ (a.u.)', fontsize=32)
ax.set_ylabel('Coincidences (a.u.)', fontsize=32)

# Limits and ticks
ax.set_ylim(0, 2.00)
ax.set_xticks(x)
ax.set_yticks(np.arange(0, 2.01, 0.5))
ax.tick_params(axis='both', labelsize=24, length=5, width=1.0)

# Boxed frame: keep all spines visible
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Purity text at upper right
ax.text(
    0.97, 0.92,
    f'Purity:\n{purity:.1f} ± {purity_err:.1f}%',
    transform=ax.transAxes,
    fontsize=20,
    fontweight='bold',
    ha='right',
    va='top'
)

plt.tight_layout()
fig.savefig("poster_c.png", dpi=300)
fig.savefig("poster_c.pdf")
plt.show()