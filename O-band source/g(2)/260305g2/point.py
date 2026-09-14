import matplotlib as mpl
import matplotlib.pyplot as plt

# 全局字体设置（以后所有图都生效）
mpl.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

data = [
    (1, 1.9141), (2, 1.9221), (3, 1.9367),
    (4, 1.9554), (5, 1.9719), (6, 1.9283),
    (7, 1.9081), (8, 1.8897),
]

x = [d[0] for d in data]
y = [d[1] for d in data]

plt.figure(figsize=(6, 4))
plt.scatter(x, y, s=60)  # 点也稍微放大
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()