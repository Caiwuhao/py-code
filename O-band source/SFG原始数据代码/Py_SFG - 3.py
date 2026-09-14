# -*- coding: utf-8 -*-
# ==== SFG 总脚本（稳健拟合 + 正确 Fig.8 叠加）====
# 用法：
# 1) 把下面两行文件名改成你的 C/L 数据；
# 2) 直接运行，会生成 fig1~fig8（同目录）。

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os # <-- 最小化增加：用于创建文件夹

# --- 1. 创建文件夹 (最小化增加) ---
# 确保保存图片的 'figures' 文件夹存在
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# --- 2. 定义函数 (与您原始代码一致) ---
def gaussian(x, amp, mu, sigma):
    return amp*np.exp(-(x-mu)**2/(2*sigma**2))

def sinc_squared(x, amp, mu, width):
    return amp*(np.sinc((x-mu)/width))**2

# --- 3. 定义文件路径 (与您原始代码一致) ---
file_path_C = 'SFG20251029105128.txt'
file_path_L = 'SFG20251029180550.txt'

# --- 4. C-band 处理 (开始修改) ---
data_C = np.loadtxt(file_path_C, skiprows=1, delimiter=None)
wave_C = data_C[:, 0]
BG00_C, BG01_C, BG10_C, SFG11_C = data_C[:, 1], data_C[:, 2], data_C[:, 3], data_C[:, 4]
std00_C, std01_C, std10_C, std11_C = data_C[:, 5], data_C[:, 6], data_C[:, 7], data_C[:, 8]

SFG_removed_C = SFG11_C - BG10_C - BG01_C + BG00_C
std_removed_C = np.sqrt(std11_C**2 + std10_C**2 + std01_C**2 + std00_C**2)

peak_value_C = np.max(SFG_removed_C)
SFG_removed_C_norm = SFG_removed_C / peak_value_C
std_removed_C_norm = std_removed_C / np.abs(peak_value_C)

# --- 核心修正点 1：定义绘图底限 ---
plot_floor_db = -50.0
plot_floor_linear = 10**(plot_floor_db / 10)

# --- 核心修正点 2：安全的 log10 ---
# 原始: SFG_removed_C_norm_dB = 10*np.log10(SFG_removed_C_norm)
SFG_removed_C_norm_dB = 10*np.log10(np.maximum(SFG_removed_C_norm, plot_floor_linear))

p0_gauss = [1, 1555, 5]
p0_sinc = [1, 1555, 10]

# --- 核心修正点 3：为 C-band 拟合添加 Bounds ---
# 原始: popt_gauss, pcov_gauss = curve_fit(gaussian, wave_C, SFG_removed_C_norm, p0=p0_gauss)
# 原始: popt_sinc, pcov_sinc = curve_fit(sinc_squared, wave_C, SFG_removed_C_norm, p0=p0_sinc)
bounds_gauss_C = ([0, 1540, 0], [2, 1570, 20])
bounds_sinc_C = ([0, 1540, 0], [2, 1570, 50])
popt_gauss, pcov_gauss = curve_fit(gaussian, wave_C, SFG_removed_C_norm, p0=p0_gauss, bounds=bounds_gauss_C)
popt_sinc, pcov_sinc = curve_fit(sinc_squared, wave_C, SFG_removed_C_norm, p0=p0_sinc, bounds=bounds_sinc_C)

y_fit_gauss = gaussian(wave_C, *popt_gauss)
y_fit_sinc = sinc_squared(wave_C, *popt_sinc)

# --- 核心修正点 2 (应用) ---
# 原始: y_fit_gauss_dB = 10*np.log10(y_fit_gauss)
# 原始: y_fit_sinc_dB = 10*np.log10(y_fit_sinc)
y_fit_gauss_dB = 10*np.log10(np.maximum(y_fit_gauss, plot_floor_linear))
y_fit_sinc_dB = 10*np.log10(np.maximum(y_fit_sinc, plot_floor_linear))

# --- C-band 绘图 (结构与您原始代码一致) ---
fig1 = plt.figure(figsize=(10, 6))
plt.plot(wave_C, SFG_removed_C_norm, '.', label='data')
plt.plot(wave_C, y_fit_gauss, label='Gaussian')
plt.plot(wave_C, y_fit_sinc, label='sinc$^2$')
plt.title('C band (background removed)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'C_linear_scale.png'))

fig2 = plt.figure(figsize=(10, 6))
plt.plot(wave_C, SFG_removed_C_norm_dB, '.', label='data')
plt.plot(wave_C, y_fit_gauss_dB, label='Gaussian')
plt.plot(wave_C, y_fit_sinc_dB, label='sinc$^2$')
plt.title('C band (background removed, dB scale)')
plt.legend()
# 核心修正点 4：统一 y 轴
# 原始: plt.ylim(-50, 1)
plt.ylim(plot_floor_db - 5, 1) 
plt.savefig(os.path.join(output_dir, 'C_dB_scale.png'))


# --- 5. L-band 处理 (应用相同修正) ---
data_L = np.loadtxt(file_path_L, skiprows=1, delimiter=None)
wave_L = data_L[:, 0]
BG00_L, BG01_L, BG10_L, SFG11_L = data_L[:, 1], data_L[:, 2], data_L[:, 3], data_L[:, 4]
std00_L, std01_L, std10_L, std11_L = data_L[:, 5], data_L[:, 6], data_L[:, 7], data_L[:, 8]

SFG_removed_L = SFG11_L - BG10_L - BG01_L + BG00_L
SFG_removed_L_norm = SFG_removed_L / peak_value_C
std_removed_L = np.sqrt(std11_L**2 + std10_L**2 + std01_L**2 + std00_L**2)

peak_value_L = np.max(SFG_removed_L)
std_removed_L_norm = std_removed_L / np.abs(peak_value_C) # 保持归一化一致性

# 核心修正点 2 (应用)
# 原始: SFG_removed_L_norm_dB = 10*np.log10(SFG_removed_L_norm)
SFG_removed_L_norm_dB = 10*np.log10(np.maximum(SFG_removed_L_norm, plot_floor_linear))

p0_gauss = [1, 1555, 5]
p0_sinc = [1, 1555, 10]

# 核心修正点 3 (应用)
bounds_gauss_L = ([0, 1540, 0], [2, 1570, 20])
bounds_sinc_L = ([0, 1540, 0], [2, 1570, 50])
popt_gauss, pcov_gauss = curve_fit(gaussian, wave_L, SFG_removed_L_norm, p0=p0_gauss, bounds=bounds_gauss_L)
popt_sinc, pcov_sinc = curve_fit(sinc_squared, wave_L, SFG_removed_L_norm, p0=p0_sinc, bounds=bounds_sinc_L)

y_fit_gauss = gaussian(wave_L, *popt_gauss)
y_fit_sinc = sinc_squared(wave_L, *popt_sinc)

# 核心修正点 2 (应用)
y_fit_gauss_dB = 10*np.log10(np.maximum(y_fit_gauss, plot_floor_linear))
y_fit_sinc_dB = 10*np.log10(np.maximum(y_fit_sinc, plot_floor_linear))

# --- L-band 绘图 (结构与您原始代码一致) ---
fig3 = plt.figure(figsize=(10, 6))
plt.plot(wave_L, SFG_removed_L_norm, '.', label='data')
plt.plot(wave_L, y_fit_gauss, label='Gaussian')
plt.plot(wave_L, y_fit_sinc, label='sinc$^2$')
plt.title('L band (background removed)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'L_linear_scale.png'))

fig4 = plt.figure(figsize=(10, 6))
plt.plot(wave_L, SFG_removed_L_norm_dB, '.', label='data')
plt.plot(wave_L, y_fit_gauss_dB, label='Gaussian')
plt.plot(wave_L, y_fit_sinc_dB, label='sinc$^2$')
plt.title('L band (background removed, dB scale)')
plt.legend()
plt.ylim(plot_floor_db - 5, 1) # 核心修正点 4
plt.savefig(os.path.join(output_dir, 'L_dB_scale.png'))


# --- 6. C+L band 合并处理 (应用相同修正) ---
wave_CL = np.concatenate((wave_C, wave_L))
SFG_removed_CL = np.concatenate((SFG_removed_C, SFG_removed_L))
std_removed_CL = np.concatenate((std_removed_C, std_removed_L))
SFG_removed_CL_norm = SFG_removed_CL / peak_value_C
std_removed_CL_norm = std_removed_CL / np.abs(peak_value_C)

# 核心修正点 2 (应用)
# 原始: SFG_removed_CL_norm_dB = 10*np.log10(SFG_removed_CL_norm)
SFG_removed_CL_norm_dB = 10*np.log10(np.maximum(SFG_removed_CL_norm, plot_floor_linear))

p0_gauss = [1, 1555, 5]
p0_sinc = [1, 1555, 10]

# 核心修正点 3 (应用)
# 原始: popt_gauss, pcov_gauss = curve_fit(gaussian, wave_CL, SFG_removed_CL_norm, p0=p0_gauss)
# 原始: popt_sinc, pcov_sinc = curve_fit(sinc_squared, wave_CL, SFG_removed_CL_norm, p0=p0_sinc)
bounds_gauss_CL = ([0, 1540, 0], [2, 1570, 20])
bounds_sinc_CL = ([0, 1540, 0], [2, 1570, 50])
popt_gauss, pcov_gauss = curve_fit(gaussian, wave_CL, SFG_removed_CL_norm, p0=p0_gauss, bounds=bounds_gauss_CL)
popt_sinc, pcov_sinc = curve_fit(sinc_squared, wave_CL, SFG_removed_CL_norm, p0=p0_sinc, bounds=bounds_sinc_CL)

y_fit_gauss = gaussian(wave_CL, *popt_gauss)
y_fit_sinc = sinc_squared(wave_CL, *popt_sinc)

# 核心修正点 2 (应用)
y_fit_gauss_dB = 10*np.log10(np.maximum(y_fit_gauss, plot_floor_linear))
y_fit_sinc_dB = 10*np.log10(np.maximum(y_fit_sinc, plot_floor_linear))

mu = popt_gauss[1]
sigma = popt_gauss[2]
fwhm = 2*np.sqrt(2*np.log(2))*sigma
DR = 10*np.log10(popt_gauss[0]/np.mean(SFG_removed_CL_norm[wave_CL>1580]))

# --- C+L 绘图 (结构与您原始代码一致) ---
fig5 = plt.figure(figsize=(10, 6))
plt.plot(wave_CL, SFG_removed_CL_norm, '.', label='data')
plt.plot(wave_CL, y_fit_gauss, label='Gaussian')
plt.plot(wave_CL, y_fit_sinc, label='sinc$^2$')
plt.title(f'C + L band (background removed)\n \
          $\mu$={mu:.3f} nm, FWHM={fwhm:.3f} nm, DR={DR:.2f} dB')
plt.legend()
plt.savefig(os.path.join(output_dir, 'C+L_linear_scale.png'))

fig6 = plt.figure(figsize=(10, 6))
plt.plot(wave_CL, SFG_removed_CL_norm_dB, '.', label='data')
plt.plot(wave_CL, y_fit_gauss_dB, label='Gaussian')
plt.plot(wave_CL, y_fit_sinc_dB, label='sinc$^2$')
plt.title(f'C + L band (background removed, dB scale)\n \
          $\mu$={mu:.3f} nm, FWHM={fwhm:.3f} nm, DR={DR:.2f} dB')
plt.legend()
plt.ylim(plot_floor_db - 5, 1) # 核心修正点 4
# 这是我们将在下面显示的关键图像
plt.savefig('C+L_dB_scale_FIXED.png') 


# --- 7. C+L band (带误差棒) (应用相同修正) ---

# 核心修正点 2 (应用)
# 原始: SFG_removed_CL_norm_dB = 10*np.log10(SFG_removed_CL_norm)
SFG_removed_CL_norm_dB = 10*np.log10(np.maximum(SFG_removed_CL_norm, plot_floor_linear))
# (注意：对数坐标下误差棒计算很复杂，原始代码中此处省略是合理的)
# std_removed_CL_norm_dB = 10*np.log10(std_removed_CL_norm) # 这行在原始代码中被注释了，保持不变

# 核心修正点 2 (应用)
y_fit_gauss_dB = 10*np.log10(np.maximum(y_fit_gauss, plot_floor_linear))
y_fit_sinc_dB = 10*np.log10(np.maximum(y_fit_sinc, plot_floor_linear))

# --- C+L 误差棒绘图 (结构与您原始代码一致) ---
fig7 = plt.figure(figsize=(10, 6))
plt.errorbar(wave_CL, SFG_removed_CL_norm, yerr=std_removed_CL_norm, fmt='.', label='data (mean $\pm$ 1$\sigma$)')
plt.plot(wave_CL, y_fit_gauss, label='Gaussian')
plt.plot(wave_CL, y_fit_sinc, label='sinc$^2$')
plt.title(f'C + L band (background removed)\n \
          $\mu$={mu:.3f} nm, FWHM={fwhm:.3f} nm, DR={DR:.2f} dB')
plt.legend()
plt.savefig(os.path.join(output_dir, 'C+L_linear_scale_errorbar.png'))

fig8 = plt.figure(figsize=(10, 6))
plt.plot(wave_CL, SFG_removed_CL_norm_dB, '.', label='data (mean only)')
# 原始代码中这里没有画 error bar，保持一致
plt.plot(wave_CL, y_fit_gauss_dB, label='Gaussian')
plt.plot(wave_CL, y_fit_sinc_dB, label='sinc$^2$')
plt.title(f'C + L band (background removed, dB scale)\n \
          $\mu$={mu:.3f} nm, FWHM={fwhm:.3f} nm, DR={DR:.2f} dB')
plt.legend()
plt.ylim(plot_floor_db - 5, 1) # 核心修正点 4
plt.savefig(os.path.join(output_dir, 'C+L_dB_scale_errorbar.png'))

print("代码执行完毕。8 张图已生成并保存到 'figures' 文件夹。")
print("用于验证修复的关键文件是 'C+L_dB_scale_FIXED.png'")