# 程序逐段说明与物理意义

## 1. 文件层级

| 相对入口的位置 | 用途 |
|---|---|
| `run_pso.py` | 参数入口、调用流程、保存和绘图 |
| `requirements.txt` | CPU 必需包；CuPy 单独安装 |
| `opln_pso/__init__.py` | 对外导出常用类和函数 |
| `opln_pso/model.py` | Sellmeier、PMF、JSA、purity、pump scan |
| `opln_pso/optimizer.py` | CPU/CUDA 后端、PSO、可选局部回归 |
| `tests/test_model.py` | 公式与坐标约定的自动测试 |
| `tests/reproduce_qute.py` | 调用原入口执行固定复现协议，不另实现优化算法 |

`run_pso.py` 必须和 `opln_pso` 文件夹处于同一级。`model.py`、
`optimizer.py`、`__init__.py` 必须位于 `opln_pso` 文件夹内。

## 2. 整体物理流程

1. 输入 pump、signal、idler 中心波长与偏振。
2. 用 LN Sellmeier 方程求折射率、群折射率和材料相位失配。
3. 确定一阶 QPM 周期、晶体包含的周期数和 duty-cycle 上下界。
4. 每个周期由一个 `+1` domain 和一个 `-1` domain 组成；duty cycle
   决定二者的分界位置。
5. 对每个 domain 精确积分，得到复数 PMF。
6. PSO 调整全部 duty cycles，使 PMF 强度接近高斯目标。
7. 对 periodic、erf initial、optimized 三种结构分别扫描最优泵浦带宽。
8. 在 60 nm、200×200 波长网格上构造 JSA，SVD 后计算 Schmidt purity。
9. 保存 duty、domain wall、PMF、JSA、带宽扫描、后端信息和图片。

图上显示的是 `abs(JSA)**2`（JSI，联合谱强度）；纯度计算输入是归一化
JSA 矩阵，不是把 JSI 直接送入 SVD。图和数据文件分别标明这一区别。

## 3. `model.py`

### 3.1 加载的包

- `from __future__ import annotations`：推迟类型注解求值，便于使用
  `float | None` 等写法，不改变数值结果。
- `dataclass`：把互相关联的物理参数集中为不可变对象。
- `asdict`：将 dataclass 转成普通字典，便于写入 JSON。
- `typing.Any/Literal`：说明函数允许的输入类型和选项，不执行物理计算。
- `numpy`：数组、广播、复指数和线性代数的基础。
- `scipy.special.erf`：论文中的 Gaussian error function 初值。
- `scipy.linalg.svd`：与 `Py_JSI_SPDC.py` 相同的 Schmidt 分解工具。

### 3.2 Sellmeier 系数和 `refractive_index()`

当前 `y -> y + z` 计算使用原 MATLAB 的 y、z 系数：

\[
n^2(\lambda)=a+\frac{b}{\lambda^2-c}-d\lambda^2,
\]

其中波长单位为 µm。`_POL_INDEX` 把 `x/y/z` 映射到相应系数。
在本例中使用 type-II 的 `y -> y + z`：pump 和 signal 取 y，idler 取 z。
本次验证针对这两个偏振分支，不将其结果外推到未核对的其它晶体/偏振配置。

### 3.3 `group_index()`

程序对 Sellmeier 方程解析求导，再计算

\[
n_g=n-\lambda\frac{dn}{d\lambda}.
\]

群折射率决定三个脉冲包络在晶体中的走离，也决定高斯 PMF 宽度与泵浦
带宽之间的可分离条件。

### 3.4 `energy_conserving_idler()`

在一维 PMF 优化切片上固定 pump 中心波长，并由

\[
\frac{1}{\lambda_p}=\frac{1}{\lambda_s}+\frac{1}{\lambda_i}
\]

求 idler。这样每个 signal 采样点都有唯一的能量守恒 idler 点。

### 3.5 `material_mismatch()` 与 `qpm_period()`

材料相位失配为

\[
\Delta k_{\rm mat}=k_p-k_s-k_i,
\qquad k_\mu=\frac{2\pi n_\mu}{\lambda_\mu}.
\]

这里没有再次减去 `2*pi/Lambda`，因为周期翻转已经显式包含在
`g(z)=+1/-1` 中。若同时在失配中减 grating vector，会把 QPM 计算两次。

标准一阶周期估计为

\[
\Lambda=\frac{2\pi}{|\Delta k_{\rm mat}|}.
\]

Sellmeier 在 1.75 → 3.5+3.5 µm 给出约 15.50037 µm；QUTE Table B1
列出 15.497 µm。`build_geometry()` 支持显式输入论文值，并在 summary 中
同时保留计算值和采用值，避免暗中覆盖差异。

### 3.6 `gaussian_pump_parameters()`

泵浦复振幅写为

\[
\alpha(\Omega)=\exp[-\Omega^2/(2\sigma_\omega^2)].
\]

若输入强度脉宽，变换极限高斯脉冲满足

\[
\sigma_\omega=\frac{2\sqrt{\ln 2}}{\tau_{I,\rm FWHM}}.
\]

程序也可直接输入实测的强度光谱 FWHM；实验设计优先使用实测光谱。

### 3.7 `DesignGeometry`、`JSAConfig` 和 `build_geometry()`

`DesignGeometry` 保存一次设计的完整派生量，包括实际周期数、实际长度、
群折射率、初始泵浦带宽和制造边界。`JSAConfig` 保存 200×200、60 nm、
泵浦粗细两级扫描等默认值。

若没有指定 `target_sigma_z_mm`，程序用一阶双高斯可分离条件估计

\[
\sigma_z=\frac{1}{\sigma_\omega\sqrt{-ab}},\quad
a=\frac{n_{g,p}-n_{g,s}}{c},\quad
b=\frac{n_{g,p}-n_{g,i}}{c}.
\]

自动晶体长度为 `alpha*sigma_z`；指定 30 mm 时则使用给定长度。周期数
默认向下取整，和 thesis 的 `floor(30000/gamma)` 一致。

最小 domain `d_min` 直接转为 PSO 边界：

\[
A_{\min}=\frac{d_{\min}}{\Lambda},\qquad
A_{\max}=1-A_{\min}.
\]

因此约束在搜索过程中始终满足，并非优化完成后才裁剪。

### 3.8 `paper_erf_initial_duties()`

复现附录的初始形状：

\[
A_j=0.5+0.45\operatorname{erf}\left[
\frac{j/M-0.5}{0.45}\right].
\]

它只是 PSO 的初始粒子，不是频域 cost 的 target。

### 3.9 `cumulative_erf_target()`

频域高斯的逆傅里叶变换仍为实空间高斯；对实空间高斯从晶体起点累计
积分后得到 erf。因此这个函数适合下一版的 `+1/-1` 朝向规划，而当前
duty-cycle PSO 仍在频域拟合 Gaussian PMF。

### 3.10 `domain_table()`

第 j 个周期边界为

\[
z_0=j\Lambda,\quad z_m=(j+A_j)\Lambda,\quad z_1=(j+1)\Lambda.
\]

`z0` 到 `zm` 取 `+1`，`zm` 到 `z1` 取 `-1`。输出表可直接检查每个
物理 domain 的长度。

### 3.11 `pmf_from_duties()`：最重要的傅里叶变换

定义与论文正文一致：

\[
\phi(\Delta k)=\frac{1}{L}\int_0^L g(z)e^{-i\Delta k z}\,dz.
\]

对一个周期的两个 domain 精确积分并求和，得到

\[
\phi=\frac{1}{i\Delta k L}\sum_j
\left[e^{-i\Delta k z_{0,j}}
-2e^{-i\Delta k z_{m,j}}
+e^{-i\Delta k z_{1,j}}\right].
\]

这里没有 FFT 网格近似；domain wall 位于任意连续位置，积分是解析的。
`chunk_size` 只控制内存，不改变公式。
在 `k=0` 或极小失配处，程序使用等价的 sinc 表达以避免 0/0；其极限为
`mean(2*A-1)`，即正负 domain 长度的净比例。

### 3.12 JSA 网格和 `pump_amplitude_py_jsi()`

`jsa_wavelength_grid()` 生成均匀波长网格，数组行是 idler、列是 signal，
所以图片横轴 signal、纵轴 idler。

`pump_amplitude_py_jsi()` 逐行复现 `Py_JSI_SPDC.py` 的约定：

\[
\omega_s=2\pi c/\lambda_s,\quad
\omega_i=2\pi c/\lambda_i,
\]

\[
\sigma_p=\frac{\sqrt2}{2.355}
\frac{2\pi c\,\Delta\lambda_{I,\rm FWHM}}{\lambda_{p0}^2},
\]

\[
\alpha=\exp[-(\omega_s+\omega_i-\omega_{p0})^2/(2\sigma_p^2)].
\]

### 3.13 `evaluate_jsa_from_pmf()` 与 `schmidt_purity()`

QUTE 复现模式先取 `abs(PMF)`，再构造

\[
f(\omega_s,\omega_i)=\alpha(\omega_s+\omega_i)|\phi|.
\]

JSA 先作 L2 归一化，再进行奇异值分解。若奇异值为 `s_j`，则

\[
P=\sum_j s_j^4.
\]

程序仍保留 `pmf_mode="complex"` 供以后研究谱相位，但 QUTE 回归固定使用
`magnitude` 作为用户脚本口径的主指标，并另报相同带宽下的复 PMF 纯度。
不能把去除 PMF 相位后的数值当成所有相位效应均已计入的物理纯度。

### 3.14 `scan_optimal_pump_bandwidth()`

每种晶体的二维 PMF 只算一次。随后按照 `Py_JSI_SPDC.py`：先从 0.5 到
8.0 nm 以 0.1 nm 粗扫，再在粗扫最优值附近 ±1 nm 以 0.01 nm 精扫。
扫描过程静默，最终带宽、purity 和扫描数据写入文件。
本程序将精扫范围限制在用户指定的上下界内；若最优点落在端点，summary
会标记。相比原脚本允许精扫越过上界，这是一项显式的边界保护。

### 3.15 其余封装函数与单位

| 函数 | 输入、输出与作用 |
|---|---|
| `DesignGeometry.to_dict()` | 将所有派生量转换成 summary 可保存的字典，不重新计算物理量 |
| `qute_3500_geometry()` | 明确生成 3.5 µm、30 mm、sigma_z=6 mm 的基准几何；不覆盖通用入口默认值 |
| `pmf_on_jsa_grid()` | 每个 signal/idler 对按能量守恒求对应 pump 波长，计算二维材料失配，再调用精确 PMF |
| `evaluate_jsa()` | 先生成二维 PMF，再委托 `evaluate_jsa_from_pmf()` 形成一套 JSA 结果 |
| `evaluate_bandwidths()` | pump scan 内部的小函数；对候选带宽重复乘泵浦包络并求 SVD，复用同一二维 PMF |

`_um` 是 µm，`_nm` 是 nm，`_fs` 是 fs，`_rad_um` 是 rad/µm；乘积
`mismatch_rad_um * z_um` 必须无量纲。群速度匹配中长度使用 m，角频率
使用 rad/s，因此那里使用 SI 光速。`alpha=L/sigma_z` 是无量纲截断参数，
不是泵浦包络函数本身。

纯度主指标严格采用用户脚本的均匀波长离散法；保留复 PMF 的补充指标
也在同一网格上计算。因此补充指标解决的是相位差异，不自动解决任意
大带宽下波长/频率测度变换的全部问题。需要比较更宽频谱时仍要明确
积分测度、Jacobian 与窗口，并检查收敛。

## 4. `optimizer.py`

此模块加载 `numpy`（CPU 数组）、`math`（标量常数/函数）、`dataclass`
（配置）、`Any/Callable`（数组和回调类型说明）；`scipy.optimize.minimize`
只在显式启用 L-BFGS-B 时参与计算。`.model` 是包内相对导入，从同目录
`model.py` 取得几何、能量守恒、相位失配和 erf 初值。`cupy` 在选择后端
时才导入，因此没有安装它的 CPU 环境仍可运行。

### 4.1 `_load_backend()`

- `cpu`：强制 NumPy；
- `cuda`：必须成功导入 CuPy 并找到 GPU，否则报错；
- `auto`：优先 CuPy/GPU，失败才回退 NumPy。

NumPy 和 CuPy 使用几乎相同的数组 API，因此同一套 PSO 数学代码可在
CPU 或 CUDA 上运行。

### 4.2 `PSOConfig`

保存粒子数、每 cycle 迭代数、cycle 数、惯性权重、认知/社会系数、随机
种子、初始扰动、扰动衰减、对称约束和设备。QUTE 基线关闭强制对称，
因为论文没有施加 `A[j]+A[M-1-j]=1` 约束。

### 4.3 `PMFObjective`

默认在 3.47–3.53 µm 的 601 个均匀 signal 波长点优化，与论文 60 nm
切片相同。每一点先求能量守恒 idler 和材料失配，再定义

\[
q=\Delta k_{\rm mat}(\lambda_s,\lambda_i)-
\Delta k_{\rm mat}(\lambda_{s0},\lambda_{i0}).
\]

高斯目标为

\[
\phi_{\rm tgt}(q)=e^{-(q\sigma_z)^2/2},\qquad
I_{\rm tgt}(q)=e^{-(q\sigma_z)^2}.
\]

target 的中心由目标波长的材料失配决定，不再错误地强行绑定到
`2*pi/qpm_period`。Table B1 周期与 Sellmeier 周期的微小差异因此不会把
高斯 target 平移到错误波长。

`pmf_batch()` 同时处理全部粒子。每个粒子的一行包含完整 duty 向量，
输出为 `particles × wavelengths` 的 PMF 矩阵。

`legacy_intensity` cost 为归一化 PMF 强度与高斯强度的均方误差；这是本次
QUTE 复现默认。`complex_amplitude` 是保留的研究选项，不参与本表复现。

每个粒子的默认 cost 具体为

\[
C=\frac1{K}\sum_{l=1}^{K}
\left(\frac{|\phi(k_l)|^2}{\max_r |\phi(k_r)|^2}
-e^{-q_l^2\sigma_z^2}\right)^2.
\]

这里 K 是 PMF 采样点数。分母设有很小的正下限，避免除零。按峰值归一化
会消掉绝对转换效率信息；这解释了为何很低 cost 的解仍可能很暗。

`__init__()` 建立取样、target 和后端；`_compute_fixed_terms()` 预计算
不随 duty 移动的周期起止端点相位；`pmf_batch()` 只重算移动 wall；
`__call__()` 让 objective 对象可直接像函数一样被 PSO 调用。

### 4.4 `optimize_duty_cycles()`：PSO 实现

每个粒子的位置向量就是所有 duty cycles：

\[
\mathbf{x}_r=(A_{r,0},A_{r,1},\ldots,A_{r,M-1}).
\]

标准 PSO 更新为

\[
\mathbf{v}\leftarrow w\mathbf{v}
+c_1\mathbf{r}_1(\mathbf{p}_{best}-\mathbf{x})
+c_2\mathbf{r}_2(\mathbf{g}_{best}-\mathbf{x}),
\]

\[
\mathbf{x}\leftarrow\mathbf{x}+\mathbf{v}.
\]

惯性权重从 0.9 线性降到 0.4。超出 duty 边界的位置被投影回可制造区间，
相应速度清零。

一个 cycle 完成后，以当前 global best 为中心重建局部 swarm，并用
`noise_decay` 缩小半径。这仍然是 PSO，不使用梯度；它对应论文反复执行
duty 优化的思想，也便于保存 checkpoint 后继续运行。

每个 cycle 中约 75% 的粒子在最好解附近加高斯扰动，其余粒子仍在全边界
内随机分布，第一颗粒子保留当前最好解。`p_best` 保存每个粒子历史上最好
的位置，`g_best` 保存全群最好位置；它们都只在 cost 改善时更新。
`callback` 是可选进度回调，主程序默认不传逐轮打印函数。

`_expand_symmetric()` 仅在打开对称选项时，将半个向量补成反向互补结构
`A[j]+A[M-1-j]=1`；奇数周期的中心 duty 固定为 0.5。它减少自由度，
因此本次主基准关闭该选项。`_to_numpy()` 在保存、比较标量等必要位置
把 CuPy 数据转回主内存，不会把整套 GPU PMF 运算偷偷切换到 CPU。

### 4.5 `polish_duty_cycles_lbfgsb()`

这是明确分开的可选回归工具。domain-wall 解析积分对 duty 的导数为

\[
\frac{\partial\phi}{\partial A_j}
=\frac{2}{M}e^{-i\Delta k(j+A_j)\Lambda}.
\]

利用该导数可迅速验证模型是否能达到目标。它默认关闭；只有显式设置
`--polish-iterations` 才执行，并在 summary 中标成 L-BFGS-B/CPU，绝不
记作纯 PSO 或 CUDA 结果。

内部 `value_and_gradient()` 同时返回损失和解析梯度，使局部算法不必对
1935 个 duty 分别作有限差分。此诊断分支在求导时采用中心点强度归一化；
结束后再用默认 PMFObjective 的峰值归一化重新计算并记录 cost，两者的
差异在偏离中心峰很远的结构上可能有影响，不能混作同一优化过程。

## 5. `run_pso.py`

### 5.1 标准库和绘图库

- `argparse`：命令行参数；
- `inspect`：从子模块读取默认值，消除重复定义；
- `pathlib`：跨平台路径；
- `json/csv`：保存可读结果；
- `matplotlib`：绘图；
- `LogNorm`：JSA 显示 10⁻⁴ 到 1 的动态范围；
- `MultipleLocator`：x/y 使用相同 20 nm 主刻度；
- `colorsys`：复现此前蓝→青→绿→黄→红的 Hue 配色。
- `platform/sys/time`：记录运行时 Python、解释器和耗时，不参与物理计算。
- `numpy/scipy`：前者负责保存数组与后处理，后者在 summary 中提供版本号；
  `Any` 用作类型注解；`mcolors` 创建自定义色图。

### 5.2 `parse_args()`

物理、JSA、PSO 和 objective 的默认值分别来自 `build_geometry`、
`JSAConfig`、`PSOConfig`、`PMFObjective`。主程序只负责把它们映射为命令行
选项。因此在 IDE 直接运行时，修改子程序默认值即可，不需要同步修改
主程序第二份数值。

### 5.3 `main()`

执行顺序为：建模 → 解析 CPU/CUDA → 生成三种 duty → PSO/断点/可选回归
→ 三种结构分别扫描泵浦 → 保存结果 → 绘图。

程序开始即输出实际 PSO 后端，但不打印每次迭代。`summary.json` 同时保存
`device_requested` 和 `device_resolved`，避免 `auto` 的实际选择不透明。

### 5.4 绘图

`plot_overview()` 画 duty、能量守恒 PMF 切片、收敛曲线和三条泵浦扫描。
`plot_jsas()` 固定横轴 signal、纵轴 idler、相同范围、相同刻度间隔和
`aspect="equal"`。旧图刻度不同是 Matplotlib 对两个轴分别自动选刻度造成
的显示选择；同时旧程序采用均匀频率网格、转为波长后本来就不是均匀
波长。两者是不同问题，新版分别统一显示刻度和物理采样约定。

### 5.5 保存、默认值与设备辅助函数

| 函数 | 作用 |
|---|---|
| `_default()` | 用 `inspect.signature()` 读取指定子函数/配置构造器的参数默认值 |
| `save_csv()` | 写标题和数据行；CSV 是便于外部检查的交换格式 |
| `load_vector()` | 加载 `.npy` duty 或历史；检查有限值和所需向量长度 |
| `create_qute_colormap()` | 将 Hue 序列转换为 RGB，并生成连续色图；不改变 JSA 数据 |
| `cuda_name()` | 读取实际 CUDA 设备名称；字节串转成可读文字 |

`__init__.py` 只定义包对外暴露的函数/类，不会自动启动 PSO。
`requirements.txt` 声明 NumPy、SciPy、Matplotlib 的依赖；其最小版本不是
要求将已经正常工作的环境全部降级。测试程序可直接以 Python 模块运行，
不需要为了运行测试再安装 pytest。

`tests/reproduce_qute.py` 用 `argparse` 接收协议参数，用 `subprocess`
调用同一个主入口，使用 `pathlib` 管理阶段目录、`json` 读取结果、
`os/sys` 传递当前解释器与子进程环境。`--stop-after 1000` 可重跑已报告的
500 nm 结果；默认 1700 是固定基准协议，不是另一处通用物理默认值。

## 6. 为什么比原 MATLAB 快

速度差异主要不是“Python 语言天然更快”，而是实现方式和工作量不同：

1. 原附录在每次 cost 中用 MATLAB 循环逐 domain 累加；这里把粒子、
   domain 和波长组织成批量数组。
2. 与 duty 无关的两个周期端点相位只预计算一次；每次只更新移动 wall。
3. CuPy 把大量彼此独立的复指数同时交给 RTX 4090。
4. 波长轴分 chunk，避免一次生成超出显存的大张量。
5. 早期 Python 测试只有 200 次 duty 更新，而论文报告约 60,000 次总优化，
   所以当时的运行时间不能直接当成同等工作量下的加速比。

短晶体、小粒子数时，CUDA kernel 启动和数据传输可能让 GPU 优势不明显；
30 mm、1935 变量和较大 swarm 才是 GPU 最适合的规模。

## 7. 怎样确认 GPU 确实在用

1. 程序启动行必须是：

   ```text
   PSO backend resolved: cuda (NVIDIA GeForce RTX 4090)
   ```

2. `summary.json` 中必须有：

   ```json
   "device_requested": "auto",
   "device_resolved": "cuda"
   ```

3. 另开 Anaconda Prompt，在 PSO 阶段运行：

   ```bash
   nvidia-smi -l 1
   ```

   应看到 Python 进程、显存占用变化和间歇性 GPU-Util。Windows WDDM 下
   进程列表可能把显存显示成 `N/A`，这不等于没有使用 CUDA。

4. PSO 完成后进入 CPU 的 JSA/SVD 扫描，GPU-Util 降低是正常现象。

你此前的 `cp.show_config()` 已识别 RTX 4090，且百万点 CuPy 复指数测试
返回 `CUDA test passed`，说明驱动、CuPy 和基本 kernel 已能工作。无需仅因
显示的 toolkit/runtime 小版本不完全相同就主动降级；先以完整程序稳定运行
为准。
