# 3.12 um OPLN 第一版设计说明

## 1. 原 MATLAB、O-band 和 custom-poling 的 target 到底是什么

原 MATLAB 程序里实际有三件不同的事：

1. `ML.m` 用 erf 形状初始化 duty-cycle 数组。附录中的 sigmoid 变换可
   代数化简为

   \[
   A(x)=0.5+0.45\,\mathrm{erf}[(x-0.5)/0.45].
   \]

2. `arrayloop.m` 计算并归一化的是 \(|\phi|^2\)，不是复 PMF 振幅。
3. `effcalc.m` 给出直接的高斯频谱，`cost.m` 对二者逐点作平方差。因此，
   qute 方案的 erf 只是初值，真正的频域 target 仍是直接高斯，而且旧
   cost 不约束谱相位。

O-band 推导和 `custom-poling` 并不矛盾。两者都先指定理想频域振幅

\[
\phi_I(\delta k)=\exp[-\delta k^2\sigma_z^2/2].
\]

其逆傅里叶变换是空间高斯非线性分布。再把空间分布从晶体入口积分到
当前位置，累计振幅自然就是 erf。O-band 直接写出了这个解析式；
`custom-poling` 则先输入复高斯 PMF，再由 `Target.compute_amplitude()`
数值积分得到沿 z 的累计目标，随后逐个选择 domain 方向。

所以两种形式的适用范围是：

- 连续 duty-cycle PSO：直接计算宽带复 PMF，拟合复高斯频谱更合适。
- 固定网格上的 `+1/-1` 朝向设计：累计 erf target 更合适，因为截至第 n
  个单元的实际累计振幅对二进制方向变量是线性的。
- 后续混合算法：先用 erf/0-1 规划确定朝向，再用复高斯频谱微调允许的
  domain wall 或 duty cycle；也可以把二者组合成 hybrid cost。

因此，后续完全可以采用高斯误差函数，但应把它定义成“空间累计振幅
target”，而不是直接拿 erf 替换现有频域 cost 中的高斯函数。

## 2. 第一版相对旧程序的主要改变

- 保留原 Sellmeier 系数和 `y -> y + z` Type-II 配置。
- 中心波长改为 `1.560 um -> 3.120 um + 3.120 um`。
- 第一版没有照搬旧程序中交替进行的 `PSO(omega_p, sigma_p)`：既然目标是
  直接使用现有 1560 nm 激光器，中心和实测带宽应作为输入，而不应让算法
  为了降低 cost 擅自“选择另一台泵浦”。程序改为由泵浦带宽计算匹配的
  `sigma_z` 和建议晶体长度；之后可单独扫描脉宽与长度。
- 每个周期内两个物理 domain 都满足

  \[
  A\Lambda\ge d_{\min},\qquad(1-A)\Lambda\ge d_{\min}.
  \]

  默认 `d_min=0.500 um`，不是优化结束后再裁剪。
- 同时提供 `legacy_intensity` 和 `complex_amplitude` 两种 cost。正式默认
  使用后者，并去除不影响物理的整体相位和线性传播相位。
- 默认强制 `A[j]+A[M-1-j]=1`，避免旧 PSO 自由搜索造成的非对称谱相位；
  可用 `--no-symmetry` 关闭。
- PSO 对粒子和频谱点批量计算；NumPy 走 CPU，CuPy 走 NVIDIA GPU。
- 输出完整 domain 边界、PMF、JSA、Schmidt purity、相对峰值效率和多窗口
  purity，而不只输出一条 duty-cycle 曲线。

## 3. 当前 3.12 um 结果及其假设

用附录 Sellmeier 公式计算：

- 中心 QPM 周期：15.3994107 um。
- 500 nm 约束对应 duty-cycle 范围：0.0324688-0.9675312。
- 若把此前候选的 1560 nm、60 fs 光源近似成变换极限高斯脉冲，其光谱
  强度 FWHM 约为 59.72 nm。
- 一阶群速度匹配给出的高斯空间宽度为 0.313225 mm；取
  `L/sigma_z=5`，按完整周期取整后晶体为 1.570740 mm，共 102 个周期。

这解释了为什么不能直接沿用旧例子的 30 mm。对当前色散，30 mm 与约
1.15 ps、约 3 nm 泵浦带宽更匹配。若实际 60 fs 激光更接近 sech-squared
脉冲，不能用 59.72 nm 这个高斯换算值；应把实测光谱 FWHM 用
`--pump-fwhm-nm` 输入。比如输入 43 nm 时，自动建议长度约为 2.17 mm。

正式设置（48 particles、200 iterations、复振幅、对称、频域 cost 覆盖
`+-15 sigma`）的 seed 7 示例为：

- 最短物理 domain：0.500 um。
- 优化后中心峰值效率：标准 50% duty PPLN 的约 23.34%；有限长度理想
  高斯目标本身约为 24.38%。
- 在 JSA 两轴 `+-4.5 pump sigma`（约 2.585-3.934 um）窗口内，Schmidt
  purity 为 0.99706。
- 将积分窗口扩大到 `+-6 sigma` 和 `+-8 sigma`，purity 分别为 0.99153
  和 0.96191。

最后一组数字非常重要：purity 必须连同滤波器、探测器响应和积分范围
一起给出。这里只能把 0.99706 称为指定窗口内的结果，不能称为无限频域
的无滤波 purity。

作为对照，旧式窄范围强度 cost 可以降到 `6.06e-7`，但相同 4.5-sigma
窗口内 purity 只有 0.94213，相对峰值效率只有 6.86%。这说明“主峰强度
看起来完美贴合高斯”并不足以证明设计正确。

## 4. GPU 能否加速

可以。PSO 每一代的粒子彼此独立，且每个频谱点也可并行，本程序用 CuPy
对这两个维度批量计算：

```bash
python run_pso.py --device cuda --wavelength-chunk 128
```

当前运行环境没有 CUDA/CuPy，所以这里只完成了 CPU 数值验证，不能声称
GPU 路径已经实机验证。约 102 个周期的 60 fs 设计在 CPU 上已经较快；
若强制 30 mm，变量约 1948 个，GPU 的价值会明显提高，同时应根据显存
调整 `--wavelength-chunk`。

## 5. 下一版 0-1 + 连续微调建议

1. 由厂商给出最小 domain、domain-wall 网格和总长度，取固定单元
   `d >= d_min`，用 `s_j in {-1,+1}` 表示方向。相邻同号单元自动合并，
   因而所有最终物理 domain 都是 d 的整数倍。
2. 构造累计复振幅

   \[
   F_n=\sum_{j\le n}s_j h_j
   \]

   并最小化它与 erf target 的平方差。这是二进制二次规划；也可同时加入
   若干频谱点上的复高斯残差和翻转次数/鲁棒性惩罚。
3. 以该二进制解为初值，只移动允许移动的 domain wall，连续微调宽带复
   PMF。每次移动都显式保留最小 domain 约束。
4. 最后对 domain-wall 随机误差、温度、泵浦中心和实测泵浦谱做 Monte
   Carlo 鲁棒性分析，再决定交给厂商的结构。

第一版尚未实现第 1-4 步的混合整数部分；当前代码专门用于先判断纯 PSO
在明确制造约束下能达到什么效果。
