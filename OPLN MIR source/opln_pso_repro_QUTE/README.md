# OPLN duty-cycle optimization：QUTE 复现修正版

本版本先固定复现 QUTE 论文的 3.5 µm 简并 SPDC：pump 1.75 µm，
signal/idler 3.5 µm，名义晶体长度 30 mm，Table B1 周期 15.497 µm，
1935 个 duty-cycle 变量。

JSA 使用论文的 60 nm、200×200 均匀波长网格。纯度计算与
`Py_JSI_SPDC.py` 一致：以 `abs(PMF)` 构造 JSA，归一化后做 SVD。
periodic、erf initial 和 optimized 三种晶体分别扫描各自的最优泵浦
intensity FWHM，扫描过程不在终端逐点输出。

## 验证目标与口径

QUTE Table B1 的参考值是 pump intensity FWHM 约 3.8 nm、purity
0.997571。已保存结果和精确运行协议见 `VALIDATION_QUTE_ZH.md`；每次
运行最终数值以对应 `summary.json` 为准，不用较小 cost 替代纯度验收。

主指标按用户指定的 Py_JSI 约定以 `abs(PMF)` 构造 JSA。新版还保存
`complex_purity_at_same_pump`，保留 PMF 相位，两者不可混称。

已保存的两组纯 PSO 结果（CPU、seed=7、每轴 60 nm 窗口）：

| 最小 domain | 累计 PSO 迭代 | 最优 pump FWHM / nm | Py_JSI 纯度 | 同泵浦保留复 PMF |
|---|---:|---:|---:|---:|
| 不限制 | 1700 | 3.78 | 0.997679977 | 0.996382591 |
| 500 nm | 1000 | 3.79 | 0.996466143 | 0.995671253 |

第一行达到指定的幅度比较指标；这不表示完整谱相位和宽窗口的论文结果
已经全部复现。无约束结构有零长度 domain；两组相对理想标准 PPLN 的
峰值强度分别约为 4.88% 和 7.76%，不是已经完成制造与亮度权衡的最终设计。

纯 PSO 使用 24 粒子、分阶段缩小搜索半径，没有使用梯度精修。论文报告
的总优化次数约为 60,000，因此不能把一次 200 iteration 的缩小测试与
论文最终结果直接比较，也不能据此宣称 Python 天然快几十倍。

“快速数值回归”是可选的解析梯度 L-BFGS-B 局部精修，只用于快速判断
傅里叶模型和制造边界的可达性。`PSOConfig.polish_iterations` 默认是 0，
所以普通运行仍是纯 PSO，不会静默换算法。

## 安装

```bash
python -m pip install -r requirements.txt
```

CuPy 是可选项，不在 `requirements.txt` 中强行固定 CUDA 版本。你的 Conda
环境已经安装并通过 CUDA 测试，无需重装。若新建环境，可用：

```bash
conda install -c conda-forge cupy
```

## 运行

Windows Anaconda Prompt、PowerShell、macOS/Linux 均可直接复制以下单行
命令。先进入包含 `run_pso.py` 的文件夹；IDE 解释器选择安装 CuPy 的
`my_env`，不要将两个子模块移到包目录之外。

自动执行明确的纯 PSO 复现协议（不调用 L-BFGS-B）：

```bash
python -m tests.reproduce_qute --device auto --min-domain-um 0
```

重跑已报告的 500 nm、1000 次迭代协议：

```bash
python -m tests.reproduce_qute --device auto --min-domain-um 0.5 --stop-after 1000
```

只检查已保存的无约束 PSO 解、重算三种晶体的最优泵浦和 JSA：

```bash
python run_pso.py --crystal-length-mm 30 --target-sigma-mm 6 --qpm-period-um 15.497 --min-domain-um 0 --initial-duties-npy output/stage_1700/optimized_duties.npy --evaluate-only --output output/recheck_saved_pso
```

程序会保存分阶段 checkpoint，并在最后如实显示是否达到参考值。CPU 和
CUDA 的随机序列不同，不保证每次得到同一个 duty 向量。模型测试命令：

```bash
python -m tests.test_model
```

严格采用论文 Table B1 的周期，做 500 nm 约束的独立模型回归：

```bash
python run_pso.py --crystal-length-mm 30 --target-sigma-mm 6 --qpm-period-um 15.497 --min-domain-um 0.5 --particles 1 --iterations 0 --polish-iterations 1500 --device auto --output output/new_500nm_regression
```

纯 PSO 示例：

```bash
python run_pso.py --crystal-length-mm 30 --target-sigma-mm 6 --qpm-period-um 15.497 --min-domain-um 0 --particles 24 --iterations 100 --cycles 5 --initial-noise 0.06 --noise-decay 0.6 --device auto --output output/new_pure_pso
```

断点续跑：

```bash
python run_pso.py --crystal-length-mm 30 --target-sigma-mm 6 --qpm-period-um 15.497 --min-domain-um 0 --particles 24 --initial-duties-npy output/new_pure_pso/optimized_duties.npy --prior-history-npy output/new_pure_pso/pso_history.npy --prior-pso-iterations 500 --initial-noise 0.005 --iterations 100 --cycles 5 --noise-decay 0.6 --device auto --output output/new_pure_pso_resume
```

`device="auto"` 在 CuPy/GPU 可用时自动采用 CUDA。程序开始时会显示
`PSO backend resolved: cuda (NVIDIA GeForce RTX 4090)`；若显示 `cpu`，则
PSO 没有走 GPU。JSA 绘图和最后的 SVD 扫描属于后处理，仍在 CPU 上执行。
如果你希望 IDE 运行时“必须 CUDA，失败就报错”，在 `PSOConfig` 中将
`device: str = "auto"` 改成 `device: str = "cuda"` 即可；主程序无需再改。

## 默认值的位置

主程序不复制子程序已经定义的默认值：

- 物理参数：`opln_pso/model.py` 的 `build_geometry()`；
- JSA 和 pump scan：`opln_pso/model.py` 的 `JSAConfig`；
- PSO：`opln_pso/optimizer.py` 的 `PSOConfig`；
- cost 采样：`opln_pso/optimizer.py` 的 `PMFObjective.__init__()`。

`run_pso.py` 使用 `inspect.signature()` 自动读取它们。以后在 IDE 中修改
中心波长、脉宽、最小 domain、粒子数等，只需要改子程序的一处。

## 输出

- `summary.json`：几何、算法、实际 CPU/CUDA 后端、三种晶体各自最优泵浦
  与 purity；
- `optimized_duties.npy`、`pso_history.npy`：断点续跑；
- `duty_cycles.csv`、`domains.csv`：duty 与物理 domain；
- `pump_bandwidth_scans.csv`：三种晶体独立的精细泵浦扫描；
- `optimization_overview.png`：duty、PMF/target、cost、带宽扫描；
- `jsa_comparison.png`：横轴 signal、纵轴 idler，相同刻度与旧配色；
- `result_arrays.npz`：进一步分析所需数组。
- `jsa_arrays.npz`：最优泵浦下真正用于绘图的三套 JSA/PMF/PEF 及扫描数组。

预置结果：`output/stage_1700` 为无约束纯 PSO，
`output/qute_500nm_pure_pso_1000` 为 500 nm 纯 PSO；
`output/qute_500nm_regression` 为明确单列的 L-BFGS-B 数值回归。
中间阶段保留 duty、history 和 summary，便于检查与续跑。

3.12 µm / 60 fs 设计不能沿用此 QUTE 的固定 60 nm JSA 窗口和 0.5–8 nm
pump scan。调整波长/脉宽后，需同步在 `JSAConfig` 检查取样范围并做收敛
测试。`geometry` 中的带宽是初始设计值，`structures` 中的是最终优化值。

详细说明见 `PROGRAM_GUIDE_ZH.md` 和 `VALIDATION_QUTE_ZH.md`。
