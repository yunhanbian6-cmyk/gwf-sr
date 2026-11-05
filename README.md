# GWF Image Super-Resolution

基于 GWF（Geodesic Weighted Flow）方法论的图像超分辨率验证程序。项目实现了从概率分布几何、图曲率约束到连续梯度流求解器的完整方法论验证链路。

## 项目结构

```text
gwf_sr/                          # 核心库
├─ config.py                     # 配置管理
├─ synthetic.py                  # 合成灰度图像生成
├─ degradation.py                # 固定退化（高斯模糊+下采样+Bicubic）
├─ data.py                       # 数据生成与保存
├─ geometry.py                   # 概率分布、熵、KL、Hilbert 距离
├─ graph.py                      # 像素图、稀疏边表、归一化扩散
├─ energy.py                     # Dirichlet 能量、熵能量、总变差
├─ curvature.py                  # 熵产生、曲率商、曲率函数
├─ solver.py                     # 超分求解器（Euler/Strang/EM）
├─ metrics.py                    # PSNR、SSIM、LPIPS、高频指标
├─ visualization.py              # 收敛曲线、对比图、JSON/CSV 保存
└─ design/                       # 设计契约文档

scripts/                         # 实验脚本
├─ generate_data.py              # 合成数据生成
├─ run_demo.py                   # 端到端演示
├─ run_ablation.py               # λ 消融实验
└─ run_stability.py              # 稳定性实验

tests/                           # 单元测试（201 个）
configs/                         # 实验配置
```

## 环境准备

```powershell
python -m pip install torch numpy Pillow matplotlib PyYAML pytest
```

注意：如果 Anaconda 的 PyTorch DLL 损坏，使用系统 Python 或虚拟环境。

## 快速开始

### 1. 运行全部测试

```powershell
python -m pytest -q
```

### 2. 端到端超分演示

```powershell
python scripts/run_demo.py --config configs/smoke.yaml
```

生成 4 种合成图像，运行 Baseline（λ=0）和 GWF-C（λ>0），输出对比图、收敛曲线和指标表。

### 3. λ 消融实验

```powershell
python scripts/run_ablation.py --config configs/smoke.yaml
```

扫描 λ=0, 0.1, 1, 5, 10，输出每个 λ 的 PSNR/SSIM/能量/曲率。

### 4. 稳定性实验

```powershell
python scripts/run_stability.py --config configs/smoke.yaml
```

用多种初始化运行，记录 Hilbert 距离收缩曲线。

### 5. 生成合成数据

```powershell
python scripts/generate_data.py --config configs/smoke.yaml
```

## 配置参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `scale` | 超分倍率 | 2 |
| `seed` | 随机种子 | 42 |
| `image_size` | 图像边长 | 128 |
| `sigma` | 高斯模糊标准差 | 1.0 |
| `noise_std` | 噪声标准差 | 0.0 |
| `iterations` | 迭代次数 | 300 |
| `dt` | 时间步长 | 0.05 |
| `lambda_values` | λ 消融列表 | [0, 0.1, 1, 5, 10] |
| `epsilon` | 分布正则化常数 | 1e-8 |
| `device` | 运行设备 | auto |

## 方法论到代码映射

| GWF 论文概念 | 代码模块 | 具体接口 |
|---|---|---|
| 概率单纯形 | `geometry.py` | `distribution_from_image` |
| 熵 H(ρ) | `geometry.py` | `entropy` |
| KL 散度 | `geometry.py` | `kl_divergence` |
| Hilbert 射影距离 | `geometry.py` | `hilbert_distance` |
| 图结构 G=(V,E,w) | `graph.py` | `build_pixel_graph` |
| 归一化扩散算子 | `graph.py` | `normalized_adjacency_action` |
| Dirichlet 能量 | `energy.py` | `dirichlet_energy` |
| 熵能量 H_λ | `energy.py` | `entropy_energy` |
| 总变差 TV | `energy.py` | `total_variation` |
| 熵产生 I(ρ) | `curvature.py` | `entropy_production` |
| 曲率商 K_λ | `curvature.py` | `curvature_quotient` |
| 曲率函数 a(R) | `curvature.py` | `curvature_acceleration` |
| 数据保真梯度流 | `solver.py` | `data_fidelity_gradient` |
| 曲率正则梯度流 | `solver.py` | `regularization_gradient` |
| Euler 离散化 | `solver.py` | `SuperResolutionSolver(integrator="euler")` |
| Strang 分裂 | `solver.py` | `SuperResolutionSolver(integrator="strang")` |
| 图像投影 P_[0,1] | `solver.py` | `project_image` |
| PSNR | `metrics.py` | `psnr` |
| SSIM | `metrics.py` | `ssim` |
| LPIPS | `metrics.py` | `lpips_distance` |

## 数学定义

### 退化模型

y = D↓(Bσ(x))

### 数据保真梯度流

dx/dt = -A*(A(x) - y)

### Strang 分裂

每步执行：曲率半步 → 数据全步 → 曲率半步，每步投影到 [0,1]。

## 输出结构

```text
results/<config>/
├─ data/                          # HR/LR/Bicubic 图像 + manifest
└─ metrics/<sample>/<method>/     # 指标、对比图、收敛曲线、轨迹
```

## 测试

```powershell
python -m pytest -q
```

201 个测试覆盖配置、合成图像、退化、几何、图、能量、曲率、求解器、指标、可视化。

## 已知限制

1. CPU 版 PyTorch，GPU 加速待配置；
2. 首轮使用合成灰度图像，真实数据集扩展进行中；
3. LPIPS 为可选依赖，未安装时返回 None；
4. λ 需根据图像尺寸和迭代次数调整（小图建议 λ≤0.1）。
