# GWF 图像超分辨率项目进度报告（第二版）

**项目名称：** 基于 GWF 方法论的图像超分辨率验证程序  
**报告日期：** 2026年10月24日  
**当前阶段：** 阶段四完成，阶段五准备开始  
**运行环境：** Windows，Python 3.14.6，PyTorch 2.14.1+CPU  
**Git 仓库：** `git@github.com:yunhanbian6-cmyk/gwf-sr.git`  
**最新提交：** `5e5e83f4aef542dac8fda195a305039a9e61700f`

---

## 一、项目目标

本项目按照老师提供的 GWF 方法论，实现一个可复现的图像超分辨率程序。程序不仅要输出超分结果，还要验证以下方法论环节：

1. 图像退化过程；
2. 概率分布与熵；
3. KL 散度和 Hilbert 射影距离；
4. 图结构、曲率约束和能量泛函；
5. 连续梯度流及其离散化；
6. Baseline 与 GWF-C 方法比较；
7. 数值收敛和稳定性验证；
8. 图像超分质量评价；
9. 后续扩展到真实数据集和神经网络 SR 模型。

---

## 二、总体进度

| 阶段 | 内容 | 状态 | Commit |
|---|---|---|---|
| 阶段一 | 项目骨架、配置体系、目录结构 | 已完成 | `a052482` |
| 阶段二 | 合成图像、固定退化、Bicubic 初始化、数据保存 | 已完成并验收 PASS | `a052482` → `2a5e617` → `8792678` |
| 阶段三 | 熵、KL、Hilbert 距离、严格正分布 | 已完成并验收 PASS | `67a7ce7` → `3931b2a` |
| 阶段四 | 图结构、曲率、正则能量 | 已完成并验收 PASS | `be70a06` → `321be32` → `5e5e83f` |
| 阶段五 | 超分求解器和迭代轨迹 | 待开始 | — |
| 阶段六 | PSNR、SSIM、LPIPS 和可视化 | 未开始 | — |
| 阶段七 | 端到端超分演示 | 未开始 | — |
| 阶段八 | λ 消融实验 | 未开始 | — |
| 阶段九 | 步长收敛阶验证 | 未开始 | — |
| 阶段十 | 稳定性实验 | 未开始 | — |
| 阶段十一 | 完整测试体系 | 未开始 | — |
| 阶段十二 | README 和方法论文档映射 | 未开始 | — |
| 阶段十三 | 真实图像和标准数据集扩展 | 未开始 | — |
| 阶段十四 | EDSR、SRGAN、扩散式 SR 扩展 | 未开始 | — |

---

## 三、Git 提交历史

```text
5e5e83f  fix(graph): correct vertical edge raster order and update gitignore
321be32  feat(graph,energy,curvature): implement frozen stage-four operators
be70a06  stage4-contract: add graph curvature and energy interface contract
3931b2a  feat(geometry): implement frozen stage-three probability geometry
8792678  fix(data): normalize manifest image paths
2a5e617  fix(synthetic): handle small piecewise sizes
a052482  stage2: freeze project skeleton and reproducible degradation baseline
```

阶段二基线标签：

```text
stage2-baseline → a0524824c2ed4adf1db3dd9ead7e97190e672bbe
```

---

## 四、阶段一完成情况

### 4.1 已完成内容

- 核心 Python 包 `gwf_sr`；
- 配置模块 `gwf_sr/config.py`；
- 实验脚本目录 `scripts`；
- 测试目录 `tests`；
- 配置目录 `configs`；
- 数据目录 `data`；
- 结果目录 `results`；
- 默认配置 `configs/default.yaml`；
- Smoke 配置 `configs/smoke.yaml`；
- 依赖文件 `requirements-core.txt`、`requirements-stage2.txt`、`requirements.txt`；
- README。

### 4.2 配置能力

`GWFConfig` 支持：

- 超分倍率；
- 随机种子；
- 合成图像尺寸；
- 高斯模糊标准差；
- 噪声标准差；
- 求解器迭代次数；
- 时间步长；
- λ 参数列表；
- 边权模式；
- epsilon；
- 输出目录；
- 运行设备；
- YAML/JSON 读写；
- 参数合法性校验；
- 运行目录自动创建。

---

## 五、阶段二完成情况

### 5.1 已实现模块

| 模块 | 文件 | 功能 |
|---|---|---|
| 合成图像 | `gwf_sr/synthetic.py` | 四种可复现灰度图：`piecewise`、`edges`、`texture`、`smooth` |
| 退化流程 | `gwf_sr/degradation.py` | 高斯模糊、×2 下采样、Bicubic 上采样、可选噪声 |
| 数据生成 | `gwf_sr/data.py` | HR/LR/Bicubic 生成、PNG 保存、manifest 清单 |
| 生成脚本 | `scripts/generate_data.py` | 配置驱动批量数据生成 |

### 5.2 退化流程

```text
HR → 固定高斯模糊 → ×2 平均下采样 → 可选固定噪声 → LR
LR → 抗锯齿 Bicubic 上采样 → 初始 SR 图像
```

### 5.3 输出文件

每个样本生成：

- `*_hr.png`：高分辨率参考；
- `*_lr.png`：低分辨率输入；
- `*_bicubic.png`：Bicubic 初始化；
- `manifest.json`：退化参数和文件清单。

### 5.4 阶段二缺陷修复

| 缺陷 | Commit | 修复内容 |
|---|---|---|
| `size=4` IndexError | `2a5e617` | 小尺寸分块 stride 改为 `max(1, size//4)` 和 `max(1, size//3)` |
| manifest 绝对路径 | `8792678` | 路径改为 data 目录相对文件名，跨 root 字节一致 |

### 5.5 阶段二验收结果

Verifier 独立验收（V0）：

```text
功能正确性：PASS
边界与异常：PASS（有限制：size=4 缺陷已修复）
可复现性：PASS（manifest 路径限制已修复）
文档与映射：PASS
总体：PASS
```

---

## 六、阶段三完成情况

### 6.1 设计契约

文件：`gwf_sr/design/geometry_contract.md`  
Commit：`67a7ce7`

冻结内容：

- 输入形状 `H×W` 或 `B×H×W`；
- 仅允许 `float32`/`float64`；
- 同 device 输出；
- NaN/Inf 抛 `ValueError`；
- 负值截断为 0；
- epsilon 严格正化；
- 全黑输入输出均匀分布；
- 熵 \(H=-\sum\rho\log\rho\)；
- KL \(D_{\mathrm{KL}}(\rho\|\eta)=\sum\rho\log(\rho/\eta)\)；
- Hilbert \(d_H=\log\max(\rho/\eta)-\log\min(\rho/\eta)\)；
- 诊断量：总质量、最小/最大概率、有效状态数 \(N_{\mathrm{eff}}=\exp(H)\)；
- \(O(N)\) 复杂度，禁止 \(N\times N\) 矩阵。

### 6.2 实现

文件：`gwf_sr/geometry.py`  
Commit：`3931b2a`

接口：

```python
distribution_from_image(image, epsilon=1e-8) -> Tensor
entropy(distribution) -> Tensor
kl_divergence(rho, eta) -> Tensor
hilbert_distance(rho, eta) -> Tensor
total_mass(distribution) -> Tensor
min_probability(distribution) -> Tensor
max_probability(distribution) -> Tensor
effective_state_count(distribution) -> Tensor
```

### 6.3 阶段三验收结果

Verifier 独立验收：

```text
功能正确性：PASS
边界与异常：PASS
复杂度与性能：PASS
可复现性：PASS
总体：PASS
```

报告 SHA256：

```text
7a50b477544894d1a7f0a055a69d983b75b63ae42c91c217fdafc28af08c0684
```

pytest：`61 passed`

---

## 七、阶段四完成情况

### 7.1 设计契约

文件：`gwf_sr/design/graph_curvature_energy_contract.md`  
Commit：`be70a06`

冻结内容：

- 像素网格图，节点索引 `index=r*W+c`；
- 稀疏边表，directed incidence `E=2*edge_count`；
- 边序：水平 forward → 垂直 forward → 水平 reverse → 垂直 reverse；
- `uniform` 边权模式；
- 对称归一化扩散算子；
- Dirichlet 能量 \(E_D(f)=\frac12\sum_{\mathrm{directed}}w(f_u-f_v)^2\)；
- 熵能量 \(H_\lambda(\rho)=H(\rho)+\frac{\lambda}{2}E_D(\log\rho)\)；
- 总变差 \(TV(f)=\frac12\sum_{\mathrm{directed}}w|f_u-f_v|\)；
- 熵产生 \(I(\rho)=\sum\widehat w\left(\frac{\rho_u-\rho_v}{\sqrt{\rho_u\rho_v}}\right)^2\)；
- 曲率商 \(K_\lambda=\lambda I/H\)；
- 曲率函数 \(a(R)=R^2e^R/(e^R-1-R)\)，\(a(0)=2\)；
- 衰减率诊断；
- \(O(N)\) 复杂度，禁止 \(N\times N\) 矩阵。

### 7.2 实现

| 文件 | 功能 |
|---|---|
| `gwf_sr/graph.py` | `PixelGraph`、`PixelGraphBatch`、`build_pixel_graph`、`build_batch_pixel_graphs`、`normalized_adjacency_action`、`graph_degree`、`graph_total_weight`、`graph_spectral_radius_bound` |
| `gwf_sr/energy.py` | `dirichlet_energy`、`entropy_energy`、`total_variation`、`log_ratio_smoothness` |
| `gwf_sr/curvature.py` | `entropy_production`、`curvature_quotient`、`curvature_acceleration`、`entropy_decay_rate` |

Commit：`321be32` → `5e5e83f`

### 7.3 阶段四验收结果

Verifier 独立验收：

```text
图结构与邻接关系：PASS
曲率计算及边界：PASS
能量与正则项数学定义：PASS
O(N) 复杂度与可复现性：PASS
总体：PASS
```

报告 SHA256：

```text
4541e0a9adce13c9c621319ac8525ede28a7b92cc4d1628278476aa7956c373c
```

pytest：`119 passed`

---

## 八、当前项目文件结构

```text
E:\bishe\
├─ gwf_sr/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ synthetic.py
│  ├─ degradation.py
│  ├─ data.py
│  ├─ geometry.py
│  ├─ graph.py
│  ├─ energy.py
│  ├─ curvature.py
│  └─ design/
│     ├─ geometry_contract.md
│     └─ graph_curvature_energy_contract.md
├─ scripts/
│  └─ generate_data.py
├─ tests/
│  ├─ test_synthetic.py
│  ├─ test_degradation.py
│  ├─ test_data.py
│  ├─ test_geometry.py
│  ├─ test_graph.py
│  ├─ test_energy.py
│  └─ test_curvature.py
├─ configs/
│  ├─ default.yaml
│  └─ smoke.yaml
├─ data/
├─ results/
├─ verify_stage2/
├─ verify_stage3/
├─ verify_stage4/
├─ requirements-core.txt
├─ requirements-stage2.txt
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## 九、测试汇总

| 阶段 | 测试文件 | 测试数 | 结果 |
|---|---|---|---|
| 阶段二 | `test_synthetic.py`、`test_degradation.py`、`test_data.py` | 20 | 全部通过 |
| 阶段三 | `test_geometry.py` | 61 | 全部通过 |
| 阶段四 | `test_graph.py`、`test_energy.py`、`test_curvature.py` | 119 | 全部通过 |

当前全套测试：

```text
119 passed in 2.53s
```

---

## 十、独立验收汇总

| 阶段 | 验收维度 | 结论 | 报告 SHA256 |
|---|---|---|---|
| 阶段二 | 功能/边界/可复现/文档 | PASS | — |
| 阶段三 | 功能/边界/复杂度/可复现 | PASS | `7a50b477...` |
| 阶段四 | 图结构/曲率/能量/复杂度 | PASS | `4541e0a9...` |

---

## 十一、已知限制

1. 当前仍为 CPU 版 PyTorch，Arc B390 XPU 尚未启用；
2. `gradient` 边权模式尚未实现（契约预留）；
3. 尚未实现超分求解器；
4. 尚未实现 PSNR、SSIM、LPIPS；
5. 尚未接入真实数据集；
6. 尚未接入神经网络 SR 模型；
7. 阶段五~十四全部未开始。

---

## 十二、下一步：阶段五

阶段五目标：实现超分求解器和迭代轨迹。

计划内容：

1. 数据保真梯度流；
2. 曲率正则梯度流；
3. Euler 更新；
4. Strang 分裂更新；
5. 图像值域投影；
6. 迭代轨迹记录；
7. 统一 `SuperResolutionSolver` 接口。

验收标准：

- [ ] 能从 LR 图像得到同尺寸 SR 图像；
- [ ] λ=0 和 λ>0 可比较；
- [ ] 每步记录残差、熵、Hilbert、曲率；
- [ ] 步长过大时能检测并报告；
- [ ] 求解器接口不依赖具体实验脚本；
- [ ] 全套测试通过；
- [ ] verifier 独立验收 PASS。

---

## 十三、团队协作记录

| 角色 | 职责 | 状态 |
|---|---|---|
| Lead | 冻结基线、调度 DAG、执行门禁、最终汇总 | 活跃 |
| architect | 设计契约（geometry、graph/energy/curvature） | 已完成两份契约 |
| developer | 实现核心模块和测试 | 已完成阶段三、四 |
| verifier | 独立验收 | 已完成阶段二、三、四验收 |

---

## 十四、结论

截至本报告日期，项目已完成四个阶段：

1. 阶段一建立了完整的项目工程骨架和配置体系；
2. 阶段二实现了合成图像、固定退化和数据保存，并通过独立验收；
3. 阶段三实现了概率分布、熵、KL、Hilbert 距离，并通过独立验收；
4. 阶段四实现了图结构、曲率、正则能量，并通过独立验收。

当前项目状态可以概括为：

> **理论基础层已完成，求解器层尚未开始。**

下一步将进入阶段五，实现超分求解器和迭代轨迹，为后续图像质量评价和端到端演示奠定基础。
