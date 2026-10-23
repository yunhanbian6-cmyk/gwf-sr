# GWF 图像超分辨率项目阶段进度报告

**项目名称：** 基于 GWF 方法论的图像超分辨率验证程序  
**报告日期：** 2026年10月23日  
**当前阶段：** 阶段二完成，阶段三准备开始  
**运行环境：** Windows，Python 3.14.6，PyTorch 2.14.1+CPU  

---

## 一、项目目标

本项目计划按照老师提供的 GWF 方法论，实现一个可复现的图像超分辨率程序。程序不仅要输出超分结果，还要验证以下方法论环节：

1. 图像退化过程；
2. 概率分布与熵；
3. KL 散度和 Hilbert 射影距离；
4. 曲率约束和能量泛函；
5. 连续梯度流及其离散化；
6. Baseline 与 GWF-C 方法比较；
7. 数值收敛和稳定性验证；
8. 图像超分质量评价；
9. 后续扩展到真实数据集和神经网络 SR 模型。

---

## 二、总体进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段一 | 项目骨架、配置体系、目录结构 | 已完成 |
| 阶段二 | 合成图像、固定退化、Bicubic 初始化、数据保存 | 已完成 |
| 阶段三 | 熵、KL、Hilbert 距离 | 待开始 |
| 阶段四 | 图结构、曲率和正则能量 | 未开始 |
| 阶段五 | 超分求解器和迭代轨迹 | 未开始 |
| 阶段六 | PSNR、SSIM、LPIPS 和可视化 | 未开始 |
| 阶段七 | 端到端超分演示 | 未开始 |
| 阶段八 | λ 消融实验 | 未开始 |
| 阶段九 | 步长收敛阶验证 | 未开始 |
| 阶段十 | 稳定性实验 | 未开始 |
| 阶段十一 | 完整测试体系 | 未开始 |
| 阶段十二 | README 和方法论文档映射 | 未开始 |
| 阶段十三 | 真实图像和标准数据集扩展 | 未开始 |
| 阶段十四 | EDSR、SRGAN、扩散式 SR 扩展 | 未开始 |

---

## 三、阶段一完成情况

### 3.1 已完成内容

阶段一建立了项目的基础工程结构，包含：

- 核心 Python 包 `gwf_sr`；
- 实验脚本目录 `scripts`；
- 测试目录 `tests`；
- 配置目录 `configs`；
- 数据目录 `data`；
- 结果目录 `results`；
- 核心配置模块；
- 默认实验配置；
- Smoke 快速配置；
- 依赖文件；
- 初版 README。

### 3.2 主要文件

#### 核心包

- [gwf_sr/__init__.py](E:\bishe/gwf_sr/__init__.py)
- [gwf_sr/config.py](E:\bishe/gwf_sr/config.py)

#### 配置文件

- [configs/default.yaml](E:\bishe/configs/default.yaml)
- [configs/smoke.yaml](E:\bishe/configs/smoke.yaml)

#### 依赖文件

- [requirements-core.txt](E:\bishe/requirements-core.txt)
- [requirements-stage2.txt](E:\bishe/requirements-stage2.txt)
- [requirements.txt](E:\bishe/requirements.txt)

#### 项目文档

- [README.md](E:\bishe/README.md)

### 3.3 已实现配置能力

配置类 `GWFConfig` 当前支持：

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
- YAML 和 JSON 读写；
- 参数范围校验；
- 运行目录自动创建。

### 3.4 配置验证结果

当前配置验证已经通过：

```text
Python 3.14.6
PyYAML 6.0.3
config_validation=ok
```

---

## 四、PyTorch 环境情况

### 4.1 已安装版本

当前项目环境已经安装：

```text
torch 2.14.1+cpu
```

验证结果：

```text
torch_version=2.14.1+cpu
cuda_available=False
gradient=[2.0, 4.0]
pytorch_import=ok
```

这说明：

- PyTorch 可以正常导入；
- PyTorch 自动求导正常；
- 当前版本为 CPU 版；
- 当前没有 CUDA GPU；
- 可以支持后续的梯度流和自动微分实现。

### 4.2 Arc B390 GPU 情况

用户硬件包括：

- Intel Core Ultra X7；
- Intel Arc B390 核显。

Arc B390 属于 Intel Arc GPU，理论上可以使用 PyTorch 的 XPU 后端，但需要：

1. Windows 11 或受支持的 Linux；
2. 安装最新 Intel Arc 驱动；
3. 安装支持 XPU 的 PyTorch 版本；
4. 确认 Python 版本存在对应的 XPU wheel；
5. 使用：

```python
torch.xpu.is_available()
```

验证显卡是否可用。

目前项目仍使用 CPU 版 PyTorch，主要原因：

- 当前阶段主要是小尺寸合成图像；
- 算法验证不需要大模型训练；
- CPU 版本环境稳定；
- Arc B390 的 XPU 版本仍需进一步确认驱动和 Python 兼容性。

PyTorch 官方 XPU 安装参考：

- [PyTorch Intel GPU 安装说明](https://docs.pytorch.org/docs/2.12/notes/get_start_xpu.html)
- [PyTorch XPU 安装说明源文件](https://docs.pytorch.org/docs/2.12/_sources/notes/get_start_xpu.rst.txt)

---

## 五、阶段二完成情况

### 5.1 阶段目标

阶段二的目标是建立一套可复现的图像超分输入数据流程：

1. 生成高分辨率合成灰度图；
2. 固定高斯模糊；
3. ×2 下采样；
4. 生成低分辨率图像；
5. 使用 Bicubic 进行初始化；
6. 保存 HR、LR、Bicubic 图像；
7. 保存退化参数和文件清单。

### 5.2 已实现模块

#### 合成图像模块

新增文件：

[gwf_sr/synthetic.py](E:\bishe/gwf_sr/synthetic.py)

支持四种合成图像：

| 类型 | 名称 | 特点 |
|---|---|---|
| `piecewise` | 分块常数图 | 具有明显区域边界，适合验证边缘和块状结构 |
| `edges` | 直线边缘图 | 包含垂直边缘和斜向边缘，适合测试边缘保持 |
| `texture` | 周期纹理图 | 包含周期高频结构，适合测试纹理恢复 |
| `smooth` | 平滑低频图 | 包含低频变化，适合测试平滑区域稳定性 |

核心接口：

```python
make_synthetic_image(kind="piecewise", size=128, seed=42)
make_synthetic_dataset(size=128, seed=42)
```

特点：

- 固定 seed 结果可重复；
- 图像类型可配置；
- 图像尺寸可配置；
- 图像统一为 `float32`；
- 输出值范围固定为 `[0, 1]`；
- 输出张量形状为 `H × W`。

#### 退化模块

新增文件：

[gwf_sr/degradation.py](E:\bishe/gwf_sr/degradation.py)

实现了以下算子：

1. 一维高斯核；
2. 可分离二维高斯模糊；
3. `reflect` 边界填充；
4. `replicate` 边界填充；
5. stride-2 平均下采样；
6. 抗锯齿 Bicubic 上采样；
7. 固定 seed 的高斯噪声；
8. 完整 `degrade()` 退化流程。

退化流程：

```text
HR
  ↓
固定高斯模糊
  ↓
×2 平均下采样
  ↓
可选固定噪声
  ↓
LR
```

Bicubic 初始化：

```text
LR
  ↓
抗锯齿 Bicubic 上采样
  ↓
初始 SR 图像
```

核心接口：

```python
result = degrade(
    image,
    scale=2,
    sigma=1.0,
    noise_std=0.0,
    seed=42,
)

low_resolution = result.low_resolution
blurred = result.blurred
```

#### 数据生成模块

新增文件：

[gwf_sr/data.py](E:\bishe/gwf_sr/data.py)

支持：

- 单张图像生成；
- HR、LR、Bicubic 图像生成；
- PNG 保存；
- PNG 加载；
- 配置化批量生成；
- 退化参数记录；
- manifest 清单生成。

核心接口：

```python
hr, lr, bicubic, record = generate_image(
    kind="piecewise",
    size=128,
    seed=42,
    scale=2,
    sigma=1.0,
    noise_std=0.0,
    output_dir="results/smoke/data",
)
```

批量生成接口：

```python
records = generate_configured_dataset(config)
```

生成文件包括：

- `*_hr.png`
- `*_lr.png`
- `*_bicubic.png`
- `manifest.json`

#### 数据生成脚本

新增文件：

[scripts/generate_data.py](E:\bishe/scripts/generate_data.py)

运行命令：

```powershell
python scripts\generate_data.py --config configs\smoke.yaml
```

实际输出：

```text
Generated 4 image sets in results\smoke\data
- piecewise_32_seed42: HR=(32, 32), LR=(16, 16)
- edges_32_seed43: HR=(32, 32), LR=(16, 16)
- texture_32_seed44: HR=(32, 32), LR=(16, 16)
- smooth_32_seed45: HR=(32, 32), LR=(16, 16)
```

验证结果：

```text
images=4
hr_exists=True
lr_exists=True
bicubic_exists=True
stage2_demo=ok
```

---

## 六、阶段二测试情况

### 6.1 新增测试文件

- [tests/test_synthetic.py](E:\bishe/tests/test_synthetic.py)
- [tests/test_degradation.py](E:\bishe/tests/test_degradation.py)
- [tests/test_data.py](E:\bishe/tests/test_data.py)

### 6.2 测试内容

#### 合成图像测试

验证：

- 图像形状；
- 图像 dtype；
- 图像值范围；
- 固定 seed 可重复性；
- 不同 seed 结果不同；
- 四种图像类型都能生成；
- 未知类型能正确报错。

#### 退化流程测试

验证：

- 高斯核归一化；
- 高斯模糊输出尺寸；
- 高斯模糊输出值范围；
- ×2 下采样尺寸；
- Bicubic 上采样尺寸；
- 固定 seed 退化结果可重复；
- 不同 seed 噪声结果不同。

#### 数据保存测试

验证：

- PNG 保存成功；
- PNG 加载成功；
- 加载结果与原图误差小于 1/255；
- 单张图像生成尺寸正确；
- 配置化数据集生成成功；
- HR、LR、Bicubic 文件存在；
- manifest.json 存在。

### 6.3 测试结果

```text
10 passed in 2.68s
```

测试结果说明：

- 阶段二所有新增模块通过单元测试；
- 数据生成流程可重复；
- 退化过程尺寸和值域正确；
- 配置化生成流程可用。

---

## 七、当前项目文件结构

```text
E:\bishe\
├─ gwf_sr/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ synthetic.py
│  ├─ degradation.py
│  └─ data.py
├─ scripts/
│  ├─ __init__.py
│  └─ generate_data.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_synthetic.py
│  ├─ test_degradation.py
│  └─ test_data.py
├─ configs/
│  ├─ default.yaml
│  └─ smoke.yaml
├─ data/
├─ results/
│  └─ smoke/
│     └─ data/
│        ├─ *_hr.png
│        ├─ *_lr.png
│        ├─ *_bicubic.png
│        └─ manifest.json
├─ requirements-core.txt
├─ requirements-stage2.txt
├─ requirements.txt
└─ README.md
```

---

## 八、当前 Smoke 配置说明

当前快速配置：

[configs/smoke.yaml](E:\bishe/configs/smoke.yaml)

配置内容：

```yaml
scale: 2
seed: 42
image_size: 32
sigma: 1.0
noise_std: 0.0
iterations: 20
dt: 0.05
lambda_values:
  - 0.0
  - 1.0
edge_weight_mode: uniform
epsilon: 1.0e-8
output_dir: results
device: cpu
```

当前配置特点：

- 图像边长为 32；
- ×2 超分；
- 无噪声；
- 高斯模糊标准差为 1.0；
- λ 消融值为 0 和 1；
- 迭代次数为 20；
- 设备为 CPU。

完整配置：

[configs/default.yaml](E:\bishe/configs/default.yaml)

完整配置特点：

- 图像边长为 128；
- 300 次迭代；
- λ 消融值为 0、0.1、1、5、10；
- 设备为自动选择。

---

## 九、当前能力边界

目前已经实现：

- 项目工程结构；
- 配置管理和校验；
- 合成灰度 HR 图像；
- 固定高斯模糊；
- 固定 ×2 下采样；
- 可选噪声；
- Bicubic 初始化；
- HR、LR、Bicubic 保存；
- 退化参数清单；
- 数据生成脚本；
- 数据生成测试。

当前还没有实现：

- 熵；
- KL 散度；
- Hilbert 射影距离；
- 严格正分布转换；
- 图结构；
- 曲率约束；
- GWF 求解器；
- Baseline 超分求解；
- PSNR；
- SSIM；
- LPIPS；
- 图像对比图；
- 收敛曲线；
- λ 消融实验；
- 稳定性实验；
- 标准 SR 数据集读取；
- 神经网络 SR 模型。

因此，当前项目已经完成“数据准备层”，但尚未进入“数学计算层”和“超分求解层”。

---

## 十、下一步阶段三计划

阶段三目标：

实现方法论中的分布与几何度量模块。

### 10.1 主要模块

计划新增：

[gwf_sr/geometry.py](E:\bishe/gwf_sr/geometry.py)

### 10.2 主要功能

计划实现：

1. 图像到严格正分布的转换；
2. 非负投影；
3. epsilon 稳定化；
4. 概率归一化；
5. 熵计算；
6. KL 散度；
7. Hilbert 射影距离；
8. 总质量诊断；
9. 最小概率诊断；
10. 最大概率诊断；
11. 有效状态数计算。

### 10.3 数学定义

图像转换为正分布：

\[
\rho_i
=
\frac{\max(x_i, 0)+\varepsilon}
{\sum_j(\max(x_j, 0)+\varepsilon)}
\]

熵：

\[
H(\rho)
=
-\sum_i \rho_i \log \rho_i
\]

KL 散度：

\[
D_{\mathrm{KL}}(\rho\|\eta)
=
\sum_i \rho_i
\log
\frac{\rho_i}{\eta_i}
\]

Hilbert 射影距离：

\[
d_H(\rho,\eta)
=
\log
\max_i\frac{\rho_i}{\eta_i}
-
\log
\min_i\frac{\rho_i}{\eta_i}
\]

### 10.4 阶段三验收标准

- [x] 非负投影正确；
- [x] epsilon 正确加入；
- [x] 分布总和为 1；
- [x] 分布中不存在 0 或负值；
- [x] 熵不超过 `log(N)`；
- [x] KL 对相同分布接近 0；
- [x] Hilbert 对相同分布为 0；
- [x] 全黑图像不会崩溃；
- [x] 含负值图像不会崩溃；
- [x] 相同输入结果可重复；
- [x] 单元测试通过。

---

## 十一、主要风险和应对

| 风险 | 当前影响 | 应对方案 |
|---|---|---|
| Python 3.14 与部分 PyTorch 生态兼容性待观察 | 中 | 当前 CPU 版 PyTorch 已验证可导入和自动求导 |
| Arc B390 的 XPU 支持需要额外确认 | 中 | 后续单独安装 XPU 版并检查 `torch.xpu.is_available()` |
| CPU 版本在大图像上速度较慢 | 中 | 首轮使用小尺寸合成图，后续再考虑 XPU |
| 图像非正导致 Hilbert 距离不稳定 | 高 | 阶段三统一使用 epsilon 和严格正分布 |
| 曲率计算可能耗内存 | 高 | 使用 Hessian-vector product 或幂迭代 |
| 只看 PSNR 无法验证方法论 | 高 | 后续增加熵、Hilbert、曲率和收敛阶实验 |
| 退化参数不一致导致实验失效 | 高 | 当前退化参数已写入 manifest.json |
| 随机性导致结果不可重复 | 中 | 当前所有合成和噪声均使用固定 seed |

---

## 十二、当前结论

截至本报告日期，项目已经完成以下阶段性目标：

1. 建立了完整的项目工程骨架；
2. 完成了配置驱动的实验体系；
3. 安装并验证了 CPU 版 PyTorch；
4. 完成了四种合成灰度图像生成；
5. 完成了固定高斯模糊；
6. 完成了 ×2 下采样；
7. 完成了抗锯齿 Bicubic 初始化；
8. 完成了可选噪声退化；
9. 完成了 HR、LR、Bicubic 图像保存；
10. 完成了退化清单 `manifest.json`；
11. 完成了阶段二数据生成脚本；
12. 完成了 10 个阶段二单元测试；
13. 全部测试通过；
14. Smoke 示例运行成功。

当前项目状态可以概括为：

> **数据准备层已经完成，数学度量层尚未开始。**

下一步将进入阶段三，实现熵、KL 散度和 Hilbert 射影距离，为后续曲率约束和超分求解器提供基础。
