# GWF Image Super-Resolution

这是一个基于老师给出的 GWF 方法论构建的图像超分辨率实验项目。第一阶段已完成项目骨架和配置体系，第二阶段已完成合成灰度图像与固定 ×2 退化流程。

## 当前进度

- [x] 建立 `gwf_sr` 核心包；
- [x] 建立 `scripts`、`tests`、`configs`、`data`、`results` 目录；
- [x] 定义统一配置模块；
- [x] 提供完整实验配置 `configs/default.yaml`；
- [x] 提供快速冒烟配置 `configs/smoke.yaml`；
- [x] 添加依赖清单 `requirements.txt`；
- [x] 实现图像数据和固定 ×2 退化；
- [ ] 实现熵、KL 和 Hilbert 距离；
- [ ] 实现曲率约束求解器；
- [ ] 实现图像评价和实验脚本；
- [ ] 接入真实图像数据集和神经 SR 模型。

## 项目结构

```text
E:\bishe\
├─ gwf_sr/
│  ├─ config.py
│  ├─ synthetic.py
│  ├─ degradation.py
│  └─ data.py
├─ scripts/
│  └─ generate_data.py
├─ tests/
│  ├─ test_synthetic.py
│  ├─ test_degradation.py
│  └─ test_data.py
├─ configs/
│  ├─ default.yaml
│  └─ smoke.yaml
├─ data/
└─ results/
```

## 环境准备

安装当前阶段所需的最小依赖：

```powershell
python -m pip install -r requirements-stage2.txt
```

完整实验依赖仍可使用：

```powershell
python -m pip install -r requirements.txt
```

当前机器的 Python 为 3.14.6，已经安装 CPU 版 PyTorch；如果需要使用 Arc B390 核显，应另行安装并验证 PyTorch XPU 版本。

## 配置说明

配置统一使用 YAML 文件，参数由 `gwf_sr.config.GWFConfig` 校验。当前支持：

| 参数 | 含义 |
|---|---|
| `scale` | 超分辨率倍率，首轮固定为 2 |
| `seed` | 随机种子 |
| `image_size` | 合成 HR 图像边长，必须能被 `scale` 整除 |
| `sigma` | 高斯退化核标准差 |
| `noise_std` | 退化噪声标准差 |
| `iterations` | 求解器迭代次数 |
| `dt` | 连续梯度流的离散步长 |
| `lambda_values` | 曲率约束强度消融列表 |
| `edge_weight_mode` | 图边权模式，当前支持 `uniform` |
| `epsilon` | 分布严格正化的小常数 |
| `output_dir` | 结果根目录 |
| `device` | `auto`、`cpu`、`cuda` 或 `mps` |

## 运行入口

生成配置中的合成数据集：

```powershell
python scripts\generate_data.py --config configs\smoke.yaml
```

生成结果包含：

- `*_hr.png`：高分辨率参考图像；
- `*_lr.png`：固定高斯模糊和 ×2 下采样后的低分辨率图像；
- `*_bicubic.png`：LR 图像的 Bicubic 上采样初始化；
- `manifest.json`：数据集和退化参数清单。

验证阶段二：

```powershell
python -m pytest tests
```

当前数据生成使用四种内置灰度图：

- `piecewise`：分块常数图；
- `edges`：直线边缘图；
- `texture`：周期纹理图；
- `smooth`：平滑低频图。

## 结果目录约定

每次运行使用独立目录：

```text
results/<config_name>/
├─ data/
├─ results/
├─ checkpoints/
└─ logs/
```

## 方法论对应

本项目将按照以下链路逐步实现：

1. 固定高斯模糊和 ×2 下采样；
2. Bicubic 初始化；
3. 熵梯度流基线；
4. 曲率约束 GWF-C；
5. λ 消融；
6. 步长收敛验证；
7. 不同初始化和扰动的稳定性验证；
8. 标准 SR 数据集扩展。

## 已知限制

第二阶段只完成合成数据、固定退化、Bicubic 初始化和数据保存，不包含 GWF 求解器、图像指标或实验图表。
