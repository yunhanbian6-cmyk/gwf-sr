# 超分求解器与迭代轨迹契约（阶段五设计草案）

**状态：等待用户确认后再实现；本文件只冻结接口、数值语义和验收边界，不包含核心实现代码。**  
**实现目标模块：`gwf_sr/solver.py`。**  
**依赖边界：复用阶段二固定退化/双三次初始化、阶段三 `geometry`、阶段四 `graph`/`energy`/`curvature`；不修改这些已通过模块。**

---

## 1. 阶段五目的与范围

阶段五将固定退化得到的低分辨率图像转换为超分迭代求解器，并冻结以下链路：

1. 把候选超分图像转换为严格正概率分布；
2. 计算低分辨率观测与候选图像降采样结果之间的数据保真项；
3. 计算数据保真梯度流；
4. 计算曲率正则梯度流；
5. 以 Euler、Euler–Maruyama、Strang 分裂等方式离散时间；
6. 每步把图像投影到 `[0,1]`；
7. 记录完整迭代轨迹和诊断量；
8. 统一 `SuperResolutionSolver` 接口；
9. 保证输出 shape、dtype、device、步长校验和复杂度满足冻结要求。

阶段五不负责：

- 生成 HR/LR/Bicubic 数据文件；
- 评价 PSNR、SSIM、LPIPS；
- 构造图结构或计算曲率商；
- 读取 YAML/JSON 配置文件；
- 修改阶段二、三、四已有模块。

---

## 2. 术语与张量约定

### 2.1 基本符号

- `N_lr = H_lr * W_lr`：低分辨率观测的状态数；
- `N = H * W`：候选超分图像的状态数；
- `s`：超分倍率，当前配置固定为 2；
- `y`：低分辨率观测张量，shape 为 `H_lr × W_lr` 或 `B × H_lr × W_lr`；
- `x`：候选超分图像，shape 为 `H × W` 或 `B × H × W`；
- `rho`：由 `x` 得到的严格正概率分布；
- `A`：固定退化算子的前向近似，即高斯模糊后 stride-2 平均下采样；
- `A*`：与 `A` 同实现的伴随算子，用于数据项梯度；
- `lambda_value`：曲率正则强度，非负；
- `dt`：离散时间步长，正有限标量；
- `k`：从 0 开始的迭代索引；
- `P_[0,1]`：逐像素投影到闭区间 `[0,1]`。

### 2.2 输入形状

阶段五公开求解器函数只接受：

1. `y`：`H_lr × W_lr` 或 `B × H_lr × W_lr`；
2. `x0`：`H × W` 或 `B × H × W`；
3. 批量输入时，`B` 必须完全一致；
4. 空 batch、空图像和 1-D/4-D 张量抛出 `ValueError`；
5. 非 tensor 输入抛出 `TypeError`。

对 `y`：

\[
H = sH_{lr}, \qquad W = sW_{lr}.
\]

若该条件不成立，抛出 `ValueError`。

### 2.3 dtype 与 device

支持 dtype：

1. `torch.float32`；
2. `torch.float64`。

规则：

1. 输入、内部轨迹、最终输出和标量诊断都保持同一 dtype；
2. 不允许静默降精度；
3. `float16`、`bfloat16`、整数、布尔、复数输入抛出 `TypeError`；
4. 所有输出保持 `x0` 的 device；
5. 求解器不自动把输入搬到 CPU 或其他设备；
6. `GWFConfig.device` 只能由实验层预先移动输入，求解器本身服从 tensor device；
7. 图、退化算子和诊断量的 device 必须与 `x0`、`y` 一致，否则抛出 `ValueError`。

### 2.4 有限值与图像负值

1. `y` 和 `x0` 必须有限；NaN/Inf 抛出 `ValueError`；
2. `y` 可以包含任意有限值，包括阶段二噪声导致的轻微越界值；
3. `x0` 允许包含负值或大于 1 的有限值，求解器在进入迭代前立即执行 `P_[0,1]`；
4. 求解器内部不允许 NaN/Inf 出现在轨迹或诊断量中；
5. 任何无效输入都不能静默修复为有限值。

---

## 3. 固定退化算子与数据保真

### 3.1 前向算子

对候选图像 `x`，冻结前向近似为：

\[
A(x) = D_s(B_\sigma(x)),
\]

其中：

1. `B_sigma` 为阶段二 `gaussian_blur`，使用 `reflect` 边界；
2. `D_s` 为阶段二 stride-2 平均下采样；
3. `s = 2`；
4. `sigma` 为阶段二配置中的模糊标准差；
5. `A` 不包含噪声；
6. `A` 不修改输入，输出 shape 为 `y.shape`；
7. `A` 保持 dtype 和 device。

### 3.2 伴随算子

由于 `B_sigma` 是线性可分离卷积，`D_s` 是线性平均下采样，冻结伴随实现为：

\[
A^*(z) = B_\sigma^*(D_s^*(z)).
\]

其中：

1. `D_s^*` 是 stride-2 平均下采样的伴随；
2. `B_sigma^*` 是使用 `replicate` 边界的可分离高斯卷积伴随；
3. `A*` 必须满足离散内积关系：

\[
\langle A(x), z\rangle
=
\langle x, A^*(z)\rangle,
\]

在 `float32` 容差 `1e-5`、`float64` 容差 `1e-12` 内成立；
4. `A*` 输入 `z` 的 shape 必须为 `y` 的 shape；
5. `A*` 输出 shape 必须为 `x0` 的 shape；
6. `A*` 必须使用显式稀疏/填充/转置操作实现；禁止使用未声明的近似算子；
7. 若实现不能证明伴随关系，禁止用该梯度进入求解器。

### 3.3 数据保真项

数据保真项定义为：

\[
\mathcal{L}_{data}(x)
=
\frac{1}{2}
\left\|
A(x)-y
\right\|_2^2.
\]

等价展开为：

\[
\mathcal{L}_{data}(x)
=
\frac{1}{2}
\sum_i
\left(
A(x)_i-y_i
\right)^2.
\]

要求：

1. 不除以 `N_lr`；
2. 不加入额外权重；
3. 不把数据项转为 KL 散度；
4. 不对 `y` 做概率归一化；
5. 对 batch 返回 shape 为 `(B,)`；
6. 对单图返回 0-D tensor；
7. `lambda_value` 不影响数据项。

数据保真梯度为：

\[
\nabla_x \mathcal{L}_{data}(x)
=
A^*(A(x)-y).
\]

该梯度必须通过 `A*` 计算，不允许对 `A` 做有限差分替代。

---

## 4. 概率分布与曲率正则

### 4.1 分布入口

每次需要概率几何量时，统一调用阶段三入口：

\[
\rho
=
\operatorname{distribution\_from\_image}(x,\epsilon).
\]

规则：

1. `epsilon` 必须为有限正标量；
2. `epsilon` 使用 `GWFConfig.epsilon` 或显式传入值；
3. `x` 在进入分布入口前先投影到 `[0,1]`；
4. `rho` 严格正且按每个样本独立归一化；
5. 不允许求解器自行实现第二套归一化公式。

### 4.2 曲率正则项

曲率正则能量采用阶段四冻结定义：

\[
\mathcal{R}_\lambda(x)
=
\lambda_{\mathrm{value}}
\left[
-\sum_u \rho_u \log \rho_u
+
\frac{\lambda_{\mathrm{value}}}{2}
\sum_{(u,v)}
\widehat w_{uv}
(\log\rho_u-\log\rho_v)^2
\right].
\]

其中：

1. `rho = distribution_from_image(x, epsilon)`；
2. `graph` 是与 `x` shape 匹配的阶段四 `PixelGraph` 或 `PixelGraphBatch`；
3. `edge_weight_mode` 使用 `GWFConfig.edge_weight_mode`；
4. `lambda_value >= 0`；
5. 曲率正则项不对 `rho` 再次归一化；
6. `lambda_value=0` 时，正则项严格为零；
7. 曲率正则梯度采用自动微分获得，但不得构造 `N×N` Hessian。

### 4.3 曲率正则梯度

定义：

\[
g_{reg}(x)
=
\nabla_x \mathcal{R}_\lambda(x).
\]

要求：

1. 自动微分输入是当前投影后的候选图像 `x`；
2. 梯度 shape 必须与 `x` 完全相同；
3. 梯度 dtype 和 device 必须与 `x` 一致；
4. 梯度必须有限；
5. 曲率正则梯度不能包含数据保真项；
6. 曲率正则梯度不能对 `rho` 使用硬 argmax、离散采样或随机估计；
7. 若使用 Hessian-vector product，只能用于未来分析，不得改变梯度定义。

---

## 5. 数据保真梯度流

### 5.1 连续流

数据保真梯度流定义为：

\[
\frac{dx}{dt}
=
-\nabla_x \mathcal{L}_{data}(x)
=
-A^*(A(x)-y).
\]

该流只包含数据项，不包含曲率正则项。

初始条件：

\[
x(0)=P_{[0,1]}(x_0).
\]

### 5.2 离散 Euler 更新

冻结 Euler 更新为：

\[
x_{k+1}
=
P_{[0,1]}
\left(
x_k
-
dt\,
A^*(A(x_k)-y)
\right).
\]

其中 `k = 0, ..., iterations-1`。

要求：

1. `x_0` 先投影到 `[0,1]`；
2. 每个子步都立即投影；
3. 不在数据流中调用分布入口；
4. 不加入 `lambda_value`；
5. 不加入随机噪声；
6. 不使用自适应步长。

### 5.3 Euler–Maruyama 数据流

可选的随机基线为：

\[
x_{k+1}
=
P_{[0,1]}
\left(
x_k
-
dt\,
A^*(A(x_k)-y)
+
\sigma_{noise}\sqrt{dt}\,\xi_k
\right),
\]

其中：

1. `sigma_noise` 即求解器配置中的 `noise_std`；
2. `noise_std >= 0`；
3. `xi_k` 为标准高斯噪声；
4. 噪声 shape 与 `x_k` 相同；
5. `noise_std=0` 时必须与 Euler 结果逐元素一致；
6. 随机模式必须使用显式 seed；
7. 随机模式不能改变确定性模式的可重复性契约。

---

## 6. 曲率正则梯度流

### 6.1 连续流

曲率正则梯度流定义为：

\[
\frac{dx}{dt}
=
-g_{reg}(x).
\]

初始条件：

\[
x(0)=P_{[0,1]}(x_0).
\]

该流不包含数据保真项。

### 6.2 离散更新

曲率流 Euler 更新为：

\[
x_{k+1}
=
P_{[0,1]}
\left(
x_k
-
dt\,
g_{reg}(x_k)
\right).
\]

要求：

1. 每步重新计算分布入口和曲率正则梯度；
2. 每步投影到 `[0,1]`；
3. 不加入数据项；
4. 不加入噪声；
5. `lambda_value=0` 时，`g_reg=0`，所以 `x_{k+1}=P_[0,1](x_k)`；
6. `lambda_value>0` 时，不保证能量单调下降，但必须记录能量、梯度范数和图像变化量。

---

## 7. Strang 分裂

### 7.1 分裂定义

完整 GWF-C 求解器在每个离散步内使用以下顺序：

\[
x^{(0)}
=
P_{[0,1]}(x_k);
\]

曲率半步：

\[
x^{(1)}
=
P_{[0,1]}
\left(
x^{(0)}
-
\frac{dt}{2}
g_{reg}(x^{(0)})
\right);
\]

数据全步：

\[
x^{(2)}
=
P_{[0,1]}
\left(
x^{(1)}
-
dt\,
A^*(A(x^{(1)})-y)
\right);
\]

曲率半步：

\[
x_{k+1}
=
P_{[0,1]}
\left(
x^{(2)}
-
\frac{dt}{2}
g_{reg}(x^{(2)})
\right).
\]

### 7.2 分裂要求

1. Strang 分裂只用于 `lambda_value` 可能为零或正数的统一求解器；
2. `lambda_value=0` 时，Strang 结果必须等同于纯数据 Euler 更新，允许浮点容差；
3. 每个半步和全步后都执行图像投影；
4. 不允许把两个曲率半步合并成一个不投影的全步；
5. 不允许把数据项和曲率项在同一个梯度表达式中隐式相加后只投影一次；
6. `dt` 在整个求解过程中固定；
7. 迭代次数固定为 `iterations`；
8. 轨迹记录点必须是完整 Strang 步之后的状态，而不是半步中间状态。

---

## 8. 图像值域投影

### 8.1 投影定义

冻结投影为：

\[
P_{[0,1]}(x)
=
\min(\max(x,0),1).
\]

规则：

1. 逐像素投影；
2. 保持 shape、dtype、device；
3. 不归一化；
4. 不平移；
5. 不改变概率质量；
6. `x` 包含 NaN/Inf 时不能被投影掩盖，而是在输入校验阶段报错；
7. 负值和大于 1 的有限值可以被投影；
8. 投影函数本身必须是公开的诊断/测试辅助函数。

```python
def project_image(image: torch.Tensor) -> torch.Tensor
```

---

## 9. 统一求解器接口

### 9.1 配置输入

求解器不读取全局配置文件，但允许从 `GWFConfig` 显式接收以下值：

1. `iterations`；
2. `dt`；
3. `lambda_value`；
4. `epsilon`；
5. `sigma`；
6. `scale`；
7. `edge_weight_mode`；
8. `noise_std`，只在随机数据基线中使用；
9. `seed`。

这些值必须在调用求解器前完成校验。

### 9.2 `SuperResolutionSolver`

冻结类接口：

```python
class SuperResolutionSolver:
    def __init__(
        self,
        low_resolution: torch.Tensor,
        initialization: torch.Tensor,
        *,
        scale: int = 2,
        sigma: float = 1.0,
        iterations: int = 300,
        dt: float = 0.05,
        lambda_value: float = 0.0,
        epsilon: float = 1e-8,
        edge_weight_mode: str = "uniform",
        integrator: str = "euler",
        record_every: int = 1,
        noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
```

构造器职责：

1. 校验 `low_resolution` 和 `initialization`；
2. 验证 shape、dtype、device、有限性；
3. 验证 `scale`、`sigma`、`iterations`、`dt`、`lambda_value`、`epsilon`、`edge_weight_mode`、`integrator`、`record_every`、`noise_std`；
4. 建立固定退化算子；
5. 建立与 initialization 匹配的阶段四图对象；
6. 不执行迭代，不修改输入；
7. 不读取文件。

### 9.3 求解方法

冻结方法：

```python
def solve(
    self,
    x0: torch.Tensor | None = None,
) -> SolverResult
```

规则：

1. `x0=None` 时使用构造器传入的 `initialization`；
2. `x0` 不为 None 时必须与 `initialization` shape、dtype、device 一致；
3. 构造器中的 `low_resolution` 固定为观测值；
4. 求解器不得在 `solve` 中重新生成数据；
5. 求解器不得修改调用方传入的 tensor；
6. 返回 `SolverResult`，不返回裸 tensor。

### 9.4 增量单步接口

冻结辅助接口：

```python
def step(
    self,
    image: torch.Tensor,
) -> torch.Tensor
```

规则：

1. 执行一个完整 Euler 或 Strang 步；
2. 返回投影后的候选图像；
3. shape、dtype、device 与输入一致；
4. 不返回轨迹；
5. 不修改内部状态；
6. 不改变构造器中的 `iterations`。

---

## 10. `SolverResult` 与轨迹记录

### 10.1 结果对象

冻结返回对象：

```python
@dataclass(frozen=True)
class SolverResult:
    reconstruction: torch.Tensor
    initial_image: torch.Tensor
    low_resolution: torch.Tensor
    trajectory: torch.Tensor
    times: torch.Tensor
    data_fidelity: torch.Tensor
    entropy: torch.Tensor
    regularization_energy: torch.Tensor
    total_energy: torch.Tensor
    curvature: torch.Tensor
    min_probability: torch.Tensor
    max_probability: torch.Tensor
    effective_state_count: torch.Tensor
    update_norm: torch.Tensor
    final_update_norm: torch.Tensor
    iterations: int
    dt: float
    lambda_value: float
    epsilon: float
    integrator: str
```

### 10.2 轨迹张量

轨迹定义为：

\[
T
=
[x_0, x_1, \ldots, x_{\mathrm{iterations}}],
\]

shape 为：

\[
(\mathrm{iterations}+1,\;B,\;H,\;W)
\]

或单图模式：

\[
(\mathrm{iterations}+1,\;H,\;W).
\]

规则：

1. `trajectory[0]` 是投影后的初始图像；
2. `trajectory[-1]` 等于 `reconstruction`；
3. `trajectory` dtype 与输入相同；
4. `trajectory` device 与输入相同；
5. 所有轨迹值有限；
6. 所有轨迹值位于 `[0,1]`，允许 `1e-6` 的浮点越界；
7. 不允许轨迹中混入未投影的中间状态；
8. 不允许把 batch 维与迭代维交换。

### 10.3 时间张量

时间张量冻结为：

\[
t_k = k \cdot dt,
\qquad k=0,\ldots,\mathrm{iterations}.
\]

shape：

- 单图：`(iterations+1,)`；
- batch：`(iterations+1,)`，因为所有 batch 样本共享求解器时间。

规则：

1. `times` dtype 与输入相同；
2. `times` device 与输入相同；
3. `times[0]=0`；
4. `times[-1]=iterations*dt`；
5. 不允许使用真实墙钟时间；
6. 不允许使用自适应时间。

### 10.4 逐迭代诊断

每个记录点必须包含以下诊断，shape 为 `(iterations+1)` 或 `(iterations+1, B)`：

1. `data_fidelity`：

\[
\frac{1}{2}\|A(x_k)-y\|_2^2;
\]

2. `entropy`：

\[
H(\rho_k);
\]

3. `regularization_energy`：

\[
\mathcal{R}_\lambda(x_k);
\]

4. `total_energy`：

\[
\mathcal{L}_{data}(x_k)+\mathcal{R}_\lambda(x_k);
\]

5. `curvature`：阶段四 `curvature_quotient(rho_k, graph, lambda_value)`；
6. `min_probability`：阶段三 `min_probability(rho_k)`；
7. `max_probability`：阶段三 `max_probability(rho_k)`；
8. `effective_state_count`：阶段三 `effective_state_count(rho_k)`；
9. `update_norm`：

\[
\|x_k-x_{k-1}\|_2,
\]

其中 `k=0` 的值定义为零；
10. `final_update_norm`：`update_norm[-1]`。

规则：

1. 所有诊断量不得跨 batch 平均；
2. 单图诊断 shape 为 `(iterations+1,)`；
3. batch 诊断 shape 为 `(iterations+1, B)`；
4. 所有诊断量有限；
5. 诊断量必须使用与求解器相同的 `rho_k`、graph、`lambda_value`、`epsilon`；
6. 诊断计算不得改变下一迭代状态。

---

## 11. 步长与迭代校验

### 11.1 `dt` 校验

`dt` 必须满足：

1. 是 Python `int` 或 `float`；
2. 不是 bool；
3. 有限；
4. 严格大于 0；
5. `dt <= 0` 抛出 `ValueError`；
6. NaN/Inf 抛出 `ValueError`；
7. 不设置全局理论上限；
8. 不使用配置文件中未声明的缩放因子。

### 11.2 `iterations` 校验

`iterations` 必须满足：

1. 是 Python 整数；
2. 不是 bool；
3. 严格大于 0；
4. `iterations <= 0` 抛出 `ValueError`；
5. 轨迹长度必须为 `iterations+1`；
6. `record_every` 不能改变求解迭代次数。

### 11.3 `lambda_value` 校验

`lambda_value` 必须满足：

1. 有限；
2. 非负；
3. `lambda_value < 0` 抛出 `ValueError`；
4. `lambda_value=0` 必须能运行；
5. `lambda_value` 大于零时，曲率正则梯度必须有限。

### 11.4 `integrator` 与 `record_every` 校验

`integrator` 当前只允许：

1. `"euler"`；
2. `"euler_maruyama"`；
3. `"strang"`；
4. 未知值抛出 `ValueError`。

为保证轨迹完整，`record_every` 冻结为：

1. 只允许 `1`；
2. 其他值抛出 `ValueError`；
3. 当前阶段五必须记录每一步；
4. 若未来启用稀疏记录，必须另建契约，不能改变当前 `trajectory` shape；
5. 阶段五实现不得默认丢弃中间轨迹。

---

## 12. 可重复性与随机模式

### 12.1 确定性模式

1. `noise_std=0` 时，相同输入、配置和 dtype/device 必须得到相同结果；
2. 相同输入重复调用 `solve` 必须逐元素一致；
3. 不允许调用未设 seed 的全局随机数生成器；
4. 不允许根据时间、线程或内存状态改变算法；
5. 图边顺序、退化算子和投影必须确定性。

### 12.2 随机基线

若 `integrator="euler_maruyama"`：

1. `noise_std` 必须有限且非负；
2. 噪声只作用于数据保真梯度流；
3. 噪声标准差为 `noise_std`；
4. 时间缩放为 `sqrt(dt)`；
5. 所有随机张量使用 `seed`、`dtype`、`device` 显式生成；
6. `noise_std=0` 必须退化为 Euler；
7. 曲率正则流不加入额外噪声；
8. 随机轨迹必须记录 `noise_std` 和 `seed`。

---

## 13. 复杂度与内存目标

### 13.1 单步复杂度

令：

- `N = H*W`；
- `E = O(N)`；
- `K` 为高斯核长度，`K = O(1)`；
- `B` 为 batch size。

冻结目标：

1. 前向退化 `A(x)`：`O(BN)`；
2. 伴随退化 `A*(z)`：`O(BN)`；
3. 数据保真项：`O(BN)`；
4. 分布入口：`O(BN)`；
5. 曲率正则项和梯度：`O(BN)`；
6. 一个 Euler 数据步：`O(BN)`；
7. 一个 Strang 步：`O(BN)`；
8. 单次完整求解：`O(BN * iterations)`；
9. 轨迹内存：`O(BN * iterations)`；
10. 诊断内存：`O(B * iterations)`。

### 13.2 禁止复杂度

禁止：

1. 构造 `N×N` Hessian；
2. 构造 `N×N` Gram 矩阵；
3. 对每个像素执行 Python 循环；
4. 使用稠密矩阵表示高斯核；
5. 使用完整特征分解求曲率或梯度；
6. 在轨迹记录中保存每一步的梯度全图，除非该梯度本身就是最终重建或必要诊断；
7. 使用 `torch.autograd.functional.hessian`；
8. 通过有限差分逐像素计算数据梯度。

### 13.3 轨迹内存警告

完整轨迹需要 `O(iterations*N)` 内存。阶段五默认允许完整记录，因为实验图像尺寸为 32 或 128；但实现必须在 `iterations` 很大时由调用方负责控制内存。求解器不得静默丢弃轨迹。

---

## 14. 模块接口冻结

### 14.1 `gwf_sr/solver.py` 公开名称

阶段五实现必须提供：

```python
project_image
data_fidelity
data_fidelity_gradient
regularization_gradient
step
SuperResolutionSolver
SolverResult
```

允许额外提供：

```python
degradation_forward
degradation_adjoint
build_solver_graph
```

但不得改变上述公开名称的语义。

### 14.2 函数签名

```python
def project_image(
    image: torch.Tensor,
) -> torch.Tensor

def data_fidelity(
    image: torch.Tensor,
    low_resolution: torch.Tensor,
    *,
    scale: int = 2,
    sigma: float = 1.0,
) -> torch.Tensor

def data_fidelity_gradient(
    image: torch.Tensor,
    low_resolution: torch.Tensor,
    *,
    scale: int = 2,
    sigma: float = 1.0,
) -> torch.Tensor

def regularization_gradient(
    image: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
    *,
    lambda_value: float,
    epsilon: float = 1e-8,
) -> torch.Tensor
```

其中 `step` 是 `SuperResolutionSolver` 的实例方法。

---

## 15. 配置映射

| 配置字段 | 阶段五含义 |
|---|---|
| `scale` | 固定退化与候选图像 shape 的倍率；当前必须为 2 |
| `sigma` | 数据保真前向/伴随高斯模糊标准差 |
| `iterations` | 求解步数，轨迹长度为 `iterations+1` |
| `dt` | 固定时间步长 |
| `lambda_values` | 消融时逐值传入 `lambda_value` |
| `epsilon` | 图像到严格正分布的平滑常数 |
| `edge_weight_mode` | 曲率正则所用图边权模式 |
| `device` | 实验层负责把输入移动到指定设备 |
| `seed` | 随机基线噪声种子 |
| `noise_std` | Euler–Maruyama 噪声标准差 |
| `output_dir` | 阶段五求解器本身不使用 |
| `config_name` | 阶段五求解器本身不使用 |

---

## 16. 验收测试契约

阶段五实现至少需要以下测试。

### 16.1 输入校验

1. 单图和 batch shape 正确；
2. `y` 与 `x0` shape 不匹配时报错；
3. `scale` 不匹配时报错；
4. 非 tensor、非法 dtype、NaN/Inf 报错；
5. 负值 `x0` 被投影，不崩溃；
6. `dt<=0`、`iterations<=0`、负 `lambda_value`、非法 integrator、非法 `record_every` 报错。

### 16.2 投影与退化

1. `project_image` 把负值投影为 0，把大于 1 的值投影为 1；
2. 投影保持 shape、dtype、device；
3. `A` 保持 shape；
4. `A*` 与 `A` 满足内积伴随关系；
5. `A` 和 `A*` 不修改输入。

### 16.3 数据流

1. 数据 Euler 更新公式与手算一致；
2. `lambda_value=0` 的 Strang 更新等于数据 Euler 更新；
3. 常数图像和常数观测的梯度行为正确；
4. `noise_std=0` 的 Euler–Maruyama 等于 Euler；
5. 数据梯度 shape、dtype、device 正确；
6. 数据梯度有限。

### 16.4 曲率流

1. 曲率梯度 shape、dtype、device 正确；
2. `lambda_value=0` 时梯度为零；
3. 曲率流单步不包含数据项；
4. 曲率流每步后图像位于 `[0,1]`；
5. 曲率流不产生 NaN/Inf。

### 16.5 轨迹与结果

1. 轨迹 shape 为 `(iterations+1,...)`；
2. 轨迹首项等于投影后的 `x0`；
3. 轨迹末项等于 `reconstruction`；
4. 时间张量为 `k*dt`；
5. 所有诊断 shape 正确；
6. `total_energy = data_fidelity + regularization_energy`；
7. 相同输入重复求解结果一致；
8. `SolverResult` 不包含裸 Python float 形式的 batch 诊断；
9. 最终图像和轨迹全部位于 `[0,1]`。

### 16.6 复杂度

1. 求解器不创建 `N×N` 矩阵；
2. 求解器不使用逐像素 Python 循环；
3. 单步时间随 `N` 近似线性增长；
4. 完整求解时间和内存随 `iterations` 近似线性增长。

---

## 17. 冻结决策摘要

| 决策项 | 阶段五冻结结果 |
|---|---|
| 输入形状 | `H_lr×W_lr`/`B×H_lr×W_lr` 与 `H×W`/`B×H×W` |
| dtype | 仅 `float32`、`float64` |
| device | 输出跟随输入 tensor |
| NaN/Inf | 输入即报错，不允许传播 |
| 前向退化 | `D_s(B_sigma(x))` |
| 伴随退化 | `B_sigma*(D_s*(z))` |
| 数据项 | `1/2 ||A(x)-y||^2` |
| 数据梯度 | `A*(A(x)-y)` |
| 数据流 | `-A*(A(x)-y)` |
| 曲率流 | `-grad_x R_lambda(x)` |
| Euler | 每步投影到 `[0,1]` |
| Strang | 曲率半步、数据全步、曲率半步 |
| 投影 | `min(max(x,0),1)` |
| 求解器类 | `SuperResolutionSolver` |
| 结果对象 | `SolverResult` |
| 轨迹长度 | `iterations+1` |
| 时间 | `t_k=k*dt` |
| `dt` | 正有限标量，无默认全局上限 |
| `iterations` | 正整数 |
| `lambda_value` | 有限非负 |
| `record_every` | 仅允许 1，必须记录每一步 |
| 轨迹记录 | 完整、投影后、不可静默丢弃 |
| 随机模式 | Euler–Maruyama，`sqrt(dt)` 缩放 |
| 单步复杂度 | `O(BN)` |
| 完整复杂度 | `O(BN*iterations)` |
| 禁止对象 | `N×N` Hessian、Gram、稠密核 |
| 核心实现 | 只在用户确认后由 developer 编写 |

---

## 18. 完成条件

阶段五设计在满足以下条件后可进入实现：

1. 用户确认本文档；
2. Lead 向 developer 传达冻结契约；
3. developer 仅新增 `gwf_sr/solver.py` 和阶段五测试；
4. 不修改阶段二、三、四模块；
5. 实现不得引入未冻结的求解器模式；
6. 实现后由 verifier 独立验收；
7. 任何契约变更必须由 architect 重新起草并由 Lead 提交。
