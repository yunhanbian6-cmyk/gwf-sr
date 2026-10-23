# Geometry 接口与数学契约（阶段三冻结版）

**状态：冻结，待阶段三实现 `gwf_sr/geometry.py` 时逐条遵守**  
**约束：本文档只定义接口和数学语义，不包含阶段三核心实现；不修改阶段二已通过模块。**

---

## 1. 目的与范围

阶段三需要把阶段二产出的图像张量转换为严格正概率分布，并提供熵、KL 散度、Hilbert 射影距离以及诊断量。本文档冻结以下内容：

1. 张量形状、dtype、device 和数值校验规则；
2. 图像到概率分布的唯一转换公式；
3. 熵、KL 和 Hilbert 射影距离的唯一数学定义；
4. NaN、Inf、负值、全零/全黑输入的处理语义；
5. `total_mass`、最小概率、最大概率、有效状态数等诊断量定义；
6. O(N) 复杂度、内存复杂度和禁止 N×N 中间矩阵的实现约束；
7. 后续单元测试必须遵守的行为契约。

其中 `N` 固定表示单张图像中的状态数，即 `N = H * W`。批量输入时，每个 batch 元素独立拥有 `N` 个状态。

---

## 2. 术语与约定

### 2.1 形状约定

允许的输入形状只有：

1. `H × W`：单张灰度图；
2. `B × H × W`：包含 `B` 张灰度图的 batch。

不允许以下行为：

1. 接受 1-D、4-D 或更高维张量；
2. 把通道维混入 geometry 输入；
3. 静默 squeeze 掉大小为 1 的 batch 维；
4. 对 `H`、`W` 的取值作额外限制；只要每张图满足 `N >= 1` 即可。

形状校验规则：

1. 输入必须是 `torch.Tensor`；
2. 输入维数必须是 2 或 3；
3. 对 2-D 输入，形状固定为 `(H, W)`；
4. 对 3-D 输入，形状固定为 `(B, H, W)`；
5. 维数错误时抛出 `ValueError`，错误信息必须包含期望形状 `H×W` 或 `B×H×W`；
6. 非 Tensor 输入抛出 `TypeError`。

### 2.2 dtype 约定

允许 dtype：

1. `torch.float32`；
2. `torch.float64`。

规则：

1. `float32` 输入保持 `float32` 输出；
2. `float64` 输入保持 `float64` 输出；
3. 函数不得在无说明的情况下把 `float64` 降为 `float32`；
4. 其他 dtype（包括 `float16`、`bfloat16`、整数、布尔、复数）一律抛出 `TypeError`；
5. 不允许静默 cast；阶段三函数是纯几何计算，输入 dtype 必须由调用方显式控制。

默认推荐：阶段二图像和后续求解器默认使用 `torch.float32`；需要更高精度的数值稳定性实验必须显式使用 `torch.float64`。

### 2.3 device 约定

规则：

1. 所有输出与输入 tensor 位于同一 device；
2. 函数不得隐式把输入搬到 CPU；
3. 函数不得隐式创建跨 device 的中间 tensor；
4. device 不匹配是调用方错误，不由 geometry 函数自动修复；
5. `epsilon` 是 Python 标量，由实现时转换为当前输入 dtype/device 的 tensor；
6. 若输入 tensor 位于非 CPU 设备，所有逐元素操作必须保持该设备；
7. 阶段三测试可只在 CPU 上验证，但接口不能写死 `cpu`。

`gwf_sr.config.GWFConfig.device` 是实验级设备选择；geometry 函数本身不读取全局配置，只服从输入 tensor 的 device。后续求解器负责把输入移动到配置指定设备。

---

## 3. 输入数值校验

### 3.1 有限值规则：NaN/Inf 报错

阶段三 geometry 函数采用“有限输入才计算”的契约：

1. `torch.isfinite(x).all()` 为真才允许计算；
2. 输入包含任意 `NaN` 时，抛出 `ValueError`；
3. 输入包含任意 `+Inf` 或 `-Inf` 时，抛出 `ValueError`；
4. 错误信息必须指出“input contains NaN or Inf”；
5. 不允许 NaN/Inf 传播到熵、KL、Hilbert 或诊断量输出；
6. 也不允许静默裁剪 NaN/Inf。

选择报错而非传播的原因：本项目需要冻结实验可重复性和失败定位；静默传播会把数据生成、初始化或求解器错误延后到指标阶段，难以审计。求解器若未来需要“故障样本级”传播，必须在求解器层单独设计，不得改变本契约。

### 3.2 负值规则：不崩溃

负值是允许出现的输入情况，尤其是数值求解过程中可能产生小的负数。

1. 函数不得因输入存在负值而报错；
2. 负值在转换为分布时使用 `max(x_i, 0)` 截断；
3. 截断是逐元素操作，不改变 shape、dtype、device；
4. 负值截断后的质量缺口由 `epsilon` 项补足；

### 3.3 空张量与零维

1. `H = 0` 或 `W = 0` 的输入抛出 `ValueError`；
2. 3-D batch 中 `B = 0` 允许与否固定为：抛出 `ValueError`，因为 batch 统计没有样本定义；
3. 任何 shape 至少保证一个像素状态。

---

## 4. 图像到概率分布的唯一转换

### 4.1 公式冻结

对任意位置像素值 `x_i`，先做非负截断：

\[
x_i^{+} = \max(x_i, 0)
\]

再做严格正化和归一化：

\[
\rho_i
=
\frac{x_i^{+} + \epsilon}
{\sum_{j=1}^{N} \left(x_j^{+} + \epsilon\right)}
\]

等价地：

\[
\rho_i
=
\frac{x_i^{+} + \epsilon}
{\sum_j x_j^{+} + N\epsilon}
\]

其中：

1. `i` 是该图像中的像素索引；
2. `N = H * W`；
3. `epsilon` 是统一的严格正化常数；
4. `sum` 按每张图像单独计算，不跨 batch 归一化。

该公式必须是后续所有熵、KL、Hilbert、诊断量计算的唯一入口。

### 4.2 epsilon 语义

`epsilon` 的语义冻结如下：

1. `epsilon` 是概率平滑常数，不是正则化系数，也不是图像亮度偏置；
2. 它为每个像素加入相同的绝对质量；
3. 它保证任意有限输入下，每个 `rho_i > 0`；
4. 它保证任意有限输入下，`rho` 可以参与 `log(rho)`、`log(rho/eta)` 和比值运算；
5. 它不负责处理 NaN/Inf；
6. 它不负责替代非负截断；
7. 它不保证近似均匀性：当输入像素质量远大于 `N*epsilon` 时，`epsilon` 只是尾部下限；
8. 它是无量纲的分布质量常数，不随图像亮度自动缩放。

阶段二配置 `GWFConfig.epsilon` 默认是 `1e-8`，且已要求 `epsilon > 0`。阶段三实现不得引入第二套默认 epsilon；所有调用必须显式传入同一个 epsilon，或使用与配置一致的参数。

实现必须校验：

1. `epsilon` 必须是有限的 Python 标量（`int` 或 `float`，不允许 `bool`）；
2. `epsilon > 0`；
3. `epsilon <= 0` 抛出 `ValueError`；
4. 实现阶段将其转换为当前输入 dtype/device 的 tensor。

### 4.3 全黑输入

全黑图像定义为所有 `x_i = 0`。

代入公式：

\[
\rho_i = \frac{\epsilon}{N\epsilon} = \frac{1}{N}
\]

契约：

1. 全黑输入不崩溃；
2. 输出必须是均匀分布 `rho_i = 1/N`；
3. 熵必须等于 `log(N)`；
4. 有效状态数必须等于 `N`；
5. 最小概率等于最大概率，均为 `1/N`；
6. 3-D batch 中每个全黑样本独立得到均匀分布。

全负输入同样满足该语义：所有 `max(x_i,0) = 0`，结果也是均匀分布。

### 4.4 分布基本性质

对任意通过 `distribution_from_image` 得到的有限输入分布：

1. shape 与输入相同；
2. dtype 与输入相同；
3. device 与输入相同；
4. 所有 `rho_i > 0`；
5. 所有 `rho_i < 1` 只在 `N >= 2` 时可保证；当 `N = 1` 时，唯一状态的概率必须为 1；
6. 对 `N >= 1`，每个 batch 元素满足 `sum_i rho_i = 1`，允许浮点误差；
7. 不允许返回负概率、零概率或未归一化概率。

验收容差：

- `float32`：`abs(sum(rho) - 1) <= 1e-5`；
- `float64`：`abs(sum(rho) - 1) <= 1e-12`。

---

## 5. 函数接口冻结

以下名称、参数顺序和返回语义在阶段三实现中不得改变。

### 5.1 `distribution_from_image`

```python
def distribution_from_image(
    image: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
```

职责：

1. 校验 shape、dtype、device、finite、epsilon；
2. 对单图或 batch 执行非负截断；
3. 按每个图像独立归一化；
4. 返回严格正概率分布。

参数：

- `image`：`H×W` 或 `B×H×W`，允许负值，但必须有限；
- `epsilon`：严格正化常数，固定为正有限标量。

返回：

- 与输入同 shape、同 dtype、同 device 的严格正概率分布。

### 5.2 `entropy`

```python
def entropy(
    distribution: torch.Tensor,
) -> torch.Tensor:
```

输入：

- 已归一化或近似归一化的分布，shape 为 `H×W` 或 `B×H×W`；
- 不负责再次归一化；
- 不负责把负值截断为零；
- 不负责替换 NaN/Inf。

数学定义：

\[
H(\rho) = -\sum_i \rho_i \log(\rho_i)
\]

返回规则：

1. 对 `H×W` 输入返回 0-D tensor 标量；
2. 对 `B×H×W` 输入返回 shape 为 `(B,)` 的 tensor，每个元素是对应样本的熵；
3. 不返回 Python float，避免 batch/单图 API 类型不一致；
4. 不跨 batch 求平均，平均值由调用方自行决定。

边界规则：

1. `rho_i = 0` 时，该项按数学极限 `0*log(0)=0` 处理，不产生 NaN；
2. `rho_i > 1` 时是非法分布，不静默修正，测试应通过分布入口保证不会出现；
3. 全黑图像经过分布入口后，熵必须为 `log(N)`。

### 5.3 `kl_divergence`

```python
def kl_divergence(
    rho: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
```

输入：

- `rho`：查询分布；
- `eta`：参考分布；
- 两者 shape 必须完全一致；
- 两者必须是同 dtype、同 device 的有限 tensor。

数学定义：

\[
D_{\mathrm{KL}}(\rho\|\eta)
=
\sum_i \rho_i \log \frac{\rho_i}{\eta_i}
\]

实现允许等价分解：

\[
D_{\mathrm{KL}}(\rho\|\eta)
=
\sum_i \rho_i (\log \rho_i - \log \eta_i)
\]

返回规则：

1. `H×W` 输入返回 0-D tensor 标量；
2. `B×H×W` 输入返回 `(B,)` tensor；
3. 不跨 batch 求平均。

契约：

1. `rho == eta` 时，结果必须在容差内等于 0；
2. 不对输入做二次归一化；
3. 不对 `rho` 或 `eta` 做非负截断；
4. 对严格正分布，结果数值必须有限；
5. 若 `eta` 含零或负值，调用方应先通过分布入口保证严格正性；geometry 函数不静默修复。

### 5.4 `hilbert_distance`

```python
def hilbert_distance(
    rho: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
```

输入约束同 KL。

数学定义：

\[
d_H(\rho,\eta)
=
\log\left(\max_i \frac{\rho_i}{\eta_i}\right)
-
\log\left(\min_i \frac{\rho_i}{\eta_i}\right)
\]

实现必须按比值的最大值和最小值计算，不得替换为任意其他距离。允许使用等价但数值更稳定的形式：

\[
d_H(\rho,\eta)
=
\max_i(\log\rho_i-\log\eta_i)
-
\min_i(\log\rho_i-\log\eta_i)
\]

返回规则：

1. `H×W` 输入返回 0-D tensor 标量；
2. `B×H×W` 输入返回 `(B,)` tensor；
3. 不跨 batch 求平均。

契约：

1. `rho == eta` 时，结果必须等于 0（在浮点容差内）；
2. `d_H(rho, eta) = d_H(eta, rho)`；
3. 对严格正分布，结果必须有限且非负；
4. 不做二次归一化；
5. 不做负值截断；
6. 不跨 batch 计算比值。

### 5.5 诊断接口

诊断量必须共享同一套 shape 规则：

```python
def total_mass(distribution) -> torch.Tensor
def min_probability(distribution) -> torch.Tensor
def max_probability(distribution) -> torch.Tensor
def effective_state_count(distribution) -> torch.Tensor
```

`H×W` 输入返回 0-D tensor，`B×H×W` 输入返回 `(B,)` tensor。

#### 5.5.1 `total_mass`

唯一定义：

\[
M(\rho)=\sum_i \rho_i
\]

用途：

1. 验证分布入口是否正确归一化；
2. 发现调用方手工构造分布时的数值异常。

验收：

- `float32` 误差不超过 `1e-5`；
- `float64` 误差不超过 `1e-12`。

#### 5.5.2 `min_probability`

唯一定义：

\[
p_{\min}(\rho)=\min_i \rho_i
\]

用途：

1. 检查严格正性；
2. 诊断 epsilon 对尾部概率的下限贡献。

对通过分布入口得到的分布，必须满足：

\[
p_{\min} \ge \frac{\epsilon}{\sum_j x_j^{+} + N\epsilon} > 0
\]

#### 5.5.3 `max_probability`

唯一定义：

\[
p_{\max}(\rho)=\max_i \rho_i
\]

用途：

1. 诊断分布集中程度；
2. 检查是否存在异常质量尖峰。

#### 5.5.4 `effective_state_count`

必须固定选择“熵指数”定义：

\[
N_{\mathrm{eff}}(\rho)=\exp(H(\rho))
=
\exp\left(-\sum_i \rho_i\log\rho_i\right)
\]

不允许同时把 participation ratio 当作本函数的返回值。participation ratio：

\[
\frac{1}{\sum_i \rho_i^2}
\]

可以作为未来独立指标，但阶段三冻结 API 不引入它，避免“有效状态数”出现两个互不等价的定义。

`effective_state_count` 的唯一语义是：

1. 全黑输入：返回 `N`；
2. 均匀分布：返回 `N`；
3. 单点集中分布：趋近于 1；
4. 返回值范围在理想数学定义下满足 `1 <= N_eff <= N`；
5. 浮点误差导致的极小越界可以由测试容差吸收，但实现不得主动 clip 到 `[1,N]`。

---

## 6. 复杂度与内存契约

### 6.1 计算复杂度

对单张 `H×W` 图像，所有阶段三函数的时间复杂度必须是：

\[
O(N), \qquad N=H\cdot W
\]

对 batch 输入，复杂度为：

\[
O(B N)
\]

禁止：

1. 隐式构造 `N×N` 协方差、Gram、相似度或距离矩阵；
2. 为了最大值/最小值以外的相关性统计构造全像素两两矩阵；
3. 通过 Python 循环逐像素计算；
4. 对 `H` 或 `W` 做二次方级别的 reshape 操作。

### 6.2 内存复杂度

除输入和输出本身外，允许使用的额外内存必须是：

\[
O(N)
\]

或 batch 情况下的：

\[
O(BN)
\]

允许的典型中间量：

1. 非负截断结果；
2. 分母或总和标量；
3. `log(rho)`、`log(eta)`；
4. 比值或差值；
5. 逐元素 reduce 结果。

禁止的中间量：

1. `N×N` tensor；
2. 每个像素到每个像素的显式比值矩阵；
3. 跨 batch 的全量 broadcast 矩阵；
4. 为求 Hilbert 距离而构造的 pairwise scatter 矩阵。

### 6.3 归约顺序约束

为保证 CPU、未来 XPU 与不同 PyTorch 版本下的结果尽量稳定：

1. 单张图内部的 `sum`、`min`、`max` 都沿最后一组空间维 `(H,W)` 执行；
2. batch 维只用于独立输出，不用于内部跨样本归约；
3. 不允许在 batch 与空间维混合后先求一个全局统计量再广播回单图；
4. 归约函数必须使用 PyTorch tensor 操作，不得用 Python `sum`、`max`、`min` 替代。

---

## 7. 数值稳定性契约

### 7.1 概率对数

1. `log` 必须作用在严格正分布上；
2. 若调用方绕过分布入口传入零或负概率，函数不静默修复；
3. 阶段三测试必须通过分布入口构造测试分布；
4. 不允许把 `log(rho)` 替换为 `log(rho+delta)`；否则会破坏冻结的数学定义。

### 7.2 比值与 Hilbert

实现优先使用：

\[
\log\rho_i-\log\eta_i
\]

而不是显式先算 `rho_i/eta_i` 再取对数，以降低溢出/下溢风险。两种写法在数学上等价，但测试只检验最终 `d_H` 值，不检验内部中间量。

### 7.3 epsilon 与精度

1. `epsilon=1e-8` 在 `float32` 下仍大于 0；
2. 但 `1e-8` 对 `float32` 的极端小质量输入可能接近下溢边界；
3. 若未来求解器需要更小 epsilon，必须先扩展配置校验并增加数值稳定性测试，不得在阶段三静默替换；
4. 任何实验报告中的 epsilon 必须从 `GWFConfig.epsilon` 或显式参数记录。

---

## 8. 非目标范围

阶段三 geometry 模块不得包含以下内容：

1. 图结构构造；
2. 曲率或 Hessian 计算；
3. 能量泛函或梯度流求解；
4. PSNR、SSIM、LPIPS；
5. 真实数据集读取；
6. 神经网络 SR 模型；
7. 修改阶段二 `gwf_sr/synthetic.py`、`gwf_sr/degradation.py`、`gwf_sr/data.py`；
8. 为了实现 geometry 而改动阶段二已有单元测试。

---

## 9. 建议的阶段三测试契约

实现 `gwf_sr/geometry.py` 后，至少覆盖以下测试：

### 9.1 输入校验

1. `H×W` 和 `B×H×W` 均可处理；
2. 1-D、4-D 输入抛出 `ValueError`；
3. 非 Tensor 输入抛出 `TypeError`；
4. `float16`、整数、复数输入抛出 `TypeError`；
5. `float32` 保持 `float32`，`float64` 保持 `float64`；
6. NaN 输入抛出 `ValueError`；
7. `+Inf`/`-Inf` 输入抛出 `ValueError`；
8. `epsilon=0` 或负数抛出 `ValueError`。

### 9.2 分布性质

1. 输出 shape、dtype、device 与输入一致；
2. 所有概率严格大于 0；
3. 总质量在容差内等于 1；
4. 全黑输入得到均匀分布；
5. 全负输入得到均匀分布；
6. 含少量负值输入不崩溃；
7. 相同输入重复调用结果一致；
8. batch 各样本独立归一化。

### 9.3 几何度量

1. 熵对全黑输入等于 `log(N)`；
2. 熵对均匀分布不超过 `log(N)`；
3. `KL(rho, rho)` 接近 0；
4. `d_H(rho, rho)` 等于 0；
5. `d_H(rho, eta) == d_H(eta, rho)`；
6. `H×W` 输入返回标量，`B×H×W` 输入返回 `(B,)`；
7. 诊断量 `total_mass`、`min_probability`、`max_probability`、`effective_state_count` 与公式一致；
8. 有效状态数使用 `exp(H)` 定义，而不是 participation ratio。

### 9.4 复杂度约束

1. 通过代码审查或小型 benchmark 保证无 `N×N` 中间 tensor；
2. `N` 增大时运行时间近似线性增长；
3. 单张 128×128 图像的 geometry 计算在 CPU 上应明显低于秒级；
4. 不得通过构造全像素 pairwise 矩阵取巧实现任何指标。

---

## 10. 实现阶段的唯一入口约束

实现 `geometry.py` 时，必须遵循以下依赖顺序：

1. 先实现并冻结 `distribution_from_image`；
2. 再让熵、KL、Hilbert 和诊断量复用该分布；
3. 所有数学函数只接受 tensor，不接受路径、配置对象或文件名；
4. `GWFConfig.epsilon` 只作为调用方参数来源，不在 geometry 内部读取全局配置；
5. 测试不得依赖阶段二模块内部实现，只依赖阶段二公开函数行为。

---

## 11. 冻结决策摘要

| 决策项 | 冻结结果 |
|---|---|
| 输入形状 | `H×W` 或 `B×H×W` |
| 支持 dtype | 仅 `float32`、`float64` |
| dtype 处理 | 输入输出保持，不静默 cast |
| device 处理 | 输入输出同设备，不自动搬运 |
| NaN/Inf | `ValueError`，不传播、不静默裁剪 |
| 负值 | 允许输入，使用 `max(x,0)` 截断，不崩溃 |
| 全黑输入 | 均匀分布 `1/N` |
| 分布公式 | `rho_i=(max(x_i,0)+epsilon)/sum_j(max(x_j,0)+epsilon)` |
| epsilon 语义 | 每像素正平滑常数，保证严格正，不处理 NaN/Inf |
| 熵 | `H=-sum rho log rho` |
| 熵返回 shape | 单图 0-D，batch `(B,)` |
| KL | `D_KL(rho||eta)=sum rho log(rho/eta)` |
| Hilbert | `log max(rho/eta)-log min(rho/eta)` |
| `total_mass` | `sum rho` |
| 最小概率 | `min rho` |
| 最大概率 | `max rho` |
| 有效状态数 | 唯一采用 `exp(H)`，不采用 participation ratio |
| 计算复杂度 | `O(N)` 或 `O(BN)` |
| 额外内存 | `O(N)` 或 `O(BN)` |
| 禁止对象 | 任何 `N×N` 中间矩阵 |

---

## 12. 阶段三完成条件

只有在以下条件全部满足时，阶段三才可视为完成：

1. `gwf_sr/geometry.py` 按本文档实现；
2. 所有阶段三单元测试通过；
3. 不修改阶段二已 PASS 模块；
4. 不改变阶段二配置语义；
5. 不引入 `geometry.py` 之外的求解器或图结构代码；
6. README 或进度报告中的阶段三状态可更新为完成。
