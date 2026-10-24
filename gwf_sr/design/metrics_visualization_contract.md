# 图像质量评价与可视化契约（阶段六设计草案）

**状态：等待用户确认后再实现；本文件只冻结接口、数学定义、输出语义、文件格式和验收边界，不包含核心实现代码。**  
**实现目标模块：`gwf_sr/metrics.py`、`gwf_sr/visualization.py`。**  
**依赖边界：复用阶段二数据保存工具、阶段三 `geometry`、阶段四图诊断、阶段五 `SolverResult`；不修改阶段二至阶段五已通过模块。**

---

## 1. 阶段六目的与范围

阶段六将超分求解结果转换为可复现实验评价和可视化，冻结以下内容：

1. PSNR、SSIM、LPIPS 的数学定义、shape、dtype、device 和输出格式；
2. 高频能量、边缘保持和相对高频保持指标；
3. 求解器迭代收敛曲线；
4. LR、Bicubic、Baseline、GWF-C、HR 对比图；
5. CSV、JSON、PNG 的保存结构；
6. LPIPS 缺失时的可选依赖策略；
7. 输入校验、参数校验和 O(BN) 复杂度目标；
8. 测试、可重复性和文件清单验收要求。

阶段六不负责：

- 修改求解器或图几何；
- 改变阶段三概率定义；
- 在指标计算中重新归一化或重新投影图像；
- 训练神经网络；
- 下载数据集或权重；
- 生成阶段七端到端演示脚本。

---

## 2. 模块边界与公开接口

### 2.1 `gwf_sr/metrics.py`

阶段六指标模块必须提供：

```python
psnr
ssim
lpips_distance
high_frequency_energy
high_frequency_energy_ratio
edge_preservation_index
edge_preservation_ratio
compare_images
compute_image_metrics
```

允许额外提供：

```python
gradient_magnitude
validate_image_pair
validate_metric_config
```

但不得改变上述公开名称的语义。

### 2.2 `gwf_sr/visualization.py`

可视化和持久化模块必须提供：

```python
plot_convergence_curves
plot_method_comparison
save_metrics_json
save_metrics_csv
save_image_comparison
save_evaluation_bundle
```

允许额外提供：

```python
tensor_to_uint8
ensure_output_directory
validate_visualization_config
```

但不得改变上述公开名称的语义。

### 2.3 模块职责

`metrics.py`：

1. 只接受 tensor；
2. 只做数值评价；
3. 不读写文件；
4. 不依赖 matplotlib；
5. 不依赖 LPIPS 以外的可选包；
6. 不修改输入图像。

`visualization.py`：

1. 接受 tensor、`SolverResult`、mapping 或 dataclass；
2. 负责绘图和文件保存；
3. 不重新定义 PSNR、SSIM、LPIPS；
4. 不改变求解器轨迹；
5. 可以调用 `metrics.py` 的公开 API；
6. 可以调用阶段二 `save_image_tensor`，但不得修改其实现。

---

## 3. 通用张量契约

### 3.1 输入形状

所有图像指标只接受：

1. `H × W`；
2. `B × H × W`。

规则：

1. 参考图像和预测图像必须是同一个 shape；
2. 空图像或空 batch 抛出 `ValueError`；
3. 1-D、4-D 或更高维输入抛出 `ValueError`；
4. 非 tensor 输入抛出 `TypeError`；
5. batch 模式下每个样本独立计算指标，不跨样本平均。

### 3.2 dtype

阶段六计算指标时支持：

1. `torch.float32`；
2. `torch.float64`。

规则：

1. 不接受 `float16`、`bfloat16`、整数、布尔、复数；
2. 不允许静默 cast；
3. 输入为 `float32` 时输出为 `float32`；
4. 输入为 `float64` 时输出为 `float64`；
5. 图像文件保存时才允许转为 `uint8`；
6. CSV/JSON 中的数值序列化为 Python `float`，不受 tensor dtype 影响。

### 3.3 device

1. 参考图像和预测图像必须位于同一 device；
2. 所有数值指标输出保持该 device；
3. 可视化和文件保存必须先把 tensor 移动到 CPU；
4. 不允许保存函数隐式改变计算指标的 device；
5. LPIPS 模型和输入必须位于同一 device；若模型无法移动到输入 device，必须报错，不允许静默回退到 CPU。

### 3.4 有限值与值域

1. 所有指标输入必须有限；NaN/Inf 抛出 `ValueError`；
2. 负值和大于 1 的有限值不被指标函数静默修复；
3. PSNR、SSIM、LPIPS 输入通常应位于 `[0,1]`；
4. 若调用方需要投影，应显式调用阶段五 `project_image`；
5. 指标函数不得修改输入 tensor；
6. 图像保存函数必须先把有效值域投影到 `[0,1]`，再量化为 8-bit。

### 3.5 输出 shape

冻结输出规则：

1. 单图输入返回 0-D tensor；
2. batch 输入返回 `(B,)` tensor；
3. 不返回 Python `float` 或 list；
4. 不跨 batch 平均；
5. 不在指标函数内部生成 Python 标量诊断。

---

## 4. PSNR

### 4.1 数学定义

对参考图像 `x` 和预测图像 `y`，均值平方误差为：

\[
\operatorname{MSE}(x,y)
=
\frac{1}{N}
\sum_i (x_i-y_i)^2.
\]

PSNR 定义为：

\[
\operatorname{PSNR}(x,y)
=
10\log_{10}
\left(
\frac{\operatorname{MAX}^2}{\operatorname{MSE}(x,y)}
\right).
\]

当前阶段冻结：

\[
\operatorname{MAX}=1.
\]

因此：

\[
\operatorname{PSNR}(x,y)
=
-10\log_{10}(\operatorname{MSE}(x,y)).
\]

### 4.2 零误差约定

当：

\[
\operatorname{MSE}(x,y)=0
\]

时，PSNR 无穷大不是可保存的有限值。冻结处理为：

1. 计算得到零 MSE 时返回有限哨兵值 `+100.0`；
2. 该值不是数学无穷大，而是实验记录约定；
3. JSON/CSV 中必须原样记录 `100.0`；
4. 不允许返回 `inf` 或 `NaN`；
5. 该约定适用于单图和 batch 每个样本。

### 4.3 接口

```python
def psnr(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor
```

规则：

1. `data_range` 必须是有限正 Python 标量；
2. `data_range <= 0` 抛出 `ValueError`；
3. 当前实验默认 `data_range=1.0`；
4. MSE 在空间维上求平均，batch 维保留；
5. 单图返回 0-D tensor；
6. batch 返回 `(B,)` tensor；
7. 不允许 NaN/Inf 输出；
8. 不对输入做投影。

---

## 5. SSIM

### 5.1 冻结定义

当前阶段冻结为单通道灰度 SSIM。对每张图像：

1. 像素值按 `[0,1]` 解释；
2. 常数 `C1=(0.01)^2`；
3. 常数 `C2=(0.03)^2`；
4. 窗口为 `11 × 11`；
5. 窗口类型为可分离一维高斯窗口；
6. 高斯标准差 `1.5`；
7. 卷积边界为 `reflect`；
8. 逐窗口计算局部 SSIM；
9. 最终输出为所有窗口 SSIM 的算术平均。

### 5.2 局部项

局部均值为 `mu_x` 和 `mu_y`，局部方差和协方差为：

\[
\sigma_x^2,\qquad
\sigma_y^2,\qquad
\sigma_{xy}.
\]

局部 SSIM 为：

\[
\operatorname{SSIM}_{local}
=
\frac{
(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)
}.
\]

最终：

\[
\operatorname{SSIM}(x,y)
=
\frac{1}{M}
\sum_{\text{windows}}
\operatorname{SSIM}_{local}.
\]

### 5.3 协方差归一化

冻结为非无偏估计：

1. 局部方差和协方差除以窗口元素总数；
2. 不使用 `N-1` 校正；
3. 所有局部统计使用同一个高斯窗口；
4. 不把 `reflect` 边界产生的部分窗口单独排除；
5. 不使用色度通道；
6. 不把灰度复制为 RGB 后再计算 SSIM。

### 5.4 数值与输出

1. SSIM 数学值通常位于 `[-1,1]`，但函数不做人工 clip；
2. `x == y` 时必须在容差内等于 1；
3. 单图返回 0-D tensor；
4. batch 返回 `(B,)` tensor；
5. 不允许 NaN/Inf；
6. 不返回窗口图或热力图；
7. 若未来需要 SSIM map，必须另建接口，不能改变 `ssim` 输出。

### 5.5 接口

```python
def ssim(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor
```

规则：

1. `data_range`、`win_sigma` 必须有限且大于 0；
2. `win_size` 必须是奇数且不小于 3；
3. 奇数校验失败抛出 `ValueError`；
4. 当前默认参数为冻结值；
5. 实现必须检查 `H >= win_size` 且 `W >= win_size`，否则抛出 `ValueError`；
6. 参数可由调用方显式覆盖，但实验结果必须记录所用参数。

---

## 6. LPIPS

### 6.1 可选依赖策略

LPIPS 是可选感知指标，不是阶段六主流程硬依赖。

1. `lpips` 包未安装时，`import lpips` 不得发生在 `gwf_sr/metrics.py` 顶层；
2. LPIPS 必须延迟导入；
3. 未安装时调用 `lpips_distance(..., require=False)` 返回 `None`；
4. 未安装时调用 `lpips_distance(..., require=True)` 抛出 `ImportError`；
5. 未安装不得阻止 PSNR、SSIM、高频和可视化流程；
6. 不在阶段六自动下载权重；
7. 权重缺失必须抛出包含缓存目录和安装提示的 `RuntimeError`；
8. 不允许 LPIPS 缺失时用其他指标冒充 LPIPS；
9. LPIPS 结果缺失必须在 JSON/CSV 中显式记录为 `null` 或空值。

### 6.2 范围选择

冻结默认 LPIPS 配置：

1. 网络为 `alex`；
2. 版本为 `0.1`；
3. 输入范围为 `[0,1]`；
4. 灰度输入复制到三通道 RGB；
5. 模型与输入同 device；
6. 模型在计算前切换到 eval 模式；
7. 计算时不保留计算图；
8. 不对输入做额外归一化；
9. 不执行数据增强；
10. 模型权重缓存路径由 LPIPS 库默认管理。

### 6.3 接口

```python
def lpips_distance(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    net: str = "alex",
    version: str = "0.1",
    require: bool = False,
    cache_dir: str | Path | None = None,
) -> torch.Tensor | None
```

规则：

1. `reference` 和 `prediction` 必须通过通用图像校验；
2. 单图返回 0-D tensor；
3. batch 返回 `(B,)` tensor；
4. 可选依赖不可用且 `require=False` 时返回 `None`；
5. `require=True` 时必须得到 tensor；
6. `net` 当前只允许 `"alex"`、`"vgg"`、`"squeeze"`，默认 `"alex"`；
7. `version` 当前只允许 `"0.1"`；
8. 未知网络或版本抛出 `ValueError`；
9. LPIPS 模型缓存不得写入项目结果目录；
10. LPIPS 输出必须有限；
11. 若计算中出现 NaN/Inf，抛出 `ValueError`；
12. LPIPS 是距离，越小越好。

### 6.4 可选依赖错误信息

若 `require=True`：

- 缺少包时错误信息必须包含 `pip install lpips`；
- 缺少权重时错误信息必须说明网络名、版本和缓存位置；
- 不允许吞掉底层异常后返回零。

---

## 7. 高频能量与边缘保持

### 7.1 梯度算子

冻结使用中心差分：

\[
G_x(i,j)
=
\frac{x(i,j+1)-x(i,j-1)}{2},
\]

\[
G_y(i,j)
=
\frac{x(i+1,j)-x(i-1,j)}{2}.
\]

边界规则：

1. 采用 `replicate` 边界；
2. 不使用 Sobel；
3. 不使用 Prewitt；
4. 不使用学习型边缘算子；
5. 算子逐通道仅针对灰度单通道；
6. batch 样本独立。

梯度幅值定义为：

\[
G(i,j)=\sqrt{G_x(i,j)^2+G_y(i,j)^2}.
\]

### 7.2 高频能量

高频能量定义为梯度幅值平方的均值：

\[
\operatorname{HFE}(x)
=
\frac{1}{N}
\sum_{i,j}
G(i,j)^2.
\]

接口：

```python
def high_frequency_energy(
    image: torch.Tensor,
) -> torch.Tensor
```

规则：

1. 单图返回 0-D tensor；
2. batch 返回 `(B,)` tensor；
3. 非负；
4. 有限；
5. 不归一化；
6. 不使用图边权；
7. 不修改输入。

### 7.3 高频能量比

预测图像相对于参考图像的高频能量比为：

\[
\operatorname{HFER}(x,y)
=
\frac{\operatorname{HFE}(y)}
{\operatorname{HFE}(x)}.
\]

接口：

```python
def high_frequency_energy_ratio(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor
```

规则：

1. `epsilon` 必须是有限正标量；
2. 分母实际使用 `HFE(reference) + epsilon`；
3. `epsilon` 只防止零除，不改变主要数值；
4. 参考高频为零时，若预测高频也为零，结果约等于 0；
5. 结果非负；
6. 不 clip 到 `[0,1]`；
7. 越接近 1 表示高频总量越接近参考，但不代表结构对齐。

### 7.4 边缘保持指数

冻结边缘保持指数为：

\[
\operatorname{EPI}(x,y)
=
\frac{
\sum_i G_x(i)G_y(i)
}{
\sqrt{
\left(\sum_i G_x(i)^2\right)
\left(\sum_i G_y(i)^2\right)
}
}.
\]

其中 `G_x` 是参考图像梯度幅值，`G_y` 是预测图像梯度幅值，求和遍历单张图全部像素。

接口：

```python
def edge_preservation_index(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor
```

规则：

1. 分母实际使用 `sqrt(...) + epsilon`；
2. 输出位于 `[-1,1]`，但不人工 clip；
3. 理想结构一致时接近 1；
4. batch 逐样本计算；
5. 不计算方向角差，只比较梯度幅值相关性；
6. 不使用阈值二值边缘。

### 7.5 边缘保持比

冻结边缘保持比为：

\[
\operatorname{EPR}(x,y)
=
\frac{\operatorname{EPI}(x,y)+1}{2}.
\]

接口：

```python
def edge_preservation_ratio(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor
```

规则：

1. 理想一致时接近 1；
2. 完全不相关时接近 0.5；
3. 反向相关时接近 0；
4. 不 clip；
5. 单图返回 0-D tensor，batch 返回 `(B,)` tensor。

---

## 8. 指标组合接口

### 8.1 单对比较

冻结结果对象：

```python
@dataclass(frozen=True)
class ImageMetrics:
    psnr: torch.Tensor
    ssim: torch.Tensor
    lpips: torch.Tensor | None
    high_frequency_energy: torch.Tensor
    high_frequency_energy_ratio: torch.Tensor
    edge_preservation_index: torch.Tensor
    edge_preservation_ratio: torch.Tensor
```

冻结函数：

```python
def compare_images(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    lpips_net: str = "alex",
    lpips_version: str = "0.1",
    require_lpips: bool = False,
    lpips_cache_dir: str | Path | None = None,
) -> ImageMetrics
```

规则：

1. 一次完成所有可用指标；
2. LPIPS 缺失且 `require_lpips=False` 时 `lpips=None`；
3. 其他指标必须照常计算；
4. 所有张量指标遵循通用 shape 规则；
5. 不返回 Python float；
6. 不读写文件。

### 8.2 批量指标矩阵

冻结函数：

```python
def compute_image_metrics(
    references: torch.Tensor,
    predictions: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    lpips_net: str = "alex",
    lpips_version: str = "0.1",
    require_lpips: bool = False,
    lpips_cache_dir: str | Path | None = None,
) -> ImageMetrics
```

语义：

1. 输入为 `B×H×W` 或 `H×W`；
2. `B×H×W` 时计算逐样本指标；
3. 单图时计算单样本指标；
4. 不跨样本平均；
5. batch CSV 序列化后每行对应一个样本；
6. JSON 汇总中必须区分逐样本值和可选均值。

---

## 9. 收敛曲线数据

### 9.1 输入类型

`plot_convergence_curves` 接受：

1. `SolverResult`；
2. mapping；
3. 可选的多组结果 mapping。

冻结字段：

1. `times`；
2. `data_fidelity`；
3. `entropy`；
4. `regularization_energy`；
5. `total_energy`；
6. `curvature`；
7. `update_norm`；
8. 可选 `kl`；
9. 可选 `hilbert`。

### 9.2 曲线定义

阶段六必须至少支持以下曲线：

1. `data_fidelity`；
2. `entropy`；
3. `regularization_energy`；
4. `total_energy`；
5. `curvature`；
6. `update_norm`；
7. 可选 `kl`；
8. 可选 `hilbert`。

横轴统一为：

\[
t_k = k \cdot dt.
\]

规则：

1. `times` 必须是一维；
2. 每个曲线必须和 `times` 长度一致；
3. batch 数据在绘图前必须由调用方选择：
   - 第 0 个样本；
   - batch 均值；
4. 当前默认冻结为：
   - `batch_mode="mean"`；
5. 可选 `batch_mode="first"`；
6. 未知 `batch_mode` 抛出 `ValueError`；
7. 曲线数据不跨多个 lambda 自动拼接；
8. 多 lambda 曲线使用独立 method label。

### 9.3 曲线接口

```python
def plot_convergence_curves(
    results: Mapping[str, SolverResult | Mapping[str, Any]],
    output_path: str | Path,
    *,
    fields: Sequence[str] | None = None,
    batch_mode: str = "mean",
    dpi: int = 150,
    title: str | None = None,
) -> Path
```

规则：

1. `results` 至少包含一个方法；
2. method label 必须是非空字符串；
3. `fields` 默认包含冻结核心字段；
4. 输出目录自动创建；
5. 返回保存后的 `Path`；
6. 必须关闭 matplotlib figure；
7. 不显示交互窗口；
8. 必须使用 `matplotlib` 的非交互后端；
9. 不改变输入结果对象。

### 9.4 保留字段

可选 KL/Hilbert 曲线只能从 `SolverResult` 或 mapping 中读取，不在阶段六内部重新计算：

1. 若 mapping 有 `kl`，绘制为 `KL`；
2. 若 mapping 有 `hilbert`，绘制为 `Hilbert`；
3. 没有则跳过；
4. 不得伪造为零；
5. 曲线字段名在 JSON 图表清单中必须原样记录。

---

## 10. 对比图

### 10.1 固定布局

阶段六必须生成一个水平五列灰度对比图：

1. `LR`；
2. `Bicubic`；
3. `Baseline`；
4. `GWF-C`；
5. `HR`。

布局冻结为：

1. 一行五列；
2. 所有图像统一放大到 HR 尺寸；
3. 上采样使用阶段二 `bicubic_upsample`；
4. 列顺序不可改变；
5. 每列下方必须有标题；
6. 标题使用 UTF-8；
7. 图像之间使用固定间隔；
8. 不加入颜色条；
9. 不加入伪彩色；
10. 所有子图使用灰度 colormap；
11. 默认输出尺寸为宽 20 英寸、高 4 英寸；
12. 默认 DPI 为 150。

### 10.2 缺失方法

若某个方法不存在：

1. 抛出 `ValueError`；
2. 不用零图像或黑图占位；
3. 不自动重排列；
4. 可由调用方显式传入占位张量后再调用。

### 10.3 接口

```python
def plot_method_comparison(
    images: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    order: Sequence[str] = ("LR", "Bicubic", "Baseline", "GWF-C", "HR"),
    dpi: int = 150,
    figsize: tuple[float, float] = (20.0, 4.0),
    title: str | None = None,
) -> Path
```

规则：

1. `images` 必须包含 `order` 中全部键；
2. 每张图像必须是 `H×W`、`float32`、`float64` 或可由阶段六接受的图像 tensor；
3. 图像必须有限；
4. 保存前允许投影到 `[0,1]`；
5. 图像必须 resize 到 HR 尺寸；
6. 所有图像保存前转为 `uint8`；
7. 输出必须是 PNG；
8. 不保存中间临时文件；
9. 返回 `Path`。

### 10.4 图像保存

灰度图像文件冻结为：

1. PNG；
2. 8-bit；
3. 模式 `L`；
4. 值域 `[0,1]`；
5. 保存前投影到 `[0,1]`；
6. 使用 `round(x*255)`；
7. 超出 `[0,1]` 的有限值投影后保存；
8. NaN/Inf 输入报错；
9. 不改变阶段二 PNG 格式；
10. 图像保存必须使用阶段二 `save_image_tensor` 的公开行为，或复用完全相同语义。

---

## 11. JSON 输出结构

### 11.1 指标 JSON

冻结函数：

```python
def save_metrics_json(
    data: Mapping[str, Any],
    path: str | Path,
) -> Path
```

要求：

1. UTF-8；
2. 缩进为 2；
3. 末尾换行；
4. `ensure_ascii=False`；
5. 禁止 NaN/Infinity JSON 标记；
6. `Path`、dataclass、tensor 在写入前转换为 JSON 可表示对象；
7. tensor 转换为 Python `float` 或 list；
8. 不压缩；
9. 父目录自动创建。

### 11.2 冻结 JSON 顶层字段

指标 JSON 至少包含：

```json
{
  "schema_version": "1.0",
  "config_name": "",
  "sample_name": "",
  "method": "GWF-C",
  "lambda_value": 0.0,
  "iterations": 0,
  "dt": 0.0,
  "metrics": {
    "psnr": 0.0,
    "ssim": 0.0,
    "lpips": null,
    "high_frequency_energy": 0.0,
    "high_frequency_energy_ratio": 0.0,
    "edge_preservation_index": 0.0,
    "edge_preservation_ratio": 0.0
  },
  "files": {
    "hr": "",
    "lr": "",
    "bicubic": "",
    "baseline": "",
    "gwf_c": "",
    "comparison_png": "",
    "convergence_png": "",
    "metrics_csv": ""
  },
  "dependencies": {
    "lpips_available": false
  }
}
```

规则：

1. `schema_version` 当前固定为 `"1.0"`；
2. 缺失 LPIPS 使用 JSON `null`；
3. 文件路径写入相对于运行目录的 POSIX 风格相对路径；
4. 不允许把绝对路径写入最终 JSON；
5. 不写入运行时间戳；
6. 不写入随机墙钟时间；
7. 配置字段由调用方显式传入；
8. 未知顶层字段必须写入 `"extra"`，不得静默丢弃。

---

## 12. CSV 输出结构

### 12.1 轨迹 CSV

冻结文件：

```text
trajectory_metrics.csv
```

列顺序冻结为：

```text
time,iteration,data_fidelity,entropy,regularization_energy,total_energy,curvature,update_norm,lambda_value
```

规则：

1. UTF-8；
2. 首行为 header；
3. 每行对应一个迭代点；
4. `iteration` 从 0 开始；
5. `time=iteration*dt`；
6. 数值使用 Python `repr` 或等价无损十进制序列化；
7. 不包含 NaN/Inf；
8. batch 数据必须在写入前完成聚合；
9. 默认聚合为 batch 均值；
10. CSV 中不得混入图像路径；
11. CSV 末尾必须换行；
12. 父目录自动创建。

### 12.2 方法指标 CSV

冻结文件：

```text
method_metrics.csv
```

列顺序冻结为：

```text
sample_name,method,lambda_value,psnr,ssim,lpips,high_frequency_energy,high_frequency_energy_ratio,edge_preservation_index,edge_preservation_ratio
```

规则：

1. 一行一个样本和方法；
2. 缺失 LPIPS 写入空字段，不写字符串 `nan`；
3. `lambda_value`、`psnr`、`ssim` 等字段写入有限浮点数；
4. 字符串字段不能包含未转义逗号；
5. CSV 写入必须正确处理逗号、引号和换行；
6. 不允许把 Python list 直接写入单元格；
7. batch 时按 `sample_name` 排序，同一样本按方法顺序排序；
8. 方法顺序默认为 `Baseline`、`GWF-C`；
9. 样本顺序默认为输入顺序；
10. 不改变输入结果顺序以外的语义；
11. CSV 末尾换行。

---

## 13. 可视化保存接口

### 13.1 图像对比保存

```python
def save_image_comparison(
    images: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    order: Sequence[str] = ("LR", "Bicubic", "Baseline", "GWF-C", "HR"),
    dpi: int = 150,
    figsize: tuple[float, float] = (20.0, 4.0),
    title: str | None = None,
) -> Path
```

语义：

1. 调用 `plot_method_comparison`；
2. 不重新定义布局；
3. 返回保存路径；
4. 允许与 `plot_method_comparison` 等价。

### 13.2 结果包保存

```python
def save_evaluation_bundle(
    output_dir: str | Path,
    *,
    config_name: str,
    sample_name: str,
    method: str,
    lambda_value: float,
    iterations: int,
    dt: float,
    hr: torch.Tensor,
    lr: torch.Tensor,
    bicubic: torch.Tensor,
    baseline: torch.Tensor,
    gwf_c: torch.Tensor,
    solver_result: SolverResult | Mapping[str, Any] | None = None,
    lpips_available: bool = False,
    require_lpips: bool = False,
) -> EvaluationBundle
```

冻结返回对象：

```python
@dataclass(frozen=True)
class EvaluationBundle:
    metrics: ImageMetrics
    metrics_json: Path
    metrics_csv: Path
    comparison_png: Path
    convergence_png: Path | None
```

规则：

1. 输出目录自动创建；
2. `save_evaluation_bundle` 是唯一负责一次运行完整文件集合的入口；
3. `solver_result=None` 时不生成收敛 CSV/PNG，但 `convergence_png=None`；
4. `solver_result` 存在时必须生成：
   - `trajectory_metrics.csv`；
   - `convergence.png`；
5. `metrics.csv` 永远生成；
6. `metrics.json` 永远生成；
7. `comparison.png` 永远生成；
8. 不覆盖其他样本或方法的文件；
9. 同一样本同方法重复保存时允许覆盖本方法目录内文件；
10. 返回路径全部是绝对路径还是相对路径必须固定为绝对 `Path`，但 JSON 内写入相对路径；
11. 不生成日志文件；
12. 不修改求解器结果。

### 13.3 输出目录结构

冻结目录结构：

```text
results/<config_name>/metrics/
  <sample_name>/
    <method>/
      metrics.json
      metrics.csv
      comparison.png
      convergence.png
      trajectory_metrics.csv
```

规则：

1. `<method>` 当前允许 `Baseline`、`GWF-C`；
2. `<sample_name>` 和 `<config_name>` 必须是安全文件名；
3. 路径中禁止空字符串、`.`、`..`、绝对路径、Windows 盘符；
4. 非法名称抛出 `ValueError`；
5. `trajectory_metrics.csv` 只在存在求解结果时写入；
6. 文件名冻结，不添加时间戳后缀；
7. 不在其他目录创建临时文件。

---

## 14. 参数校验

### 14.1 图像 pair 校验

所有图像 pair 必须校验：

1. tensor 类型；
2. 2-D 或 3-D；
3. 同 shape；
4. 同 dtype；
5. 同 device；
6. 非空；
7. 有限；
8. batch 一致；
9. 不允许 broadcast。

### 14.2 标量参数校验

以下参数必须是有限 Python `int` 或 `float`，不允许 bool：

1. `data_range`；
2. `win_sigma`；
3. `epsilon`；
4. `dpi`；
5. `lambda_value`；
6. `iterations`；
7. `dt`。

规则：

1. `data_range > 0`；
2. `win_sigma > 0`；
3. `epsilon > 0`；
4. `dpi > 0`；
5. `lambda_value >= 0`；
6. `iterations > 0`；
7. `dt > 0`；
8. NaN/Inf 抛出 `ValueError`；
9. 类型错误抛出 `TypeError`；
10. 不允许布尔值被当作整数。

### 14.3 离散参数校验

1. `win_size` 必须是奇数且 `>=3`；
2. `lpips_net` 只允许冻结枚举；
3. `lpips_version` 只允许 `"0.1"`；
4. `batch_mode` 只允许 `"mean"`、`"first"`；
5. `order` 必须包含 `LR`、`Bicubic`、`Baseline`、`GWF-C`、`HR`；
6. `fields` 只允许冻结曲线字段；
7. `dpi` 必须是正整数；
8. `figsize` 必须是两个正有限浮点数；
9. `config_name`、`sample_name`、`method` 必须是非空安全文件名；
10. `require_lpips` 必须是 bool。

---

## 15. LPIPS 缺失时的主流程

### 15.1 默认行为

1. `lpips_available=False` 时不计算 LPIPS；
2. `ImageMetrics.lpips` 为 `None`；
3. `metrics.json` 的 `metrics.lpips` 为 `null`；
4. `method_metrics.csv` 的 `lpips` 为空字段；
5. 其他指标必须正常计算；
6. 图像对比图必须正常生成；
7. 收敛曲线必须正常生成；
8. `save_evaluation_bundle` 不得因为 LPIPS 缺失失败。

### 15.2 显式要求

若调用方设置 `require_lpips=True`：

1. LPIPS 包缺失时抛出 `ImportError`；
2. 权重缺失时抛出 `RuntimeError`；
3. 不生成部分结果文件；
4. 不把缺失结果写成零；
5. 不降低验收标准。

### 15.3 可用性查询

允许实现提供：

```python
def lpips_available() -> bool
```

规则：

1. 只检查包和必要导入；
2. 不加载模型权重；
3. 不改变全局状态；
4. 不下载文件。

---

## 16. 收敛和对比图要求

### 16.1 收敛图

必须生成：

1. `convergence.png`；
2. 多子图布局；
3. 至少包含：
   - `Data Fidelity`；
   - `Entropy`；
   - `Regularization Energy`；
   - `Total Energy`；
   - `Curvature`；
   - `Update Norm`；
4. 若字段存在，可包含：
   - `KL`；
   - `Hilbert`；
5. x 轴标签为 `Time`；
6. 每个子图有 y 轴标签；
7. 每个子图有网格；
8. 方法名称写入图例；
9. 单次保存一个 PNG；
10. 不生成 SVG、PDF 或动画；
11. 不显示窗口。

### 16.2 方法对比图

必须生成：

1. `comparison.png`；
2. 五列顺序固定；
3. 每列标题固定；
4. 图像统一灰度显示；
5. 所有列 resize 到 HR 尺寸；
6. 不叠加文本水印；
7. 不加入误差热力图；
8. 不改变像素值；
9. 不显示窗口。

---

## 17. 复杂度与内存目标

### 17.1 数值指标复杂度

令 `N=H*W`，batch 大小为 `B`。

1. PSNR：`O(BN)`；
2. SSIM：`O(BN)`；
3. LPIPS：由模型决定，阶段六不承诺 `O(BN)`，必须记录为可选神经指标；
4. 高频能量：`O(BN)`；
5. 高频能量比：`O(BN)`；
6. 边缘保持：`O(BN)`；
7. batch 指标：`O(BN)`；
8. 指标内存额外开销：`O(BN)`。

### 17.2 可视化复杂度

1. 五图对比保存：`O(BN)`，其中 `B` 是展示图像数量；
2. 收敛曲线：`O(TB)`，`T` 为迭代数；
3. CSV/JSON 写入：`O(TB)` 或 `O(B)`，取决于文件类型；
4. 不创建 `N×N` 矩阵；
5. 不对每个像素使用 Python 循环；
6. 不在保存时保留全部历史梯度。

### 17.3 复杂度验收

阶段六测试必须验证：

1. PSNR、SSIM、HFE、EPI 等核心指标在 `N` 增大时近似线性；
2. 不生成全像素 pairwise 矩阵；
3. 不使用逐像素 Python 循环；
4. CSV/JSON 行数与迭代数或样本数线性相关；
5. 可视化函数不保留未关闭 figure；
6. 单次运行只写入约定文件，不生成额外大文件。

---

## 18. 依赖处理

### 18.1 必需依赖

阶段六必需：

1. `torch`；
2. `numpy`；
3. `Pillow`；
4. `matplotlib`；
5. `PyYAML`；
6. `pytest`，测试环境必需。

### 18.2 可选依赖

LPIPS：

1. `lpips`；
2. 其网络权重依赖；
3. 仅在调用 `lpips_distance(..., require=True)` 时强依赖；
4. 不改变阶段六主流程安装要求；
5. 不在 `requirements.txt` 中强制加入；
6. 若未来加入单独 `requirements-metrics.txt`，必须由 Lead 提交并另行验收。

### 18.3 禁止依赖

阶段六不得引入：

1. `torchvision` 作为 PSNR/SSIM 的隐式依赖；
2. OpenCV；
3. scikit-image；
4. 在线下载；
5. 交互式 GUI 后端；
6. 外部绘图服务。

---

## 19. 验收测试契约

### 19.1 PSNR

1. 相同图像 PSNR 为 `100.0`；
2. 不同图像 PSNR 有限；
3. 单图返回 0-D tensor；
4. batch 返回 `(B,)` tensor；
5. `data_range` 非法时报错；
6. NaN/Inf 输入报错；
7. dtype/device 保持正确。

### 19.2 SSIM

1. 相同图像 SSIM 接近 1；
2. 明显不同图像 SSIM 小于 1；
3. 单图返回 0-D tensor；
4. batch 返回 `(B,)` tensor；
5. 非奇数 `win_size` 报错；
6. 图像尺寸小于窗口时报错；
7. 高斯窗口、边界、协方差归一化符合冻结定义；
8. dtype/device 保持正确。

### 19.3 LPIPS

1. `lpips` 缺失时 `require=False` 返回 `None`；
2. `lpips` 缺失时 `require=True` 抛出 `ImportError`；
3. 其他指标在 LPIPS 缺失时继续工作；
4. LPIPS 存在时结果有限；
5. 灰度输入自动复制为三通道；
6. 未知网络和版本报错；
7. 不下载权重；
8. 不把缺失结果写成零。

### 19.4 高频和边缘

1. 常数图像 HFE 为 0；
2. 梯度增大时 HFE 非降；
3. 相同图像 HFER 接近 1；
4. 相同图像 EPI 接近 1；
5. 相同图像 EPR 接近 1；
6. batch shape 正确；
7. 结果有限；
8. 不修改输入；
9. 边界规则和中心差分与冻结定义一致。

### 19.5 CSV/JSON/图像

1. JSON 可被标准解析器读取；
2. JSON 中不存在 NaN/Infinity；
3. 缺失 LPIPS 在 JSON 中为 null；
4. CSV header 顺序正确；
5. 轨迹 CSV 行数等于迭代点数；
6. 方法 CSV 行数等于样本数乘方法数；
7. PNG 文件存在且可被 PIL 打开；
8. 图像保存使用灰度 8-bit；
9. 输出目录结构符合冻结路径；
10. 重复调用不产生额外临时文件；
11. 返回路径正确。

### 19.6 收敛和对比图

1. 收敛图字段缺失时跳过而不是伪造；
2. 收敛图 x 轴为 times；
3. 对比图包含五个方法；
4. 缺失方法报错；
5. 所有子图 resize 到 HR 尺寸；
6. 标题和列顺序正确；
7. 不显示窗口；
8. figure 被关闭；
9. 输出文件可被 PIL 打开。

### 19.7 复杂度

1. 指标不创建 `N×N` tensor；
2. 指标不使用逐像素 Python 循环；
3. `N` 增大时 PSNR/SSIM/HFE 近似线性；
4. 保存路径只写入约定文件；
5. 文件数量和内容符合预期；
6. batch 计算不跨样本平均。

---

## 20. 冻结决策摘要

| 决策项 | 阶段六冻结结果 |
|---|---|
| 输入 shape | `H×W` 或 `B×H×W` |
| dtype | 仅 `float32`、`float64` |
| device | 数值指标跟随输入；保存前转 CPU |
| 输出格式 | 单图 0-D tensor，batch `(B,)` tensor |
| PSNR | `10 log10(MAX/MSE)`，默认 `MAX=1` |
| 零 MSE | PSNR 哨兵值 `100.0` |
| SSIM | 单通道，`11×11` 高斯窗，`sigma=1.5` |
| SSIM 边界 | `reflect` |
| LPIPS | 可选，默认 `alex`、`0.1`、输入 `[0,1]` |
| LPIPS 缺失 | `None`，JSON null，CSV 空字段 |
| LPIPS 强依赖 | `require=True` 时 `ImportError`/`RuntimeError` |
| 高频能量 | 中心差分梯度幅值平方均值 |
| 边缘保持 | 梯度幅值相关性 |
| 收敛曲线 | 时间横轴，冻结诊断字段 |
| 对比图 | LR、Bicubic、Baseline、GWF-C、HR 五列 |
| 图像格式 | 8-bit 灰度 PNG |
| JSON | UTF-8、2 空格、无 NaN/Infinity |
| CSV | UTF-8、固定 header、无 NaN/Infinity |
| 输出目录 | `results/<config>/metrics/<sample>/<method>/` |
| 复杂度 | PSNR/SSIM/HFE/EPI 为 `O(BN)` |
| 禁止依赖 | torchvision、OpenCV、skimage、在线下载 |
| 核心实现 | 只在用户确认后由 developer 编写 |

---

## 21. 完成条件

阶段六设计在满足以下条件后可进入实现：

1. 用户确认本文档；
2. Lead 向 developer 传达冻结契约；
3. developer 只新增 `gwf_sr/metrics.py`、`gwf_sr/visualization.py` 和阶段六测试；
4. 不修改阶段二至阶段五模块；
5. LPIPS 可选依赖不得阻断主流程；
6. 实现后由 verifier 独立验收；
7. 任何契约变更必须由 architect 重新起草并由 Lead 提交。
