# Graph Structure, Curvature, and Regularized Energy Contract

**Status:** Draft for user confirmation; no stage-four core implementation is included.  
**Scope boundary:** This document freezes the mathematical and API contract only. It does not modify the stage-two data pipeline, the accepted stage-three geometry module, or any existing tests. The planned implementation module is `gwf_sr/graph.py`; this document does not create or modify that file.

---

## 1. Purpose

Stage four introduces the discrete graph geometry needed by the GWF-C solver while keeping all numerical objects compatible with the frozen stage-three probability geometry:

- stage two produces `H×W` or `B×H×W` image tensors;
- stage three converts finite image tensors into strictly positive probability distributions;
- stage four constructs a finite undirected pixel graph, its normalized diffusion operator, its entropy/diffusion energy, its curvature quotient, and its O(N) diagnostics;
- stage five will implement the continuous flow and its time discretization using the frozen stage-four operators.

The stage-four module must be a pure geometry layer. It must not perform super-resolution iterations, read files, read global configuration, or modify existing modules.

---

## 2. Frozen symbols and conventions

For one image:

- `H`: image height;
- `W`: image width;
- `N = H * W`: number of graph nodes and probability states;
- `i`, `j`, `k`: zero-based pixel/node indices;
- `rho`, `eta`: finite probability tensors with the stage-three shape contract;
- `lambda`: nonnegative regularization strength, matching `GWFConfig.lambda_values`;
- `epsilon`: strictly positive stage-three smoothing constant;
- `degree(v) = sum_{u} w_{vu}`: weighted degree of node `v`;
- `edge_count`: number of undirected edges after de-duplication.

For batch inputs, every sample in `B×H×W` is treated as an independent image graph with its own `H`, `W`, and `N`. No operation may share reductions across the batch dimension.

All graph quantities are defined for one undirected, finite, positive-weight graph. A disconnected graph is permitted unless a specific API says otherwise; curvature and spectral diagnostics must not assume connectedness silently. Unless an API explicitly operates on node-level fields, graph metrics require at least one undirected edge.

---

## 3. Input, dtype, and device contract

### 3.1 Shape

Every stage-four tensor input accepts only:

1. `H×W`;
2. `B×H×W`.

The edge index table has a separate explicit shape, described in Section 5.3. Invalid shapes raise `ValueError`; non-tensors raise `TypeError`.

### 3.2 dtype

Only `torch.float32` and `torch.float64` are accepted for floating-point tensors. Integer, boolean, complex, `float16`, and `bfloat16` inputs raise `TypeError`.

Shape, dtype, and device preservation rules are inherited from the stage-three geometry contract:

- a `float32` input produces `float32` outputs;
- a `float64` input produces `float64` outputs;
- no function silently casts a tensor;
- all outputs remain on the input device.

### 3.3 Finite values and negatives

All tensor inputs must be finite. NaN or Inf raises `ValueError` before any graph computation.

Negative values are treated as follows:

1. image and distribution tensors are never silently clipped by graph construction;
2. edge weights must be nonnegative by definition and are validated explicitly;
3. a distribution used with entropy, energy, or curvature must be strictly positive because logarithms are part of the frozen definitions;
4. a negative or zero edge weight, degree, denominator, or curvature quotient raises `ValueError` rather than propagating a silently invalid graph.

This preserves the stage-three rule that negative image values do not crash conversion, while making invalid graph geometry fail at the graph API boundary.

### 3.4 Empty tensors

`H=0`, `W=0`, `B=0`, an empty edge set where an edge set is required, or an empty graph for a metric requiring at least one edge raises `ValueError`. A single-node graph is valid for distributions but has no undirected edge and therefore cannot provide edge-based total variation or diffusion energy.

---

## 4. Pixel graph construction

### 4.1 Node indexing

The graph has exactly one node per pixel. The frozen raster mapping is:

\[
\operatorname{index}(r,c)=rW+c,\qquad 0\le r<H,\quad 0\le c<W.
\]

The inverse mapping is the floor division and remainder:

\[
r=\operatorname{index}(r,c)\operatorname{div}W,\qquad
c=\operatorname{index}(r,c)\operatorname{mod}W.
\]

No diagonal edges, wrapping edges, or boundary-to-boundary edges are introduced. The graph is undirected: if `(i,j)` exists, `(j,i)` must be represented as the reverse directed incidence for computation but must count as one undirected edge in `edge_count`.

### 4.2 Uniform mode

`GWFConfig.edge_weight_mode == "uniform"` is the stage-four default and the only mode required for the initial implementation.

For each horizontally or vertically adjacent pixel pair:

\[
w_{ij}=1.
\]

This produces:

- `2H(W-1)` horizontal edges;
- `2W(H-1)` vertical edges as directed incidences;
- `edge_count = H(W-1) + W(H-1)` undirected edges.

The uniform graph is regular in its interior, with boundary degree 3 or 4 and corner degree 2.

### 4.3 Gradient mode

`GWFConfig.edge_weight_mode == "gradient"` is a reserved future mode. Its contract is frozen now so that stage four does not later change the meaning of configuration values:

1. gradient mode is not required for the first stage-four implementation;
2. if implemented later, it must use a nonnegative, deterministic edge-weight rule;
3. it must be documented as a separate weight formula before use in experiments;
4. it must not silently fall back to uniform mode;
5. any unknown mode raises `ValueError`.

No stage-four test may require gradient mode unless that mode is explicitly implemented.

### 4.4 Edge validation

For every directed incidence `(u,v)`:

1. `0 <= u < N` and `0 <= v < N`;
2. `u != v`;
3. `w_{uv}` is finite;
4. `w_{uv} >= 0`;
5. `w_{uv} = w_{vu}` for the undirected graph;
6. `degree(v) > 0` for every node used by a normalized diffusion operator.

`edge_weight` must use a supported floating dtype; integer weights raise `TypeError`. An isolated node is allowed only when the API is explicitly diagnostic and does not divide by its degree. Any normalized operator must either reject isolated nodes or define its behavior before accepting them; it may not emit NaN or Inf.

---

## 5. Frozen graph interface

The following API names and return semantics are frozen. Stage five and later stages may reuse them without redefining their meaning.

### 5.1 `build_pixel_graph`

```python
def build_pixel_graph(
    shape: tuple[int, int],
    edge_weight_mode: str = "uniform",
) -> PixelGraph:
```

`shape` is `(H, W)`. The function returns a `PixelGraph` descriptor with frozen fields:

- `height`;
- `width`;
- `node_count`;
- `edge_count`;
- `edge_weight_mode`;
- `dtype`;
- `device`;
- `source_index`;
- `target_index`;
- `edge_weight`.

The descriptor must not be a dense `N×N` matrix. Its floating-point tensors use `torch.float32` by default and remain on the current default tensor device unless an implementation API explicitly selects `torch.float64`; metric outputs then follow the input distribution or value tensor, not the cached graph.

### 5.2 `build_batch_pixel_graphs`

```python
def build_batch_pixel_graphs(
    images: torch.Tensor,
    edge_weight_mode: str = "uniform",
) -> PixelGraphBatch:
```

This is a convenience wrapper for `B×H×W` tensors. It uses the dtype and device of `images` for all edge tables and constructs one graph per sample. It returns an object with:

- `batch_size`;
- `height`;
- `width`;
- `node_count`;
- `edge_count`;
- `edge_weight_mode`;
- `dtype`;
- `device`;
- `source_index`;
- `target_index`;
- `edge_weight`;
- `batch_index`.

`batch_index` is required when edges are concatenated across samples. Each edge belongs to exactly one sample and cannot connect two different batch elements.

### 5.3 Edge index representation

For `E` directed incidences, the edge index table is:

- `source_index`: integer tensor of shape `(E,)`;
- `target_index`: integer tensor of shape `(E,)`;
- `edge_weight`: floating tensor of shape `(E,)`;
- optional `batch_index`: integer tensor of shape `(E,)` for batched graphs.

The edge index table is the only permitted representation of graph sparsity. It must satisfy:

- `E = 2 * edge_count` for an undirected graph, because both directions are explicit;
- integer index tensors use `torch.int64`;
- `edge_weight` uses a supported floating dtype;
- indices are valid for the graph's `node_count`;
- no self-loops;
- reverse pairs have equal weights;
- the edge order is deterministic, with horizontal edges first in raster order followed by vertical edges in raster order, unless a future contract revision explicitly freezes a different order.

A dense adjacency tensor is allowed only inside a small test and never as a public stage-four output or intermediate required by a metric.

---

## 6. Normalized diffusion operator

For each directed incidence `(u,v)` with weight `w_{uv}`, define the symmetric normalization:

\[
\widehat w_{uv}
=
\frac{w_{uv}}{\sqrt{\operatorname{degree}(u)\operatorname{degree}(v)}}.
\]

Define the symmetric normalized adjacency by:

\[
(A_{\mathrm{norm}}f)_u
=
\sum_{v\sim u}\widehat w_{uv} f_v.
\]

The following properties are frozen:

1. `A_norm` is self-adjoint under the standard Euclidean inner product;
2. every eigenvalue lies in `[-1,1]`;
3. a connected bipartite component may have eigenvalue `-1`;
4. a connected non-bipartite component has a strictly positive Perron eigenvector;
5. a disconnected graph has one spectral result per component.

### 6.1 `normalized_adjacency_action`

```python
def normalized_adjacency_action(
    values: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
) -> torch.Tensor:
```

The function applies the sparse normalized adjacency to `values`. It returns a tensor with the same shape as `values`, and it must not form `N×N` matrices.

### 6.2 Degree and normalization diagnostics

```python
def graph_degree(graph) -> torch.Tensor
def graph_total_weight(graph) -> torch.Tensor
def graph_spectral_radius_bound(graph) -> torch.Tensor
```

`graph_degree` returns one degree per node. For a batched graph, it returns shape `(B,H,W)`. `graph_total_weight` returns the sum over unique undirected edges, not twice the directed sum; for a single graph it is a 0-D tensor and for a batch it is shape `(B,)`. `graph_spectral_radius_bound` returns a tensor containing `1.0` for the symmetric normalized adjacency and is not an eigensolver call.

---

## 7. Entropy and Dirichlet energy

### 7.1 Dirichlet energy

For a scalar field `f` on one graph, the undirected Dirichlet energy is:

\[
\mathcal{E}_{D}(f)
=
\frac{1}{2}\sum_{(u,v)\in E}w_{uv}(f_u-f_v)^2.
\]

The factor `1/2` counts each undirected edge once even though the implementation stores both directions. For a batch, the result is one value per sample, shape `(B,)`.

### 7.2 Normalized graph entropy

For a strictly positive probability vector `rho` and symmetric normalized adjacency `A_norm`, define:

\[
\mathcal{H}(\rho)
=
-\sum_u \rho_u\log\rho_u
+
\frac{\lambda}{2}
\sum_{(u,v)\in E}
\widehat w_{uv}
(\log\rho_u-\log\rho_v)^2.
\]

Equivalently, the second term is:

\[
\frac{\lambda}{2}
\mathcal{E}_{D}(\log\rho)
\]

with the symmetric normalized weights. This is the frozen regularized entropy for stage four. It is not the same object as the unregularized stage-three entropy; stage-three entropy remains unchanged.

At `lambda = 0`, the stage-four regularized entropy reduces exactly to stage-three entropy, up to floating-point error.

### 7.3 Energy API

```python
def entropy_energy(
    distribution: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
    lambda_value: float,
) -> torch.Tensor
```

Rules:

1. `distribution` must be strictly positive and finite;
2. `lambda_value` must be a finite nonnegative Python scalar;
3. single-image input returns a 0-D tensor;
4. batch input returns shape `(B,)`;
5. no cross-sample averaging occurs inside the function;
6. `lambda_value=0` is exactly the unregularized entropy of the same distribution.

### 7.4 Diffusion energy API

```python
def dirichlet_energy(
    values: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
) -> torch.Tensor
```

This returns the unweighted-by-`lambda` energy above. It does not include entropy, data fidelity, or a normalization by total mass.

---

## 8. Entropy production and curvature quotient

### 8.1 Entropy production

For a strictly positive probability vector, define the nonnegative entropy production:

\[
\mathcal{I}(\rho)
=
\sum_{(u,v)\in E}
\widehat w_{uv}
\left(
\frac{\rho_u-\rho_v}{\sqrt{\rho_u\rho_v}}
\right)^2.
\]

This is the frozen stage-four production diagnostic for the symmetric normalized graph diffusion. It is nonnegative, vanishes exactly on each connected component when `rho` is constant on that component, and is invariant to batch-independent scaling only when the input is already normalized.

For a batch, it returns shape `(B,)`.

### 8.2 Curvature quotient

For `lambda >= 0`, define the curvature quotient:

\[
K_\lambda(\rho)
=
\frac{
\lambda\,\mathcal{I}(\rho)
}{
\mathcal{H}(\rho)
}
\quad\text{when }\mathcal{H}(\rho)>0,
\]

and:

\[
K_0(\rho)=0.
\]

The quotient is not defined when the denominator is nonpositive or nonfinite; such inputs raise `ValueError` rather than returning Inf or NaN.

The implementation must use the same `rho`, graph, and normalization as `entropy_energy`. It must not re-normalize the distribution or switch to an unnormalized graph Laplacian.

### 8.3 Curvature API

```python
def entropy_production(
    distribution: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
) -> torch.Tensor

def curvature_quotient(
    distribution: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
    lambda_value: float,
) -> torch.Tensor
```

Return rules:

1. single image returns a 0-D tensor;
2. batch returns `(B,)`;
3. `lambda_value=0` returns zero;
4. no clipping to a prescribed theoretical upper or lower bound is permitted;
5. a finite but negative quotient is an error only if it results from an invalid graph or distribution; the formulas above should produce a nonnegative value for valid inputs.

---

## 9. Total variation and smoothing diagnostics

For a scalar field `f`, the undirected total variation is:

\[
\operatorname{TV}(f)
=
\frac{1}{2}
\sum_{(u,v)\in E}
w_{uv}|f_u-f_v|.
\]

For a probability distribution, the frozen smoothing diagnostic is:

\[
\operatorname{Smoothness}(\rho)
=
\frac{\operatorname{TV}(\log\rho)}
{\operatorname{TV}(\rho)+\epsilon_{\mathrm{diag}}}.
\]

Here `epsilon_diag` is a diagnostic-only floor with the same positivity and dtype/device conversion rules as the stage-three `epsilon`. It must never replace the epsilon used in `distribution_from_image`.

The following APIs are frozen:

```python
def total_variation(
    values: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
) -> torch.Tensor

def log_ratio_smoothness(
    distribution: torch.Tensor,
    graph: PixelGraph | PixelGraphBatch,
    epsilon: float = 1e-8,
) -> torch.Tensor
```

Both return one value per image. They are diagnostics, not solver objectives.

---

## 10. Curvature function and exponential decay diagnostic

The report's curvature function is frozen as:

\[
a(R)
=
\frac{R^2e^R}{e^R-1-R},
\qquad a(0)=2.
\]

For `R > 0`, this function is finite and strictly greater than `2`. The implementation must use a numerically stable branch near `R=0` and must not divide by zero. Negative `R` is outside the domain of the stage-four diagnostic and raises `ValueError`.

```python
def curvature_acceleration(ratio: torch.Tensor) -> torch.Tensor
```

For a tensor of finite nonnegative ratios, it applies `a(R)` elementwise and preserves shape and dtype.

The associated decay diagnostic is:

\[
D_\lambda(R_0,t)
=
\lambda
+
\lambda\,a(R_0e^{-\lambda t}),
\]

with:

```python
def entropy_decay_rate(
    lambda_value: float,
    initial_ratio: torch.Tensor | float,
    time: torch.Tensor | float,
) -> torch.Tensor
```

`time` must be finite and nonnegative. `initial_ratio` must be finite and nonnegative. For `lambda_value=0`, the rate is exactly zero. This function is a diagnostic for later convergence plots; it does not evolve a distribution by itself.

---

## 11. Numerical and complexity constraints

### 11.1 Sparse representation

All graph metrics must operate through edge lists or equivalent sparse reductions. No stage-four public function may return or require a dense `N×N` adjacency, covariance, kernel, or Hessian matrix.

### 11.2 Complexity

For one image with `N` nodes and `E = O(N)` grid edges:

1. graph construction is `O(N)`;
2. every energy, production, total-variation, and curvature-quotient evaluation is `O(E)=O(N)`;
3. memory overhead beyond inputs and outputs is `O(N)`;
4. for a batch, the corresponding bound is `O(BN)` time and memory;
5. no power iteration, full eigendecomposition, or dense Hessian is required in stage four.

### 11.3 Stable reductions

1. use tensor reductions over the edge list;
2. do not use Python scalar loops over pixels or edges;
3. preserve the input device;
4. keep the undirected-edge factor `1/2` in exactly one place per metric;
5. use log-domain differences for `log(rho_u)-log(rho_v)` when possible;
6. do not add a new epsilon to the mathematical energy terms.

### 11.4 Determinism

For the same input tensor, shape, mode, lambda, and epsilon, outputs must be deterministic on the same device and dtype. Edge ordering must be deterministic. No random initialization is permitted in stage-four graph or curvature APIs.

---

## 12. Configuration mapping

`GWFConfig` remains the source of experiment-level values:

| Configuration field | Stage-four use |
|---|---|
| `edge_weight_mode` | `build_pixel_graph` mode; `uniform` is required |
| `lambda_values` | values passed as `lambda_value` to energy and curvature APIs |
| `epsilon` | distribution smoothing and diagnostic floor |
| `device` | solver-level placement; graph functions follow input tensors |
| `iterations` | not used by stage-four geometry; reserved for stage five |
| `dt` | not used by stage-four geometry; reserved for stage five |

No stage-four function reads a YAML file or a global configuration object. The caller passes the selected field values explicitly.

---

## 13. Non-goals and module boundaries

Stage four must not:

1. implement a solver or time integrator;
2. compute PSNR, SSIM, or LPIPS;
3. load or save images;
4. alter the stage-two degradation pipeline;
5. alter the accepted stage-three geometry module or its tests;
6. use dense graph matrices in public APIs;
7. claim that a finite numerical graph satisfies a continuous curvature theorem without an explicit experiment;
8. change `GWFConfig` or existing YAML files.

The planned implementation module is `gwf_sr/graph.py`. It may import `torch` and reuse the frozen stage-three geometry helpers, but it must not create a circular dependency. This document does not create or modify that implementation file.

---

## 14. Required test contract

Stage-four tests must cover the following frozen behavior.

### 14.1 Graph structure

1. `H×W` uniform graph has `N=H*W` nodes;
2. uniform edge count is `H(W-1)+W(H-1)`;
3. every undirected edge appears exactly once in `edge_count` and twice in the directed incidence table;
4. no self-loops or diagonal edges exist;
5. node indexing obeys `index=r*W+c`;
6. invalid mode, negative weights, and out-of-range indices raise `ValueError`;
7. batched edge indices never connect different samples.

### 14.2 Energy and entropy

1. constant fields have zero Dirichlet energy and total variation;
2. `lambda=0` reproduces stage-three entropy;
3. positive distributions yield finite energy and production;
4. invalid distributions raise `ValueError`;
5. single-image and batch return shapes are `0-D` and `(B,)`;
6. energy values are deterministic and nonnegative where required.

### 14.3 Curvature and diagnostics

1. `K_0=0`;
2. `K_lambda` is nonnegative for valid inputs;
3. identical distributions on each connected component give zero production;
4. curvature function satisfies `a(0)=2` within the selected dtype tolerance;
5. `entropy_decay_rate` at `lambda=0` returns zero;
6. no API returns NaN or Inf for finite valid inputs;
7. total variation and log-smoothness preserve batch shape.

### 14.4 Complexity

1. no test or implementation path creates an `N×N` tensor;
2. a simple benchmark or structural inspection confirms linear scaling in `N`;
3. `float32` and `float64` outputs preserve dtype and device.

---

## 15. Frozen decision summary

| Decision | Frozen contract |
|---|---|
| Node set | one node per pixel |
| Node index | `r*W+c` |
| Graph type | finite undirected grid graph |
| Uniform edge weight | `1.0` |
| Uniform edge count | `H(W-1)+W(H-1)` |
| Edge storage | sparse edge list, no dense `N×N` public matrix |
| Batch isolation | independent graph and reductions per sample |
| dtype | only `float32` and `float64` |
| device | output follows input tensor device |
| NaN/Inf | raise `ValueError` |
| Negative image values | accepted by stage three; graph weights must still be nonnegative |
| Normalized operator | symmetric degree-normalized adjacency |
| Dirichlet energy | `1/2 sum w(f_u-f_v)^2` |
| Entropy energy | stage-three entropy plus `lambda/2` normalized log-gradient energy |
| Entropy production | nonnegative normalized squared log-density flux |
| Curvature quotient | `K_lambda=lambda*I/H`, with `K_0=0` |
| Curvature function | `a(R)=R^2e^R/(e^R-1-R)`, `a(0)=2` |
| Complexity | `O(N)` time and memory per image, since `E=O(N)` |
| Forbidden intermediate | any `N×N` matrix |
| Solver scope | stage five only |

---

## 16. Stage-four completion criteria

Stage four is complete only when:

1. `gwf_sr/graph.py` implements the frozen APIs;
2. all stage-four tests pass;
3. no stage-two or stage-three module is modified;
4. no dense graph matrix is used in the production path;
5. configuration fields map to the frozen meanings above;
6. the resulting commit message is recorded as `stage4-contract: ...` or a later implementation-specific message approved by the Lead;
7. the design contract is preserved unless a later revision explicitly freezes a change.
