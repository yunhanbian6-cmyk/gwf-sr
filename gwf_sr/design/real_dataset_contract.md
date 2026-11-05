# Real Image & Standard Dataset Contract (Stage 13)

**Status:** Frozen — implemented in `gwf_sr/dataset.py`  
**Commit:** `44c45d0f685c96591bdcf4ef03bf01a20b7a2b5e`

---

## 1. Scope

Stage 13 extends the GWF-SR pipeline from synthetic grayscale images to real images and standard SR benchmarks.

**In scope:**
- Local image loading (PNG/JPG/BMP/TIF/WebP)
- Grayscale and RGB support
- Standard datasets: Set5, Set14, BSD100, Urban100
- Degradation pipeline integration
- Batch loading and preparation

**Out of scope:**
- Neural network SR models (stage 14)
- Training loops
- Online dataset downloads

---

## 2. Image Loading Contract

### 2.1 `load_image_tensor(path, grayscale=True, max_size=None) -> Tensor`

| Property | Value |
|---|---|
| Grayscale output | `H×W` float32 |
| RGB output | `3×H×W` float32 (channel-first) |
| Value range | `[0, 1]` |
| `max_size` | Resize longer side to this size (BICUBIC), only if `scale < 1.0` |
| Supported formats | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp` |
| Missing file | `FileNotFoundError` |
| Unsupported format | `ValueError` |
| Non-8-bit input | Converted via PIL `convert("L")` or `convert("RGB")` |

### 2.2 `save_image_tensor(tensor, path) -> Path`

| Input | Output |
|---|---|
| `H×W` float32 | 8-bit grayscale PNG (mode `L`) |
| `3×H×W` float32 | 8-bit RGB PNG (mode `RGB`) |
| Other shapes | `ValueError` |
| Value range | Clamped to `[0, 1]`, then `round(x*255)` |

---

## 3. Dataset Loading Contract

### 3.1 `load_image_directory(directory, grayscale=True, max_size=None) -> list[(name, Tensor)]`

- Loads all supported images from a directory;
- Sorted by filename;
- `name` = filename stem;
- Empty directory → `ValueError`;
- Non-directory → `NotADirectoryError`.

### 3.2 `load_standard_dataset(name, root, grayscale=True, max_size=None) -> list[(name, Tensor)]`

| Dataset | Expected path | Images |
|---|---|---|
| Set5 | `<root>/Set5/` | 5 |
| Set14 | `<root>/Set14/` | 14 |
| BSD100 | `<root>/BSD100/` | 100 |
| Urban100 | `<root>/Urban100/` | 100 |

- Unknown dataset → `ValueError`;
- Missing directory → `FileNotFoundError` with download hint;
- Images loaded via `load_image_directory`.

### 3.3 `load_local_images(paths, grayscale=True, max_size=None) -> list[(name, Tensor)]`

- Loads specific files by path;
- `name` = filename stem.

---

## 4. Degradation Contract

### 4.1 `degrade_and_prepare(hr, scale=2, sigma=1.0, noise_std=0.0, seed=42) -> (lr, bicubic)`

| Input | LR shape | Bicubic shape |
|---|---|---|
| `H×W` | `H/scale × W/scale` | `H×W` |
| `3×H×W` | `3×H/scale × W/scale` | `3×H×W` |

- Grayscale: single-pass degradation;
- RGB: per-channel independent degradation;
- Uses `gwf_sr.degradation.degrade` + `bicubic_upsample`;
- Deterministic for fixed seed;
- Reproducible across calls.

---

## 5. Complexity

- Image loading: `O(H×W)` per image;
- Degradation: `O(H×W)` per channel;
- No `N×N` intermediate matrices;
- Batch loading: `O(B×H×W)`.

---

## 6. Testing Contract

| Test class | Coverage |
|---|---|
| `TestLoadImageTensor` | Grayscale, RGB, max_size, missing file, unsupported format |
| `TestSaveImageTensor` | Grayscale, RGB, bad shape |
| `TestLoadImageDirectory` | Multiple images, empty dir, non-dir |
| `TestStandardDataset` | Unknown name, missing directory |
| `TestLoadLocalImages` | Multiple paths |
| `TestDegradeAndPrepare` | Grayscale, RGB, reproducibility |

---

## 7. Non-Goals

- No EXIF auto-rotation (PIL default behavior);
- No silent rescaling (only explicit `max_size`);
- No color space conversion beyond L/RGB;
- No dataset downloading;
- No modification of stages 2–6 modules.
