# Siamese Contrastive Encoder for Cryo-EM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Siamese 对比学习编码器（双分支：实空间+频域），用模拟数据跑通数据生成→训练→检索→评估完整链路。

**Architecture:** 标准 Python 包 `siamese/`，包含 models、data、losses、training、eval、utils 六个子包。脚本在 `scripts/`，配置在 `configs/`。使用 ConvNeXt-Tiny 作为 backbone，双分支共享架构但权重独立，融合后投影到 128 维 embedding，InfoNCE loss 训练，FAISS 检索评估。

**Tech Stack:** Python 3.12, PyTorch, torchvision, timm (ConvNeXt), FAISS, healpy, mrcfile, numpy, pyyaml, tqdm, matplotlib

---

## 文件结构总览

```
Siamese/
├── siamese/                          # Python 包
│   ├── __init__.py                   # 空
│   ├── models/
│   │   ├── __init__.py               # 导出 encoder, backbone
│   │   ├── backbone.py              # build_backbone(): ConvNeXt/ViT/Swin 工厂函数
│   │   ├── encoder.py               # SiameseEncoder: 双分支 + 融合 + 投影头
│   │   └── fusion.py                # FusionHead: concat + Linear+BN+ReLU + Linear
│   ├── data/
│   │   ├── __init__.py               # 导出 MicProjDataset, 生成函数
│   │   ├── dataset.py               # MicProjDataset: 从预处理数据加载 (mic, proj)
│   │   ├── generate.py              # 生成模拟数据: HEALPix投影 + CTF + noise
│   │   └── transforms.py            # 归一化 + FFT 变换
│   ├── losses/
│   │   ├── __init__.py               # 导出 InfoNCE
│   │   └── infonce.py               # InfoNCELoss: 对称 NT-Xent
│   ├── training/
│   │   ├── __init__.py               # 导出 Trainer, TrainConfig
│   │   ├── config.py                # TrainConfig: 训练超参数 dataclass
│   │   └── trainer.py               # Trainer: 训练循环 + 验证 + 保存
│   ├── eval/
│   │   ├── __init__.py               # 导出检索和指标函数
│   │   ├── retrieval.py             # build_faiss_index(), retrieve_topk()
│   │   └── metrics.py               # compute_accuracy_at_k(), plot_retrieval()
│   └── utils/
│       ├── __init__.py               # 导出 ctf, fft 工具
│       ├── ctf.py                    # compute_ctf(): 生成 CTF 调制
│       └── fft.py                    # image_to_freq(): 实部+虚部2通道输出
├── scripts/
│   ├── generate_data.py              # 命令行: 生成模拟数据到 data/
│   ├── train.py                      # 命令行: 训练模型
│   └── eval.py                       # 命令行: 评估检索性能
├── configs/
│   └── default.yaml                  # 默认配置文件
├── tests/
│   ├── test_encoder.py               # 编码器测试
│   ├── test_dataset.py               # 数据集测试
│   ├── test_loss.py                  # Loss 测试
│   ├── test_ctf.py                   # CTF 生成测试
│   └── test_retrieval.py             # 检索测试
├── pyproject.toml                    # 项目配置
├── README.md                         # 项目说明
├── data/                             # 数据目录 (gitignore)
├── checkpoints/                      # 模型保存 (gitignore)
└── project.py                        # 已有: 投影函数 (保留不动)
```

所有 TODO 标记集中在文件顶部或函数 docstring 中，便于后续查找。

---

### Task 1: 项目初始化 — pyproject.toml + 依赖安装 + 目录结构 + .gitignore

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: 所有空 `__init__.py` 和目录

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "siamese-cryoem"
version = "0.1.0"
description = "Siamese contrastive encoder for cryo-EM particle retrieval"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.5",
    "torchvision",
    "timm>=0.9",
    "numpy",
    "healpy",
    "mrcfile",
    "pyyaml",
    "tqdm",
    "matplotlib",
    "faiss-cpu",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["siamese*"]
```

- [ ] **Step 2: 创建 .gitignore**

```
data/
checkpoints/
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
*.egg-info/
dist/
build/
.DS_Store
*.swp
*.swo
```

- [ ] **Step 3: 创建目录结构和空 __init__.py**

```bash
mkdir -p siamese/models siamese/data siamese/losses siamese/training siamese/eval siamese/utils
mkdir -p scripts configs tests data checkpoints
touch siamese/__init__.py
touch siamese/models/__init__.py siamese/data/__init__.py siamese/losses/__init__.py
touch siamese/training/__init__.py siamese/eval/__init__.py siamese/utils/__init__.py
```

- [ ] **Step 4: 创建 README.md**

```markdown
# Siamese Cryo-EM Particle Retrieval

Siamese contrastive encoder for matching low-SNR micrographs to clean projections.

## Installation

```bash
uv pip install -e .  # in existing .venv
```

## Quick Start

```bash
python scripts/generate_data.py emd_19110.map 8 --output-dir data/
python scripts/train.py --config configs/default.yaml
python scripts/eval.py --checkpoint checkpoints/best.pt --data-dir data/
```
```

- [ ] **Step 5: 安装依赖**

```bash
cd /Data/Work/Siamese
uv pip install -e ".[dev]"
```

- [ ] **Step 6: 验证安装**

```bash
.venv/bin/python -c "import torch; import timm; import healpy; import mrcfile; import faiss; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml .gitignore README.md siamese/ configs/ scripts/ tests/
git commit -m "chore: initialize project structure and dependencies"
```

---

### Task 2: CTF 生成工具

**Files:**
- Create: `siamese/utils/ctf.py`
- Modify: `siamese/utils/__init__.py`
- Create: `tests/test_ctf.py`

- [ ] **Step 1: 编写 CTF 生成的失败测试**

创建 `tests/test_ctf.py`:

```python
"""CTF 生成函数的测试。"""

import torch
import pytest
from siamese.utils.ctf import compute_ctf


def test_compute_ctf_shape_and_dtype():
    """测试 CTF 输出的形状和数据类型。"""
    D = 128
    ctf = compute_ctf(
        image_size=D,
        pixel_size=1.0,       # Å/pixel
        defocus=2.0,          # μm
        cs=2.7,               # mm, 球差系数
        voltage=300.0,        # kV
        amplitude_contrast=0.1,
    )
    # 输出应为复数 CTF，形状 [D, D]
    assert ctf.shape == (D, D), f"Expected shape ({D}, {D}), got {ctf.shape}"
    assert ctf.dtype in (torch.complex64, torch.complex128), \
        f"Expected complex dtype, got {ctf.dtype}"


def test_compute_ctf_batch_shape():
    """测试批量 CTF 输出形状。"""
    N = 4
    D = 64
    ctf = compute_ctf(
        image_size=D,
        pixel_size=1.0,
        defocus=torch.tensor([1.0, 1.5, 2.0, 2.5]),  # [N]
        cs=2.7,
        voltage=300.0,
        amplitude_contrast=0.1,
    )
    assert ctf.shape == (N, D, D), f"Expected shape ({N}, {D}, {D}), got {ctf.shape}"


def test_compute_ctf_values_in_range():
    """测试 CTF 值在 [-1, 1] 范围内。"""
    D = 64
    ctf = compute_ctf(
        image_size=D,
        pixel_size=1.0,
        defocus=2.0,
        cs=2.7,
        voltage=300.0,
        amplitude_contrast=0.1,
    )
    assert torch.all(ctf.real >= -1.0) and torch.all(ctf.real <= 1.0), \
        "CTF values should be in [-1, 1]"
    assert torch.all(ctf.imag.abs() < 1e-6), \
        "CTF should be nearly real-valued (imag ~ 0)"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_ctf.py -v
```
Expected: FAIL (ImportError or ModuleNotFoundError)

- [ ] **Step 3: 实现 CTF 生成函数**

创建 `siamese/utils/ctf.py`:

```python
"""
CTF (Contrast Transfer Function) 生成工具。

CTF 调制是 cryo-EM 图像形成模型的核心部分。在频域中，CTF 描述了
不同空间频率成分的相位对比传递特性。

参考: Rohou & Grigorieff (2015) JSB
"""

import math
from typing import Union

import torch
from torch.fft import fftshift, fftfreq


def compute_ctf(
    image_size: int,
    pixel_size: float = 1.0,
    defocus: Union[float, torch.Tensor] = 2.0,
    cs: float = 2.7,
    voltage: float = 300.0,
    amplitude_contrast: float = 0.1,
    b_factor: float = 0.0,
    device: Union[torch.device, str, None] = None,
) -> torch.Tensor:
    """
    计算 CTF (Contrast Transfer Function)。

    参数:
        image_size: int, 图像边长 D
        pixel_size: float, 像素大小 (Å/pixel)
        defocus: float 或 shape [N] 的 tensor, 欠焦值 (μm)
        cs: float, 球差系数 (mm)
        voltage: float, 加速电压 (kV)
        amplitude_contrast: float, 振幅对比度比例 (0~1)
        b_factor: float, B因子衰减 (Å²), 默认 0 表示不衰减
        device: 计算设备

    返回:
        ctf: 形状 [D, D] 或 [N, D, D] 的复数 CTF。
             CTF 是实值函数（虚部为 0），返回复数类型以方便直接与 FFT 结果相乘。
    """
    if device is None:
        device = torch.device("cpu")

    # 电子波长 (Å), 非相对论近似
    # λ = h / sqrt(2 * m * e * V)
    # 简化: λ ≈ 12.2643 / sqrt(V + 0.97845e-6 * V^2)  (包含相对论修正)
    voltage_rel = voltage * (1.0 + voltage * 0.97845e-6)  # 相对论修正
    wavelength = 12.2643 / math.sqrt(voltage_rel)  # Å

    # 处理 defocus: 可以是标量或 batch
    if isinstance(defocus, float):
        defocus = torch.tensor([defocus], device=device)
    elif not isinstance(defocus, torch.Tensor):
        defocus = torch.as_tensor(defocus, device=device)
    else:
        defocus = defocus.to(device=device)
    # defocus 形状: [N]

    N = defocus.shape[0]
    defocus_um = defocus  # [N]

    # 空间频率坐标 (Å⁻¹)
    # fftfreq 返回 [0, 1/(2*pixel), -1/(2*pixel), ..., -1/(N*pixel)]
    kx = fftfreq(image_size, d=pixel_size, device=device)  # [D]
    ky = fftfreq(image_size, d=pixel_size, device=device)  # [D]
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")  # 各 [D, D]
    k2 = kx_grid ** 2 + ky_grid ** 2  # [D, D], 空间频率平方 (Å⁻²)
    k = torch.sqrt(k2)  # [D, D], 空间频率幅值 (Å⁻¹)

    # 将 k 扩展到 batch 维度: [1, D, D]
    k = k.unsqueeze(0)  # [1, D, D]
    k2 = k2.unsqueeze(0)  # [1, D, D]

    # 相位偏移: χ = π * λ * k^2 * (Δz - 0.5 * λ^2 * Cs * k^2)
    # defocus_um 转换为 Å: 1 μm = 10000 Å
    defocus_A = defocus_um * 10000.0  # [N]
    defocus_A = defocus_A.view(-1, 1, 1)  # [N, 1, 1]

    cs_A = cs * 1e7  # mm -> Å (1 mm = 1e7 Å)

    # χ(k) = π * λ * (Δz * k² - 0.5 * Cs * λ² * k⁴)
    chi = math.pi * wavelength * (
        defocus_A * k2 - 0.5 * cs_A * (wavelength ** 2) * (k2 ** 2)
    )  # [N, D, D]

    # CTF = -sqrt(1 - A²) * sin(χ) - A * cos(χ)
    ctf = -math.sqrt(1.0 - amplitude_contrast ** 2) * torch.sin(chi) \
          - amplitude_contrast * torch.cos(chi)  # [N, D, D]

    # B因子衰减
    if b_factor > 0.0:
        envelope = torch.exp(-b_factor * k2 / 4.0)  # [1, D, D]
        ctf = ctf * envelope

    # 复数化: 转换为复数便于和 FFT 结果相乘
    ctf = ctf.to(torch.complex64)

    # 如果输入是标量 defocus，squeeze 掉 batch 维度
    if N == 1 and isinstance(defocus, torch.Tensor) and defocus.numel() == 1:
        ctf = ctf.squeeze(0)  # [D, D]

    return ctf
```

更新 `siamese/utils/__init__.py`:

```python
"""工具函数模块。"""

from siamese.utils.ctf import compute_ctf

__all__ = ["compute_ctf"]
```

- [ ] **Step 4: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_ctf.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add siamese/utils/ctf.py siamese/utils/__init__.py tests/test_ctf.py
git commit -m "feat: add CTF computation utility"
```

---

### Task 3: FFT 工具函数

**Files:**
- Create: `siamese/utils/fft.py`
- Modify: `siamese/utils/__init__.py`
- Create: `tests/test_fft.py` (测试合并到已有测试或单独文件)

- [ ] **Step 1: 实现 FFT 工具函数**

创建 `siamese/utils/fft.py`:

```python
"""
FFT 工具函数。

将实空间图像转换为频域表示（2通道：实部+虚部），供频域分支使用。
"""

import torch
from torch.fft import fft2, fftshift


def image_to_freq_channels(
    image: torch.Tensor,
    shift: bool = True,
) -> torch.Tensor:
    """
    将实空间图像转换为频域 2 通道表示（实部 + 虚部）。

    参数:
        image: 形状 [..., D, D] 的实空间图像（batch 维度可选）
        shift: 是否先做 fftshift 将低频移到中心（默认 True）

    返回:
        freq: 形状 [..., 2, D, D] 的频域表示。
              freq[..., 0, :, :] = 实部
              freq[..., 1, :, :] = 虚部
    """
    if shift:
        image = fftshift(image, dim=(-2, -1))

    # FFT2, 输出复数
    ft = fft2(image, dim=(-2, -1), norm="ortho")  # [..., D, D] complex

    if shift:
        ft = fftshift(ft, dim=(-2, -1))

    # 拆成 2 通道: 实部 + 虚部
    freq = torch.stack([ft.real, ft.imag], dim=-3)  # [..., 2, D, D]

    return freq


def normalize_image(
    image: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对图像做均值-标准差归一化。

    对每张图像独立计算 mean 和 std，归一化到 mean=0, std=1。

    参数:
        image: 形状 [..., D, D] 的图像
        eps: 防止除零的小常数

    返回:
        normalized: 形状 [..., D, D] 的归一化图像
    """
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True)
    return (image - mean) / (std + eps)
```

更新 `siamese/utils/__init__.py`:

```python
"""工具函数模块。"""

from siamese.utils.ctf import compute_ctf
from siamese.utils.fft import image_to_freq_channels, normalize_image

__all__ = ["compute_ctf", "image_to_freq_channels", "normalize_image"]
```

- [ ] **Step 2: 编写快速验证脚本**

```bash
.venv/bin/python -c "
import torch
from siamese.utils.fft import image_to_freq_channels, normalize_image

# 测试 normalize_image
img = torch.randn(3, 64, 64) * 5.0 + 10.0
normed = normalize_image(img)
print('Normalize: mean ≈', normed[0].mean().item(), 'std ≈', normed[0].std().item())
assert abs(normed[0].mean().item()) < 1e-6, 'Mean should be ~0'
assert abs(normed[0].std().item() - 1.0) < 1e-5, 'Std should be ~1'

# 测试 image_to_freq_channels
freq = image_to_freq_channels(img)
print('Freq shape:', freq.shape)
assert freq.shape == (3, 2, 64, 64), f'Expected (3, 2, 64, 64), got {freq.shape}'
print('All OK')
"
```
Expected: `All OK`

- [ ] **Step 3: 提交**

```bash
git add siamese/utils/fft.py siamese/utils/__init__.py
git commit -m "feat: add FFT and normalization utilities"
```

---

### Task 4: 配置类

**Files:**
- Create: `siamese/training/config.py`
- Modify: `siamese/training/__init__.py`

- [ ] **Step 1: 实现配置 dataclass**

创建 `siamese/training/config.py`:

```python
"""
训练配置 dataclass。

所有可配置的超参数集中管理，支持从 YAML 文件加载。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import yaml


@dataclass
class TrainConfig:
    """训练和模型配置。

    TODO: 后续支持多尺寸模型 (128/192/256/512)
    """

    # --- 数据 ---
    data_dir: str = "data"
    image_size: int = 128
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split = 1 - train_split - val_split

    # --- 模型 ---
    backbone: Literal["convnext_tiny", "convnext_small", "convnext_base",
                       "vit_small", "swin_t"] = "convnext_tiny"
    embedding_dim: int = 128   # TODO: 后续可调
    real_in_channels: int = 1
    freq_in_channels: int = 2
    # ConvNeXt 各 stage 维度 (Tiny 默认值)
    convnext_depths: tuple = (3, 3, 9, 3)
    convnext_dims: tuple = (96, 192, 384, 768)

    # --- 训练 ---
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07

    # scheduler
    scheduler_t0: int = 50   # CosineAnnealingWarmRestarts 的 T_0
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6

    # --- 系统 ---
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = False  # 冒烟测试先不用
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10  # 每 N 个 batch 打印一次 loss

    # --- TODO: 后续支持 ---
    # deepspeed_config: Optional[str] = None  # TODO: DeepSpeed 多卡训练
    # hard_negative: bool = False             # TODO: Hard negative mining

    def save(self, path: str) -> None:
        """保存配置到 YAML 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """从 YAML 文件加载配置，文件中的值覆盖默认值。"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)
```

更新 `siamese/training/__init__.py`:

```python
"""训练模块。"""

from siamese.training.config import TrainConfig

__all__ = ["TrainConfig"]
```

- [ ] **Step 2: 验证**

```bash
.venv/bin/python -c "
from siamese.training.config import TrainConfig
cfg = TrainConfig()
print('Default image_size:', cfg.image_size)
print('Default embedding_dim:', cfg.embedding_dim)
cfg2 = TrainConfig(image_size=256, batch_size=128)
print('Custom image_size:', cfg2.image_size)
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add siamese/training/config.py siamese/training/__init__.py
git commit -m "feat: add training configuration dataclass"
```

---

### Task 5: Backbone 工厂函数

**Files:**
- Create: `siamese/models/backbone.py`
- Modify: `siamese/models/__init__.py`

- [ ] **Step 1: 实现 backbone 工厂函数**

创建 `siamese/models/backbone.py`:

```python
"""
Backbone 工厂函数。

支持 ConvNeXt (默认)、ViT-Small、Swin-T 等多种 backbone。
通过工厂函数 build_backbone() 统一创建，返回 nn.Module。
"""

from typing import Literal, Tuple

import torch.nn as nn

# TODO: 后续可添加 ViT / Swin-T 支持
# from timm.models.vision_transformer import vit_small_patch16_224
# from timm.models.swin_transformer import swin_tiny_patch4_window7_224


def build_backbone(
    name: Literal["convnext_tiny", "convnext_small", "convnext_base",
                   "vit_small", "swin_t"] = "convnext_tiny",
    in_channels: int = 1,
    image_size: int = 128,
    depths: Tuple[int, ...] = (3, 3, 9, 3),
    dims: Tuple[int, ...] = (96, 192, 384, 768),
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """
    创建 backbone 特征提取器。

    参数:
        name: backbone 名称
        in_channels: 输入通道数 (实空间 1, 频域 2)
        image_size: 输入图像尺寸
        depths: ConvNeXt 各 stage 的 block 数量
        dims: ConvNeXt 各 stage 的通道数
        drop_path_rate: stochastic depth rate

    返回:
        backbone: nn.Module, 输出特征图 (不含 head 和 GAP)

    TODO: 支持 ViT-small / Swin-T backbone
    """
    if name.startswith("convnext"):
        from timm.models.convnext import ConvNeXt

        # 映射名称到 timm 的 convnext 变体
        if "tiny" in name:
            depths = (3, 3, 9, 3)
            dims = (96, 192, 384, 768)
        elif "small" in name:
            depths = (3, 3, 27, 3)
            dims = (96, 192, 384, 768)
        elif "base" in name:
            depths = (3, 3, 27, 3)
            dims = (128, 256, 512, 1024)

        backbone = ConvNeXt(
            in_chans=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            head_hidden_dim=None,  # 不使用分类头
            num_classes=0,          # 返回特征
        )
        # 移除分类头: ConvNeXt 的 head 是 nn.Identity 当 num_classes=0
        # 但我们需要保留 stem + stages，去掉最后的 head 和 global_pool
        return backbone

    elif name in ("vit_small", "swin_t"):
        raise NotImplementedError(
            f"Backbone '{name}' not yet implemented. TODO: 添加 ViT/Swin 支持"
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
```

更新 `siamese/models/__init__.py`:

```python
"""模型模块。"""

from siamese.models.backbone import build_backbone

__all__ = ["build_backbone"]
```

- [ ] **Step 2: 验证**

```bash
.venv/bin/python -c "
import torch
from siamese.models.backbone import build_backbone

# 测试实空间分支 (1 通道)
bb = build_backbone('convnext_tiny', in_channels=1, image_size=128)
x = torch.randn(2, 1, 128, 128)
# ConvNeXt 返回特征 (不含 head)，检查输出形状
with torch.no_grad():
    y = bb.forward_features(x)
print('Real branch output shape:', y.shape)
# 期望: [2, 768, 4, 4] (128/32=4, dims[-1]=768)

# 测试频域分支 (2 通道)
bb2 = build_backbone('convnext_tiny', in_channels=2, image_size=128)
x2 = torch.randn(2, 2, 128, 128)
with torch.no_grad():
    y2 = bb2.forward_features(x2)
print('Freq branch output shape:', y2.shape)
print('OK')
"
```
Expected: 输出形状为 `[2, 768, 4, 4]`，OK

- [ ] **Step 3: 提交**

```bash
git add siamese/models/backbone.py siamese/models/__init__.py
git commit -m "feat: add backbone factory function with ConvNeXt support"
```

---

### Task 6: 融合层

**Files:**
- Create: `siamese/models/fusion.py`
- Modify: `siamese/models/__init__.py`

- [ ] **Step 1: 实现融合头**

创建 `siamese/models/fusion.py`:

```python
"""
融合层 (Fusion Head)。

将实空间和频域两支特征拼接后，通过 MLP 投影到 embedding 空间。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """
    双分支特征融合 + 投影头。

    实空间特征 + 频域特征 → concat → Linear+BN+ReLU → Linear → L2 norm → embedding
    """

    def __init__(
        self,
        real_dim: int,       # 实空间分支输出维度 (ConvNeXt-Tiny: 768)
        freq_dim: int,       # 频域分支输出维度 (同上)
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        """
        参数:
            real_dim: 实空间分支特征维度
            freq_dim: 频域分支特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出 embedding 维度
        """
        super().__init__()
        input_dim = real_dim + freq_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, real_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        """
        参数:
            real_feat: 形状 [N, real_dim] 的实空间特征
            freq_feat: 形状 [N, freq_dim] 的频域特征

        返回:
            embedding: 形状 [N, output_dim] 的 L2 归一化 embedding
        """
        combined = torch.cat([real_feat, freq_feat], dim=-1)  # [N, real_dim+freq_dim]
        embedding = self.fusion(combined)  # [N, output_dim]
        embedding = F.normalize(embedding, p=2, dim=-1)  # L2 normalize
        return embedding
```

更新 `siamese/models/__init__.py`:

```python
"""模型模块。"""

from siamese.models.backbone import build_backbone
from siamese.models.fusion import FusionHead

__all__ = ["build_backbone", "FusionHead"]
```

- [ ] **Step 2: 验证**

```bash
.venv/bin/python -c "
import torch
from siamese.models.fusion import FusionHead

head = FusionHead(real_dim=768, freq_dim=768, hidden_dim=256, output_dim=128)
r = torch.randn(4, 768)
f = torch.randn(4, 768)
emb = head(r, f)
print('Embedding shape:', emb.shape)
# 检查 L2 norm
print('L2 norm:', emb.norm(dim=-1))
assert torch.allclose(emb.norm(dim=-1), torch.ones(4), atol=1e-5), 'Should be L2 normalized'
print('OK')
"
```
Expected: shape `[4, 128]`, L2 norm ≈ 1.0, OK

- [ ] **Step 3: 提交**

```bash
git add siamese/models/fusion.py siamese/models/__init__.py
git commit -m "feat: add fusion head for dual-branch features"
```

---

### Task 7: Siamese 编码器

**Files:**
- Create: `siamese/models/encoder.py`
- Modify: `siamese/models/__init__.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: 编写编码器测试**

创建 `tests/test_encoder.py`:

```python
"""SiameseEncoder 测试。"""

import torch
import pytest
from siamese.models.encoder import SiameseEncoder


@pytest.fixture
def encoder():
    """创建默认编码器实例。"""
    return SiameseEncoder(
        backbone_name="convnext_tiny",
        image_size=128,
        real_in_channels=1,
        freq_in_channels=2,
        embedding_dim=128,
    )


def test_encoder_output_shape(encoder):
    """测试编码器输出形状。"""
    N = 4
    D = 128
    x = torch.randn(N, 1, D, D)  # 实空间输入
    emb = encoder(x)
    assert emb.shape == (N, 128), f"Expected (4, 128), got {emb.shape}"


def test_encoder_l2_normalized(encoder):
    """测试编码器输出 L2 归一化。"""
    x = torch.randn(8, 1, 128, 128)
    emb = encoder(x)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), \
        f"Expected L2 norm ≈ 1.0, got {norms}"


def test_encoder_consistency(encoder):
    """测试同一输入产生相同 embedding。"""
    x = torch.randn(1, 1, 128, 128)
    encoder.eval()
    with torch.no_grad():
        emb1 = encoder(x)
        emb2 = encoder(x)
    assert torch.allclose(emb1, emb2, atol=1e-6), \
        "Same input should produce same embedding in eval mode"


def test_encoder_different_inputs_different_embeddings(encoder):
    """测试不同输入产生不同 embedding。"""
    x1 = torch.randn(2, 1, 128, 128)
    x2 = torch.randn(2, 1, 128, 128)
    encoder.eval()
    with torch.no_grad():
        emb1 = encoder(x1)
        emb2 = encoder(x2)
    # 不同输入应有不同 embedding
    assert not torch.allclose(emb1, emb2, atol=1e-3), \
        "Different inputs should produce different embeddings"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_encoder.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 Siamese 编码器**

创建 `siamese/models/encoder.py`:

```python
"""
Siamese 双分支编码器。

实空间分支 (1 通道) + 频域分支 (2 通道: 实部+虚部)
→ 各自经过 backbone 提取特征 → GAP → FusionHead → L2 embedding
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn

from siamese.models.backbone import build_backbone
from siamese.models.fusion import FusionHead
from siamese.utils.fft import image_to_freq_channels, normalize_image


class SiameseEncoder(nn.Module):
    """
    Siamese 双分支对比编码器。

    输入: [N, 1, D, D] 实空间图像
    输出: [N, embedding_dim] L2 归一化 embedding

    工作流程:
        1. 归一化输入图像
        2. 实空间分支: 归一化图像 → backbone → GAP → real_feat
        3. 频域分支: 归一化图像 → FFT2 → backbone → GAP → freq_feat
        4. FusionHead: concat(real_feat, freq_feat) → MLP → L2 embedding
    """

    def __init__(
        self,
        backbone_name: Literal[
            "convnext_tiny", "convnext_small", "convnext_base",
            "vit_small", "swin_t"] = "convnext_tiny",
        image_size: int = 128,
        real_in_channels: int = 1,
        freq_in_channels: int = 2,
        embedding_dim: int = 128,   # TODO: 后续可调
        hidden_dim: int = 256,
        convnext_depths: Tuple[int, ...] = (3, 3, 9, 3),
        convnext_dims: Tuple[int, ...] = (96, 192, 384, 768),
    ):
        """
        参数:
            backbone_name: backbone 架构名称
            image_size: 输入图像尺寸
            real_in_channels: 实空间分支输入通道数
            freq_in_channels: 频域分支输入通道数
            embedding_dim: 输出 embedding 维度
            hidden_dim: 融合头隐藏层维度
            convnext_depths: ConvNeXt 各 stage 的 block 数
            convnext_dims: ConvNeXt 各 stage 的通道数
        """
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim

        # 实空间分支
        self.backbone_real = build_backbone(
            name=backbone_name,
            in_channels=real_in_channels,
            image_size=image_size,
            depths=convnext_depths,
            dims=convnext_dims,
        )

        # 频域分支（权重独立）
        self.backbone_freq = build_backbone(
            name=backbone_name,
            in_channels=freq_in_channels,
            image_size=image_size,
            depths=convnext_depths,
            dims=convnext_dims,
        )

        # 特征维度 (ConvNeXt 最后一个 stage 的输出通道数)
        feature_dim = convnext_dims[-1]  # 默认 768

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 融合头
        self.fusion_head = FusionHead(
            real_dim=feature_dim,
            freq_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 形状 [N, 1, D, D] 的实空间图像

        返回:
            embedding: 形状 [N, embedding_dim] 的 L2 归一化 embedding
        """
        # 1. 归一化 (per-image mean-std)
        x_norm = normalize_image(x)  # [N, 1, D, D]

        # 2. 实空间分支
        real_feat = self.backbone_real.forward_features(x_norm)  # [N, C, H, W]
        real_feat = self.gap(real_feat)  # [N, C, 1, 1]
        real_feat = real_feat.flatten(1)  # [N, C]

        # 3. 频域分支
        # 对归一化后的图像做 FFT
        x_freq = image_to_freq_channels(x_norm.squeeze(1))  # [N, D, D] -> [N, 2, D, D]
        freq_feat = self.backbone_freq.forward_features(x_freq)  # [N, C, H, W]
        freq_feat = self.gap(freq_feat)  # [N, C, 1, 1]
        freq_feat = freq_feat.flatten(1)  # [N, C]

        # 4. 融合 + 投影
        embedding = self.fusion_head(real_feat, freq_feat)  # [N, embedding_dim]

        return embedding
```

更新 `siamese/models/__init__.py`:

```python
"""模型模块。"""

from siamese.models.backbone import build_backbone
from siamese.models.fusion import FusionHead
from siamese.models.encoder import SiameseEncoder

__all__ = ["build_backbone", "FusionHead", "SiameseEncoder"]
```

- [ ] **Step 4: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_encoder.py -v
```
Expected: 4 PASS

- [ ] **Step 5: 提交**

```bash
git add siamese/models/encoder.py siamese/models/__init__.py tests/test_encoder.py
git commit -m "feat: add Siamese dual-branch encoder"
```

---

### Task 8: InfoNCE Loss

**Files:**
- Create: `siamese/losses/infonce.py`
- Modify: `siamese/losses/__init__.py`
- Create: `tests/test_loss.py`

- [ ] **Step 1: 编写 Loss 测试**

创建 `tests/test_loss.py`:

```python
"""InfoNCE loss 测试。"""

import torch
import pytest
from siamese.losses.infonce import InfoNCELoss


def test_infonce_loss_shape():
    """测试 loss 输出为标量。"""
    loss_fn = InfoNCELoss(temperature=0.07)
    z_mic = torch.randn(8, 128)
    z_proj = torch.randn(8, 128)
    loss = loss_fn(z_mic, z_proj)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


def test_infonce_loss_positive():
    """测试 loss 值为正。"""
    loss_fn = InfoNCELoss(temperature=0.07)
    z_mic = torch.randn(8, 128)
    z_proj = torch.randn(8, 128)
    loss = loss_fn(z_mic, z_proj)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


def test_infonce_loss_perfect_match():
    """测试当 mic 和 proj embedding 相同时 loss 最小。"""
    loss_fn = InfoNCELoss(temperature=0.07)
    z = torch.randn(8, 128)
    z = torch.nn.functional.normalize(z, dim=-1)
    loss_same = loss_fn(z, z)

    # 打乱 proj (破坏配对)
    z_shuffled = z[torch.randperm(8)]
    loss_shuffled = loss_fn(z, z_shuffled)

    assert loss_same.item() < loss_shuffled.item(), \
        "Loss should be lower when embeddings are aligned"


def test_infonce_loss_with_hard_negatives():
    """测试带 hard negatives 的 loss。"""
    # TODO: 后续实现 hard negative mining 后完善此测试
    pass
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_loss.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 InfoNCE Loss**

创建 `siamese/losses/infonce.py`:

```python
"""
InfoNCE (NT-Xent) 对比损失。

对称版本: 同时计算 mic→proj 和 proj→mic 两个方向的交叉熵损失。

TODO: 添加 hard negative mining 支持
  - hard negatives: 相邻 viewing direction, 相似 silhouette, 对称相关 projection
  - 接口: loss_fn(z_mic, z_proj, hard_negatives=None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent 对比损失。

    给定 batch 中 N 对 (mic_i, proj_i):
      S[i,j] = z_mic[i] · z_proj[j] / τ
      loss = (CE(S, labels) + CE(S.T, labels)) / 2

    正样本: 对角线位置 (i, i)
    负样本: batch 内其他样本 (i, j) for j ≠ i
    """

    def __init__(self, temperature: float = 0.07):
        """
        参数:
            temperature: 温度参数 τ, 控制 softmax 的锐度。
                        值越小, 对负样本的区分越严格。
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_mic: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            z_mic: 形状 [N, D] 的 mic embedding (应已 L2 归一化)
            z_proj: 形状 [N, D] 的 proj embedding (应已 L2 归一化)

        返回:
            loss: 标量, 对称 InfoNCE loss
        """
        N = z_mic.shape[0]

        # 相似度矩阵: S[i,j] = cos_sim(mic_i, proj_j) / τ
        # 由于 embedding 已 L2 归一化, 点积 = 余弦相似度
        logits = torch.matmul(z_mic, z_proj.T) / self.temperature  # [N, N]

        # 正样本标签: 对角线位置
        labels = torch.arange(N, device=z_mic.device)

        # 对称 InfoNCE: mic→proj 和 proj→mic
        loss_mic = F.cross_entropy(logits, labels)      # 每行分类
        loss_proj = F.cross_entropy(logits.T, labels)   # 每列分类

        loss = (loss_mic + loss_proj) / 2.0
        return loss
```

更新 `siamese/losses/__init__.py`:

```python
"""损失函数模块。"""

from siamese.losses.infonce import InfoNCELoss

__all__ = ["InfoNCELoss"]
```

- [ ] **Step 4: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_loss.py -v
```
Expected: 4 PASS (1 skipped — hard negative test)

- [ ] **Step 5: 提交**

```bash
git add siamese/losses/infonce.py siamese/losses/__init__.py tests/test_loss.py
git commit -m "feat: add InfoNCE contrastive loss"
```

---

### Task 9: 数据预处理变换

**Files:**
- Create: `siamese/data/transforms.py`
- Modify: `siamese/data/__init__.py`

- [ ] **Step 1: 实现预处理变换**

创建 `siamese/data/transforms.py`:

```python
"""
数据预处理变换。

包括图像归一化、频域转换等。作为可组合的变换函数，供 Dataset 使用。
"""

import torch
from siamese.utils.fft import normalize_image, image_to_freq_channels


class PreprocessTransform:
    """
    预处理变换: 归一化 + 可选的频域转换。

    在 Dataset 的 __getitem__ 中调用，对每个样本独立处理。
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        参数:
            image: 形状 [D, D] 的图像

        返回:
            image: 形状 [1, D, D] 的归一化图像 (添加 channel 维度)
        """
        if self.normalize:
            image = normalize_image(image)  # [D, D], mean=0, std=1

        # 添加 channel 维度: [D, D] -> [1, D, D]
        image = image.unsqueeze(0)
        return image
```

更新 `siamese/data/__init__.py`:

```python
"""数据模块。"""

from siamese.data.transforms import PreprocessTransform

__all__ = ["PreprocessTransform"]
```

- [ ] **Step 2: 验证**

```bash
.venv/bin/python -c "
import torch
from siamese.data.transforms import PreprocessTransform

transform = PreprocessTransform(normalize=True)
img = torch.randn(64, 64) * 5.0 + 10.0
out = transform(img)
print('Output shape:', out.shape)
assert out.shape == (1, 64, 64), f'Expected (1, 64, 64), got {out.shape}'
print('OK')
"
```
Expected: shape `(1, 64, 64)`, OK

- [ ] **Step 3: 提交**

```bash
git add siamese/data/transforms.py siamese/data/__init__.py
git commit -m "feat: add data preprocessing transforms"
```

---

### Task 10: 模拟数据生成

**Files:**
- Create: `siamese/data/generate.py`
- Modify: `siamese/data/__init__.py`

- [ ] **Step 1: 实现数据生成函数**

创建 `siamese/data/generate.py`:

```python
"""
模拟数据生成。

从 3D volume 用 HEALPix 采样生成 proj，再添加 CTF、shift、噪声生成 mic。
一次生成所有数据并保存到硬盘，避免训练时重复计算。

输出文件:
  - projs.pt: [N, D, D] clean projections
  - mics.pt:  [M, D, D] noisy micrographs (M = N * num_mics_per_proj)
  - axisang.pt: [N, 3] 每个 proj 对应的轴角
  - pairs.pt: [M, 2] 每个 mic 对应的 (proj_idx, mic_idx) 配对索引
  - metadata.yaml: 生成参数记录
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from siamese.utils.ctf import compute_ctf

# 复用 project.py 中的投影函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from project import healpix_project, project


def generate_simulated_data(
    map_path: str,
    nside: int,
    output_dir: str,
    image_size: int = 128,
    pixel_size: float = 1.0,
    num_mics_per_proj: int = 2,
    snr_range: Tuple[float, float] = (0.001, 0.01),
    defocus_range: Tuple[float, float] = (0.5, 4.0),
    max_shift_pixels: float = 5.0,
    cs: float = 2.7,
    voltage: float = 300.0,
    amplitude_contrast: float = 0.1,
    device: str = "cuda",
    chunk_size: int = 256,
    seed: int = 42,
) -> dict:
    """
    生成模拟训练数据。

    参数:
        map_path: 3D volume .map 文件路径
        nside: HEALPix nside, 方向数为 12 * nside^2
        output_dir: 输出目录
        image_size: 图像尺寸 D
        pixel_size: 像素大小 (Å/pixel)
        num_mics_per_proj: 每个 proj 生成几个不同噪声版本的 mic
        snr_range: SNR 采样范围 (min, max)
        defocus_range: 欠焦值采样范围 (μm)
        max_shift_pixels: 最大随机平移 (pixels)
        cs: 球差系数 (mm)
        voltage: 加速电压 (kV)
        amplitude_contrast: 振幅对比度
        device: 计算设备
        chunk_size: 每次投影的方向数
        seed: 随机种子

    返回:
        dict: 包含生成统计信息的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. 读取 volume
    import mrcfile
    with mrcfile.open(map_path, permissive=True) as mrc:
        volume_np = np.asarray(mrc.data, dtype=np.float32).copy()
    # 如果需要，resize volume
    # 此处假设 volume 尺寸已经合适
    volume = torch.from_numpy(volume_np)

    num_directions = 12 * nside * nside
    print(f"Generating {num_directions} HEALPix directions (nside={nside})...")

    # 2. 生成 clean projs
    projs, axisang = healpix_project(
        volume=volume,
        nside=nside,
        device=device,
        chunk_size=chunk_size,
    )  # projs: [N, D_vol, D_vol], axisang: [N, 3]

    D_vol = projs.shape[1]
    print(f"Projections shape: {projs.shape}, dtype: {projs.dtype}")

    # 3. 如果 volume 尺寸 ≠ image_size，需要裁剪或 resize
    #    这里简单处理: 取中心 image_size x image_size
    if D_vol != image_size:
        margin = (D_vol - image_size) // 2
        projs = projs[:, margin:margin + image_size, margin:margin + image_size]
        print(f"Cropped projections to {image_size}x{image_size}")

    total_mics = num_directions * num_mics_per_proj

    # 4. 生成 mic (带随机参数)
    #    为每个 mic 随机采样 SNR, defocus, shift
    snr_values = rng.uniform(snr_range[0], snr_range[1], size=total_mics)
    snr_values = snr_values.reshape(num_directions, num_mics_per_proj)

    defocus_values = rng.uniform(defocus_range[0], defocus_range[1], size=total_mics)
    defocus_values = defocus_values.reshape(num_directions, num_mics_per_proj)

    shift_values = rng.uniform(-max_shift_pixels, max_shift_pixels, size=(total_mics, 2))
    shift_values = shift_values.reshape(num_directions, num_mics_per_proj, 2)

    mics_list = []
    pairs_list = []  # (proj_idx, mic_idx_within_proj)

    print(f"Generating {total_mics} noisy micrographs...")
    mic_idx = 0
    for i in tqdm(range(num_directions)):
        proj_i = projs[i]  # [D, D]
        for j in range(num_mics_per_proj):
            # 生成这张 mic 的参数
            snr = float(snr_values[i, j])
            defocus = float(defocus_values[i, j])
            shift = torch.from_numpy(shift_values[i, j]).float().unsqueeze(0)  # [1, 2]

            # 生成 CTF
            ctf = compute_ctf(
                image_size=image_size,
                pixel_size=pixel_size,
                defocus=defocus,
                cs=cs,
                voltage=voltage,
                amplitude_contrast=amplitude_contrast,
                device=device,
            )  # [D, D] complex

            # 使用 project() 生成带 shift + CTF + noise 的 mic
            # 我们需要一个 axisang=0 的投影（即不旋转），直接在 proj 上加 shift/CTF/noise
            mic_proj = proj_i.unsqueeze(0).to(device)  # [1, D, D]

            # 在频域应用 shift 和 CTF
            from torch.fft import fftshift, fft2, ifft2, fftfreq
            import torch.fft

            D = image_size
            mic_ft = fftshift(fft2(fftshift(mic_proj, dim=(-2, -1)), dim=(-2, -1), norm="ortho"), dim=(-2, -1))

            # 应用 shift
            shift_pix = shift.to(device) / pixel_size  # [1, 2]
            dx = shift_pix[:, 0].view(1, 1, 1)
            dy = shift_pix[:, 1].view(1, 1, 1)
            shift_freq = fftfreq(D, device=device)
            shift_freq = fftshift(shift_freq)
            shift_fy, shift_fx = torch.meshgrid(shift_freq, shift_freq, indexing="ij")
            shift_angle = -2.0 * math.pi * (
                shift_fx[None, :, :] * dx + shift_fy[None, :, :] * dy
            )
            shift_phase = torch.polar(torch.ones_like(shift_angle), shift_angle)
            mic_ft = mic_ft * shift_phase

            # 应用 CTF
            if ctf.dim() == 2:
                ctf = ctf.unsqueeze(0)  # [1, D, D]
            mic_ft = mic_ft * ctf.to(device)

            # 逆 FFT 回实空间
            mic = ifft2(fftshift(mic_ft, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
            mic = fftshift(mic, dim=(-2, -1)).real * math.sqrt(D)

            # 加噪声
            x0 = mic - mic.mean(dim=(-2, -1), keepdim=True)
            var_s = (x0 ** 2).mean(dim=(-2, -1))
            var_n = var_s / (snr + 1e-8)
            sigma_n = torch.sqrt(torch.clamp(var_n, min=1e-8))
            g = torch.Generator(device=device)
            g.manual_seed(seed + mic_idx)
            noise = torch.randn(mic.shape, generator=g, device=device, dtype=mic.dtype) * sigma_n.view(-1, 1, 1)
            mic = mic + noise

            mics_list.append(mic.squeeze(0).cpu())
            pairs_list.append((i, mic_idx))
            mic_idx += 1

    # 5. 保存
    mics = torch.stack(mics_list, dim=0)  # [M, D, D]
    pairs = torch.tensor(pairs_list, dtype=torch.long)  # [M, 2]

    torch.save(projs.cpu(), output_dir / "projs.pt")
    torch.save(mics, output_dir / "mics.pt")
    torch.save(axisang.cpu(), output_dir / "axisang.pt")
    torch.save(pairs, output_dir / "pairs.pt")

    # 保存元数据
    metadata = {
        "map_path": str(map_path),
        "nside": nside,
        "num_directions": num_directions,
        "num_mics_total": total_mics,
        "num_mics_per_proj": num_mics_per_proj,
        "image_size": image_size,
        "pixel_size": pixel_size,
        "snr_range": list(snr_range),
        "defocus_range": list(defocus_range),
        "max_shift_pixels": max_shift_pixels,
        "cs": cs,
        "voltage": voltage,
        "amplitude_contrast": amplitude_contrast,
        "seed": seed,
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    print(f"Saved {num_directions} projs, {total_mics} mics to {output_dir}")
    print(f"Proj shape: {projs.shape}, Mic shape: {mics.shape}")
    return metadata
```

更新 `siamese/data/__init__.py`:

```python
"""数据模块。"""

from siamese.data.transforms import PreprocessTransform
from siamese.data.generate import generate_simulated_data

__all__ = ["PreprocessTransform", "generate_simulated_data"]
```

- [ ] **Step 2: 提交**

```bash
git add siamese/data/generate.py siamese/data/__init__.py
git commit -m "feat: add simulated data generation pipeline"
```

---

### Task 11: MicProjDataset

**Files:**
- Create: `siamese/data/dataset.py`
- Modify: `siamese/data/__init__.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: 编写数据集测试**

创建 `tests/test_dataset.py`:

```python
"""MicProjDataset 测试。"""

import torch
import tempfile
import pytest
from pathlib import Path
from siamese.data.dataset import MicProjDataset


@pytest.fixture
def temp_data_dir():
    """创建临时数据目录用于测试。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        D = 32
        N = 20
        M = 40  # 2 mics per proj

        projs = torch.randn(N, D, D)
        mics = torch.randn(M, D, D)
        axisang = torch.randn(N, 3)
        # pairs: [proj_idx, mic_idx]
        pairs = torch.zeros(M, 2, dtype=torch.long)
        for i in range(N):
            for j in range(2):
                pairs[i * 2 + j, 0] = i
                pairs[i * 2 + j, 1] = i * 2 + j

        torch.save(projs, tmpdir / "projs.pt")
        torch.save(mics, tmpdir / "mics.pt")
        torch.save(axisang, tmpdir / "axisang.pt")
        torch.save(pairs, tmpdir / "pairs.pt")
        yield tmpdir


def test_dataset_len(temp_data_dir):
    """测试数据集长度。"""
    dataset = MicProjDataset(data_dir=str(temp_data_dir), split="train", train_split=0.7, val_split=0.15)
    # 40 mics, 0.7 train = 28
    assert len(dataset) == 28, f"Expected 28, got {len(dataset)}"


def test_dataset_getitem_shape(temp_data_dir):
    """测试 __getitem__ 返回形状。"""
    dataset = MicProjDataset(data_dir=str(temp_data_dir), split="train", train_split=0.7, val_split=0.15)
    mic, proj = dataset[0]
    # mic: [1, D, D], proj: [1, D, D]
    assert mic.shape == (1, 32, 32), f"Expected (1, 32, 32), got {mic.shape}"
    assert proj.shape == (1, 32, 32), f"Expected (1, 32, 32), got {proj.shape}"


def test_dataset_splits_disjoint(temp_data_dir):
    """测试 train/val/test 不重叠。"""
    train_ds = MicProjDataset(data_dir=str(temp_data_dir), split="train", train_split=0.7, val_split=0.15)
    val_ds = MicProjDataset(data_dir=str(temp_data_dir), split="val", train_split=0.7, val_split=0.15)
    test_ds = MicProjDataset(data_dir=str(temp_data_dir), split="test", train_split=0.7, val_split=0.15)

    assert len(train_ds) + len(val_ds) + len(test_ds) == 40, "Splits should cover all samples"
    assert len(train_ds) == 28, f"Expected 28 train, got {len(train_ds)}"
    assert len(val_ds) == 6, f"Expected 6 val, got {len(val_ds)}"
    assert len(test_ds) == 6, f"Expected 6 test, got {len(test_ds)}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 MicProjDataset**

创建 `siamese/data/dataset.py`:

```python
"""
MicProjDataset: 从预生成的模拟数据中加载 (mic, proj) 配对。
"""

from pathlib import Path
from typing import Tuple, Literal

import torch
from torch.utils.data import Dataset

from siamese.data.transforms import PreprocessTransform


class MicProjDataset(Dataset):
    """
    Mic-Proj 配对数据集。

    从预生成的 .pt 文件加载数据，支持 train/val/test 划分。

    每个样本返回 (mic, proj) 元组，mic 和 proj 已归一化并添加 channel 维度。

    划分策略: 按 HEALPix 方向顺序划分，避免按方向半球划分的复杂性
             (冒烟测试数据量小，简单顺序划分即可)
    TODO: 后续可改为按方向半球划分避免泄漏
    """

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val", "test"] = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        参数:
            data_dir: 包含 projs.pt, mics.pt, pairs.pt 的目录
            split: 数据集划分
            train_split: 训练集比例
            val_split: 验证集比例 (测试集 = 1 - train - val)
            normalize: 是否对图像做归一化
            seed: 随机种子 (用于 shuffle)
        """
        data_dir = Path(data_dir)

        self.projs = torch.load(data_dir / "projs.pt", weights_only=True)  # [N, D, D]
        self.mics = torch.load(data_dir / "mics.pt", weights_only=True)    # [M, D, D]
        self.pairs = torch.load(data_dir / "pairs.pt", weights_only=True)  # [M, 2]

        self.transform = PreprocessTransform(normalize=normalize)

        M = len(self.mics)

        # 生成索引并 shuffle
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(M, generator=g).tolist()

        # 按比例划分
        train_end = int(M * train_split)
        val_end = int(M * (train_split + val_split))

        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        elif split == "test":
            self.indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            idx: 数据集内部索引

        返回:
            mic: 形状 [1, D, D] 的归一化 noisy micrograph
            proj: 形状 [1, D, D] 的归一化 clean projection
        """
        global_idx = self.indices[idx]
        proj_idx, mic_idx = self.pairs[global_idx]

        mic = self.mics[mic_idx]    # [D, D]
        proj = self.projs[proj_idx]  # [D, D]

        mic = self.transform(mic)    # [1, D, D]
        proj = self.transform(proj)  # [1, D, D]

        return mic, proj
```

更新 `siamese/data/__init__.py`:

```python
"""数据模块。"""

from siamese.data.transforms import PreprocessTransform
from siamese.data.generate import generate_simulated_data
from siamese.data.dataset import MicProjDataset

__all__ = ["PreprocessTransform", "generate_simulated_data", "MicProjDataset"]
```

- [ ] **Step 4: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add siamese/data/dataset.py siamese/data/__init__.py tests/test_dataset.py
git commit -m "feat: add MicProjDataset with train/val/test splits"
```

---

### Task 12: Trainer 训练循环

**Files:**
- Create: `siamese/training/trainer.py`
- Modify: `siamese/training/__init__.py`

- [ ] **Step 1: 实现 Trainer**

创建 `siamese/training/trainer.py`:

```python
"""
训练循环。

封装训练和验证逻辑，支持 checkpoint 保存和恢复。
TODO: 后续添加 DeepSpeed 支持
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from siamese.training.config import TrainConfig
from siamese.losses.infonce import InfoNCELoss


class Trainer:
    """
    训练器。

    管理训练循环、验证、checkpoint 保存和恢复。
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        参数:
            model: SiameseEncoder 模型
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器 (可选)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        self.criterion = InfoNCELoss(temperature=config.temperature)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_t0,
            T_mult=config.scheduler_t_mult,
            eta_min=config.scheduler_eta_min,
        )

        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # 创建 checkpoint 目录
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        """训练一个 epoch，返回平均 loss。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        for batch_idx, (mic, proj) in enumerate(pbar):
            mic = mic.to(self.device)
            proj = proj.to(self.device)

            # 编码
            z_mic = self.model(mic)    # [N, D]
            z_proj = self.model(proj)  # [N, D]

            # 计算 loss
            loss = self.criterion(z_mic, z_proj)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """验证一个 epoch，返回平均 loss。"""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for mic, proj in self.val_loader:
            mic = mic.to(self.device)
            proj = proj.to(self.device)

            z_mic = self.model(mic)
            z_proj = self.model(proj)

            loss = self.criterion(z_mic, z_proj)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self) -> Dict[str, list]:
        """
        完整训练循环。

        返回:
            dict: {"train_losses": [...], "val_losses": [...]}
        """
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")

            # 定期保存
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # 保存最终模型
        self.save_checkpoint("last.pt")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def save_checkpoint(self, filename: str) -> None:
        """保存 checkpoint。"""
        path = Path(self.config.checkpoint_dir) / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """从 checkpoint 恢复训练状态。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        print(f"Loaded checkpoint from {path} (epoch {self.epoch + 1})")
```

更新 `siamese/training/__init__.py`:

```python
"""训练模块。"""

from siamese.training.config import TrainConfig
from siamese.training.trainer import Trainer

__all__ = ["TrainConfig", "Trainer"]
```

- [ ] **Step 2: 提交**

```bash
git add siamese/training/trainer.py siamese/training/__init__.py
git commit -m "feat: add training loop with checkpointing"
```

---

### Task 13: 评估模块 — 检索与指标

**Files:**
- Create: `siamese/eval/retrieval.py`
- Create: `siamese/eval/metrics.py`
- Modify: `siamese/eval/__init__.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: 实现检索函数**

创建 `siamese/eval/retrieval.py`:

```python
"""
FAISS 检索模块。

对 clean proj 建索引，对 noisy mic 查询 top-k 匹配。
"""

from typing import Tuple

import faiss
import numpy as np
import torch


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    构建 FAISS 内积索引 (用于余弦相似度检索)。

    参数:
        embeddings: 形状 [N, D] 的 L2 归一化 embedding (numpy)

    返回:
        index: FAISS IndexFlatIP 索引
    """
    D = embeddings.shape[1]
    index = faiss.IndexFlatIP(D)  # 内积 = 余弦相似度 (因为 L2 归一化)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve_topk(
    query_embeddings: torch.Tensor,
    index: faiss.IndexFlatIP,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    检索 top-k 最相似 proj。

    参数:
        query_embeddings: 形状 [N, D] 的 mic embedding (torch tensor, L2 归一化)
        index: 已构建的 FAISS 索引
        k: 返回的 top-k 数量

    返回:
        distances: 形状 [N, k] 的余弦相似度 (numpy)
        indices: 形状 [N, k] 的 proj 索引 (numpy)
    """
    query_np = query_embeddings.detach().cpu().numpy().astype(np.float32)
    distances, indices = index.search(query_np, k)
    return distances, indices
```

创建 `siamese/eval/metrics.py`:

```python
"""
评估指标计算。

包括 top-k 准确率、检索结果可视化等。
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_accuracy_at_k(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
) -> dict:
    """
    计算 top-k 准确率。

    参数:
        retrieved_indices: 形状 [N, max_k] 的检索结果索引
        ground_truth_indices: 形状 [N] 的 ground truth proj 索引
        k_values: 要计算的 k 值列表

    返回:
        accuracies: dict, {f"top-{k}": accuracy}
    """
    N = len(ground_truth_indices)
    accuracies = {}

    for k in k_values:
        if k > retrieved_indices.shape[1]:
            continue
        # 检查 ground truth 是否在 top-k 中
        top_k = retrieved_indices[:, :k]  # [N, k]
        matches = np.any(top_k == ground_truth_indices[:, None], axis=1)  # [N]
        acc = matches.mean()
        accuracies[f"top-{k}"] = acc

    return accuracies


def plot_retrieval_results(
    query_mic: torch.Tensor,
    retrieved_projs: torch.Tensor,
    ground_truth: torch.Tensor,
    similarities: np.ndarray,
    k: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    可视化检索结果。

    展示 query mic + top-k retrieved projs + ground truth，每张图下方标注相似度。

    参数:
        query_mic: 形状 [D, D] 的 query image
        retrieved_projs: 形状 [k, D, D] 的检索结果
        ground_truth: 形状 [D, D] 的 ground truth proj
        similarities: 形状 [k] 的余弦相似度
        k: 展示的 top-k 数量
        save_path: 保存路径 (可选)
    """
    fig, axes = plt.subplots(1, k + 2, figsize=(3 * (k + 2), 3))

    # Query mic
    axes[0].imshow(query_mic.cpu().numpy(), cmap="gray")
    axes[0].set_title("Query (mic)")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(ground_truth.cpu().numpy(), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Top-k retrieved
    for i in range(k):
        if i < len(retrieved_projs):
            axes[i + 2].imshow(retrieved_projs[i].cpu().numpy(), cmap="gray")
            is_gt = "✓" if similarities[i] == similarities.max() else ""
            axes[i + 2].set_title(f"Top-{i + 1}: {similarities[i]:.3f} {is_gt}")
        axes[i + 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved retrieval visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tsne(
    mic_embeddings: torch.Tensor,
    proj_embeddings: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    t-SNE 可视化 embedding 空间。

    参数:
        mic_embeddings: 形状 [N, D] 的 mic embedding
        proj_embeddings: 形状 [M, D] 的 proj embedding
        labels: 形状 [N] 的 mic 对应的 proj 标签 (可选)
        save_path: 保存路径 (可选)
    """
    from sklearn.manifold import TSNE

    all_emb = torch.cat([mic_embeddings, proj_embeddings], dim=0).cpu().numpy()
    n_mic = len(mic_embeddings)
    n_proj = len(proj_embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_mic + n_proj - 1))
    emb_2d = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_2d[:n_mic, 0], emb_2d[:n_mic, 1], c="blue", label="mic", alpha=0.6, s=10)
    ax.scatter(emb_2d[n_mic:, 0], emb_2d[n_mic:, 1], c="red", label="proj", alpha=0.6, s=10)
    ax.legend()
    ax.set_title("t-SNE of Embedding Space")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
```

更新 `siamese/eval/__init__.py`:

```python
"""评估模块。"""

from siamese.eval.retrieval import build_faiss_index, retrieve_topk
from siamese.eval.metrics import compute_accuracy_at_k, plot_retrieval_results, plot_tsne

__all__ = [
    "build_faiss_index",
    "retrieve_topk",
    "compute_accuracy_at_k",
    "plot_retrieval_results",
    "plot_tsne",
]
```

- [ ] **Step 2: 编写检索测试**

创建 `tests/test_retrieval.py`:

```python
"""检索和评估指标测试。"""

import numpy as np
import torch
from siamese.eval.retrieval import build_faiss_index, retrieve_topk
from siamese.eval.metrics import compute_accuracy_at_k


def test_build_and_retrieve():
    """测试建索引和检索基本功能。"""
    N, D = 100, 128
    # 创建随机 L2 归一化 embedding
    emb = torch.randn(N, D)
    emb = torch.nn.functional.normalize(emb, dim=-1)

    index = build_faiss_index(emb.numpy())

    # 用自身查询，top-1 应该返回自身
    distances, indices = retrieve_topk(emb, index, k=5)
    assert indices.shape == (N, 5), f"Expected ({N}, 5), got {indices.shape}"
    # top-1 应该 = 自身 (余弦相似度最高)
    assert np.all(indices[:, 0] == np.arange(N)), "Top-1 should be self"


def test_compute_accuracy():
    """测试准确率计算。"""
    # 模拟: 5 个 query, 检索 top-10, ground truth 在位置 0, 2, 4, 6, 8
    retrieved = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # gt=0, top-1 命中
        [5, 6, 2, 3, 4, 0, 1, 7, 8, 9],  # gt=2, top-3 命中
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # gt=4, top-6 命中
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],  # gt=6, top-8 命中
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  # gt=8, 未命中
    ])
    gt = np.array([0, 2, 4, 6, 8])

    acc = compute_accuracy_at_k(retrieved, gt, k_values=[1, 5, 10])
    assert acc["top-1"] == 0.2, f"Expected 0.2, got {acc['top-1']}"
    assert acc["top-5"] == 0.6, f"Expected 0.6, got {acc['top-5']}"
    assert acc["top-10"] == 0.8, f"Expected 0.8, got {acc['top-10']}"
```

- [ ] **Step 3: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_retrieval.py -v
```
Expected: 2 PASS

- [ ] **Step 4: 提交**

```bash
git add siamese/eval/retrieval.py siamese/eval/metrics.py siamese/eval/__init__.py tests/test_retrieval.py
git commit -m "feat: add FAISS retrieval and evaluation metrics"
```

---

### Task 14: 运行脚本 — generate_data.py

**Files:**
- Create: `scripts/generate_data.py`

- [ ] **Step 1: 实现数据生成脚本**

创建 `scripts/generate_data.py`:

```python
#!/usr/bin/env python
"""
生成模拟训练数据。

用法:
    python scripts/generate_data.py emd_19110.map 8 --output-dir data/ --image-size 128
"""

import argparse
from pathlib import Path

from siamese.data.generate import generate_simulated_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated mic-proj paired data from 3D volume."
    )
    parser.add_argument("map_path", type=str, help="Path to .map 3D volume file.")
    parser.add_argument("nside", type=int, default=8,
                        help="HEALPix nside (directions = 12 * nside^2).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for generated data.")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image size D (square).")
    parser.add_argument("--pixel-size", type=float, default=1.0,
                        help="Pixel size in Angstrom.")
    parser.add_argument("--num-mics-per-proj", type=int, default=2,
                        help="Number of noisy mics per clean proj.")
    parser.add_argument("--snr-min", type=float, default=0.001,
                        help="Minimum SNR for noise.")
    parser.add_argument("--snr-max", type=float, default=0.01,
                        help="Maximum SNR for noise.")
    parser.add_argument("--defocus-min", type=float, default=0.5,
                        help="Minimum defocus in um.")
    parser.add_argument("--defocus-max", type=float, default=4.0,
                        help="Maximum defocus in um.")
    parser.add_argument("--max-shift", type=float, default=5.0,
                        help="Maximum in-plane shift in pixels.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device.")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="HEALPix projection chunk size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    generate_simulated_data(
        map_path=args.map_path,
        nside=args.nside,
        output_dir=args.output_dir,
        image_size=args.image_size,
        pixel_size=args.pixel_size,
        num_mics_per_proj=args.num_mics_per_proj,
        snr_range=(args.snr_min, args.snr_max),
        defocus_range=(args.defocus_min, args.defocus_max),
        max_shift_pixels=args.max_shift,
        device=args.device,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/generate_data.py
git commit -m "feat: add data generation CLI script"
```

---

### Task 15: 运行脚本 — train.py

**Files:**
- Create: `scripts/train.py`
- Create: `configs/default.yaml`

- [ ] **Step 1: 创建默认配置文件**

创建 `configs/default.yaml`:

```yaml
# Siamese Cryo-EM 默认训练配置
# 冒烟测试用: 128x128 图像, ConvNeXt-Tiny, batch_size=64

# --- 数据 ---
data_dir: "data"
image_size: 128
train_split: 0.7
val_split: 0.15

# --- 模型 ---
backbone: "convnext_tiny"
embedding_dim: 128
real_in_channels: 1
freq_in_channels: 2
convnext_depths: [3, 3, 9, 3]
convnext_dims: [96, 192, 384, 768]

# --- 训练 ---
batch_size: 64
num_epochs: 200
learning_rate: 0.0001
weight_decay: 0.0001
temperature: 0.07

# scheduler
scheduler_t0: 50
scheduler_t_mult: 2
scheduler_eta_min: 0.000001

# --- 系统 ---
device: "cuda"
num_workers: 4
seed: 42
mixed_precision: false
checkpoint_dir: "checkpoints"
log_interval: 10
```

- [ ] **Step 2: 实现训练脚本**

创建 `scripts/train.py`:

```python
#!/usr/bin/env python
"""
训练 Siamese 对比编码器。

用法:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from siamese.training.config import TrainConfig
from siamese.training.trainer import Trainer
from siamese.models.encoder import SiameseEncoder
from siamese.data.dataset import MicProjDataset


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese contrastive encoder for cryo-EM particle retrieval."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    # 加载配置
    config = TrainConfig.from_yaml(args.config)
    set_seed(config.seed)

    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}, Backbone: {config.backbone}")
    print(f"Embedding dim: {config.embedding_dim}, Batch size: {config.batch_size}")

    # 创建数据集
    train_dataset = MicProjDataset(
        data_dir=config.data_dir,
        split="train",
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.seed,
    )
    val_dataset = MicProjDataset(
        data_dir=config.data_dir,
        split="val",
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.seed,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # InfoNCE 需要固定 batch size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 创建模型
    model = SiameseEncoder(
        backbone_name=config.backbone,
        image_size=config.image_size,
        real_in_channels=config.real_in_channels,
        freq_in_channels=config.freq_in_channels,
        embedding_dim=config.embedding_dim,
        convnext_depths=tuple(config.convnext_depths),
        convnext_dims=tuple(config.convnext_dims),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 训练
    history = trainer.train()

    print(f"Training complete. Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to {config.checkpoint_dir}")
    return history


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 提交**

```bash
git add scripts/train.py configs/default.yaml
git commit -m "feat: add training CLI script and default config"
```

---

### Task 16: 运行脚本 — eval.py

**Files:**
- Create: `scripts/eval.py`

- [ ] **Step 1: 实现评估脚本**

创建 `scripts/eval.py`:

```python
#!/usr/bin/env python
"""
评估训练好的 Siamese 编码器的检索性能。

用法:
    python scripts/eval.py --checkpoint checkpoints/best.pt --data-dir data/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from siamese.training.config import TrainConfig
from siamese.models.encoder import SiameseEncoder
from siamese.data.dataset import MicProjDataset
from siamese.eval.retrieval import build_faiss_index, retrieve_topk
from siamese.eval.metrics import compute_accuracy_at_k, plot_retrieval_results, plot_tsne


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Siamese encoder retrieval performance."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory.")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image size.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device.")
    parser.add_argument("--k", type=int, default=20,
                        help="Top-k for retrieval.")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory for evaluation outputs.")
    parser.add_argument("--num-visualize", type=int, default=5,
                        help="Number of retrieval examples to visualize.")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Epoch: {checkpoint['epoch'] + 1}, Best val loss: {checkpoint['best_val_loss']:.4f}")

    # 创建模型
    model = SiameseEncoder(
        backbone_name=config.backbone,
        image_size=config.image_size,
        real_in_channels=config.real_in_channels,
        freq_in_channels=config.freq_in_channels,
        embedding_dim=config.embedding_dim,
        convnext_depths=tuple(config.convnext_depths),
        convnext_dims=tuple(config.convnext_dims),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 加载数据
    test_dataset = MicProjDataset(
        data_dir=args.data_dir,
        split="test",
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 加载所有 proj (用于建索引)
    projs = torch.load(Path(args.data_dir) / "projs.pt", weights_only=True)  # [N, D, D]
    from siamese.data.transforms import PreprocessTransform
    transform = PreprocessTransform(normalize=True)

    print(f"Encoding {len(projs)} projs for index...")
    proj_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(projs), args.batch_size)):
            batch = projs[i:i + args.batch_size]
            batch = torch.stack([transform(p) for p in batch]).to(device)  # [B, 1, D, D]
            emb = model(batch)
            proj_embeddings.append(emb.cpu())
    proj_embeddings = torch.cat(proj_embeddings, dim=0)  # [N, D]

    # 建 FAISS 索引
    index = build_faiss_index(proj_embeddings.numpy())
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 编码所有 test mics
    print(f"Encoding {len(test_dataset)} test mics...")
    mic_embeddings = []
    mic_images = []
    gt_indices = []  # 每个 mic 对应的 ground truth proj 索引
    with torch.no_grad():
        for mic, proj in tqdm(test_loader):
            mic = mic.to(device)
            emb = model(mic)
            mic_embeddings.append(emb.cpu())
            mic_images.append(mic.cpu())

            # 找 ground truth: 对每个 mic 在 projs 中找匹配的 proj
            # 由于 proj 也经过了 transform，直接比较 tensor 不现实
            # 这里使用 pairs 文件中的映射
    mic_embeddings = torch.cat(mic_embeddings, dim=0)
    mic_images = torch.cat(mic_images, dim=0)

    # 从 pairs 获取 ground truth
    pairs = torch.load(Path(args.data_dir) / "pairs.pt", weights_only=True)  # [M, 2]
    test_indices = test_dataset.indices
    gt_indices = pairs[test_indices, 0].numpy()  # [N_test], proj index

    # 检索
    print(f"Retrieving top-{args.k}...")
    distances, indices = retrieve_topk(mic_embeddings, index, k=args.k)

    # 计算指标
    N = len(gt_indices)
    random_baseline = 1.0 / len(projs)
    acc = compute_accuracy_at_k(indices, gt_indices, k_values=[1, 5, 10, 20, 50, 100])

    print(f"\n{'='*50}")
    print(f"Evaluation Results (N_test={N}, N_proj={len(projs)})")
    print(f"Random baseline: {random_baseline:.6f}")
    print(f"{'='*50}")
    for k, v in acc.items():
        ratio = v / random_baseline if random_baseline > 0 else float("inf")
        print(f"  {k}: {v:.4f} ({ratio:.1f}x random)")
    print(f"{'='*50}")

    # 保存结果
    results = {
        "accuracies": acc,
        "random_baseline": random_baseline,
        "n_test": N,
        "n_proj": len(projs),
        "checkpoint": args.checkpoint,
    }
    import yaml
    with open(output_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)

    # 可视化 t-SNE
    print("Generating t-SNE visualization...")
    plot_tsne(
        mic_embeddings,
        proj_embeddings,
        save_path=str(output_dir / "tsne.png"),
    )

    # 可视化检索结果 (前几个)
    print(f"Generating retrieval examples...")
    for i in range(min(args.num_visualize, N)):
        plot_retrieval_results(
            query_mic=mic_images[i, 0],  # [D, D]
            retrieved_projs=projs[indices[i]],  # [k, D, D]
            ground_truth=projs[gt_indices[i]],  # [D, D]
            similarities=distances[i],
            k=min(args.k, 10),
            save_path=str(output_dir / f"retrieval_{i}.png"),
        )

    print(f"\nEvaluation results saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/eval.py siamese/eval/__init__.py
git commit -m "feat: add evaluation CLI script with t-SNE and retrieval viz"
```

---

### Task 17: 冒烟测试 — 端到端验证

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: 编写冒烟测试**

创建 `tests/test_smoke.py`:

```python
"""
冒烟测试: 端到端验证数据生成 → 训练 → 检索 → 评估流程。

使用小规模参数 (nside=2, 小图像, 少量 epoch) 快速验证 pipeline。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


def test_smoke_pipeline():
    """
    冒烟测试: 完整 pipeline 验证。

    注意: 此测试需要 GPU。如果 GPU 不可用，跳过。
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # 1. 创建临时 volume 和输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        checkpoint_dir = tmpdir / "checkpoints"
        data_dir.mkdir()
        checkpoint_dir.mkdir()

        # 创建一个小型测试 volume (球体)
        D = 32
        vol = np.zeros((D, D, D), dtype=np.float32)
        center = D // 2
        radius = D // 4
        for z in range(D):
            for y in range(D):
                for x in range(D):
                    if (z - center)**2 + (y - center)**2 + (x - center)**2 < radius**2:
                        vol[z, y, x] = 1.0

        import mrcfile
        map_path = str(tmpdir / "test_volume.map")
        with mrcfile.new(map_path, overwrite=True) as mrc:
            mrc.set_data(vol)

        # 2. 生成模拟数据
        from siamese.data.generate import generate_simulated_data
        metadata = generate_simulated_data(
            map_path=map_path,
            nside=2,  # 48 个方向
            output_dir=str(data_dir),
            image_size=32,
            pixel_size=1.0,
            num_mics_per_proj=2,
            snr_range=(0.001, 0.01),
            defocus_range=(0.5, 4.0),
            max_shift_pixels=3.0,
            device="cuda",
            chunk_size=48,
            seed=42,
        )

        assert metadata["num_directions"] == 48
        assert metadata["num_mics_total"] == 96
        assert (data_dir / "projs.pt").exists()
        assert (data_dir / "mics.pt").exists()
        assert (data_dir / "pairs.pt").exists()
        print(f"Step 1 PASS: Generated {metadata['num_mics_total']} mics")

        # 3. 创建数据集
        from siamese.data.dataset import MicProjDataset
        train_ds = MicProjDataset(data_dir=str(data_dir), split="train",
                                   train_split=0.7, val_split=0.15, seed=42)
        val_ds = MicProjDataset(data_dir=str(data_dir), split="val",
                                 train_split=0.7, val_split=0.15, seed=42)
        test_ds = MicProjDataset(data_dir=str(data_dir), split="test",
                                  train_split=0.7, val_split=0.15, seed=42)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        print(f"Step 2 PASS: Dataset splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        # 4. 创建模型
        from siamese.models.encoder import SiameseEncoder
        model = SiameseEncoder(
            backbone_name="convnext_tiny",
            image_size=32,
            embedding_dim=32,  # 小 embedding
            convnext_depths=(2, 2, 2, 2),     # 最小的 ConvNeXt
            convnext_dims=(32, 64, 128, 256),  # 小通道数
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Step 3 PASS: Model created with {num_params:,} parameters")

        # 5. 快速训练几个 epoch
        from torch.utils.data import DataLoader
        from siamese.training.config import TrainConfig
        from siamese.training.trainer import Trainer

        config = TrainConfig(
            data_dir=str(data_dir),
            image_size=32,
            batch_size=16,
            num_epochs=5,
            learning_rate=1e-3,
            device="cuda",
            checkpoint_dir=str(checkpoint_dir),
            log_interval=5,
        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, drop_last=False)

        trainer = Trainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader)
        history = trainer.train()

        assert len(history["train_losses"]) == config.num_epochs
        assert history["train_losses"][-1] < history["train_losses"][0], \
            "Loss should decrease during training"
        print(f"Step 4 PASS: Training loss: {history['train_losses'][0]:.4f} -> {history['train_losses'][-1]:.4f}")

        # 6. 评估检索
        from siamese.eval.retrieval import build_faiss_index, retrieve_topk
        from siamese.eval.metrics import compute_accuracy_at_k

        model.eval()
        projs_data = torch.load(data_dir / "projs.pt", weights_only=True)
        from siamese.data.transforms import PreprocessTransform
        transform = PreprocessTransform(normalize=True)

        # 编码 proj
        proj_embs = []
        with torch.no_grad():
            for i in range(0, len(projs_data), 16):
                batch = torch.stack([transform(p) for p in projs_data[i:i+16]]).cuda()
                proj_embs.append(model(batch).cpu())
        proj_embs = torch.cat(proj_embs, dim=0)

        # 编码 test mic
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        mic_embs = []
        with torch.no_grad():
            for mic, _ in test_loader:
                mic_embs.append(model(mic.cuda()).cpu())
        mic_embs = torch.cat(mic_embs, dim=0)

        # 建索引 + 检索
        index = build_faiss_index(proj_embs.numpy())
        pairs = torch.load(data_dir / "pairs.pt", weights_only=True)
        gt_indices = pairs[test_ds.indices, 0].numpy()

        distances, retrieved = retrieve_topk(mic_embs, index, k=10)
        acc = compute_accuracy_at_k(retrieved, gt_indices, k_values=[1, 5, 10])

        N = len(projs_data)
        random_baseline = 1.0 / N
        print(f"Step 5 PASS: Retrieval results:")
        print(f"  Random baseline: {random_baseline:.4f}")
        for k, v in acc.items():
            print(f"  {k}: {v:.4f} ({v / random_baseline:.1f}x random)")

        # 冒烟测试通过标准: 至少有一个 top-k 准确率 > 5x random
        assert any(v > 5 * random_baseline for v in acc.values()), \
            f"At least one top-k accuracy should be > 5x random baseline. Got: {acc}"

        print("ALL SMOKE TESTS PASSED!")


def test_smoke_pipeline_cpu():
    """CPU 版本的冒烟测试: 只验证数据生成 + 模型前向传播。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        data_dir.mkdir()

        # 创建测试 volume
        D = 16
        vol = np.ones((D, D, D), dtype=np.float32)
        vol[D//2:, :, :] = 0.5

        import mrcfile
        map_path = str(tmpdir / "test_volume.map")
        with mrcfile.new(map_path, overwrite=True) as mrc:
            mrc.set_data(vol)

        from siamese.data.generate import generate_simulated_data
        metadata = generate_simulated_data(
            map_path=map_path,
            nside=1,  # 12 个方向
            output_dir=str(data_dir),
            image_size=16,
            num_mics_per_proj=1,
            snr_range=(0.01, 0.01),
            defocus_range=(2.0, 2.0),
            max_shift_pixels=1.0,
            device="cpu",
            chunk_size=12,
            seed=42,
        )

        assert metadata["num_directions"] == 12
        print("CPU smoke test PASSED")
```

- [ ] **Step 2: 运行 CPU 冒烟测试**

```bash
.venv/bin/python -m pytest tests/test_smoke.py::test_smoke_pipeline_cpu -v
```
Expected: PASS

- [ ] **Step 3: 运行 GPU 冒烟测试 (如果 GPU 可用)**

```bash
.venv/bin/python -m pytest tests/test_smoke.py::test_smoke_pipeline -v
```
Expected: PASS (GPU) 或 SKIP (无 GPU)

- [ ] **Step 4: 提交**

```bash
git add tests/test_smoke.py
git commit -m "test: add end-to-end smoke test"
```

---

### Task 18: 最终检查 — 运行全部测试

**Files:** 无新文件

- [ ] **Step 1: 运行全部单元测试**

```bash
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_smoke.py
```
Expected: 所有测试 PASS

- [ ] **Step 2: 验证 import 完整性**

```bash
.venv/bin/python -c "
import siamese
from siamese.models import SiameseEncoder, build_backbone, FusionHead
from siamese.data import MicProjDataset, PreprocessTransform, generate_simulated_data
from siamese.losses import InfoNCELoss
from siamese.training import Trainer, TrainConfig
from siamese.eval import build_faiss_index, retrieve_topk, compute_accuracy_at_k
from siamese.utils import compute_ctf, image_to_freq_channels, normalize_image
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "chore: finalize implementation, all tests passing"
```

---

## 自审

### 1. Spec 覆盖检查

| Spec 章节 | 对应 Task |
|-----------|-----------|
| 2.1 数据源 | Task 1 (pyproject.toml 依赖), Task 10 (数据生成) |
| 2.2 生成流程 | Task 10 (generate_simulated_data) |
| 2.3 预处理 | Task 3 (FFT), Task 9 (transforms) |
| 2.4 数据集类 | Task 11 (MicProjDataset) |
| 3.1 双分支编码器 | Task 7 (SiameseEncoder) |
| 3.2 设计要点 | Task 7 (归一化), Task 3 (FFT) |
| 3.3 Backbone | Task 5 (build_backbone) |
| 3.4 可配置项 | Task 4 (TrainConfig) |
| 4.1 InfoNCE | Task 8 (InfoNCELoss) |
| 4.2 训练配置 | Task 4 (TrainConfig), Task 12 (Trainer) |
| 4.3 负样本策略 | Task 8 (TODO 标记) |
| 4.4 预留接口 | Task 4, 5, 7, 8 (各处 TODO) |
| 5.1 评估流程 | Task 13 (retrieval), Task 16 (eval.py) |
| 5.2 评估指标 | Task 13 (metrics) |
| 5.3 可视化 | Task 13 (plot_tsne, plot_retrieval_results) |
| 6. 项目结构 | Task 1 (目录结构) |
| 7. 实现顺序 | Tasks 1-18 |
| 8. 记录轴角 | Task 10 (保存 axisang.pt) |

### 2. Placeholder 扫描

- 无 "TBD" 或 "TODO" 占位（只有明确的后续功能 TODO 标记）
- 所有步骤都有完整代码
- 无 "implement later" 等模糊描述

### 3. 类型一致性检查

- `SiameseEncoder.__init__` 参数在各处一致 (backbone_name, image_size, real_in_channels, freq_in_channels, embedding_dim, convnext_depths, convnext_dims)
- `FusionHead.__init__` 参数: real_dim, freq_dim, hidden_dim, output_dim — 在 Task 7 中正确调用
- `TrainConfig` 字段在 Task 12, 15, 16 中一致使用
- `compute_ctf` 签名在 Task 2 和 Task 10 中一致
- `MicProjDataset` 参数在各处一致 (data_dir, split, train_split, val_split, seed)