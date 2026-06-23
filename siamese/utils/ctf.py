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