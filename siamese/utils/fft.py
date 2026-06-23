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