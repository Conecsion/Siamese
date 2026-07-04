"""
图像重采样工具（Fourier binning 方法）。

所有降采样统一使用 Fourier binning：
1. FFT 到频域
2. 裁剪到目标尺寸（截断高频，防止混叠）
3. 逆 FFT 回实空间

这确保了无混叠伪影的降采样，优于直接空间域插值。
"""

import numpy as np
import torch
from torch import Tensor


def fourier_crop_2d(img: np.ndarray | Tensor, new_size: int) -> np.ndarray | Tensor:
    """
    2D Fourier binning 降采样（无混叠）。

    参数:
        img: [H, W] 输入图像（numpy 或 torch）
        new_size: 目标尺寸（正方形）

    返回:
        [new_size, new_size] 降采样后的图像
    """
    is_torch = isinstance(img, Tensor)
    if is_torch:
        device = img.device
        img_np = img.cpu().numpy()
    else:
        img_np = img

    old_size = img_np.shape[0]
    assert img_np.shape[0] == img_np.shape[1], "仅支持正方形图像"

    if old_size == new_size:
        return img

    # FFT 到频域并中心化
    img_fft = np.fft.fftshift(np.fft.fft2(img_np))

    # 裁剪中心区域（截断高频）
    center = old_size // 2
    half_new = new_size // 2
    img_fft_cropped = img_fft[center-half_new:center+half_new,
                               center-half_new:center+half_new]

    # 逆 FFT 回实空间
    img_cropped = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_cropped)))

    # 能量归一化（Fourier crop 会改变总能量）
    img_cropped *= (new_size / old_size) ** 2

    if is_torch:
        return torch.from_numpy(img_cropped.astype(np.float32)).to(device)
    return img_cropped.astype(np.float32)


def fourier_crop_3d(vol: np.ndarray, new_size: int) -> np.ndarray:
    """
    3D Fourier binning 降采样（无混叠）。

    参数:
        vol: [D, D, D] 输入 volume（numpy）
        new_size: 目标尺寸（立方体）

    返回:
        [new_size, new_size, new_size] 降采样后的 volume
    """
    old_size = vol.shape[0]
    assert vol.shape == (old_size, old_size, old_size), "仅支持立方体 volume"

    if old_size == new_size:
        return vol

    # 3D FFT 到频域并中心化
    vol_fft = np.fft.fftshift(np.fft.fftn(vol))

    # 裁剪中心区域（截断高频）
    center = old_size // 2
    half_new = new_size // 2
    vol_fft_cropped = vol_fft[center-half_new:center+half_new,
                               center-half_new:center+half_new,
                               center-half_new:center+half_new]

    # 逆 FFT 回实空间
    vol_cropped = np.real(np.fft.ifftn(np.fft.ifftshift(vol_fft_cropped)))

    # 能量归一化
    vol_cropped *= (new_size / old_size) ** 3

    return vol_cropped.astype(np.float32)


def fourier_pad_2d(img: np.ndarray | Tensor, new_size: int) -> np.ndarray | Tensor:
    """
    2D Fourier padding 升采样（在频域补零）。

    参数:
        img: [H, W] 输入图像（numpy 或 torch）
        new_size: 目标尺寸（正方形）

    返回:
        [new_size, new_size] 升采样后的图像
    """
    is_torch = isinstance(img, Tensor)
    if is_torch:
        device = img.device
        img_np = img.cpu().numpy()
    else:
        img_np = img

    old_size = img_np.shape[0]
    assert img_np.shape[0] == img_np.shape[1], "仅支持正方形图像"

    if old_size == new_size:
        return img

    # FFT 到频域并中心化
    img_fft = np.fft.fftshift(np.fft.fft2(img_np))

    # 创建更大的频域数组（边缘补零）
    img_fft_padded = np.zeros((new_size, new_size), dtype=img_fft.dtype)
    start_new = (new_size - old_size) // 2
    img_fft_padded[start_new:start_new+old_size,
                   start_new:start_new+old_size] = img_fft

    # 逆 FFT 回实空间
    img_padded = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_padded)))

    # 能量归一化
    img_padded *= (new_size / old_size) ** 2

    if is_torch:
        return torch.from_numpy(img_padded.astype(np.float32)).to(device)
    return img_padded.astype(np.float32)


# 保留旧接口兼容性
DEFAULT_BUCKETS = (64, 128, 256, 384, 512)


def resample_to_working_ps(
    img: Tensor,
    orig_psize: float,
    working_ps: float,
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
) -> tuple[Tensor, int, float]:
    """
    重采样图像到工作 pixel size 并分桶。

    参数:
        img: [H, W] 输入图像
        orig_psize: 原始 pixel size (Å)
        working_ps: 目标 pixel size (Å)
        buckets: 桶尺寸列表

    返回:
        (重采样后的图像, 桶尺寸, 实际 pixel size)
    """
    scale = orig_psize / working_ps
    new_size = int(np.round(img.shape[-1] * scale))

    # 找最近的桶
    bucket = min(buckets, key=lambda b: abs(b - new_size))

    # 两步: 1) 重采样到 new_size，2) pad/crop 到 bucket
    if img.shape[-1] != new_size:
        if img.shape[-1] > new_size:
            img = fourier_crop_2d(img, new_size)
        else:
            img = fourier_pad_2d(img, new_size)

    # 再 pad/crop 到最近的桶
    if new_size > bucket:
        img_resampled = fourier_crop_2d(img, bucket)
    elif new_size < bucket:
        img_resampled = fourier_pad_2d(img, bucket)
    else:
        img_resampled = img

    # 实际 pixel size
    actual_ps = orig_psize * img.shape[-1] / bucket

    return img_resampled, bucket, actual_ps
