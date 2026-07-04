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


def bandpass_filter(
    image: torch.Tensor,
    pixel_size: float,
    low_res: float,
    high_res: float,
) -> torch.Tensor:
    """
    对 2D 图像进行傅里叶空间带通滤波。

    在频域中保留频率分量在 [1/low_res, 1/high_res] 之间的部分，
    其余频率置零。分辨率范围基于 cryo-EM 惯例，数值越大表示越低的分辨率
    （频率越低），数值越小表示越高的分辨率（频率越高）。

    分辨率范围: [奈奎斯特频率, +∞)。奈奎斯特频率 = 2 * pixel_size。
    例如 pixel_size=1.0 时，奈奎斯特分辨率 = 2.0 Å。

    参数:
        image: 形状 [..., D, D] 的实空间图像（batch 维度可选）。
        pixel_size: 像素尺寸（Å/pixel）。
        low_res: 低分辨率截止值（Å），频率低于 1/low_res 的部分被滤除。
                 例如 low_res=20.0 表示滤除 >20 Å 的低频分量。
        high_res: 高分辨率截止值（Å），频率高于 1/high_res 的部分被滤除。
                  例如 high_res=5.0 表示滤除 <5 Å 的高频分量。
                  必须满足 low_res >= high_res >= 2 * pixel_size。

    返回:
        filtered: 形状 [..., D, D] 的滤波后实空间图像。

    示例:
        >>> # 带通滤波: 保留 5-20 Å 分辨率范围
        >>> img = torch.randn(128, 128)
        >>> filtered = bandpass_filter(img, pixel_size=1.0, low_res=20.0, high_res=5.0)
    """
    D = image.shape[-1]  # 图像尺寸

    # --- 参数校验 ---
    nyquist = 2.0 * pixel_size  # 奈奎斯特分辨率 (Å)
    if high_res < nyquist:
        raise ValueError(
            f"high_res ({high_res:.2f} Å) 不能小于奈奎斯特分辨率 "
            f"({nyquist:.2f} Å = 2 × pixel_size)"
        )
    if low_res < high_res:
        raise ValueError(
            f"low_res ({low_res:.2f} Å) 必须 >= high_res ({high_res:.2f} Å)"
        )

    # --- 0. 减去均值，避免 DC 分量淹没其他频率 ---
    # 滤波后再加回均值，保证图像整体亮度不变
    mean = image.mean(dim=(-2, -1), keepdim=True)  # [..., 1, 1]
    image_zero_mean = image - mean  # [..., D, D]

    # --- 1. FFT2 变换到频域 ---
    # 使用 norm='backward' (默认) 保证 FFT → IFFT 来回后值域不变
    ft = torch.fft.fft2(image_zero_mean, dim=(-2, -1), norm="backward")  # [..., D, D] complex
    ft = torch.fft.fftshift(ft, dim=(-2, -1))  # 低频移到中心

    # --- 2. 构建频率网格 ---
    # 频率坐标: 以 0 为中心，范围 [-D/2, D/2)
    # fftfreq 返回 corner-based 频率 (DC 在 index 0), 需要 fftshift 对齐 fftshift 后的频谱
    # 频率单位: 1/Å，即每个像素对应的空间频率 = 1 / (pixel_size * D)
    freq_1d = torch.fft.fftfreq(D, d=pixel_size, dtype=torch.float32, device=image.device)  # [D]
    freq_1d = torch.fft.fftshift(freq_1d)  # 对齐 fftshift 后的频谱 (DC 在中心)
    fy, fx = torch.meshgrid(freq_1d, freq_1d, indexing="ij")  # 各 [D, D]
    freq_mag = torch.sqrt(fx**2 + fy**2)  # [D, D], 径向频率 (1/Å)

    # --- 3. 构建带通掩膜 ---
    # 分辨率 R 对应的空间频率 f = 1 / R (单位: 1/Å)
    # 保留频率范围: high_freq <= freq_mag <= low_freq
    # 注意: high_res 对应低频率, low_res 对应高频率
    high_freq = 1.0 / high_res  # 高频截止 (1/Å)
    low_freq = 1.0 / low_res    # 低频截止 (1/Å)

    # 掩膜: 1 表示保留, 0 表示滤除
    mask = torch.ones_like(freq_mag)
    mask[freq_mag < low_freq] = 0.0   # 滤除低频（低于 low_freq 的分量）
    mask[freq_mag > high_freq] = 0.0  # 滤除高频（高于 high_freq 的分量）

    # --- 4. 应用掩膜 ---
    ft_filtered = ft * mask  # [..., D, D] complex

    # --- 5. IFFT 变换回实空间 ---
    ft_filtered = torch.fft.ifftshift(ft_filtered, dim=(-2, -1))  # 低频移回角落
    filtered = torch.fft.ifft2(ft_filtered, dim=(-2, -1), norm="backward")  # [..., D, D] complex
    filtered = filtered.real  # [..., D, D], 取实部（理论上虚部为 0）

    # --- 6. 加回均值 ---
    filtered = filtered + mean  # mean 形状 [..., 1, 1], 自动 broadcast

    return filtered


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