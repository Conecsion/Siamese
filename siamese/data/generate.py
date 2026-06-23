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

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from torch.fft import fftshift, fft2, ifft2, fftfreq
from tqdm import tqdm

from siamese.utils.ctf import compute_ctf

# 将项目根目录加入 sys.path，以便导入根目录下的 project.py
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from project import healpix_project


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
    volume = torch.from_numpy(volume_np)

    num_directions = 12 * nside * nside
    print(f"Generating {num_directions} HEALPix directions (nside={nside})...")

    # 2. 生成 clean projs（不含 shift、CTF、噪声）
    #    healpix_project 返回的 projs 在 CPU 上, axisang 也在 CPU 上
    projs, axisang = healpix_project(
        volume=volume,
        nside=nside,
        device=device,
        chunk_size=chunk_size,
    )  # projs: [N, D_vol, D_vol], axisang: [N, 3]

    D_vol = projs.shape[1]
    print(f"Projections shape: {projs.shape}, dtype: {projs.dtype}")

    # 3. 如果 volume 尺寸 != image_size，裁剪中心区域
    if D_vol != image_size:
        margin = (D_vol - image_size) // 2
        projs = projs[:, margin:margin + image_size, margin:margin + image_size]
        print(f"Cropped projections to {image_size}x{image_size}")

    total_mics = num_directions * num_mics_per_proj

    # 4. 为每个 mic 随机采样 SNR, defocus, shift
    #    shift_values 单位是 pixels
    snr_values = rng.uniform(snr_range[0], snr_range[1], size=total_mics)
    snr_values = snr_values.reshape(num_directions, num_mics_per_proj)

    defocus_values = rng.uniform(defocus_range[0], defocus_range[1], size=total_mics)
    defocus_values = defocus_values.reshape(num_directions, num_mics_per_proj)

    # shift_values: [num_directions, num_mics_per_proj, 2], 单位 pixels
    shift_values = rng.uniform(-max_shift_pixels, max_shift_pixels, size=(total_mics, 2))
    shift_values = shift_values.reshape(num_directions, num_mics_per_proj, 2)

    mics_list = []
    pairs_list = []  # (proj_idx, mic_global_idx)

    print(f"Generating {total_mics} noisy micrographs...")
    mic_idx = 0
    for i in tqdm(range(num_directions)):
        proj_i = projs[i]  # [D, D], CPU

        for j in range(num_mics_per_proj):
            # 获取当前 mic 的参数
            snr = float(snr_values[i, j])
            defocus = float(defocus_values[i, j])
            # shift 已经是 pixels 单位，不需要再除以 pixel_size
            shift = torch.from_numpy(shift_values[i, j]).float().unsqueeze(0)  # [1, 2]

            # 生成 CTF（频域，DC 在 corner）
            ctf = compute_ctf(
                image_size=image_size,
                pixel_size=pixel_size,
                defocus=defocus,
                cs=cs,
                voltage=voltage,
                amplitude_contrast=amplitude_contrast,
                device=device,
            )  # [D, D] complex, DC 在 corner

            # 将 proj 移至 GPU: [D, D] -> [1, D, D]
            mic_proj = proj_i.unsqueeze(0).to(device)  # [1, D, D]

            # ---- 正向 FFT: 实空间 -> 频域，DC 居中 ----
            # 与 project.py 保持一致: fftshift -> fft2 -> fftshift
            #   1. fftshift: 实空间 DC 由 center → corner
            #   2. fft2(..., norm='ortho'): FFT
            #   3. fftshift: 频域 DC 由 corner → center
            mic_ft = fftshift(
                fft2(fftshift(mic_proj, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
                dim=(-2, -1),
            )  # [1, D, D] complex, DC 居中

            # ---- 应用 shift（频域相位调制）----
            # shift 单位 pixels, fftfreq 单位 cycles/pixel
            D = image_size
            dx = shift[:, 0].view(1, 1, 1).to(device)  # [1, 1, 1]
            dy = shift[:, 1].view(1, 1, 1).to(device)  # [1, 1, 1]
            shift_freq = fftfreq(D, device=device)  # [D], [-0.5, 0.5)
            shift_freq = fftshift(shift_freq)  # 居中，与频域数据 DC 居中一致
            shift_fy, shift_fx = torch.meshgrid(shift_freq, shift_freq, indexing="ij")
            # phase = exp(-2πi * (fx * dx + fy * dy))
            shift_angle = -2.0 * math.pi * (
                shift_fx[None, :, :] * dx + shift_fy[None, :, :] * dy
            )  # [1, D, D]
            shift_phase = torch.polar(torch.ones_like(shift_angle), shift_angle)
            mic_ft = mic_ft * shift_phase

            # ---- 应用 CTF ----
            # compute_ctf 返回的 CTF 的 DC 在 corner，需要 fftshift 到 center
            if ctf.dim() == 2:
                ctf = ctf.unsqueeze(0)  # [1, D, D]
            ctf_centered = fftshift(ctf, dim=(-2, -1))  # DC corner → center
            mic_ft = mic_ft * ctf_centered.to(device)

            # ---- 逆 FFT: 频域 -> 实空间 ----
            # 与 project.py 保持一致: fftshift -> ifft2 -> fftshift -> .real -> * sqrt(D)
            #   1. fftshift: 频域 DC 由 center → corner
            #   2. ifft2(..., norm='ortho'): IFFT
            #   3. fftshift: 实空间 DC 由 corner → center
            #   4. .real: 取实部
            #   5. * sqrt(D): 补偿 ortho 归一化的 1/sqrt(D) 因子
            mic = ifft2(
                fftshift(mic_ft, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
            )
            mic = fftshift(mic, dim=(-2, -1)).real * math.sqrt(D)  # [1, D, D]

            # ---- 加噪声 ----
            # SNR = var_signal / var_noise
            x0 = mic - mic.mean(dim=(-2, -1), keepdim=True)  # [1, D, D]
            var_s = (x0 ** 2).mean(dim=(-2, -1))  # [1,]
            var_n = var_s / (snr + 1e-8)  # [1,]
            sigma_n = torch.sqrt(torch.clamp(var_n, min=1e-8))  # [1,]
            g = torch.Generator(device=device)
            g.manual_seed(seed + mic_idx)
            noise = torch.randn(
                mic.shape, generator=g, device=device, dtype=mic.dtype
            ) * sigma_n.view(-1, 1, 1)  # [1, D, D]
            mic = mic + noise

            mics_list.append(mic.squeeze(0).cpu())  # [D, D]
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