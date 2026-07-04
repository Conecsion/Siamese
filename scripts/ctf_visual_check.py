#!/usr/bin/env python
"""
CTF 端到端可视化检查 (T20S)。

流程: 5 个随机朝向
  1. 无 CTF 干净投影 (clean)
  2. 加 CTF 版本 (ctf)
  3. 对 ctf 版本做 Wiener 校正 (corrected)
共 5x3=15 张图像 + 15 张功率谱 = 30 张, 保存供人工检查。

用途: 验证 CTF 修复 (sin/cos 同号) 后, 加 CTF 与校正的视觉/频谱行为是否合理。
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch

from siamese.data.projection import (
    project_fourier_slice_from_axis_angle,
    _compute_ctf_2d,
)

# --- 配置 (T20S J17) ---
VOL_PATH = "data/cs_processed/t20s_homerefine/J17_volume/J17/J17_006_volume_map.mrc"
OUT_DIR = Path("ctf_check")
N_PROJ = 5
PSIZE = 0.6575
KV, CS, AC = 300.0, 2.7, 0.10
DF1, DF2, DF_ANGLE = 12444.5, 12331.6, 4.6958
PHASE_SHIFT, SIGN = 0.0, -1.0
WIENER_LAMBDA = 0.1
SEED = 0


def power_spectrum(img: np.ndarray) -> np.ndarray:
    """log 功率谱 (fftshift, DC 居中)。"""
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F) ** 2)


def wiener_correct(img: np.ndarray, ctf: np.ndarray, lam: float) -> np.ndarray:
    """Wiener 校正: F' = F * CTF / (CTF^2 + lam)，DC 在角 (fft 原生序)。"""
    F = np.fft.fft2(img)
    H = ctf / (ctf ** 2 + lam)           # ctf 已是 fft 原生序 (DC 角)
    return np.fft.ifft2(F * H).real.astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.simplefilter("ignore")

    with mrcfile.open(VOL_PATH, permissive=True) as m:
        vol = torch.from_numpy(np.asarray(m.data, np.float32).copy()).to(device)
    D = vol.shape[0]
    print(f"volume {tuple(vol.shape)}, device {device}")

    # 随机朝向
    g = torch.Generator(device=device).manual_seed(SEED)
    aa = torch.randn(N_PROJ, 3, generator=g, device=device)
    aa = aa / aa.norm(dim=1, keepdim=True) * torch.rand(N_PROJ, 1, generator=g, device=device) * np.pi

    ctf_kwargs = dict(
        apply_ctf=True, psize=PSIZE, ctf_voltage=KV, ctf_cs=CS,
        ctf_amp_contrast=AC, ctf_df_u=DF1, ctf_df_v=DF2,
        ctf_df_angle=DF_ANGLE, ctf_phase_shift=PHASE_SHIFT, ctf_particle_sign=SIGN,
    )

    # CTF 2D (居中网格, 用于校正需转 fft 原生序)
    k = torch.arange(-D // 2, D // 2, dtype=torch.float32)
    kx, ky = torch.meshgrid(k, k, indexing="xy")
    ctf2d_centered = _compute_ctf_2d(kx, ky, D, PSIZE, DF1, DF2, DF_ANGLE,
                                     KV, CS, AC, PHASE_SHIFT, SIGN).numpy()
    ctf2d_fft = np.fft.ifftshift(ctf2d_centered)  # DC 移到角

    with torch.no_grad():
        clean = project_fourier_slice_from_axis_angle(
            vol, aa, pfac=2, normalize=True, noise_model="none"
        ).cpu().numpy()  # [N, D, D]
        ctfed = project_fourier_slice_from_axis_angle(
            vol, aa, pfac=2, normalize=True, noise_model="none", **ctf_kwargs
        ).cpu().numpy()  # [N, D, D]

    # 校正
    corrected = np.stack([wiener_correct(ctfed[i], ctf2d_fft, WIENER_LAMBDA)
                          for i in range(N_PROJ)])

    # --- 绘图: 每个朝向一行 6 列 (clean/ctf/corr 图像 + 各自功率谱) ---
    fig, axes = plt.subplots(N_PROJ, 6, figsize=(20, 3.4 * N_PROJ))
    col_titles = ["clean", "clean PS", "ctf", "ctf PS", "wiener", "wiener PS"]
    for i in range(N_PROJ):
        imgs = [clean[i], ctfed[i], corrected[i]]
        cells = [imgs[0], power_spectrum(imgs[0]),
                 imgs[1], power_spectrum(imgs[1]),
                 imgs[2], power_spectrum(imgs[2])]
        for j, cell in enumerate(cells):
            ax = axes[i, j]
            cmap = "gray" if j % 2 == 0 else "viridis"
            ax.imshow(cell, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=12)
            if j == 0:
                ax.set_ylabel(f"proj {i}", fontsize=11)
    fig.suptitle(
        f"T20S CTF check (df1={DF1:.0f} df2={DF2:.0f} Å, psize={PSIZE} Å, "
        f"wiener λ={WIENER_LAMBDA})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = OUT_DIR / "ctf_check_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")

    # 单独存每张 (30 张), 便于放大检查
    for i in range(N_PROJ):
        for name, img in [("clean", clean[i]), ("ctf", ctfed[i]), ("wiener", corrected[i])]:
            for kind, data in [("img", img), ("ps", power_spectrum(img))]:
                f, a = plt.subplots(figsize=(4, 4))
                a.imshow(data, cmap="gray" if kind == "img" else "viridis")
                a.set_title(f"proj{i} {name} {kind}"); a.axis("off")
                f.savefig(OUT_DIR / f"proj{i}_{name}_{kind}.png", dpi=110, bbox_inches="tight")
                plt.close(f)
    print(f"saved 30 individual images to {OUT_DIR}/")


if __name__ == "__main__":
    main()
