"""
模拟训练数据生成。

主要输出 (projs.pt / mics.pt / axisang.pt / pairs.pt / metadata.yaml):
  projs   [N, D, D] — 无 CTF / 无噪声的干净投影 (proj 分支用)
  mics    [M, D, D] — 含 CTF + 随机位移 + 高斯噪声的模拟 mic (particle 分支用)
  axisang [N, 3]    — projs 对应的轴角向量 (弧度, pyem / cryoSPARC 约定)
  pairs   [M, 2]    — (proj_idx, mic_idx) 配对索引

可选额外导出 (export_format="cryosparc" / "relion"):
  由 siamese.data.export 完成, 同时写 .mrcs + .cs / .star。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import mrcfile
import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from siamese.data.orientations import healpix_axis_angles, uniform_so3_axis_angles
from siamese.data.projection import project_fourier_slice_from_axis_angle


def generate_simulated_data(
    map_path: str,
    nside: int,
    output_dir: str,
    *,
    orientation_mode: Literal["healpix", "uniform"] = "healpix",
    n_inplane: int = 1,
    image_size: int = 128,
    pixel_size: float = 1.0,
    num_mics_per_proj: int = 2,
    snr_range: Tuple[float, float] = (0.001, 0.01),
    defocus_range: Tuple[float, float] = (0.5, 4.0),
    max_shift_pixels: float = 5.0,
    cs: float = 2.7,
    voltage: float = 300.0,
    amplitude_contrast: float = 0.07,
    device: str = "cuda",
    chunk_size: Optional[int] = None,
    seed: int = 42,
    export_format: Literal["none", "cryosparc", "relion"] = "none",
    export_mrcs_name: str = "particles.mrcs",
) -> dict:
    """
    从 3D volume 生成配对模拟训练数据并保存到 output_dir。

    参数:
        map_path: 3D volume (.map/.mrc) 路径。
        nside: HEALPix nside; 方向数 = 12*nside^2 (uniform 模式同) * n_inplane。
        output_dir: 输出目录。
        orientation_mode: "healpix" (均匀格点) 或 "uniform" (均匀随机)。
        n_inplane: 每个 HEALPix 方向的面内角数量。
        image_size: 输出图像边长 D (volume 超出时裁中心)。
        pixel_size: 像素大小 (Å/pixel)。
        num_mics_per_proj: 每投影生成几个 mic (不同噪声/CTF)。
        snr_range: mic 的 SNR 范围。
        defocus_range: 散焦范围 (μm)。
        max_shift_pixels: 最大随机平移 (像素)。
        cs, voltage, amplitude_contrast: CTF 参数。
        device: "cuda" 或 "cpu"。
        chunk_size: batch 分块大小; None = 按显存自动选。
        seed: 随机种子。
        export_format: 额外导出格式 ("none" / "cryosparc" / "relion")。
        export_mrcs_name: 导出 .mrcs 文件名。

    返回:
        metadata dict。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    # --- 1. 读取 volume ---
    with mrcfile.open(map_path, permissive=True) as mrc:
        vol = torch.from_numpy(np.asarray(mrc.data, dtype=np.float32).copy()).to(dev)
    D_vol = vol.shape[0]

    # --- 2. 采样朝向 -> 轴角 [N, 3] ---
    if orientation_mode == "healpix":
        aa = healpix_axis_angles(nside, n_inplane=n_inplane, device=dev)
    else:
        g = torch.Generator(); g.manual_seed(seed)
        aa = uniform_so3_axis_angles(12 * nside * nside, generator=g, device=dev)
    N = aa.shape[0]
    print(f"Orientations: {N} ({orientation_mode}, nside={nside})")

    # --- 3. 干净投影 [N, D, D] (无 CTF / 无噪声, CLAUDE.md 约定) ---
    print(f"Projecting {N} orientations on {device} ...")
    with torch.no_grad():
        projs = project_fourier_slice_from_axis_angle(
            vol, aa, pfac=2, normalize=True, noise_model="none", chunk_size=chunk_size,
        )  # [N, D_vol, D_vol]
    m = (D_vol - image_size) // 2 if D_vol != image_size else 0
    if D_vol != image_size:
        projs = projs[:, m : m + image_size, m : m + image_size]
    projs = projs.cpu()
    print(f"Projections: {tuple(projs.shape)}")

    # --- 4. Mics (CTF + shift + noise 逐颗粒) ---
    M = N * num_mics_per_proj
    snr_all  = rng.uniform(*snr_range,          size=(N, num_mics_per_proj))
    df_all   = rng.uniform(*defocus_range,       size=(N, num_mics_per_proj))  # μm
    shft_all = rng.uniform(-max_shift_pixels, max_shift_pixels,
                           size=(N, num_mics_per_proj, 2))
    mics_list: list[Tensor] = []
    pairs_list: list[tuple[int, int]] = []
    mic_id = 0
    for i in tqdm(range(N), desc="Mics"):
        for j in range(num_mics_per_proj):
            snr = float(snr_all[i, j])
            df_A = float(df_all[i, j]) * 1e4  # μm -> Å
            sx, sy = float(shft_all[i, j, 0]), float(shft_all[i, j, 1])
            shift_t = torch.tensor([[sx, sy]], device=dev)
            with torch.no_grad():
                mic = project_fourier_slice_from_axis_angle(
                    vol, aa[i : i + 1], shifts=shift_t, pfac=2,
                    normalize=True, noise_model="none",
                    apply_ctf=True, psize=pixel_size,
                    ctf_voltage=voltage, ctf_cs=cs,
                    ctf_amp_contrast=amplitude_contrast,
                    ctf_df_u=df_A, ctf_df_v=df_A, ctf_particle_sign=-1.0,
                )[0]  # [D_vol, D_vol]
            if D_vol != image_size:
                mic = mic[m : m + image_size, m : m + image_size]
            # 加噪声
            x0 = mic - mic.mean()
            sigma_n = torch.sqrt((x0 ** 2).mean() / (snr + 1e-8)).clamp(min=1e-8)
            g_t = torch.Generator(device=dev); g_t.manual_seed(seed + mic_id)
            mic = mic + torch.randn(mic.shape, generator=g_t, device=dev) * sigma_n
            mics_list.append(mic.cpu())
            pairs_list.append((i, mic_id))
            mic_id += 1

    mics   = torch.stack(mics_list)                       # [M, D, D]
    pairs  = torch.tensor(pairs_list, dtype=torch.long)   # [M, 2]
    axisang = aa.cpu()                                    # [N, 3]

    # --- 5. 保存训练张量 ---
    torch.save(projs,   output_dir / "projs.pt")
    torch.save(mics,    output_dir / "mics.pt")
    torch.save(axisang, output_dir / "axisang.pt")
    torch.save(pairs,   output_dir / "pairs.pt")
    meta = dict(
        map_path=str(map_path), nside=nside, orientation_mode=orientation_mode,
        n_inplane=n_inplane, num_directions=N, num_mics_total=M,
        num_mics_per_proj=num_mics_per_proj, image_size=image_size,
        pixel_size=pixel_size, snr_range=list(snr_range),
        defocus_range=list(defocus_range), max_shift_pixels=max_shift_pixels,
        cs=cs, voltage=voltage, amplitude_contrast=amplitude_contrast, seed=seed,
    )
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)
    print(f"Saved {N} projs, {M} mics to {output_dir}")

    # --- 6. 可选格式导出 ---
    if export_format != "none":
        from siamese.data.export import write_cryosparc_cs, write_relion_star
        mrcs_path = output_dir / export_mrcs_name
        with mrcfile.new(str(mrcs_path), overwrite=True) as mrc:
            mrc.set_data(projs.numpy().astype(np.float32))
            mrc.voxel_size = pixel_size
        aa_np = axisang.numpy()
        if export_format == "cryosparc":
            write_cryosparc_cs(aa_np, output_dir / "particles.cs", mrcs_path, pixel_size)
        else:
            write_relion_star(aa_np, output_dir / "particles.star", mrcs_path, pixel_size)

    return meta
