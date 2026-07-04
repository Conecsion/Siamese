#!/usr/bin/env python
"""
颗粒 CTF 校正。

读取 cryoSPARC .cs 文件，对每个颗粒图像应用 CTF 校正，
输出校正后的颗粒堆栈 (.mrcs) 和更新的 .cs 文件。

三种校正方法:
  phase_flip   : F' = F * sign(CTF)
  multiply_ctf : F' = F * CTF   (等效 CTF² 加权，同样修正相位)
  wiener       : F' = F * CTF / (CTF² + λ(k))

λ(k) 模型 (wiener 专用):
  constant    : λ(k) = λ₀
  linear      : λ(k) = λ₀ · k / k_nyq
  quadratic   : λ(k) = λ₀ · (k / k_nyq)²  ← 推荐，高频噪声惩罚更重

用法:
    python scripts/ctf_correct.py particles.cs output_dir/ \\
        --method wiener --wiener-lambda 0.1 --lambda-model quadratic \\
        --cs-project-dir /path/to/cryosparc/project
"""

import argparse
import math
import warnings
from collections import defaultdict
from pathlib import Path

import mrcfile
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CTF 计算 (DC 在 [0,0]，与 np.fft.fft2 输出顺序一致)
# ---------------------------------------------------------------------------

def _ctf_batch(
    D: int,
    psize: float,
    df1_A: np.ndarray,        # [B] float32
    df2_A: np.ndarray,        # [B]
    df_angle_rad: np.ndarray, # [B]
    voltage_kv: float,
    cs_mm: float,
    amp_contrast: float,
    phase_shift_rad: np.ndarray,  # [B]
    particle_sign: np.ndarray,    # [B]
    bfactor: np.ndarray,          # [B]
    device: torch.device,
) -> torch.Tensor:
    """
    批量计算 2D 各向异性 CTF，频率按 np.fft.fft2 约定排列 (DC=[0,0])。

    参数:
        D:      图像边长 (像素)
        psize:  像素大小 (Å/pixel)
        df1_A .. bfactor: 每颗粒 CTF 参数，各形状 [B]

    返回:
        ctf: 形状 [B, D, D] 的 float32 张量
    """
    # 空间频率网格 (Å^-1)，形状 [D, D]
    freq = torch.fft.fftfreq(D, d=psize, device=device)
    kx, ky = torch.meshgrid(freq, freq, indexing="xy")
    k2 = kx ** 2 + ky ** 2          # [D, D]
    phi = torch.atan2(ky, kx)        # [D, D]

    # 相对论电子波长 (Å)
    V = voltage_kv * 1e3
    lam = 12.2643247 / math.sqrt(V * (1.0 + 0.978466e-6 * V))
    cs_A = cs_mm * 1e7               # mm → Å

    # 广播到 [B, 1, 1]
    B = len(df1_A)
    def _v(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device).view(B, 1, 1)

    df1 = _v(df1_A); df2 = _v(df2_A)
    df_ang = _v(df_angle_rad); ps = _v(phase_shift_rad); sign = _v(particle_sign)
    bf = _v(bfactor)

    # 各向异性散焦 (Å)，[B, D, D]
    df = 0.5 * (df1 + df2 + (df1 - df2) * torch.cos(2.0 * (phi - df_ang)))

    # CTF 相位 γ(k), [B, D, D]
    gamma = (
        math.pi * lam * df * k2
        - 0.5 * math.pi * lam ** 3 * cs_A * k2 ** 2
        + ps
    )

    w1 = math.sqrt(max(1.0 - amp_contrast ** 2, 0.0))
    # CTF 两项同号 (与 RELION/EMAN2/pyem 一致); 之前 +Q*cos 异号是 bug。
    ctf = sign * (-w1 * torch.sin(gamma) - amp_contrast * torch.cos(gamma))

    # B 因子衰减（仅非零时计算）
    mask = bf != 0.0
    if mask.any():
        ctf = ctf * torch.exp(-bf * k2 / 4.0)

    return ctf  # [B, D, D]


# ---------------------------------------------------------------------------
# 校正滤波器
# ---------------------------------------------------------------------------

def _build_filters(
    ctf: torch.Tensor,   # [B, D, D]
    method: str,
    lam0: float,
    lam_model: str,
    k_nyq: float,
) -> torch.Tensor:
    """
    根据 method 构建频域校正滤波器，输出与 ctf 同形状 [B, D, D]。
    """
    if method == "phase_flip":
        return torch.sign(ctf)
    if method == "multiply_ctf":
        return ctf

    # wiener: H = CTF / (CTF² + λ(k))
    D = ctf.shape[-1]
    psize_v = 1.0 / (2.0 * k_nyq)
    freq = torch.fft.fftfreq(D, d=psize_v, device=ctf.device)
    kx, ky = torch.meshgrid(freq, freq, indexing="xy")
    k = torch.sqrt(kx ** 2 + ky ** 2)  # [D, D]

    if lam_model == "constant":
        lam: torch.Tensor = torch.full_like(k, lam0)
    elif lam_model == "linear":
        lam = lam0 * k / k_nyq
    elif lam_model == "quadratic":
        lam = lam0 * (k / k_nyq) ** 2
    else:
        raise ValueError(f"未知 lambda_model: {lam_model!r}")

    return ctf / (ctf ** 2 + lam)   # [B, D, D]


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CTF correction for cryoSPARC particles (.cs format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cs",           help="输入 .cs 文件路径")
    parser.add_argument("output_dir",   help="输出目录（自动创建）")
    parser.add_argument("--method", choices=["phase_flip", "multiply_ctf", "wiener"],
                        default="phase_flip")
    parser.add_argument("--wiener-lambda", type=float, default=0.1, metavar="LAMBDA",
                        help="Wiener λ₀ (越大越平滑)")
    parser.add_argument("--lambda-model", choices=["constant", "linear", "quadratic"],
                        default="constant", help="λ(k) 频率依赖模型")
    parser.add_argument("--cs-project-dir", default=None,
                        help="cryoSPARC 项目根目录，用于解析 blob/path")
    parser.add_argument("--output-name", default="particles_ctf_corrected",
                        help="输出文件基础名（不含扩展名）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="每批处理的颗粒数")
    args = parser.parse_args()

    cs_path  = Path(args.cs).resolve()
    out_dir  = Path(args.output_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    proj_base = Path(args.cs_project_dir) if args.cs_project_dir else cs_path.parent
    out_mrcs = out_dir / (args.output_name + ".mrcs")
    out_cs   = out_dir / (args.output_name + ".cs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 .cs 元数据
    cs_data = np.load(str(cs_path))
    N = len(cs_data)
    D = int(cs_data[0]["blob/shape"][0])
    psize = float(cs_data[0]["blob/psize_A"])
    print(f"颗粒数: {N}，尺寸: {D}×{D}，psize: {psize} Å，方法: {args.method}，设备: {device}")

    # 按源 mrc 文件分组 -> {rel_path: [global_idx, ...]}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(cs_data):
        groups[p["blob/path"].decode()].append(i)

    # 预分配输出 .mrcs
    with mrcfile.new_mmap(str(out_mrcs), shape=(N, D, D), mrc_mode=2, overwrite=True) as mrc_out:
        mrc_out.voxel_size = psize

        # 公共 CTF 参数（同批次同电镜条件，取第一个颗粒读；per-particle 参数在批内逐颗粒取）
        p0 = cs_data[0]
        voltage_kv   = float(p0["ctf/accel_kv"])
        cs_mm        = float(p0["ctf/cs_mm"])
        amp_contrast = float(p0["ctf/amp_contrast"])
        k_nyq        = 1.0 / (2.0 * psize)
        has_bfactor  = "ctf/bfactor" in cs_data.dtype.names

        pbar = tqdm(total=N, desc="Correcting", unit="ptcl")

        for rel_path, indices in groups.items():
            mrc_path = proj_base / rel_path
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with mrcfile.open(str(mrc_path), permissive=True, mode="r") as mrc_in:
                    stack = mrc_in.data  # [M, D, D] mmap

                    # 按 batch_size 分批处理
                    for batch_start in range(0, len(indices), args.batch_size):
                        batch_idx = indices[batch_start : batch_start + args.batch_size]
                        B = len(batch_idx)

                        # 读取颗粒图像 [B, D, D]
                        imgs = np.stack([
                            stack[int(cs_data[gi]["blob/idx"])].astype(np.float32)
                            for gi in batch_idx
                        ])

                        # 收集 per-particle CTF 参数
                        df1    = np.array([cs_data[gi]["ctf/df1_A"]         for gi in batch_idx], np.float32)
                        df2    = np.array([cs_data[gi]["ctf/df2_A"]         for gi in batch_idx], np.float32)
                        df_ang = np.array([cs_data[gi]["ctf/df_angle_rad"]  for gi in batch_idx], np.float32)
                        ps_rad = np.array([cs_data[gi]["ctf/phase_shift_rad"] for gi in batch_idx], np.float32)
                        sign   = np.array([cs_data[gi]["blob/sign"]         for gi in batch_idx], np.float32)
                        bf     = (np.array([cs_data[gi]["ctf/bfactor"]      for gi in batch_idx], np.float32)
                                  if has_bfactor else np.zeros(B, np.float32))

                        # 计算 CTF [B, D, D]
                        ctf = _ctf_batch(
                            D, psize, df1, df2, df_ang, voltage_kv, cs_mm,
                            amp_contrast, ps_rad, sign, bf, device,
                        )

                        # 构建校正滤波器 [B, D, D]
                        H = _build_filters(ctf, args.method, args.wiener_lambda,
                                           args.lambda_model, k_nyq)
                        H_np = H.cpu().numpy()  # [B, D, D]

                        # FFT → 校正 → IFFT，批量操作
                        F    = np.fft.fft2(imgs)                    # [B, D, D] complex
                        corr = np.fft.ifft2(F * H_np).real.astype(np.float32)  # [B, D, D]

                        # 写入输出 mrcs
                        for j, gi in enumerate(batch_idx):
                            mrc_out.data[gi] = corr[j]

                        pbar.update(B)

        pbar.close()

    # 构建输出 .cs 文件
    out_data = cs_data.copy()
    new_path = str(out_mrcs).encode()
    max_len  = out_data.dtype["blob/path"].itemsize
    out_data["blob/path"] = new_path[:max_len]  # 广播赋值
    out_data["blob/idx"]  = np.arange(N, dtype=np.uint32)
    # 相位翻转类方法校正后 sign 置为 +1（cryoSPARC 约定：暗场颗粒为 -1）
    if args.method in ("phase_flip", "wiener"):
        out_data["blob/sign"] = 1.0
    with open(out_cs, "wb") as f:
        np.save(f, out_data)

    print(f"✓ 校正完成\n  颗粒堆栈: {out_mrcs}\n  元数据:   {out_cs}")


if __name__ == "__main__":
    main()
