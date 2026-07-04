"""
前向模型 E-step 引擎 (design §5.3)。

前向模型是**打分器 + 可被优化的似然曲面**, 无任何可学习参数:
给定颗粒 I 和一组候选朝向 R_candidates, 对每个候选用傅里叶切片投影
(复用已修复 CTF 的 project_fourier_slice) 生成参考投影, 用 FFT 互相关一次性
求出所有平移下的最优 shift + 相关峰, 再算似然 logL。

精确连续 pose = 网络给离散起点 + 在此似然曲面上局部优化 (见 refine_orientation)。
本模块无网络依赖, 可用暴力搜索 (R_candidates=全 gallery) 独立验证推理链正确性。

主要接口:
    score(I, R_candidates, V, ctf_params, ...) -> ScoreResult(logL, shift, peak)
    brute_force_pose(I, gallery_aa, V, ctf_params) -> 最优朝向 + shift  (mini-cryoSPARC)
    angular_error(R_a, R_b) -> 测地角误差 (deg), 用于对 cryoSPARC pose 验证
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from siamese.data.projection import (
    axis_angle_to_matrix,
    project_fourier_slice,
)


@dataclass
class ScoreResult:
    """前向模型打分结果。"""
    logL: Tensor    # [M] 每候选的对数似然 (越大越好)
    shift: Tensor   # [M, 2] 每候选的最优平移 (像素, 亚像素精度)
    peak: Tensor    # [M] 互相关峰值


def angular_error(R_a: Tensor, R_b: Tensor) -> Tensor:
    """
    两组旋转矩阵之间的测地角误差 (度)。

    d(A,B) = arccos((trace(Aᵀ B) - 1) / 2)

    参数:
        R_a: [..., 3, 3]
        R_b: [..., 3, 3]
    返回:
        angle_deg: [...] 测地角误差 (度)
    """
    # trace(Aᵀ B) = sum_{ij} A_ij B_ij  (Frobenius 内积)
    tr = (R_a * R_b).sum(dim=(-2, -1))            # [...]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def _radial_shell_index(D: int, device: torch.device) -> Tensor:
    """每个 fft 频率点的整数半径壳索引 [D, D] (DC 在角, fft 原生序)。"""
    fy = torch.fft.fftfreq(D, device=device) * D     # [D] 整数频率
    fx = torch.fft.fftfreq(D, device=device) * D
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(kx ** 2 + ky ** 2)
    return r.round().long()                          # [D, D]


def _shell_noise_var(resid_power: Tensor, shell: Tensor, n_shell: int) -> Tensor:
    """
    逐频率壳噪声方差 σ²(k) (design §5.3)。

    早期低分辨 reference 时高频壳 σ² 大 → 高频自动不贡献 (= frequency marching)。

    参数:
        resid_power: [D, D] 残差功率谱 |I - proj|² (fft 原生序)
        shell:       [D, D] 整数壳索引
        n_shell:     壳数
    返回:
        sigma2_map:  [D, D] 每频点对应壳的平均功率 (即 σ²(k))
    """
    flat_p = resid_power.flatten()
    flat_s = shell.flatten()
    sums = torch.zeros(n_shell, device=resid_power.device).scatter_add_(0, flat_s, flat_p)
    cnts = torch.zeros(n_shell, device=resid_power.device).scatter_add_(
        0, flat_s, torch.ones_like(flat_p))
    shell_var = sums / cnts.clamp(min=1)            # [n_shell]
    return shell_var[shell]                          # [D, D]


def score(
    I: Tensor,
    R_candidates: Tensor,
    V: Tensor,
    ctf_params: Optional[dict] = None,
    *,
    shell_noise: bool = True,
    max_shift: float = 20.0,
    pfac: int = 1,
    method: str = "trilinear",
    chunk_size: Optional[int] = None,
) -> ScoreResult:
    """
    前向模型打分: 对每个候选朝向算似然 + 最优平移 (design §5.3)。

    流程 (每候选 R_k):
        proj_k = CTF · P_{R_k}(V)                          # 投影 + CTF
        corr   = IFFT( conj(FFT(proj_k)) · FFT(I) )        # 一次 FFT 互相关 → 所有平移
        t_k    = argmax(corr) + 抛物线亚像素精修
        logL_k = -Σ |I - shift(proj_k, t_k)|² / 2σ²(k)     # 逐壳或全局噪声

    参数:
        I:            颗粒图像 [D, D] (实空间)
        R_candidates: 候选朝向 [M, 3, 3] (网络 top-M 或全 gallery)
        V:            reference volume [Dv, Dv, Dv]
        ctf_params:   CTF 参数 dict (psize/df_u/df_v/df_angle/voltage/cs/amp_contrast/
                      phase_shift/particle_sign); None = 不加 CTF
        shell_noise:  True = 逐频率壳 σ²(k); False = 全局标量 σ² (NCC)
        max_shift:    平移搜索半径 (像素), 限制互相关峰位置
        pfac:         投影过采样因子。搜索默认 pfac=1 (重建才需 pfac=2)
        method:       投影插值。搜索默认 "trilinear" (比 KB-gridding 快 ~90×,
                      pose 搜索精度无损; KB 仅重建/精修需要)
        chunk_size:   投影分块 (None = 自动); 显存按 chunk 而非候选数 M 增长
    返回:
        ScoreResult(logL [M], shift [M,2], peak [M])
    """
    device = V.device
    I = I.to(device)
    D = I.shape[-1]
    M = R_candidates.shape[0]

    ctf_kwargs = {}
    if ctf_params is not None:
        ctf_kwargs = dict(
            apply_ctf=True,
            psize=ctf_params["psize"],
            ctf_voltage=ctf_params.get("voltage", 300.0),
            ctf_cs=ctf_params.get("cs", 2.7),
            ctf_amp_contrast=ctf_params.get("amp_contrast", 0.1),
            ctf_df_u=ctf_params["df_u"],
            ctf_df_v=ctf_params["df_v"],
            ctf_df_angle=ctf_params.get("df_angle", 0.0),
            ctf_phase_shift=ctf_params.get("phase_shift", 0.0),
            ctf_particle_sign=ctf_params.get("particle_sign", -1.0),
        )

    I0 = I - I.mean()
    FI = torch.fft.fft2(I0)                             # [D, D]
    ky = torch.fft.fftfreq(D, device=device)[:, None]
    kx = torch.fft.fftfreq(D, device=device)[None, :]
    if shell_noise:
        shell = _radial_shell_index(D, device)
        n_shell = int(shell.max().item()) + 1

    # 平移搜索范围掩膜: cryoSPARC shift 通常很小, 无约束互相关会被远处噪声伪峰带偏。
    # 构造 [D,D] 掩膜, 只在 |shift| <= max_shift 内找峰 (fft 原生序, 环绕)。
    dgrid = torch.fft.fftfreq(D, device=device) * D    # [D] 整数位移 (含负)
    dyy, dxx = torch.meshgrid(dgrid, dgrid, indexing="ij")
    shift_mask = (dxx ** 2 + dyy ** 2) <= max_shift ** 2  # [D, D] bool

    # 分块处理候选 (投影 + FFT 互相关 + 似然), 峰值显存与 chunk 成正比, 不随 M 爆炸
    cs = chunk_size if chunk_size is not None else 256
    cs = max(1, min(cs, M))
    logL_all = torch.empty(M, device=device)
    shift_all = torch.empty(M, 2, device=device)
    peak_all = torch.empty(M, device=device)

    for b0 in range(0, M, cs):
        b1 = min(b0 + cs, M)
        with torch.no_grad():
            projs = project_fourier_slice(
                V, R_candidates[b0:b1], shifts=None, pfac=pfac,
                method=method, chunk_size=cs, **ctf_kwargs,
            )  # [bc, Dp, Dp]
        if projs.shape[-1] != D:
            m = (projs.shape[-1] - D) // 2
            projs = projs[:, m:m + D, m:m + D]
        projs = projs - projs.mean(dim=(-2, -1), keepdim=True)
        bc = projs.shape[0]

        # FFT 互相关: 一次性求所有平移
        Fp = torch.fft.fft2(projs)                     # [bc, D, D]
        corr = torch.fft.ifft2(torch.conj(Fp) * FI[None]).real
        # 限制在 max_shift 范围内找峰 (掩膜外置 -inf)
        corr = torch.where(shift_mask[None], corr, torch.full_like(corr, float("-inf")))
        corr_flat = corr.reshape(bc, -1)
        peak_idx = corr_flat.argmax(dim=1)
        peak = corr_flat.gather(1, peak_idx[:, None]).squeeze(1)
        sy = (peak_idx // D).float()
        sx = (peak_idx % D).float()
        sy = torch.where(sy > D // 2, sy - D, sy)
        sx = torch.where(sx > D // 2, sx - D, sx)

        # 在最优平移下对齐 proj, 算判别性得分
        phase = torch.exp(-2j * torch.pi * (kx * sx[:, None, None] + ky * sy[:, None, None]))
        proj_shifted = torch.fft.ifft2(Fp * phase).real     # [bc, D, D]

        # 关键: 用归一化互相关 (NCC) 作为似然得分。
        # logL ∝ -min_a ‖I - a·proj‖² 关于幅度 a 边缘化 => 正比于 NCC²。
        # NCC 对每候选投影的幅度尺度不变, 避免"逐候选方差归一化"抹平判别性
        # (那是之前 GT pose 似然排名垫底的 bug 根因)。
        if shell_noise:
            # 逐频壳白化后再算 NCC: 高频壳噪声大 → 自动降权 (frequency marching)
            FI_full = FI
            Fp_al = Fp * phase                              # 对齐后的 proj 频谱
            flat_s = shell.flatten()
            cnts = torch.zeros(n_shell, device=device).scatter_add_(
                0, flat_s, torch.ones_like(flat_s, dtype=torch.float32))
            # 颗粒逐壳功率 (所有候选共享, 作为白化权重)
            ip = (FI_full.abs() ** 2).flatten()
            ishell = torch.zeros(n_shell, device=device).scatter_add_(0, flat_s, ip)
            w = (1.0 / (ishell / cnts.clamp(min=1)).clamp(min=1e-8))  # [n_shell] 壳白化权重
            wmap = w[shell]                                  # [D, D]
            num = (FI_full[None].conj() * Fp_al * wmap[None]).real.reshape(bc, -1).sum(1)
            ni = ((FI_full.abs() ** 2) * wmap).sum().clamp(min=1e-8).sqrt()
            npj = ((Fp_al.abs() ** 2) * wmap[None]).reshape(bc, -1).sum(1).clamp(min=1e-8).sqrt()
            logL = num / (ni * npj)                          # 白化 NCC ∈ [-1,1]
        else:
            ps = proj_shifted.reshape(bc, -1)
            iv = I0.reshape(-1)
            num = (ps * iv[None]).sum(1)
            ncc = num / (iv.norm().clamp(min=1e-8) * ps.norm(dim=1).clamp(min=1e-8))
            logL = ncc                                       # NCC ∈ [-1,1], 越大越好

        logL_all[b0:b1] = logL
        # 报告 shift 取负: 内部 sx/sy 是"把 proj 平移去对齐 particle"的位移,
        # 与 cryoSPARC 存储的 particle shift 约定相反 (实测幅度一致、符号相反)。
        shift_all[b0:b1] = torch.stack([-sx, -sy], dim=1)
        peak_all[b0:b1] = peak

    return ScoreResult(logL=logL_all, shift=shift_all, peak=peak_all)



def brute_force_pose(
    I: Tensor,
    gallery_aa: Tensor,
    V: Tensor,
    ctf_params: Optional[dict] = None,
    *,
    top_k: int = 1,
    shell_noise: bool = True,
    max_shift: float = 20.0,
    pfac: int = 1,
    method: str = "trilinear",
    chunk_size: Optional[int] = None,
) -> dict:
    """
    暴力搜索定向 (mini-cryoSPARC): 全 gallery 上找似然最大的朝向 (design §5.3)。

    无网络剪枝, 直接在整个 gallery 上评估前向似然。用于独立验证推理链正确性
    (vs cryoSPARC pose), 也是网络失败时的下界回退。

    参数:
        I:          颗粒图像 [D, D]
        gallery_aa: gallery 朝向轴角 [G, 3]
        V:          reference volume
        ctf_params: CTF 参数 dict (见 score)
        top_k:      返回前 k 个候选
        shell_noise/pfac/chunk_size: 见 score
    返回:
        dict(
          best_R [3,3], best_aa [3], best_shift [2], best_logL 标量,
          topk_idx [top_k], topk_logL [top_k], topk_shift [top_k,2],
        )
    """
    R_gal = axis_angle_to_matrix(gallery_aa.to(V.device))  # [G,3,3]
    res = score(I, R_gal, V, ctf_params, shell_noise=shell_noise,
                max_shift=max_shift, pfac=pfac, method=method, chunk_size=chunk_size)
    order = torch.argsort(res.logL, descending=True)
    topk = order[:top_k]
    best = topk[0]
    return dict(
        best_R=R_gal[best],
        best_aa=gallery_aa[best.cpu()],
        best_shift=res.shift[best],
        best_logL=res.logL[best],
        topk_idx=topk,
        topk_logL=res.logL[topk],
        topk_shift=res.shift[topk],
    )


