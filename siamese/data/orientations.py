"""
SO(3) 朝向采样。

为投影生成 [N, 3] 轴角向量 (弧度), 与 siamese.data.projection.axis_angle_to_matrix
(pyem / cryoSPARC 约定) 配套: project_fourier_slice_from_axis_angle 内部会对每个
轴角做 axis_angle_to_matrix(aa) 得到旋转矩阵。

两种采样:
  - uniform_so3_axis_angles(n): 均匀随机 SO(3) (Shoemake 四元数法), 训练默认。
  - healpix_axis_angles(nside, n_inplane): HEALPix 均匀视角方向 × 均匀面内角,
    覆盖更规整, 适合构建检索 gallery。N = 12*nside^2 * n_inplane。

辅助:
  - matrix_to_axis_angle(R): 旋转矩阵 -> 轴角, 是 axis_angle_to_matrix 的精确逆。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "uniform_so3_axis_angles",
    "healpix_axis_angles",
    "matrix_to_axis_angle",
]


def matrix_to_axis_angle(R: Tensor) -> Tensor:
    """
    旋转矩阵 -> 轴角向量 (axis_angle_to_matrix 的精确逆, pyem 约定)。

    参数:
        R: 形状 [B, 3, 3] 的旋转矩阵。

    返回:
        aa: 形状 [B, 3] 的轴角向量 (弧度)。
    """
    device, dtype = R.device, R.dtype
    eye = torch.eye(3, device=device, dtype=dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    theta = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))  # [B], ∈ [0, π]

    # 反对称部分 (标准 vee), = 2 sinθ * axis (scipy 约定)
    v = torch.stack(
        [
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ],
        dim=-1,
    )  # [B, 3]
    sin_t = torch.sin(theta)

    # 一般情形 (含小角极限 coef -> -0.5): pyem aa = -theta/(2 sinθ) * v
    small = sin_t.abs() < 1e-8
    coef = torch.where(
        small, torch.full_like(sin_t, -0.5), -theta / (2.0 * sin_t)
    )
    aa = coef.unsqueeze(-1) * v

    # θ ≈ π: v ≈ 0, 上式失效, 改用对称部分 (R+I)/2 = n nᵀ 提取轴
    near_pi = (math.pi - theta) < 1e-3
    if bool(near_pi.any()):
        B = R.shape[0]
        M = 0.5 * (R + eye)  # [B, 3, 3]
        diagM = torch.diagonal(M, dim1=-2, dim2=-1).clamp(min=0.0)  # [B, 3]
        n = torch.sqrt(diagM)  # 轴分量幅值
        k = torch.argmax(n, dim=-1)  # [B], 取最大分量定符号基准
        rows = M[torch.arange(B, device=device), k]  # [B, 3]
        signs = torch.sign(rows)
        signs[signs == 0] = 1.0
        signs[torch.arange(B, device=device), k] = 1.0
        n = n * signs
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        aa_pi = -math.pi * n  # θ=π 时 ±πn 等价, 符号无关紧要
        aa = torch.where(near_pi.unsqueeze(-1), aa_pi, aa)

    return aa


def uniform_so3_axis_angles(
    n: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    均匀随机 SO(3) 采样 (Shoemake 四元数法), 返回 [n, 3] 轴角向量 (弧度)。

    参数:
        n: 采样数量。
        generator: 可选 torch.Generator (复现用)。
        device, dtype: 输出张量设备与精度。

    返回:
        aa: 形状 [n, 3] 的轴角向量。
    """
    u = torch.rand(n, 3, generator=generator, device=device, dtype=dtype)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    two_pi = 2.0 * math.pi
    q_xyz = torch.stack(
        [
            torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2),
            torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2),
            torch.sqrt(u1) * torch.sin(two_pi * u3),
        ],
        dim=-1,
    )  # [n, 3]
    w = (torch.sqrt(u1) * torch.cos(two_pi * u3)).clamp(-1.0, 1.0)  # 四元数实部
    half = torch.acos(w)  # = θ/2
    theta = 2.0 * half
    sin_half = torch.sin(half).clamp(min=1e-12)
    axis = q_xyz / sin_half.unsqueeze(-1)
    return theta.unsqueeze(-1) * axis  # [n, 3]


def _rotz(a: Tensor) -> Tensor:
    z, o = torch.zeros_like(a), torch.ones_like(a)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


def _roty(b: Tensor) -> Tensor:
    z, o = torch.zeros_like(b), torch.ones_like(b)
    c, s = torch.cos(b), torch.sin(b)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def healpix_axis_angles(
    nside: int,
    n_inplane: int = 1,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    HEALPix 均匀视角方向 × 均匀面内角 -> [N, 3] 轴角向量, N = 12*nside^2 * n_inplane。

    用 ZYZ 欧拉角 R = Rz(phi) Ry(theta) Rz(psi) 构造旋转矩阵后转轴角:
      (theta, phi) 取自 HEALPix 像素中心 (均匀覆盖视角球),
      psi 在 [0, 2π) 上均匀取 n_inplane 个面内角。

    参数:
        nside: HEALPix nside (像素数 = 12*nside^2)。
        n_inplane: 每个方向的面内角数量 (默认 1)。
        device, dtype: 输出张量设备与精度。

    返回:
        aa: 形状 [N, 3] 的轴角向量。
    """
    import healpy as hp

    npix = hp.nside2npix(nside)
    theta_np, phi_np = hp.pix2ang(nside, np.arange(npix))  # 各 [npix]
    theta = torch.as_tensor(theta_np, dtype=dtype)  # [npix], ∈ [0, π]
    phi = torch.as_tensor(phi_np, dtype=dtype)  # [npix], ∈ [0, 2π)
    psi = torch.arange(n_inplane, dtype=dtype) * (2.0 * math.pi / n_inplane)  # [n_inplane]

    # 笛卡尔积: 方向慢变, 面内角快变
    theta = theta.repeat_interleave(n_inplane)  # [N]
    phi = phi.repeat_interleave(n_inplane)  # [N]
    psi = psi.repeat(npix)  # [N]

    R = _rotz(phi) @ _roty(theta) @ _rotz(psi)  # [N, 3, 3]
    aa = matrix_to_axis_angle(R)  # [N, 3]
    return aa.to(device=device, dtype=dtype)
