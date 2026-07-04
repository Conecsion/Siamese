"""
InfoNCE (NT-Xent) 对比损失。

InfoNCELoss: 标准对称版本。
OrientationAwareInfoNCELoss: 方向感知版本，将 SO(3) 测地距离小于阈值的
    proj 对从 negative 集合中排除，避免近邻投影被错误推开。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent 对比损失。

    给定 batch 中 N 对 (mic_i, proj_i):
      S[i,j] = z_mic[i] · z_proj[j] / τ
      loss = (CE(S, labels) + CE(S.T, labels)) / 2

    正样本: 对角线位置 (i, i)
    负样本: batch 内其他样本 (i, j) for j ≠ i
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_mic: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            z_mic:   形状 [N, D] (已 L2 归一化)
            z_proj:  形状 [N, D] (已 L2 归一化)
            axisang: 忽略，仅为接口兼容

        返回:
            loss: 标量
        """
        N = z_mic.shape[0]
        logits = torch.matmul(z_mic, z_proj.T) / self.temperature  # [N, N]
        labels = torch.arange(N, device=z_mic.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
        return loss


def _pairwise_geodesic_dist(axisang: torch.Tensor) -> torch.Tensor:
    """
    计算 batch 内所有轴角对的 SO(3) 测地距离。

    利用 trace(R_i^T @ R_j) = Frobenius 内积 R_i·R_j, 向量化实现。

    参数:
        axisang: 形状 [B, 3], 轴角向量 (pyem 约定)

    返回:
        dist: 形状 [B, B], 测地角距离 (弧度), 对角线为 0
    """
    B = axisang.shape[0]
    device, dtype = axisang.device, axisang.dtype

    theta = torch.norm(axisang, dim=-1, keepdim=True).clamp(min=1e-16)  # [B, 1]
    w = axisang / theta                                                  # [B, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    zeros = torch.zeros(B, device=device, dtype=dtype)

    # 反对称矩阵 K (pyem 约定)
    K = torch.stack([
        torch.stack([ zeros,   wz,  -wy], -1),
        torch.stack([  -wz,  zeros,  wx], -1),
        torch.stack([   wy,   -wx, zeros], -1),
    ], dim=1)  # [B, 3, 3]

    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)   # [1, 3, 3]
    sin_t = torch.sin(theta).unsqueeze(-1)                       # [B, 1, 1]
    cos_t = torch.cos(theta).unsqueeze(-1)
    R = I + sin_t * K + (1 - cos_t) * (K @ K)                   # [B, 3, 3]

    # trace(R_i^T @ R_j) = sum_{kl} R_i[k,l]*R_j[k,l] (Frobenius 内积)
    R_flat = R.reshape(B, 9)                                     # [B, 9]
    tr = R_flat @ R_flat.T                                       # [B, B]
    cos_d = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos_d)                                     # [B, B]


class OrientationAwareInfoNCELoss(nn.Module):
    """
    方向感知 InfoNCE 损失。

    在标准 InfoNCE 基础上, 将 batch 中 SO(3) 测地距离 < margin 的 proj 对
    从 negative 集合中排除 (logit 置为 -inf)，避免近邻投影被错误推开，
    从而使 embedding 空间尊重 SO(3) 的连续性。

    margin_deg=0 时退化为标准 InfoNCE。
    """

    def __init__(self, temperature: float = 0.07, margin_deg: float = 15.0):
        """
        参数:
            temperature: InfoNCE 温度参数 τ
            margin_deg:  近邻判定阈值 (度)。小于此角度的 proj 对不计入 negative。
                         15° 是经验起点，对称性分子可适当加大。
        """
        super().__init__()
        self.temperature = temperature
        self.margin_rad = math.radians(margin_deg)

    def forward(
        self,
        z_mic: torch.Tensor,
        z_proj: torch.Tensor,
        axisang: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            z_mic:   形状 [N, D] (已 L2 归一化)
            z_proj:  形状 [N, D] (已 L2 归一化)
            axisang: 形状 [N, 3] 的 proj 轴角 (pyem 约定)；
                     None 时退化为标准 InfoNCE

        返回:
            loss: 标量
        """
        N = z_mic.shape[0]
        logits = torch.matmul(z_mic, z_proj.T) / self.temperature  # [N, N]

        if axisang is not None and self.margin_rad > 0.0:
            dist = _pairwise_geodesic_dist(axisang)   # [N, N]
            near = dist < self.margin_rad              # [N, N]
            near.fill_diagonal_(False)                 # 不 mask 正样本对角线
            logits = logits.masked_fill(near, float("-inf"))

        labels = torch.arange(N, device=z_mic.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
        return loss
