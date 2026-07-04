"""
Gallery 分类损失 (design §6.4 改造 1 / §7.1 损失第 2 项)。

诊断结论: 纯 InfoNCE (batch 内 1 正 + 15 负) 与"全 gallery 检索"目标规模差 600×,
导致颗粒 embedding 在 9216 gallery 里排名近随机。分类损失让训练目标 = 检索目标:
颗粒对全 gallery 算 softmax, 监督其匹配 GT 朝向的格点 (及邻域软标签)。

gallery embedding 作分类原型 (每 epoch 由 proj 塔快照, 无梯度); 颗粒 (mic 塔) 学着
映射进该空间。proj 塔的学习与联合空间一致性由并行的 InfoNCE 维持。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_geodesic_to_gallery(R_query: torch.Tensor, R_gallery: torch.Tensor) -> torch.Tensor:
    """
    每个查询朝向到所有 gallery 朝向的测地角距离 (弧度)。

    d = arccos((trace(Qᵀ G) - 1) / 2), 用 Frobenius 内积向量化。

    参数:
        R_query:   [N, 3, 3] 查询 (GT) 旋转矩阵
        R_gallery: [G, 3, 3] gallery 旋转矩阵
    返回:
        dist: [N, G] 测地角距离 (弧度)
    """
    # trace(Qᵀ G) = sum_{ij} Q_ij G_ij; 对所有 (N,G) 对 = Qf @ Gf.T
    Qf = R_query.reshape(R_query.shape[0], 9)      # [N, 9]
    Gf = R_gallery.reshape(R_gallery.shape[0], 9)  # [G, 9]
    tr = Qf @ Gf.t()                                # [N, G]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.arccos(cos)                        # [N, G]


class GalleryClassificationLoss(nn.Module):
    """
    颗粒 → gallery 朝向的 softmax 分类损失 (软标签)。

    目标分布: GT 朝向附近的 gallery 格点按测地距离高斯加权 (邻域软正样本),
    其余为 0。这让训练目标与"全 gallery top-K 检索"完全对齐。
    """

    def __init__(self, temperature: float = 0.07, label_sigma_deg: float = 7.0):
        """
        参数:
            temperature:     logits 温度 (与检索 softmax 一致)
            label_sigma_deg: 软标签高斯宽度 (度); 控制邻域正样本范围
        """
        super().__init__()
        self.temperature = temperature
        self.label_sigma = math.radians(label_sigma_deg)

    def forward(
        self,
        z_mic: torch.Tensor,         # [N, C] L2 归一化颗粒 embedding
        gallery_emb: torch.Tensor,   # [G, C] L2 归一化 gallery 原型 (无梯度)
        R_gt: torch.Tensor,          # [N, 3, 3] 颗粒 GT 旋转矩阵
        R_gallery: torch.Tensor,     # [G, 3, 3] gallery 旋转矩阵
    ) -> torch.Tensor:
        """
        返回: 标量交叉熵损失 (软标签)。
        """
        # logits: 颗粒对全 gallery 的相似度
        logits = (z_mic @ gallery_emb.t()) / self.temperature   # [N, G]

        # 软标签: 按到 GT 的测地距离高斯加权
        with torch.no_grad():
            d = _pairwise_geodesic_to_gallery(R_gt, R_gallery)  # [N, G]
            target = torch.exp(-(d ** 2) / (2.0 * self.label_sigma ** 2))
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-12)

        logp = F.log_softmax(logits, dim=1)
        return -(target * logp).sum(dim=1).mean()
