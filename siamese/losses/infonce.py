"""
InfoNCE (NT-Xent) 对比损失。

对称版本: 同时计算 mic→proj 和 proj→mic 两个方向的交叉熵损失。

TODO: 添加 hard negative mining 支持
  - hard negatives: 相邻 viewing direction, 相似 silhouette, 对称相关 projection
  - 接口: loss_fn(z_mic, z_proj, hard_negatives=None)
"""

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
        """
        参数:
            temperature: 温度参数 τ, 控制 softmax 的锐度。
                        值越小, 对负样本的区分越严格。
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_mic: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            z_mic: 形状 [N, D] 的 mic embedding (应已 L2 归一化)
            z_proj: 形状 [N, D] 的 proj embedding (应已 L2 归一化)

        返回:
            loss: 标量, 对称 InfoNCE loss
        """
        N = z_mic.shape[0]

        # 相似度矩阵: S[i,j] = cos_sim(mic_i, proj_j) / τ
        # 由于 embedding 已 L2 归一化, 点积 = 余弦相似度
        logits = torch.matmul(z_mic, z_proj.T) / self.temperature  # [N, N]

        # 正样本标签: 对角线位置
        labels = torch.arange(N, device=z_mic.device)

        # 对称 InfoNCE: mic→proj 和 proj→mic
        loss_mic = F.cross_entropy(logits, labels)      # 每行分类
        loss_proj = F.cross_entropy(logits.T, labels)   # 每列分类

        loss = (loss_mic + loss_proj) / 2.0
        return loss