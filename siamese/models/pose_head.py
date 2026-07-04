"""
Pose 输出头 (design §6.4): 把检索器升级为「top-M 提议 + 先验分布 + 连续残差」。

现有 TwoTowerEncoder 只是检索器 (embedding 相似度)。本模块在其上加:
  1. 先验分布 π_θ: softmax over 检索分数 (零新增结构)
  2. 残差头: 锚点 -> 切空间 so(3) 残差 (连续朝向, 唯一新增可学习参数)
  3. 位移头: 粗 shift 初值 (精确 shift 交前向模型 FFT 互相关)

职责边界 (design §3.2): 网络只做「提议 + 先验」, 判别和精修在前向模型。
所有张量设备从输入推断, 不写死单卡, batch 维可分片 (为多卡/DeepSpeed 留接口)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProposalResult:
    """proposer 一次前向的输出 (design §6.4)。"""
    topk_idx: torch.Tensor      # [N, M] gallery 候选索引 (先验最高的 M 个)
    prior: torch.Tensor         # [N, M] 先验概率 π_θ (softmax over 检索分数)
    residual: Optional[torch.Tensor]  # [N, M, 3] 每候选的 so(3) 切空间残差 (可选)
    shift: Optional[torch.Tensor]     # [N, 2] 粗位移初值 (可选)


class ResidualHead(nn.Module):
    """
    切空间残差回归头 (design §6.1/§6.4)。

    输入: 颗粒特征 h [N, C] + 锚点 gallery embedding g [N, M, C]
    输出: 每候选的 so(3) 残差 δ [N, M, 3], 限制在格点胞内的局部切空间。
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256, max_residual: float = 0.15):
        """
        参数:
            feat_dim:     编码器输出维度 C
            hidden_dim:   MLP 隐藏层
            max_residual: 残差幅值上限 (弧度), tanh 限幅, 约半个粗网格胞
        """
        super().__init__()
        self.max_residual = max_residual
        # 输入是 [颗粒特征, 锚点特征] 拼接 -> δ ∈ so(3)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        参数:
            h: [N, C] 颗粒特征
            g: [N, M, C] M 个锚点候选的 embedding
        返回:
            delta: [N, M, 3] so(3) 残差 (弧度, 已限幅)
        """
        N, M, C = g.shape
        h_exp = h.unsqueeze(1).expand(N, M, C)          # [N, M, C]
        x = torch.cat([h_exp, g], dim=-1)                # [N, M, 2C]
        delta = self.mlp(x)                              # [N, M, 3]
        return self.max_residual * torch.tanh(delta)     # 限幅到局部胞


class ShiftHead(nn.Module):
    """
    粗位移回归头: 颗粒特征 -> 平面内位移初值 [N, 2] (像素)。
    精确 shift 由前向模型 FFT 互相关给, 此头只提供搜索中心。
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, max_shift: float = 20.0):
        super().__init__()
        self.max_shift = max_shift
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [N, C] -> shift [N, 2] (像素, 已限幅)。"""
        return self.max_shift * torch.tanh(self.mlp(h))


class PoseProposer(nn.Module):
    """
    Pose proposer (design §6.4): TwoTowerEncoder + gallery + 三个头。

    一次前向产出 {top-M 候选, 先验 π_θ, 残差, 粗 shift}, 交前向模型判别+精修。

    gallery embedding 由 proj 塔离线预计算 (set_gallery), 推理时只跑 mic 塔。
    多卡: 模块无单卡假设; gallery 可在各卡复制或分片, batch 维 (颗粒) 可切分。
    """

    def __init__(
        self,
        encoder: "TwoTowerEncoder",   # 精确类型，让 Pyright 知道有 encode_mic/encode_proj
        temperature: float = 0.07,    # 先验 softmax 温度
        use_residual: bool = True,
        use_shift: bool = True,
        hidden_dim: int = 256,
        max_residual: float = 0.15,
        max_shift: float = 20.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        C = encoder.embedding_dim

        self.residual_head = (
            ResidualHead(C, hidden_dim, max_residual) if use_residual else None
        )
        self.shift_head = ShiftHead(C, hidden_dim // 2, max_shift) if use_shift else None

        # gallery: embedding [G, C] 与对应轴角 [G, 3], 通过 set_gallery 注册为 buffer
        self.register_buffer("gallery_emb", torch.empty(0), persistent=False)
        self.register_buffer("gallery_aa", torch.empty(0), persistent=False)

    @torch.no_grad()
    def set_gallery(self, gallery_emb: torch.Tensor, gallery_aa: torch.Tensor) -> None:
        """
        注册 gallery (proj 塔离线预计算的 embedding + 轴角)。

        参数:
            gallery_emb: [G, C] L2 归一化的 gallery embedding
            gallery_aa:  [G, 3] 对应轴角 (弧度)
        """
        self.gallery_emb = gallery_emb
        self.gallery_aa = gallery_aa

    def encode_gallery(self, proj: torch.Tensor) -> torch.Tensor:
        """用 proj 塔编码 gallery 投影 [G,1,D,D] -> [G, C]。"""
        return self.encoder.encode_proj(proj)

    def forward(self, mic: torch.Tensor, top_m: int = 50) -> ProposalResult:
        """
        参数:
            mic:   [N, 1, D, D] 颗粒图像
            top_m: 提议候选数 M
        返回:
            ProposalResult(topk_idx [N,M], prior [N,M], residual [N,M,3]?, shift [N,2]?)
        """
        assert self.gallery_emb.numel() > 0, "需先调用 set_gallery 注册 gallery"
        z = self.encoder.encode_mic(mic)                 # [N, C] (L2 归一化)

        # 检索分数 = 内积 (= 余弦相似度); top-M
        sims = z @ self.gallery_emb.t()                  # [N, G]
        topk_sims, topk_idx = sims.topk(top_m, dim=1)    # [N, M]

        # 先验 π_θ = softmax over 检索分数 (design §6.4 改造 1)
        prior = F.softmax(topk_sims / self.temperature, dim=1)  # [N, M]

        # 残差 (改造 2): 取 top-M 锚点 embedding
        residual = None
        if self.residual_head is not None:
            g = self.gallery_emb[topk_idx]               # [N, M, C]
            residual = self.residual_head(z, g)          # [N, M, 3]

        shift = self.shift_head(z) if self.shift_head is not None else None

        return ProposalResult(topk_idx=topk_idx, prior=prior,
                              residual=residual, shift=shift)
