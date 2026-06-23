"""
融合层 (Fusion Head)。

将实空间和频域两支特征拼接后，通过 MLP 投影到 embedding 空间。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """
    双分支特征融合 + 投影头。

    实空间特征 + 频域特征 → concat → Linear+BN+ReLU → Linear → L2 norm → embedding
    """

    def __init__(
        self,
        real_dim: int,       # 实空间分支输出维度 (ConvNeXt-Tiny: 768)
        freq_dim: int,       # 频域分支输出维度 (同上)
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        """
        参数:
            real_dim: 实空间分支特征维度
            freq_dim: 频域分支特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出 embedding 维度
        """
        super().__init__()
        input_dim = real_dim + freq_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, real_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        """
        参数:
            real_feat: 形状 [N, real_dim] 的实空间特征
            freq_feat: 形状 [N, freq_dim] 的频域特征

        返回:
            embedding: 形状 [N, output_dim] 的 L2 归一化 embedding
        """
        combined = torch.cat([real_feat, freq_feat], dim=-1)  # [N, real_dim+freq_dim]
        embedding = self.fusion(combined)  # [N, output_dim]
        embedding = F.normalize(embedding, p=2, dim=-1)  # L2 normalize
        return embedding