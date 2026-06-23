"""
Siamese 双分支编码器。

实空间分支 (1 通道) + 频域分支 (2 通道: 实部+虚部)
→ 各自经过 backbone 提取特征 → GAP → FusionHead → L2 embedding
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn

from siamese.models.backbone import build_backbone
from siamese.models.fusion import FusionHead
from siamese.utils.fft import image_to_freq_channels, normalize_image


class SiameseEncoder(nn.Module):
    """
    Siamese 双分支对比编码器。

    输入: [N, 1, D, D] 实空间图像
    输出: [N, embedding_dim] L2 归一化 embedding

    工作流程:
        1. 归一化输入图像
        2. 实空间分支: 归一化图像 → backbone → GAP → real_feat
        3. 频域分支: 归一化图像 → FFT2 → backbone → GAP → freq_feat
        4. FusionHead: concat(real_feat, freq_feat) → MLP → L2 embedding
    """

    def __init__(
        self,
        backbone_name: Literal[
            "convnext_tiny", "convnext_small", "convnext_base",
            "vit_small", "swin_t"] = "convnext_tiny",
        image_size: int = 128,
        real_in_channels: int = 1,
        freq_in_channels: int = 2,
        embedding_dim: int = 128,   # TODO: 后续可调
        hidden_dim: int = 256,
        convnext_depths: Tuple[int, ...] = (3, 3, 9, 3),
        convnext_dims: Tuple[int, ...] = (96, 192, 384, 768),
    ):
        """
        参数:
            backbone_name: backbone 架构名称
            image_size: 输入图像尺寸
            real_in_channels: 实空间分支输入通道数
            freq_in_channels: 频域分支输入通道数
            embedding_dim: 输出 embedding 维度
            hidden_dim: 融合头隐藏层维度
            convnext_depths: ConvNeXt 各 stage 的 block 数
            convnext_dims: ConvNeXt 各 stage 的通道数
        """
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim

        # 实空间分支
        self.backbone_real = build_backbone(
            name=backbone_name,
            in_channels=real_in_channels,
            image_size=image_size,
            depths=convnext_depths,
            dims=convnext_dims,
        )

        # 频域分支（权重独立）
        self.backbone_freq = build_backbone(
            name=backbone_name,
            in_channels=freq_in_channels,
            image_size=image_size,
            depths=convnext_depths,
            dims=convnext_dims,
        )

        # 特征维度 (ConvNeXt 最后一个 stage 的输出通道数)
        feature_dim = convnext_dims[-1]  # 默认 768

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 融合头
        self.fusion_head = FusionHead(
            real_dim=feature_dim,
            freq_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 形状 [N, 1, D, D] 的实空间图像

        返回:
            embedding: 形状 [N, embedding_dim] 的 L2 归一化 embedding
        """
        # 1. 归一化 (per-image mean-std)
        x_norm = normalize_image(x)  # [N, 1, D, D]

        # 2. 实空间分支
        real_feat = self.backbone_real.forward_features(x_norm)  # [N, C, H, W]
        real_feat = self.gap(real_feat)  # [N, C, 1, 1]
        real_feat = real_feat.flatten(1)  # [N, C]

        # 3. 频域分支
        # 对归一化后的图像做 FFT
        x_freq = image_to_freq_channels(x_norm.squeeze(1))  # [N, D, D] -> [N, 2, D, D]
        freq_feat = self.backbone_freq.forward_features(x_freq)  # [N, C, H, W]
        freq_feat = self.gap(freq_feat)  # [N, C, 1, 1]
        freq_feat = freq_feat.flatten(1)  # [N, C]

        # 4. 融合 + 投影
        embedding = self.fusion_head(real_feat, freq_feat)  # [N, embedding_dim]

        return embedding