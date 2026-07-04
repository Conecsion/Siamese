"""
编码器模块。

SiameseEncoder: 双分支编码器 (实空间 + 频域)
    实空间分支 (1 通道) + 频域分支 (2 通道: 实部+虚部)
    → 各自经过 backbone 提取特征 → GAP → FusionHead → L2 embedding

TwoTowerEncoder: 非对称双塔编码器
    ProjEncoder (处理干净投影) + MicEncoder (处理含噪颗粒)
    两塔独立权重, 输出同维度 embedding 用于对比学习。
    推理时 proj gallery 可离线预计算。

优化:
    - stem_stride 可配置, 默认 4 (ConvNeXt 标准), 小图像建议 2
    - share_backbone 可配置, 默认 False (独立权重)
    - 支持 cross-attention 或 concat 融合
"""

from typing import Literal, Protocol, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamese.models.backbone import build_backbone


class _Backbone(Protocol):
    """timm backbone 最小接口，供 Pyright 类型推断。"""
    def forward_features(self, x: torch.Tensor) -> torch.Tensor: ...
from siamese.models.fusion import FusionHead
from siamese.utils.fft import image_to_freq_channels, normalize_image


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合层 (可选, 用于较大数据集)。

    实空间特征和频域特征通过 cross-attention 互相增强后融合。
    TODO: 在更大数据集上测试效果
    """

    def __init__(self, dim: int, num_heads: int = 4, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.cross_attn_real = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_freq = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_real = nn.LayerNorm(dim)
        self.norm_freq = nn.LayerNorm(dim)
        self.projector = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, real_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        r = real_feat.unsqueeze(1)
        f = freq_feat.unsqueeze(1)
        r_enhanced, _ = self.cross_attn_real(r, f, f)
        f_enhanced, _ = self.cross_attn_freq(f, r, r)
        r_enhanced = self.norm_real(r_enhanced + r)
        f_enhanced = self.norm_freq(f_enhanced + f)
        combined = torch.cat([r_enhanced.squeeze(1), f_enhanced.squeeze(1)], dim=-1)
        embedding = self.projector(combined)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


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
        stem_stride: int = 4,        # stem 卷积 stride, 默认 4, 小图像可设 2
        share_backbone: bool = False,  # 是否共享 backbone 权重
        use_cross_attention: bool = False,  # TODO: 是否用 cross-attention 融合
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
            stem_stride: stem 卷积 stride, 减小可保留更多空间信息
            share_backbone: 是否共享 backbone 权重 (减少参数量)
            use_cross_attention: 是否使用 cross-attention 融合
        """
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.share_backbone = share_backbone

        # 实空间分支 backbone
        self.backbone_real: _Backbone = cast(_Backbone, build_backbone(
            name=backbone_name,
            in_channels=real_in_channels,
            image_size=image_size,
            depths=convnext_depths,
            dims=convnext_dims,
            stem_stride=stem_stride,
        ))

        if share_backbone:
            # 频域分支共享 backbone, 仅用轻量 stem adapter 处理 2->1 通道
            self.backbone_freq: _Backbone = self.backbone_real
            self.freq_adapter = nn.Conv2d(freq_in_channels, real_in_channels, kernel_size=1)
        else:
            self.backbone_freq = cast(_Backbone, build_backbone(
                name=backbone_name,
                in_channels=freq_in_channels,
                image_size=image_size,
                depths=convnext_depths,
                dims=convnext_dims,
                stem_stride=stem_stride,
            ))
            self.freq_adapter = None

        # 特征维度 (ConvNeXt 最后一个 stage 的输出通道数)
        feature_dim = convnext_dims[-1]  # 默认 768

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 融合头
        if use_cross_attention:
            self.fusion_head = CrossAttentionFusion(
                dim=feature_dim,
                num_heads=4,
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
            )
        else:
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
        # FFT: [N, 1, D, D] -> squeeze -> [N, D, D] -> FFT -> [N, 2, D, D]
        x_freq = image_to_freq_channels(x_norm.squeeze(1))  # [N, 2, D, D]

        if self.freq_adapter is not None:
            # 共享 backbone: 2 通道 -> 1 通道 adapter
            x_freq = self.freq_adapter(x_freq)  # [N, 1, D, D]

        freq_feat = self.backbone_freq.forward_features(x_freq)  # [N, C, H, W]
        freq_feat = self.gap(freq_feat)  # [N, C, 1, 1]
        freq_feat = freq_feat.flatten(1)  # [N, C]

        # 4. 融合 + 投影
        embedding = self.fusion_head(real_feat, freq_feat)  # [N, embedding_dim]

        return embedding


class TwoTowerEncoder(nn.Module):
    """
    非对称双塔编码器: 独立的 proj_encoder + mic_encoder。

    两塔拥有独立权重, proj 编码器处理干净投影, mic 编码器处理含噪颗粒。
    输出同维度 L2 归一化 embedding, 配合 InfoNCE 对比损失训练。

    推理时可离线预计算整个 proj gallery 的 embedding, 只需在线跑 mic_encoder。
    """

    def __init__(self, proj_encoder: SiameseEncoder, mic_encoder: SiameseEncoder):
        """
        参数:
            proj_encoder: 处理干净投影的编码器 (通常轻量)
            mic_encoder:  处理含噪颗粒的编码器 (可更重/更宽)
        """
        super().__init__()
        assert proj_encoder.embedding_dim == mic_encoder.embedding_dim, (
            f"两塔 embedding_dim 必须相同: "
            f"proj={proj_encoder.embedding_dim}, mic={mic_encoder.embedding_dim}"
        )
        self.proj_encoder = proj_encoder
        self.mic_encoder = mic_encoder
        self.embedding_dim = proj_encoder.embedding_dim

    def encode_proj(self, proj: torch.Tensor) -> torch.Tensor:
        """
        参数:
            proj: 形状 [N, 1, D, D] 的干净投影
        返回:
            embedding: 形状 [N, embedding_dim]
        """
        return self.proj_encoder(proj)

    def encode_mic(self, mic: torch.Tensor) -> torch.Tensor:
        """
        参数:
            mic: 形状 [N, 1, D, D] 的含噪颗粒
        返回:
            embedding: 形状 [N, embedding_dim]
        """
        return self.mic_encoder(mic)

    def forward(
        self, mic: torch.Tensor, proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            mic:  形状 [N, 1, D, D]
            proj: 形状 [N, 1, D, D]
        返回:
            (z_mic, z_proj): 各形状 [N, embedding_dim]
        """
        return self.encode_mic(mic), self.encode_proj(proj)