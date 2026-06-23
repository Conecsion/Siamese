"""
Backbone 工厂函数。

支持 ConvNeXt (默认)、ViT-Small、Swin-T 等多种 backbone。
通过工厂函数 build_backbone() 统一创建，返回 nn.Module。
"""

from typing import Literal, Tuple

import torch.nn as nn

# TODO: 后续可添加 ViT / Swin-T 支持
# from timm.models.vision_transformer import vit_small_patch16_224
# from timm.models.swin_transformer import swin_tiny_patch4_window7_224


def build_backbone(
    name: Literal["convnext_tiny", "convnext_small", "convnext_base",
                   "vit_small", "swin_t"] = "convnext_tiny",
    in_channels: int = 1,
    image_size: int = 128,
    depths: Tuple[int, ...] = (3, 3, 9, 3),
    dims: Tuple[int, ...] = (96, 192, 384, 768),
    drop_path_rate: float = 0.0,
    stem_stride: int = 4,
) -> nn.Module:
    """
    创建 backbone 特征提取器。

    参数:
        name: backbone 名称
        in_channels: 输入通道数 (实空间 1, 频域 2)
        image_size: 输入图像尺寸
        depths: ConvNeXt 各 stage 的 block 数量
        dims: ConvNeXt 各 stage 的通道数
        drop_path_rate: stochastic depth rate
        stem_stride: stem 卷积 stride, 默认 4 (ConvNeXt 标准)。
                     设为 2 可减少下采样, 为小图像保留更多空间信息。

    返回:
        backbone: nn.Module, 输出特征图 (不含 head 和 GAP)

    TODO: 支持 ViT-small / Swin-T backbone
    """
    if name.startswith("convnext"):
        from timm.models.convnext import ConvNeXt

        # 映射名称到 timm 的 convnext 变体
        if "small" in name:
            depths = (3, 3, 27, 3)
            dims = (96, 192, 384, 768)
        elif "base" in name:
            depths = (3, 3, 27, 3)
            dims = (128, 256, 512, 1024)

        backbone = ConvNeXt(
            in_chans=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            head_hidden_size=None,  # 不使用分类头
            num_classes=0,          # 返回特征
        )

        # 如果 stem_stride != 4, 修改 stem 的 stride
        if stem_stride != 4:
            # ConvNeXt stem: Sequential(Conv2d(4x4, stride=4), LayerNorm)
            # 替换 stride
            backbone.stem[0].stride = (stem_stride, stem_stride)

        return backbone

    elif name in ("vit_small", "swin_t"):
        raise NotImplementedError(
            f"Backbone '{name}' not yet implemented. TODO: 添加 ViT/Swin 支持"
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")