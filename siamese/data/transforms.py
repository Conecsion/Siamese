"""
数据预处理变换。

包括图像归一化、频域转换等。作为可组合的变换函数，供 Dataset 使用。
"""

import torch
from siamese.utils.fft import normalize_image, image_to_freq_channels


class PreprocessTransform:
    """
    预处理变换: 归一化 + 可选的频域转换。

    在 Dataset 的 __getitem__ 中调用，对每个样本独立处理。
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        参数:
            image: 形状 [D, D] 的图像

        返回:
            image: 形状 [1, D, D] 的归一化图像 (添加 channel 维度)
        """
        if self.normalize:
            image = normalize_image(image)  # [D, D], mean=0, std=1

        # 添加 channel 维度: [D, D] -> [1, D, D]
        image = image.unsqueeze(0)
        return image