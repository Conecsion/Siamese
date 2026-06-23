"""
MicProjDataset: 从预生成的模拟数据中加载 (mic, proj) 配对。
"""

from pathlib import Path
from typing import Tuple, Literal

import torch
from torch.utils.data import Dataset

from siamese.data.transforms import PreprocessTransform


class MicProjDataset(Dataset):
    """
    Mic-Proj 配对数据集。

    从预生成的 .pt 文件加载数据，支持 train/val/test 划分。

    每个样本返回 (mic, proj) 元组，mic 和 proj 已归一化并添加 channel 维度。

    划分策略: 按 HEALPix 方向顺序划分，避免按方向半球划分的复杂性
             (冒烟测试数据量小，简单顺序划分即可)
    TODO: 后续可改为按方向半球划分避免泄漏
    """

    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val", "test"] = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        参数:
            data_dir: 包含 projs.pt, mics.pt, pairs.pt 的目录
            split: 数据集划分
            train_split: 训练集比例
            val_split: 验证集比例 (测试集 = 1 - train - val)
            normalize: 是否对图像做归一化
            seed: 随机种子 (用于 shuffle)
        """
        data_dir = Path(data_dir)

        self.projs = torch.load(data_dir / "projs.pt", weights_only=True)  # [N, D, D]
        self.mics = torch.load(data_dir / "mics.pt", weights_only=True)    # [M, D, D]
        self.pairs = torch.load(data_dir / "pairs.pt", weights_only=True)  # [M, 2]

        self.transform = PreprocessTransform(normalize=normalize)

        M = len(self.mics)

        # 生成索引并 shuffle
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(M, generator=g).tolist()

        # 按比例划分
        train_end = int(M * train_split)
        val_end = int(M * (train_split + val_split))

        if split == "train":
            self.indices = indices[:train_end]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        elif split == "test":
            self.indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            idx: 数据集内部索引

        返回:
            mic: 形状 [1, D, D] 的归一化 noisy micrograph
            proj: 形状 [1, D, D] 的归一化 clean projection
        """
        global_idx = self.indices[idx]
        proj_idx, mic_idx = self.pairs[global_idx]

        mic = self.mics[mic_idx]    # [D, D]
        proj = self.projs[proj_idx]  # [D, D]

        mic = self.transform(mic)    # [1, D, D]
        proj = self.transform(proj)  # [1, D, D]

        return mic, proj