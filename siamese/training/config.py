"""
训练配置 dataclass。

所有可配置的超参数集中管理，支持从 YAML 文件加载。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class TrainConfig:
    """训练和模型配置。"""

    # --- 数据 ---
    data_dir: str = "data"
    image_size: int = 128
    train_split: float = 0.7
    val_split: float = 0.15

    # --- Proj 编码器 ---
    backbone: Literal["convnext_tiny", "convnext_small", "convnext_base",
                      "vit_small", "swin_t"] = "convnext_tiny"
    embedding_dim: int = 128
    real_in_channels: int = 1
    freq_in_channels: int = 2
    convnext_depths: tuple = (3, 3, 9, 3)
    convnext_dims: tuple = (96, 192, 384, 768)

    # --- Mic 编码器 (空 = 与 proj 相同) ---
    # 可单独配置更大的 backbone 增强对噪声的表征能力
    mic_backbone: str = ""               # 空 = 与 backbone 相同
    mic_convnext_depths: tuple = ()      # 空 = 与 convnext_depths 相同
    mic_convnext_dims: tuple = ()        # 空 = 与 convnext_dims 相同

    # --- 训练 ---
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    orientation_margin_deg: float = 15.0  # 近邻方向排除阈值(度); 0 = 标准 InfoNCE

    # scheduler
    scheduler_t0: int = 50
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6

    # --- 系统 ---
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = False
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10

    def mic_backbone_resolved(self) -> str:
        return self.mic_backbone or self.backbone

    def mic_depths_resolved(self) -> tuple:
        return self.mic_convnext_depths or self.convnext_depths

    def mic_dims_resolved(self) -> tuple:
        return self.mic_convnext_dims or self.convnext_dims

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))
