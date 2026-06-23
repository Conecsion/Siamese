"""
训练配置 dataclass。

所有可配置的超参数集中管理，支持从 YAML 文件加载。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import yaml


@dataclass
class TrainConfig:
    """训练和模型配置。

    TODO: 后续支持多尺寸模型 (128/192/256/512)
    """

    # --- 数据 ---
    data_dir: str = "data"
    image_size: int = 128
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split = 1 - train_split - val_split

    # --- 模型 ---
    backbone: Literal["convnext_tiny", "convnext_small", "convnext_base",
                       "vit_small", "swin_t"] = "convnext_tiny"
    embedding_dim: int = 128   # TODO: 后续可调
    real_in_channels: int = 1
    freq_in_channels: int = 2
    # ConvNeXt 各 stage 维度 (Tiny 默认值)
    convnext_depths: tuple = (3, 3, 9, 3)
    convnext_dims: tuple = (96, 192, 384, 768)

    # --- 训练 ---
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07

    # scheduler
    scheduler_t0: int = 50   # CosineAnnealingWarmRestarts 的 T_0
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6

    # --- 系统 ---
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = False  # 冒烟测试先不用
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10  # 每 N 个 batch 打印一次 loss

    # --- TODO: 后续支持 ---
    # deepspeed_config: Optional[str] = None  # TODO: DeepSpeed 多卡训练
    # hard_negative: bool = False             # TODO: Hard negative mining

    def save(self, path: str) -> None:
        """保存配置到 YAML 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """从 YAML 文件加载配置，文件中的值覆盖默认值。"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)