#!/usr/bin/env python
"""
训练 Siamese 对比编码器。

用法:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from siamese.training.config import TrainConfig
from siamese.training.trainer import Trainer
from siamese.models.encoder import SiameseEncoder
from siamese.data.dataset import MicProjDataset


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese contrastive encoder for cryo-EM particle retrieval."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    # 加载配置
    config = TrainConfig.from_yaml(args.config)
    set_seed(config.seed)

    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}, Backbone: {config.backbone}")
    print(f"Embedding dim: {config.embedding_dim}, Batch size: {config.batch_size}")

    # 创建数据集
    train_dataset = MicProjDataset(
        data_dir=config.data_dir,
        split="train",
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.seed,
    )
    val_dataset = MicProjDataset(
        data_dir=config.data_dir,
        split="val",
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.seed,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # InfoNCE 需要固定 batch size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 创建模型
    model = SiameseEncoder(
        backbone_name=config.backbone,
        image_size=config.image_size,
        real_in_channels=config.real_in_channels,
        freq_in_channels=config.freq_in_channels,
        embedding_dim=config.embedding_dim,
        convnext_depths=tuple(config.convnext_depths),
        convnext_dims=tuple(config.convnext_dims),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 训练
    history = trainer.train()

    print(f"Training complete. Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to {config.checkpoint_dir}")
    return history


if __name__ == "__main__":
    main()