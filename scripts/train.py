#!/usr/bin/env python
"""
训练双塔对比编码器。

用法:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/last.pt
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from siamese.training.config import TrainConfig
from siamese.training.trainer import Trainer
from siamese.models.encoder import SiameseEncoder, TwoTowerEncoder
from siamese.data.dataset import MicProjDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_single_encoder(
    backbone: str,
    depths: tuple,
    dims: tuple,
    config: TrainConfig,
) -> SiameseEncoder:
    """根据 backbone 名称和 depths/dims 构建单塔 SiameseEncoder。"""
    return SiameseEncoder(
        backbone_name=backbone,  # type: ignore[arg-type]
        image_size=config.image_size,
        real_in_channels=config.real_in_channels,
        freq_in_channels=config.freq_in_channels,
        embedding_dim=config.embedding_dim,
        convnext_depths=tuple(depths),
        convnext_dims=tuple(dims),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train two-tower contrastive encoder for cryo-EM particle retrieval."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    set_seed(config.seed)

    use_axisang = config.orientation_margin_deg > 0.0

    train_dataset = MicProjDataset(
        data_dir=config.data_dir, split="train",
        train_split=config.train_split, val_split=config.val_split,
        seed=config.seed, return_axisang=use_axisang,
    )
    val_dataset = MicProjDataset(
        data_dir=config.data_dir, split="val",
        train_split=config.train_split, val_split=config.val_split,
        seed=config.seed, return_axisang=use_axisang,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    proj_encoder = build_single_encoder(
        config.backbone, config.convnext_depths, config.convnext_dims, config
    )
    mic_encoder = build_single_encoder(
        config.mic_backbone_resolved(),
        config.mic_depths_resolved(),
        config.mic_dims_resolved(),
        config,
    )
    model = TwoTowerEncoder(proj_encoder=proj_encoder, mic_encoder=mic_encoder)

    proj_params = sum(p.numel() for p in proj_encoder.parameters())
    mic_params = sum(p.numel() for p in mic_encoder.parameters())
    print(
        f"ProjEncoder: {proj_params:,}  |  MicEncoder: {mic_params:,}  |  "
        f"Total: {proj_params + mic_params:,}"
    )
    print(f"OrientationMargin: {config.orientation_margin_deg}°")

    trainer = Trainer(model=model, config=config,
                      train_loader=train_loader, val_loader=val_loader)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    print(f"Done. Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
