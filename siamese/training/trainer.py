"""
训练循环。

封装训练和验证逻辑，支持 checkpoint 保存和恢复。
TODO: 后续添加 DeepSpeed 支持
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from siamese.training.config import TrainConfig
from siamese.losses.infonce import InfoNCELoss


class Trainer:
    """
    训练器。

    管理训练循环、验证、checkpoint 保存和恢复。
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        参数:
            model: SiameseEncoder 模型
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器 (可选)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        self.criterion = InfoNCELoss(temperature=config.temperature)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_t0,
            T_mult=config.scheduler_t_mult,
            eta_min=config.scheduler_eta_min,
        )

        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # 创建 checkpoint 目录
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        """训练一个 epoch，返回平均 loss。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        for batch_idx, (mic, proj) in enumerate(pbar):
            mic = mic.to(self.device)
            proj = proj.to(self.device)

            # 编码
            z_mic = self.model(mic)    # [N, D]
            z_proj = self.model(proj)  # [N, D]

            # 计算 loss
            loss = self.criterion(z_mic, z_proj)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """验证一个 epoch，返回平均 loss。"""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for mic, proj in self.val_loader:
            mic = mic.to(self.device)
            proj = proj.to(self.device)

            z_mic = self.model(mic)
            z_proj = self.model(proj)

            loss = self.criterion(z_mic, z_proj)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self) -> Dict[str, list]:
        """
        完整训练循环。

        返回:
            dict: {"train_losses": [...], "val_losses": [...]}
        """
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")

            # 定期保存
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # 保存最终模型
        self.save_checkpoint("last.pt")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def save_checkpoint(self, filename: str) -> None:
        """保存 checkpoint。"""
        path = Path(self.config.checkpoint_dir) / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """从 checkpoint 恢复训练状态。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        print(f"Loaded checkpoint from {path} (epoch {self.epoch + 1})")