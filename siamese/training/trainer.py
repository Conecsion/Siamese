"""
训练循环。

支持 TwoTowerEncoder (encode_mic / encode_proj 接口) 和可选的方向感知 loss。
当 DataLoader 返回 (mic, proj, axisang) 三元组时自动将 axisang 传给 criterion。
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from siamese.training.config import TrainConfig
from siamese.losses.infonce import InfoNCELoss, OrientationAwareInfoNCELoss


class Trainer:
    """训练器，管理训练循环、验证、checkpoint 保存和恢复。"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # 根据配置选择损失函数
        if config.orientation_margin_deg > 0.0:
            self.criterion: Union[InfoNCELoss, OrientationAwareInfoNCELoss] = (
                OrientationAwareInfoNCELoss(
                    temperature=config.temperature,
                    margin_deg=config.orientation_margin_deg,
                )
            )
        else:
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

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _encode(self, mic: torch.Tensor, proj: torch.Tensor):
        """统一编码接口：TwoTowerEncoder 用 encode_mic/encode_proj，旧接口直接调用。"""
        if hasattr(self.model, "encode_mic"):
            return self.model.encode_mic(mic), self.model.encode_proj(proj)
        # 兼容旧版 SiameseEncoder
        return self.model(mic), self.model(proj)

    def _step(self, batch: tuple) -> torch.Tensor:
        """从一个 batch 计算 loss。支持 (mic, proj) 和 (mic, proj, axisang)。"""
        mic = batch[0].to(self.device)
        proj = batch[1].to(self.device)
        axisang = batch[2].to(self.device) if len(batch) == 3 else None

        z_mic, z_proj = self._encode(mic, proj)
        return self.criterion(z_mic, z_proj, axisang)  # type: ignore[arg-type]

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        for i, batch in enumerate(pbar):
            loss = self._step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if i % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg = total_loss / len(self.train_loader)
        self.train_losses.append(avg)
        return avg

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            total_loss += self._step(batch).item()
        avg = total_loss / len(self.val_loader)
        self.val_losses.append(avg)
        return avg

    def train(self) -> dict:
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")
        self.save_checkpoint("last.pt")
        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def save_checkpoint(self, filename: str) -> None:
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
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["val_losses"]
        print(f"Loaded checkpoint: {path} (epoch {self.epoch + 1})")
