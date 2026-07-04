"""
Pose proposer 训练脚本 - DeepSpeed 多卡版本

使用 DeepSpeed 进行分布式训练，支持：
- ZeRO Stage 2 (优化器+梯度分片)
- FP16 混合精度
- 梯度累积
- 4 x RTX 3090 (24GB each)

启动:
    deepspeed --num_gpus=4 scripts/train_proposer_ds.py --config configs/proposer_ribosome_multi.yaml --deepspeed configs/ds_config.json
"""

import argparse
import os
import random
import warnings
from pathlib import Path

import deepspeed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from siamese.data.cryosparc import CryoSparcParticleDataset
from siamese.data.orientations import healpix_axis_angles
from siamese.models.encoder import SiameseEncoder, TwoTowerEncoder
from siamese.models.pose_head import PoseProposer
from siamese.losses.infonce import OrientationAwareInfoNCELoss
from siamese.losses.gallery_ce import GalleryClassificationLoss
from siamese.eval.forward_model import angular_error
from siamese.data.projection import axis_angle_to_matrix


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    """判断是否为主进程 (rank 0)。"""
    return not dist.is_initialized() or dist.get_rank() == 0


@torch.no_grad()
def evaluate_recall(
    model_engine,  # DeepSpeed engine
    loader: DataLoader,
    gallery_aa: torch.Tensor,
    device: torch.device,
    m_values=(1, 5, 10, 50),
    grid_tol_deg: float = 7.0,
) -> dict:
    """
    Recall@M 评估 (仅在主进程执行)。
    """
    if not is_main_process():
        return {}

    model_engine.eval()
    proposer = model_engine.module  # 解包 DeepSpeed wrapper

    n_tot = 0
    hits = {m: 0 for m in m_values}

    for particle, _, axisang, _ in loader:
        particle = particle.to(device)
        axisang = axisang.to(device)

        z_mic = torch.nn.functional.normalize(
            proposer.encoder.encode_mic(particle), dim=1)

        # top-M 候选索引
        gal_emb = proposer.gallery_emb
        scores = z_mic @ gal_emb.t()
        top_m = scores.topk(max(m_values), dim=1).indices

        # 真 pose 最近的 gallery 点
        R_gt = axis_angle_to_matrix(axisang)
        R_gallery = axis_angle_to_matrix(gallery_aa.to(device))
        ang_err = angular_error(
            R_gt.unsqueeze(1), R_gallery.unsqueeze(0))
        nearest_gal = ang_err.argmin(dim=1)

        # 命中统计
        for m in m_values:
            hit = (top_m[:, :m] == nearest_gal.unsqueeze(1)).any(dim=1)
            hits[m] += hit.sum().item()
        n_tot += len(particle)

    return {f"Recall@{m}": hits[m] / n_tot for m in m_values}


def visualize_particle_projection_pairs(
    particle: torch.Tensor,
    proj: torch.Tensor,
    writer: SummaryWriter,
    epoch: int,
    max_pairs: int = 4,
) -> None:
    """TensorBoard 可视化颗粒-投影配对 (仅主进程)。"""
    if not is_main_process():
        return

    n = min(max_pairs, len(particle))
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        mic = particle[i, 0].cpu().numpy()
        prj = proj[i, 0].cpu().numpy()

        axes[0, i].imshow(mic, cmap='gray')
        axes[0, i].set_title(f'Particle {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(prj, cmap='gray')
        axes[1, i].set_title(f'Projection {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    writer.add_figure('train/particle_projection_pairs', fig, epoch)
    plt.close(fig)


@torch.no_grad()
def build_gallery(
    model_engine,
    gallery_aa: torch.Tensor,
    ref_vol: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    构建 gallery embeddings (仅主进程，然后广播)。

    返回: [G, C] gallery 特征 (所有进程都有)
    """
    if is_main_process():
        model_engine.eval()
        proposer = model_engine.module

        G = len(gallery_aa)
        batch_size = 128
        embs = []

        for i in range(0, G, batch_size):
            aa_batch = gallery_aa[i:i+batch_size].to(device)

            # 投影
            from siamese.data.projection import project_fourier_slice_from_axis_angle
            proj_batch = project_fourier_slice_from_axis_angle(
                ref_vol, aa_batch, device=device)

            # 编码
            z_proj = torch.nn.functional.normalize(
                proposer.encoder.encode_proj(proj_batch), dim=1)
            embs.append(z_proj.cpu())

        gallery_emb = torch.cat(embs, 0).to(device)
    else:
        # 非主进程创建占位符
        proposer = model_engine.module
        C = proposer.encoder.embedding_dim
        gallery_emb = torch.empty(len(gallery_aa), C, device=device)

    # 广播到所有进程
    if dist.is_initialized():
        dist.broadcast(gallery_emb, src=0)

    return gallery_emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pose proposer with DeepSpeed")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (auto-set by DeepSpeed)")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    warnings.simplefilter("ignore")

    # DeepSpeed 初始化分布式环境
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if is_main_process():
        print(f"🚀 DeepSpeed 分布式训练")
        print(f"   World size: {dist.get_world_size()}")
        print(f"   Backend: {dist.get_backend()}")

    # --- 数据 ---
    train_dss: list[CryoSparcParticleDataset] = []
    val_dss: list[CryoSparcParticleDataset] = []
    if "datasets" in cfg:
        for ds_cfg in cfg["datasets"]:
            common = dict(
                cs_path=ds_cfg["cs_path"],
                reference_path=ds_cfg["reference_path"],
                project_dir=ds_cfg["project_dir"],
                working_ps=cfg.get("working_ps", 2.0),
                bucket=cfg.get("bucket"),
                apply_ctf_to_proj=cfg.get("apply_ctf_to_proj", False),
                device="cuda", seed=cfg.get("seed", 42),
            )
            tr = CryoSparcParticleDataset(split="train", **common)
            va = CryoSparcParticleDataset(split="val", **common)
            if cfg.get("cache_samples", True):
                tr.enable_cache(); va.enable_cache()
            train_dss.append(tr); val_dss.append(va)
            if is_main_process():
                print(f"  {ds_cfg['name']}: train {len(tr)} val {len(va)}")
        train_ds = ConcatDataset(train_dss)
        val_ds = ConcatDataset(val_dss)
        ref_vol = train_dss[0].vol
    else:
        common = dict(
            cs_path=cfg["cs_path"], reference_path=cfg["reference_path"],
            project_dir=cfg["project_dir"], working_ps=cfg.get("working_ps", 2.0),
            bucket=cfg.get("bucket"),
            apply_ctf_to_proj=cfg.get("apply_ctf_to_proj", False),
            device="cuda", seed=cfg.get("seed", 42),
        )
        train_ds = CryoSparcParticleDataset(split="train", **common)
        val_ds = CryoSparcParticleDataset(split="val", **common)
        if cfg.get("cache_samples", True):
            train_ds.enable_cache(); val_ds.enable_cache()
        ref_vol = train_ds.vol

    if is_main_process():
        print(f"train {len(train_ds)} val {len(val_ds)} | working_ps {cfg.get('working_ps', 2.0)}")

    # 分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, shuffle=False, drop_last=False)

    # DataLoader (batch_size 由 DeepSpeed 管理，这里用 micro_batch_size)
    micro_batch_size = cfg.get("batch_size", 16)  # 每个 GPU 的 batch size
    train_loader = DataLoader(train_ds, batch_size=micro_batch_size,
                               sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=micro_batch_size,
                             sampler=val_sampler, num_workers=2, pin_memory=True)

    # --- Gallery ---
    nside = cfg.get("gallery_nside", 8)
    inplane = cfg.get("gallery_inplane", 12)
    gallery_aa = healpix_axis_angles(nside, n_inplane=inplane)
    if is_main_process():
        print(f"gallery: {len(gallery_aa)} 朝向")

    # --- 模型 ---
    backbone_name = cfg.get("backbone", "convnext_tiny")
    emb_dim = cfg.get("embedding_dim", 128)
    stem_stride = cfg.get("stem_stride", 4)
    share_backbone = cfg.get("share_backbone", False)

    # 创建双塔编码器
    from siamese.models.encoder import SiameseEncoder
    proj_encoder = SiameseEncoder(
        backbone_name=backbone_name,
        embedding_dim=emb_dim,
        stem_stride=stem_stride,
        share_backbone=share_backbone)
    mic_encoder = SiameseEncoder(
        backbone_name=backbone_name,
        embedding_dim=emb_dim,
        stem_stride=stem_stride,
        share_backbone=share_backbone)
    encoder = TwoTowerEncoder(proj_encoder=proj_encoder, mic_encoder=mic_encoder)

    # PoseProposer 配置
    temperature = cfg.get("temperature", 0.07)
    proposer = PoseProposer(
        encoder=encoder,
        temperature=temperature,
        use_residual=True,
        use_shift=True)

    if is_main_process():
        n_params = sum(p.numel() for p in proposer.parameters())
        print(f"参数量 {n_params:,}")

    # --- DeepSpeed 初始化 ---
    # DeepSpeed 会接管 optimizer, lr_scheduler, fp16
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=proposer,
        model_parameters=proposer.parameters(),
    )

    # --- 损失函数 ---
    criterion = OrientationAwareInfoNCELoss(
        temperature=cfg.get("temperature", 0.07))
    gallery_ce = GalleryClassificationLoss(
        temperature=cfg.get("temperature", 0.07),
        label_sigma_deg=cfg.get("label_sigma_deg", 7.0))
    w_ce = cfg.get("w_gallery_ce", 1.0)
    w_nce = cfg.get("w_infonce", 0.5)

    R_gallery = axis_angle_to_matrix(gallery_aa.to(device))

    # --- Checkpoint ---
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints_proposer"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_recall = 0.0

    # TensorBoard (仅主进程)
    writer = None
    if is_main_process():
        tb_dir = ckpt_dir / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard: {tb_dir}")

    # --- 训练循环 ---
    num_epochs = cfg.get("num_epochs", 50)
    eval_interval = cfg.get("eval_interval", 5)
    global_step = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # 重要：打乱数据
        model_engine.train()

        # 重建 gallery (每个 epoch 开始时)
        gal_emb = build_gallery(model_engine, gallery_aa, ref_vol, device)
        model_engine.module.set_gallery(gal_emb, gallery_aa)

        tot_loss, tot_ce, tot_nce, nb = 0.0, 0.0, 0.0, 0
        import time; t0 = time.time()

        for batch_idx, (particle, proj, axisang, _) in enumerate(train_loader):
            particle = particle.to(device); proj = proj.to(device)
            axisang = axisang.to(device)
            R_gt = axis_angle_to_matrix(axisang)

            # 前向
            z_mic = torch.nn.functional.normalize(
                model_engine.module.encoder.encode_mic(particle), dim=1)
            z_proj = torch.nn.functional.normalize(
                model_engine.module.encoder.encode_proj(proj), dim=1)
            loss_ce = gallery_ce(z_mic, gal_emb, R_gt, R_gallery)
            loss_nce = criterion(z_mic, z_proj, axisang)
            loss = w_ce * loss_ce + w_nce * loss_nce

            # NaN 检测
            if not torch.isfinite(loss):
                if is_main_process():
                    print(f"  ⚠ NaN/Inf loss at epoch {epoch+1} batch {batch_idx}, skipping", flush=True)
                continue

            # 反向传播 (DeepSpeed 自动处理 fp16, 梯度裁剪, optimizer step)
            model_engine.backward(loss)
            model_engine.step()

            tot_loss += loss.item()
            tot_ce += loss_ce.item()
            tot_nce += loss_nce.item()
            nb += 1
            global_step += 1

            # TensorBoard (仅主进程)
            if is_main_process() and global_step % 100 == 0 and writer is not None:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/loss_ce', loss_ce.item(), global_step)
                writer.add_scalar('train/loss_nce', loss_nce.item(), global_step)

            # 第一个 batch 可视化
            if batch_idx == 0 and is_main_process() and writer is not None:
                visualize_particle_projection_pairs(particle, proj, writer, epoch, max_pairs=4)

        elapsed = time.time() - t0
        avg_loss = tot_loss / nb if nb > 0 else float('nan')

        # 评估
        recall_metrics = {}
        if (epoch + 1) % eval_interval == 0:
            recall_metrics = evaluate_recall(
                model_engine, val_loader, gallery_aa, device,
                m_values=(1, 5, 10, 50), grid_tol_deg=cfg.get("grid_tol_deg", 15.0))

            # 保存最佳模型 (仅主进程)
            if is_main_process() and recall_metrics:
                r10 = recall_metrics.get("Recall@10", 0.0)
                if r10 > best_recall:
                    best_recall = r10
                    # DeepSpeed 保存
                    model_engine.save_checkpoint(str(ckpt_dir), tag="best")

                # TensorBoard
                if writer is not None:
                    for k, v in recall_metrics.items():
                        writer.add_scalar(f'val/{k}', v, epoch+1)

        # 打印 (仅主进程)
        if is_main_process():
            metrics_str = " | ".join([f"{k} {v:.3f}" for k, v in recall_metrics.items()])
            print(f"epoch {epoch+1}: loss {avg_loss:.4f} ({int(elapsed)}s)" +
                  (f" | {metrics_str}" if metrics_str else ""), flush=True)

    if is_main_process():
        print(f"\n✅ 训练完成！最佳 Recall@10: {best_recall:.3f}")
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
