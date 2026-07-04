#!/usr/bin/env python
"""
Proposer 训练 (design §6.4 / §7.1 Phase A) —— 单蛋白可行性验证。

训练 TwoTowerEncoder 对比学习: 真实颗粒 ↔ 其 pose+CTF 投影。
评估指标 Recall@M: 真 pose 对应的 gallery 朝向是否落在网络 top-M 内
(这是 proposer 的核心职责, 决定前向模型精修能否找到正确 pose)。

用法:
    python scripts/train_proposer.py --config configs/proposer_ribosome.yaml

多卡: 当前单卡 sanity; 数据并行维度是颗粒, 后续可接 DeepSpeed (见 design §13)。
"""

import argparse
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from siamese.data.cryosparc import CryoSparcParticleDataset
from siamese.data.orientations import healpix_axis_angles
from siamese.models.encoder import SiameseEncoder, TwoTowerEncoder
from siamese.models.pose_head import PoseProposer
from siamese.losses.infonce import OrientationAwareInfoNCELoss
from siamese.eval.forward_model import angular_error
from siamese.data.projection import axis_angle_to_matrix


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_recall(
    proposer: PoseProposer,
    loader: DataLoader,
    gallery_aa: torch.Tensor,     # [G, 3] gallery 朝向
    device: torch.device,
    m_values=(1, 5, 10, 50),
    grid_tol_deg: float = 7.0,    # 真 pose 与 gallery 点的容差 (网格分辨率)
) -> dict:
    """
    Recall@M: 真 pose 的最近 gallery 朝向是否在网络 top-M 候选内。

    因 gallery 是离散的, "命中"定义为: top-M 候选里存在一个与真 pose
    测地距离 < grid_tol 的朝向。
    """
    proposer.eval()
    R_gal = axis_angle_to_matrix(gallery_aa.to(device))  # [G,3,3]
    max_m = max(m_values)
    hits = {m: 0 for m in m_values}
    total = 0

    for batch in loader:
        particle, _, axisang, _, _ = batch
        particle = particle.to(device)
        out = proposer(particle, top_m=max_m)
        topk_idx = out.topk_idx                      # [N, max_m]
        R_gt = axis_angle_to_matrix(axisang.to(device))  # [N,3,3]

        # 每个颗粒: top-M 候选朝向与真 pose 的测地距离
        cand_R = R_gal[topk_idx]                     # [N, max_m, 3, 3]
        N = R_gt.shape[0]
        err = angular_error(
            cand_R.reshape(-1, 3, 3),
            R_gt[:, None].expand(N, max_m, 3, 3).reshape(-1, 3, 3),
        ).reshape(N, max_m)                          # [N, max_m] deg
        for m in m_values:
            hit = (err[:, :m] < grid_tol_deg).any(dim=1)  # [N]
            hits[m] += int(hit.sum())
        total += N

    proposer.train()
    return {f"Recall@{m}": hits[m] / max(total, 1) for m in m_values}


@torch.no_grad()
def visualize_particle_projection_pairs(particle, proj, writer, step, max_pairs=4):
    """
    可视化颗粒和投影配对，记录到 TensorBoard。

    参数:
        particle: [N, 1, H, W] 真实颗粒图像
        proj: [N, 1, H, W] 投影图像
        writer: TensorBoard SummaryWriter
        step: 当前训练步数
        max_pairs: 最多可视化多少对
    """
    import matplotlib
    matplotlib.use('Agg')

    n = min(max_pairs, particle.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # 颗粒
        pa = particle[i, 0].cpu().numpy()
        axes[0, i].imshow(pa, cmap='gray')
        axes[0, i].set_title(f'Particle {i}')
        axes[0, i].axis('off')

        # 投影
        pr = proj[i, 0].cpu().numpy()
        axes[1, i].imshow(pr, cmap='gray')
        axes[1, i].set_title(f'Projection {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    writer.add_figure('particle_projection_pairs', fig, global_step=step)
    plt.close(fig)


@torch.no_grad()
def build_gallery_embeddings(proposer, gallery_aa, vol, ds, device, chunk=256):
    """用 proj 塔编码 gallery: 按 gallery 朝向投影 reference -> embedding [G,C]。"""
    from siamese.data.projection import project_fourier_slice
    from siamese.data.resample import resample_to_working_ps, DEFAULT_BUCKETS
    from siamese.utils.fft import normalize_image

    G = gallery_aa.shape[0]
    R = axis_angle_to_matrix(gallery_aa.to(device))
    embs = []
    for b0 in range(0, G, chunk):
        rc = R[b0:b0 + chunk]
        projs = project_fourier_slice(vol, rc, shifts=None, pfac=1,
                                      method="trilinear", chunk_size=chunk)  # [bc,Dv,Dv]
        # 裁到颗粒尺寸
        if projs.shape[-1] != ds.image_size:
            m = (projs.shape[-1] - ds.image_size) // 2
            projs = projs[:, m:m + ds.image_size, m:m + ds.image_size]
        # 重采样到工作 ps (与训练输入一致)
        if ds.working_ps is not None:
            buckets = (ds.fixed_bucket,) if ds.fixed_bucket else DEFAULT_BUCKETS
            projs = torch.stack([
                resample_to_working_ps(projs[i], ds.psize, ds.working_ps, buckets)[0]
                for i in range(projs.shape[0])
            ])
        projs = normalize_image(projs).unsqueeze(1)  # [bc,1,B,B]
        embs.append(proposer.encoder.encode_proj(projs))
    return torch.cat(embs, 0)  # [G, C]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pose proposer (single protein).")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    warnings.simplefilter("ignore")
    device = torch.device(cfg.get("device", "cuda"))

    # --- 数据 ---
    train_dss: list[CryoSparcParticleDataset] = []   # 顶层声明，避免 possibly-unbound
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
            print(f"  {ds_cfg['name']}: train {len(tr)} val {len(va)}")
        train_ds = ConcatDataset(train_dss)
        val_ds = ConcatDataset(val_dss)
        # gallery 用第一个数据集的 volume (同一蛋白的不同 refine, volume 近似)
        ref_vol = train_dss[0].vol
    else:
        # 单数据集 (向后兼容)
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
    # num_workers=0: 投影在 GPU 上做, 不能跨进程
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32),
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32),
                            shuffle=False, num_workers=0)
    print(f"train {len(train_ds)} val {len(val_ds)} | working_ps {cfg.get('working_ps', 2.0)}")

    # --- gallery 朝向 (评估用) ---
    gallery_aa = healpix_axis_angles(nside=cfg.get("gallery_nside", 6),
                                     n_inplane=cfg.get("gallery_inplane", 6),
                                     device=device)
    print(f"gallery: {gallery_aa.shape[0]} 朝向")

    # --- 模型 ---
    C = cfg.get("embedding_dim", 128)
    bb = cfg.get("backbone", "convnext_tiny")
    img = cfg.get("image_size", 256)  # 桶尺寸
    enc = TwoTowerEncoder(
        SiameseEncoder(backbone_name=bb, image_size=img, embedding_dim=C),
        SiameseEncoder(backbone_name=bb, image_size=img, embedding_dim=C),
    )
    proposer = PoseProposer(enc, temperature=cfg.get("temperature", 0.07),
                            use_residual=True, use_shift=True).to(device)
    print(f"参数量 {sum(p.numel() for p in proposer.parameters()):,}")

    criterion = OrientationAwareInfoNCELoss(
        temperature=cfg.get("temperature", 0.07),
        margin_deg=cfg.get("orientation_margin_deg", 7.0))
    from siamese.losses.gallery_ce import GalleryClassificationLoss
    gallery_ce = GalleryClassificationLoss(
        temperature=cfg.get("temperature", 0.07),
        label_sigma_deg=cfg.get("label_sigma_deg", 7.0))
    w_ce = cfg.get("w_gallery_ce", 1.0)
    w_nce = cfg.get("w_infonce", 0.5)
    R_gallery = axis_angle_to_matrix(gallery_aa.to(device))  # [G,3,3] 缓存
    opt = torch.optim.AdamW(proposer.parameters(), lr=cfg.get("lr", 1e-4),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    use_amp = cfg.get("mixed_precision", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints_proposer"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_recall = 0.0

    # TensorBoard
    tb_dir = ckpt_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: {tb_dir}")

    global_step = 0
    for epoch in range(cfg.get("num_epochs", 50)):
        import time as _t; _t0 = _t.time()
        # 每 epoch: 用当前 proj 塔快照 gallery embedding 作分类原型 (无梯度)
        proposer.eval()
        with torch.no_grad():
            gal_emb = build_gallery_embeddings(proposer, gallery_aa, ref_vol, train_ds if not isinstance(train_ds, ConcatDataset) else train_dss[0], device)
            gal_emb = torch.nn.functional.normalize(gal_emb, dim=1)
        proposer.train()
        tot_loss = 0.0; tot_ce = 0.0; tot_nce = 0.0; nb = 0
        for batch_idx, (particle, proj, axisang, _, _) in enumerate(train_loader):
            particle = particle.to(device); proj = proj.to(device)
            axisang = axisang.to(device)
            R_gt = axis_angle_to_matrix(axisang)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore[attr-defined]
                z_mic = torch.nn.functional.normalize(
                    proposer.encoder.encode_mic(particle), dim=1)
                z_proj = torch.nn.functional.normalize(   # 显式归一化，防止 AMP 下精度漂移
                    proposer.encoder.encode_proj(proj), dim=1)
                loss_ce = gallery_ce(z_mic, gal_emb, R_gt, R_gallery)
                loss_nce = criterion(z_mic, z_proj, axisang)
                loss = w_ce * loss_ce + w_nce * loss_nce

            # NaN 检测：跳过坏 batch，避免污染权重
            if not torch.isfinite(loss):
                print(f"  ⚠ NaN/Inf loss at epoch {epoch+1} batch {batch_idx}, skipping", flush=True)
                opt.zero_grad()
                continue

            scaler.scale(loss).backward()
            # 梯度裁剪（unscale 后再 clip，与 AMP 兼容）
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
            tot_loss += loss.item()
            tot_ce += loss_ce.item()
            tot_nce += loss_nce.item()
            nb += 1
            global_step += 1

            # TensorBoard: 每 100 batch 记录一次 loss
            if global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/loss_ce', loss_ce.item(), global_step)
                writer.add_scalar('train/loss_nce', loss_nce.item(), global_step)

            # TensorBoard: 每个 epoch 第一个 batch 可视化颗粒-投影配对
            if batch_idx == 0:
                visualize_particle_projection_pairs(particle, proj, writer, epoch, max_pairs=4)

        # 记录 epoch 平均 loss
        avg_loss = tot_loss / nb
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_loss_ce', tot_ce / nb, epoch)
        writer.add_scalar('train/epoch_loss_nce', tot_nce / nb, epoch)

        # 评估 Recall@M (每 eval_interval epoch)
        if (epoch + 1) % cfg.get("eval_interval", 5) == 0:
            gal_emb = build_gallery_embeddings(proposer, gallery_aa, ref_vol, train_ds if not isinstance(train_ds, ConcatDataset) else train_dss[0], device)
            proposer.set_gallery(gal_emb, gallery_aa)
            rec = evaluate_recall(proposer, val_loader, gallery_aa, device,
                                  grid_tol_deg=cfg.get("grid_tol_deg", 7.0))

            # TensorBoard: 记录 Recall
            for metric, value in rec.items():
                writer.add_scalar(f'val/{metric}', value, epoch)

            msg = " ".join(f"{k} {v:.3f}" for k, v in rec.items())
            print(f"epoch {epoch+1}: loss {avg_loss:.4f} ({_t.time()-_t0:.0f}s) | {msg}", flush=True)
            if rec.get("Recall@10", 0) > best_recall:
                best_recall = rec["Recall@10"]
                torch.save({"model": proposer.state_dict(), "epoch": epoch,
                            "recall": rec}, ckpt_dir / "best.pt")
        else:
            print(f"epoch {epoch+1}: loss {avg_loss:.4f} ({_t.time()-_t0:.0f}s)", flush=True)

    print(f"完成. best Recall@10 = {best_recall:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
