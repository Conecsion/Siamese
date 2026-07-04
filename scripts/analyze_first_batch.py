#!/usr/bin/env python3
"""
分析训练中第一个 batch 的投影值分布

直接从训练数据加载第一个 batch，检查投影的实际像素值分布，
找出是否真的有"白光"（高值区域过大）的投影。
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt


def analyze_first_batch(config_path: str):
    """分析训练第一个 batch 的投影"""

    print("=" * 70)
    print("分析训练第一个 Batch 的投影值")
    print("=" * 70)
    print()

    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 加载数据集
    from siamese.data.cryosparc import CryoSparcParticleDataset

    datasets = []
    for ds_cfg in cfg['datasets']:
        dataset = CryoSparcParticleDataset(
            cs_path=ds_cfg['cs_path'],
            reference_path=ds_cfg['reference_path'],
            project_dir=ds_cfg['project_dir'],
            working_ps=cfg['working_ps'],
            apply_ctf_to_proj=cfg.get('apply_ctf_to_proj', False),
            device='cpu'
        )
        datasets.append(dataset)

    full_dataset = ConcatDataset(datasets)

    # 创建 DataLoader（与训练时相同的设置）
    dataloader = DataLoader(
        full_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,  # 不打乱，获取训练时的第一个 batch
        num_workers=0
    )

    print(f"Batch size: {cfg['batch_size']}")
    print()

    # 获取第一个 batch
    print("加载第一个 batch...")
    batch = next(iter(dataloader))
    particle, proj, axisang, shift, ps_work = batch

    print(f"  Particle shape: {particle.shape}")  # [B, 1, D, D]
    print(f"  Projection shape: {proj.shape}")    # [B, 1, D, D]
    print()

    # 分析每个投影
    print("分析每个投影的像素值分布:")
    print("-" * 70)

    save_dir = Path("first_batch_analysis")
    save_dir.mkdir(exist_ok=True)

    for i in range(len(proj)):
        proj_i = proj[i, 0]  # [D, D]
        particle_i = particle[i, 0]

        # 统计信息
        proj_min = proj_i.min().item()
        proj_max = proj_i.max().item()
        proj_mean = proj_i.mean().item()
        proj_std = proj_i.std().item()

        # 计算亮像素比例（使用不同阈值）
        p95 = torch.quantile(proj_i.flatten(), 0.95).item()
        p90 = torch.quantile(proj_i.flatten(), 0.90).item()
        p99 = torch.quantile(proj_i.flatten(), 0.99).item()

        bright_95 = (proj_i > p95).float().mean().item()
        bright_90 = (proj_i > p90).float().mean().item()
        bright_99 = (proj_i > p99).float().mean().item()

        # 判断是否异常
        is_abnormal = bright_95 > 0.5 or proj_std < 0.01

        status = "❌ 异常" if is_abnormal else "✅ 正常"

        print(f"\n投影 {i+1} {status}:")
        print(f"  值范围: [{proj_min:.4f}, {proj_max:.4f}]")
        print(f"  均值/标准差: {proj_mean:.4f} ± {proj_std:.4f}")
        print(f"  高亮像素比例:")
        print(f"    > P90 ({p90:.4f}): {bright_90*100:.1f}%")
        print(f"    > P95 ({p95:.4f}): {bright_95*100:.1f}%")
        print(f"    > P99 ({p99:.4f}): {bright_99*100:.1f}%")

        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始投影
        ax = axes[0, 0]
        im = ax.imshow(proj_i.numpy(), cmap='gray')
        ax.set_title(f'Projection {i+1}\n{status}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 颗粒图像
        ax = axes[0, 1]
        im = ax.imshow(particle_i.numpy(), cmap='gray')
        ax.set_title(f'Particle {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 直方图
        ax = axes[0, 2]
        proj_values = proj_i.flatten().numpy()
        ax.hist(proj_values, bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(p95, color='red', linestyle='--', label=f'P95={p95:.3f}')
        ax.axvline(p90, color='orange', linestyle='--', label=f'P90={p90:.3f}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count')
        ax.set_title('Value Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 高亮掩码 (P95)
        ax = axes[1, 0]
        mask_95 = (proj_i > p95).numpy()
        im = ax.imshow(mask_95, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'Bright Mask (>P95)\n{bright_95*100:.1f}% coverage')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 高亮掩码 (P90)
        ax = axes[1, 1]
        mask_90 = (proj_i > p90).numpy()
        im = ax.imshow(mask_90, cmap='Oranges', vmin=0, vmax=1)
        ax.set_title(f'Bright Mask (>P90)\n{bright_90*100:.1f}% coverage')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 统计文本
        ax = axes[1, 2]
        ax.axis('off')
        stats_text = f"""
统计信息:

Min:  {proj_min:.4f}
Max:  {proj_max:.4f}
Mean: {proj_mean:.4f}
Std:  {proj_std:.4f}

百分位值:
P90:  {p90:.4f}
P95:  {p95:.4f}
P99:  {p99:.4f}

高亮比例:
>P90: {bright_90*100:.1f}%
>P95: {bright_95*100:.1f}%
>P99: {bright_99*100:.1f}%

姿态:
[{axisang[i,0]:.3f},
 {axisang[i,1]:.3f},
 {axisang[i,2]:.3f}]
"""
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        output_path = save_dir / f"projection_{i+1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  💾 已保存: {output_path}")

    print()
    print("=" * 70)
    print(f"分析完成！结果保存在: {save_dir}")
    print()
    print("说明:")
    print("  - P90/P95/P99 表示第 90/95/99 百分位的像素值")
    print("  - 如果 >P95 的像素超过 50%，说明投影异常（大面积高亮）")
    print("  - 如果标准差 < 0.01，说明投影几乎没有变化（全白或全黑）")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="分析第一个 batch 的投影")
    parser.add_argument('--config', type=str,
                        default='configs/proposer_ribosome_multi.yaml',
                        help="训练配置文件")
    args = parser.parse_args()

    analyze_first_batch(args.config)
