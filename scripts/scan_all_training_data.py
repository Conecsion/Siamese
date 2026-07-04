#!/usr/bin/env python3
"""
全面扫描所有训练数据，找出异常投影

检查所有 47,139 个训练样本，识别：
1. 标准差过小的投影（几乎无变化）
2. 高亮区域过大的投影（>50%）
3. 值范围异常的投影

这将帮助找出 TensorBoard 中看到的"白光"投影的来源。
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def full_dataset_scan(config_path: str, save_dir: str = "full_dataset_scan"):
    """扫描所有训练数据，找出异常投影"""

    print("=" * 70)
    print("全面扫描训练数据 - 查找异常投影")
    print("=" * 70)
    print()

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 加载数据集
    from siamese.data.cryosparc import CryoSparcParticleDataset

    datasets = []
    dataset_names = []
    for ds_cfg in cfg['datasets']:
        print(f"加载数据集: {ds_cfg['name']}")
        dataset = CryoSparcParticleDataset(
            cs_path=ds_cfg['cs_path'],
            reference_path=ds_cfg['reference_path'],
            project_dir=ds_cfg['project_dir'],
            working_ps=cfg['working_ps'],
            apply_ctf_to_proj=cfg.get('apply_ctf_to_proj', False),
            device='cpu'
        )
        datasets.append(dataset)
        dataset_names.append(ds_cfg['name'])
        print(f"  样本数: {len(dataset)}")

    full_dataset = ConcatDataset(datasets)
    total_samples = len(full_dataset)
    print(f"\n总样本数: {total_samples}")
    print()

    # 创建 DataLoader
    dataloader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4  # 并行加载
    )

    # 统计结果
    abnormal_projections = []
    stats_list = []

    # 阈值
    std_threshold = 0.01      # 标准差阈值
    bright_threshold = 0.5    # 高亮区域比例阈值

    print(f"开始扫描 {total_samples} 个样本...")
    print(f"检测条件:")
    print(f"  - 标准差 < {std_threshold}")
    print(f"  - 高亮像素比例 > {bright_threshold*100}%")
    print()

    # 扫描
    for idx, batch in enumerate(tqdm(dataloader, total=total_samples)):
        particle, proj, axisang, shift, ps_work = batch

        # 去除 batch 维度
        proj = proj.squeeze(0).squeeze(0)  # [D, D]

        # 统计信息
        proj_min = proj.min().item()
        proj_max = proj.max().item()
        proj_mean = proj.mean().item()
        proj_std = proj.std().item()

        # 计算高亮像素比例
        if proj_std > 0:
            p95 = torch.quantile(proj.flatten(), 0.95).item()
            bright_ratio = (proj > p95).float().mean().item()
        else:
            bright_ratio = 1.0

        stats = {
            'index': idx,
            'min': proj_min,
            'max': proj_max,
            'mean': proj_mean,
            'std': proj_std,
            'bright_ratio': bright_ratio,
        }

        stats_list.append(stats)

        # 检查是否异常
        is_abnormal = proj_std < std_threshold or bright_ratio > bright_threshold

        if is_abnormal:
            abnormal_projections.append({
                'index': idx,
                'stats': stats,
                'projection': proj.clone(),
                'particle': particle.squeeze(0).squeeze(0).clone(),
                'axisang': axisang.squeeze(0).tolist(),
                'reason': []
            })

            if proj_std < std_threshold:
                abnormal_projections[-1]['reason'].append(f'低标准差 ({proj_std:.4f})')
            if bright_ratio > bright_threshold:
                abnormal_projections[-1]['reason'].append(f'高亮区域过大 ({bright_ratio*100:.1f}%)')

    # 打印结果
    print()
    print("=" * 70)
    print("扫描结果")
    print("=" * 70)
    print()

    num_abnormal = len(abnormal_projections)
    abnormal_rate = num_abnormal / total_samples

    print(f"总样本数: {total_samples}")
    print(f"异常投影: {num_abnormal} ({abnormal_rate*100:.3f}%)")
    print(f"正常投影: {total_samples - num_abnormal} ({(1-abnormal_rate)*100:.3f}%)")
    print()

    if abnormal_projections:
        print(f"发现 {num_abnormal} 个异常投影！")
        print()
        print("前 20 个异常样本:")
        for i, sample in enumerate(abnormal_projections[:20]):
            print(f"  样本 #{sample['index']}: {', '.join(sample['reason'])}")
            print(f"    统计: min={sample['stats']['min']:.3f}, "
                  f"max={sample['stats']['max']:.3f}, "
                  f"mean={sample['stats']['mean']:.3f}, "
                  f"std={sample['stats']['std']:.4f}")

        # 保存异常样本
        print()
        print(f"保存前 50 个异常投影...")
        for i, sample in enumerate(abnormal_projections[:50]):
            save_abnormal_sample(sample, i, save_dir)

        # 生成统计报告
        generate_statistics_report(stats_list, abnormal_projections, save_dir)

    else:
        print("✅ 未发现异常投影！所有数据正常。")

    print()
    print("=" * 70)
    print(f"结果已保存到: {save_dir}")


def save_abnormal_sample(sample, sample_idx, save_dir):
    """保存异常样本"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 投影
    ax = axes[0]
    im = ax.imshow(sample['projection'].numpy(), cmap='gray')
    ax.set_title(f'Projection (Sample #{sample["index"]})\n❌ 异常', color='red')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 颗粒
    ax = axes[1]
    im = ax.imshow(sample['particle'].numpy(), cmap='gray')
    ax.set_title(f'Particle (Sample #{sample["index"]})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 统计
    ax = axes[2]
    ax.axis('off')

    stats = sample['stats']
    info_text = f"""
样本信息:
  Index: {sample['index']}

投影统计:
  Min:  {stats['min']:.4f}
  Max:  {stats['max']:.4f}
  Mean: {stats['mean']:.4f}
  Std:  {stats['std']:.4f}

  Bright Ratio: {stats['bright_ratio']*100:.1f}%

异常原因:
"""
    for reason in sample['reason']:
        info_text += f"  ❌ {reason}\n"

    info_text += f"\n姿态 (axis angle):\n"
    info_text += f"  [{sample['axisang'][0]:.3f},\n"
    info_text += f"   {sample['axisang'][1]:.3f},\n"
    info_text += f"   {sample['axisang'][2]:.3f}]"

    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    output_path = save_dir / f"abnormal_{sample_idx:03d}_sample{sample['index']}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_statistics_report(stats_list, abnormal_projections, save_dir):
    """生成统计报告"""

    # 生成分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 标准差分布
    ax = axes[0, 0]
    stds = [s['std'] for s in stats_list]
    ax.hist(stds, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0.01, color='red', linestyle='--', label='Threshold=0.01')
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Count')
    ax.set_title('Projection Std Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 高亮比例分布
    ax = axes[0, 1]
    bright_ratios = [s['bright_ratio'] for s in stats_list]
    ax.hist(bright_ratios, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', label='Threshold=50%')
    ax.set_xlabel('Bright Area Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Bright Area Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 均值分布
    ax = axes[1, 0]
    means = [s['mean'] for s in stats_list]
    ax.hist(means, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Value')
    ax.set_ylabel('Count')
    ax.set_title('Projection Mean Distribution')
    ax.grid(True, alpha=0.3)

    # 异常样本索引分布
    ax = axes[1, 1]
    if abnormal_projections:
        abnormal_indices = [s['index'] for s in abnormal_projections]
        ax.scatter(abnormal_indices, [1]*len(abnormal_indices), alpha=0.5, s=10)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Abnormal')
        ax.set_title(f'Abnormal Sample Distribution ({len(abnormal_projections)} samples)')
        ax.set_ylim([0, 2])
    else:
        ax.text(0.5, 0.5, 'No abnormal samples', ha='center', va='center')
        ax.set_title('No Abnormal Samples Found')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = save_dir / "statistics_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 统计分布图已保存: {output_path}")

    # 生成文本报告
    report_path = save_dir / "scan_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("全面数据扫描报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"总样本数: {len(stats_list)}\n")
        f.write(f"异常样本数: {len(abnormal_projections)}\n")
        f.write(f"异常比例: {len(abnormal_projections)/len(stats_list)*100:.3f}%\n\n")

        if abnormal_projections:
            f.write("异常样本列表:\n")
            f.write("-" * 70 + "\n")
            for sample in abnormal_projections:
                f.write(f"样本 #{sample['index']}:\n")
                f.write(f"  原因: {', '.join(sample['reason'])}\n")
                f.write(f"  统计: std={sample['stats']['std']:.4f}, "
                       f"bright_ratio={sample['stats']['bright_ratio']*100:.1f}%\n")
                f.write(f"  姿态: {sample['axisang']}\n\n")

    print(f"  ✅ 扫描报告已保存: {report_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="全面扫描训练数据")
    parser.add_argument('--config', type=str,
                        default='configs/proposer_ribosome_multi.yaml',
                        help="训练配置文件")
    parser.add_argument('--save-dir', type=str,
                        default='full_dataset_scan',
                        help="保存目录")
    args = parser.parse_args()

    full_dataset_scan(args.config, args.save_dir)
