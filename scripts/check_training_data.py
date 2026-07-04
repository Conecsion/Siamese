#!/usr/bin/env python3
"""
训练数据质量检查工具

直接从 DataLoader 采样，检查：
1. 真实颗粒图像质量
2. 配对投影质量（使用真实姿态）
3. 是否有异常样本导致 NaN

这个工具可以帮助找出 TensorBoard 中"白光"投影的来源
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


def check_training_data_quality(config_path: str, num_samples: int = 100):
    """检查训练数据质量"""

    print("=" * 70)
    print("训练数据质量检查")
    print("=" * 70)
    print()

    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"配置文件: {config_path}")
    print(f"数据集数量: {len(cfg['datasets'])}")
    print()

    # 导入必要的模块
    from siamese.data.cryosparc import CryoSparcParticleDataset

    # 加载所有数据集
    datasets = []
    for ds_cfg in cfg['datasets']:
        print(f"加载数据集: {ds_cfg['name']}")
        dataset = CryoSparcParticleDataset(
            cs_path=ds_cfg['cs_path'],
            reference_path=ds_cfg['reference_path'],
            project_dir=ds_cfg['project_dir'],
            working_ps=cfg['working_ps'],
            apply_ctf_to_proj=cfg.get('apply_ctf_to_proj', False),
            device='cpu'  # 使用 CPU 避免 CUDA 内存问题
        )
        datasets.append(dataset)
        print(f"  样本数: {len(dataset)}")

    # 合并数据集
    full_dataset = ConcatDataset(datasets)
    print(f"\n总样本数: {len(full_dataset)}")

    # 创建 DataLoader
    dataloader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0  # 单线程，方便调试
    )

    # 统计结果
    stats = {
        'total': 0,
        'valid_particle': 0,
        'valid_proj': 0,
        'invalid_particle': [],
        'invalid_proj': [],
        'particle_stats': [],
        'proj_stats': []
    }

    print(f"\n开始采样 {num_samples} 个样本...")
    print()

    save_dir = Path("training_data_analysis")
    save_dir.mkdir(exist_ok=True)

    # 采样并检查
    for i, batch in enumerate(tqdm(dataloader, total=num_samples)):
        if i >= num_samples:
            break

        particle, proj, axisang, idx = batch

        # 去除 batch 维度
        particle = particle.squeeze(0)  # [1, D, D]
        proj = proj.squeeze(0)          # [1, D, D]
        axisang = axisang.squeeze(0)    # [3]

        stats['total'] += 1

        # 检查颗粒图像
        particle_result = check_sample_quality(particle, f"particle_{i}")
        stats['particle_stats'].append(particle_result)

        if particle_result['is_valid']:
            stats['valid_particle'] += 1
        else:
            stats['invalid_particle'].append({
                'index': i,
                'idx': idx.item(),
                'issues': particle_result['issues'],
                'stats': particle_result['stats']
            })

            # 保存异常样本（前 10 个）
            if len(stats['invalid_particle']) <= 10:
                save_sample_image(
                    particle.squeeze(0),
                    f"invalid_particle_{i}",
                    particle_result,
                    save_dir
                )

        # 检查投影图像
        proj_result = check_sample_quality(proj, f"proj_{i}")
        stats['proj_stats'].append(proj_result)

        if proj_result['is_valid']:
            stats['valid_proj'] += 1
        else:
            stats['invalid_proj'].append({
                'index': i,
                'idx': idx.item(),
                'axisang': axisang.tolist(),
                'issues': proj_result['issues'],
                'stats': proj_result['stats']
            })

            # 保存异常投影（前 10 个）
            if len(stats['invalid_proj']) <= 10:
                save_sample_image(
                    proj.squeeze(0),
                    f"invalid_proj_{i}",
                    proj_result,
                    save_dir
                )

                # 同时保存对应的颗粒图像
                save_sample_image(
                    particle.squeeze(0),
                    f"invalid_proj_{i}_particle",
                    particle_result,
                    save_dir
                )

    # 打印统计结果
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)
    print()

    print(f"总样本数: {stats['total']}")
    print()

    print(f"颗粒图像:")
    print(f"  有效: {stats['valid_particle']} ({stats['valid_particle']/stats['total']*100:.1f}%)")
    print(f"  异常: {len(stats['invalid_particle'])} ({len(stats['invalid_particle'])/stats['total']*100:.1f}%)")

    if stats['invalid_particle']:
        print(f"\n  前 5 个异常样本:")
        for sample in stats['invalid_particle'][:5]:
            print(f"    #{sample['index']} (idx={sample['idx']}): {', '.join(sample['issues'])}")

    print()

    print(f"投影图像:")
    print(f"  有效: {stats['valid_proj']} ({stats['valid_proj']/stats['total']*100:.1f}%)")
    print(f"  异常: {len(stats['invalid_proj'])} ({len(stats['invalid_proj'])/stats['total']*100:.1f}%)")

    if stats['invalid_proj']:
        print(f"\n  前 5 个异常样本:")
        for sample in stats['invalid_proj'][:5]:
            print(f"    #{sample['index']} (idx={sample['idx']}): {', '.join(sample['issues'])}")
            print(f"      Axisang: [{sample['axisang'][0]:.3f}, {sample['axisang'][1]:.3f}, {sample['axisang'][2]:.3f}]")

    print()
    print("=" * 70)

    # 生成汇总报告
    generate_summary_report(stats, save_dir)

    print(f"\n✅ 分析完成！结果已保存到: {save_dir}")

    # 建议
    if len(stats['invalid_proj']) > stats['total'] * 0.05:
        print("\n⚠️  警告: 超过 5% 的投影异常！")
        print("\n建议:")
        print("  1. 在训练中添加数据过滤:")
        print("     - 过滤标准差 < 0.01 的投影")
        print("     - 裁剪值范围: torch.clamp(proj, -10, 10)")
        print("  2. 检查姿态参数是否合理")
        print("  3. 检查参考体积质量")


def check_sample_quality(tensor: torch.Tensor, name: str, threshold_std: float = 0.01) -> dict:
    """检查单个样本的质量"""

    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    result = {
        'name': name,
        'shape': tensor.shape,
        'stats': {
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
        },
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'issues': []
    }

    # 检查异常
    if result['has_nan']:
        result['issues'].append("NaN")

    if result['has_inf']:
        result['issues'].append("Inf")

    if result['stats']['std'] < threshold_std:
        result['issues'].append(f"几乎无变化 (std={result['stats']['std']:.4f})")

    if abs(result['stats']['mean']) > 100:
        result['issues'].append(f"均值异常 (mean={result['stats']['mean']:.2f})")

    if result['stats']['std'] > 100:
        result['issues'].append(f"标准差异常 (std={result['stats']['std']:.2f})")

    result['is_valid'] = len(result['issues']) == 0

    return result


def save_sample_image(tensor: torch.Tensor, name: str, result: dict, save_dir: Path):
    """保存样本图像"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图像
    ax = axes[0]
    im = ax.imshow(tensor.cpu().numpy(), cmap='gray')
    ax.set_title(f'{name}\n{"INVALID" if not result["is_valid"] else "VALID"}',
                 color='red' if not result['is_valid'] else 'green')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 统计信息
    ax = axes[1]
    ax.axis('off')

    stats = result['stats']
    info_text = f"""
统计信息:
  Min:  {stats['min']:.4f}
  Max:  {stats['max']:.4f}
  Mean: {stats['mean']:.4f}
  Std:  {stats['std']:.4f}

问题:
"""
    if result['issues']:
        for issue in result['issues']:
            info_text += f"  ❌ {issue}\n"
    else:
        info_text += "  ✅ 无问题\n"

    ax.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
            verticalalignment='center')

    output_path = save_dir / f"{name}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_summary_report(stats: dict, save_dir: Path):
    """生成汇总报告"""

    report_path = save_dir / "summary_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("训练数据质量检查报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"总样本数: {stats['total']}\n\n")

        f.write("颗粒图像:\n")
        f.write(f"  有效: {stats['valid_particle']} ({stats['valid_particle']/stats['total']*100:.1f}%)\n")
        f.write(f"  异常: {len(stats['invalid_particle'])} ({len(stats['invalid_particle'])/stats['total']*100:.1f}%)\n\n")

        if stats['invalid_particle']:
            f.write("  异常样本列表:\n")
            for sample in stats['invalid_particle']:
                f.write(f"    #{sample['index']}: {', '.join(sample['issues'])}\n")
            f.write("\n")

        f.write("投影图像:\n")
        f.write(f"  有效: {stats['valid_proj']} ({stats['valid_proj']/stats['total']*100:.1f}%)\n")
        f.write(f"  异常: {len(stats['invalid_proj'])} ({len(stats['invalid_proj'])/stats['total']*100:.1f}%)\n\n")

        if stats['invalid_proj']:
            f.write("  异常样本列表:\n")
            for sample in stats['invalid_proj']:
                f.write(f"    #{sample['index']}: {', '.join(sample['issues'])}\n")
                f.write(f"      Axisang: {sample['axisang']}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"✅ 汇总报告已保存: {report_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="训练数据质量检查")
    parser.add_argument('--config', type=str,
                        default='configs/proposer_ribosome_multi.yaml',
                        help="训练配置文件")
    parser.add_argument('--num-samples', type=int, default=100,
                        help="采样数量")
    args = parser.parse_args()

    check_training_data_quality(args.config, args.num_samples)
