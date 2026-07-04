#!/usr/bin/env python3
"""
检测高亮投影（"白光"）工具

使用 .cs 文件中的真实姿态进行投影，检测白色面积（像素值高于阈值）
超过 50% 的异常投影图像。

这些异常投影可能导致训练不稳定和 NaN。
"""

import torch
import numpy as np
import mrcfile
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml


def detect_bright_projections(
    cs_path: str,
    volume_path: str,
    threshold_percentile: float = 95,  # 亮度阈值（百分位）
    bright_area_ratio: float = 0.5,    # 白色面积比例阈值
    num_check: int = 1000,              # 检查的样本数
    save_dir: str = "bright_projection_analysis"
):
    """
    检测高亮投影

    Args:
        cs_path: CryoSPARC .cs 文件路径
        volume_path: 参考体积路径
        threshold_percentile: 亮度阈值百分位（例如 95 表示高于 95% 像素值的被认为是"白"）
        bright_area_ratio: 白色面积比例阈值（例如 0.5 表示超过 50% 为异常）
        num_check: 检查的样本数
        save_dir: 保存目录

    Returns:
        检测结果统计
    """

    print("=" * 70)
    print("高亮投影检测工具")
    print("=" * 70)
    print()

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 1. 加载 .cs 文件
    print(f"加载 .cs 文件: {cs_path}")
    cs_data = np.load(cs_path)
    total_particles = len(cs_data)
    print(f"  总颗粒数: {total_particles}")

    # 随机采样
    if num_check < total_particles:
        indices = np.random.choice(total_particles, num_check, replace=False)
    else:
        indices = np.arange(total_particles)
        num_check = total_particles

    print(f"  检查数量: {num_check}")
    print()

    # 2. 加载参考体积
    print(f"加载参考体积: {volume_path}")
    with mrcfile.open(volume_path, mode='r', permissive=True) as mrc:
        vol_np = np.asarray(mrc.data, dtype=np.float32)
    vol = torch.from_numpy(vol_np)
    print(f"  体积形状: {vol.shape}")
    print()

    # 3. 提取姿态参数
    print("提取姿态参数...")
    axis_angles = []
    for idx in indices:
        aa = cs_data[idx]['alignments3D/pose']
        axis_angles.append(aa)

    axis_angles = torch.from_numpy(np.array(axis_angles, dtype=np.float32))
    print(f"  姿态形状: {axis_angles.shape}")
    print()

    # 4. 批量投影
    print("生成投影...")
    from siamese.data.projection import project_fourier_slice_from_axis_angle

    batch_size = 50
    all_projections = []

    for i in tqdm(range(0, len(axis_angles), batch_size)):
        aa_batch = axis_angles[i:i+batch_size]
        proj_batch = project_fourier_slice_from_axis_angle(vol, aa_batch)
        all_projections.append(proj_batch)

    projections = torch.cat(all_projections, dim=0)
    print(f"  投影形状: {projections.shape}")
    print()

    # 5. 分析每个投影
    print("分析投影亮度...")
    results = []
    bright_samples = []

    # 计算全局阈值（采样以避免内存问题）
    all_values = projections.flatten()
    sample_size = min(100000, all_values.numel())
    sampled_values = all_values[torch.randperm(all_values.numel())[:sample_size]]
    global_threshold = torch.quantile(sampled_values, threshold_percentile / 100.0).item()
    print(f"  全局亮度阈值 (P{threshold_percentile}, 采样 {sample_size} 值): {global_threshold:.4f}")
    print()

    for i, proj in enumerate(tqdm(projections)):
        # 计算高于阈值的像素比例
        bright_pixels = (proj > global_threshold).sum().item()
        total_pixels = proj.numel()
        bright_ratio = bright_pixels / total_pixels

        result = {
            'index': int(indices[i]),
            'axis_angle': axis_angles[i].tolist(),
            'min': proj.min().item(),
            'max': proj.max().item(),
            'mean': proj.mean().item(),
            'std': proj.std().item(),
            'bright_ratio': bright_ratio,
            'is_abnormal': bright_ratio > bright_area_ratio
        }

        results.append(result)

        if result['is_abnormal']:
            bright_samples.append(result)

    # 6. 统计结果
    print()
    print("=" * 70)
    print("检测结果")
    print("=" * 70)
    print()

    num_abnormal = len(bright_samples)
    abnormal_rate = num_abnormal / num_check

    print(f"检查样本数: {num_check}")
    print(f"异常投影数: {num_abnormal} ({abnormal_rate*100:.2f}%)")
    print(f"正常投影数: {num_check - num_abnormal} ({(1-abnormal_rate)*100:.2f}%)")
    print()

    if bright_samples:
        print(f"前 10 个异常样本:")
        for i, sample in enumerate(bright_samples[:10]):
            print(f"  #{sample['index']}: 白色面积 {sample['bright_ratio']*100:.1f}%, "
                  f"mean={sample['mean']:.3f}, std={sample['std']:.3f}")
        print()

    # 7. 可视化分布
    print("生成可视化...")
    visualize_brightness_distribution(results, global_threshold, bright_area_ratio, save_dir)

    # 8. 保存异常样本
    if bright_samples:
        print(f"\n保存前 20 个异常投影...")
        for i, sample in enumerate(bright_samples[:20]):
            idx_in_batch = results.index(sample)
            proj = projections[idx_in_batch]
            save_projection_with_stats(
                proj,
                sample,
                global_threshold,
                f"abnormal_{i}_idx{sample['index']}",
                save_dir
            )

    # 9. 生成报告
    generate_report(results, global_threshold, bright_area_ratio, save_dir)

    print()
    print("=" * 70)

    # 10. 建议
    if abnormal_rate > 0.05:
        print()
        print("⚠️  警告: 超过 5% 的投影存在大面积高亮区域！")
        print()
        print("可能的原因:")
        print("  1. 某些姿态投影到体积的空白/低密度区域")
        print("  2. 参考体积归一化问题")
        print("  3. 投影算法的数值问题")
        print()
        print("建议的解决方案:")
        print("  1. 在训练中添加投影过滤:")
        print("     - 过滤白色面积 > 50% 的投影")
        print("     - 在 Dataset.__getitem__ 中检查并跳过")
        print("  2. 对投影值进行裁剪:")
        print("     proj = torch.clamp(proj, -3*std, 3*std)")
        print("  3. 检查参考体积的质量和归一化")
    else:
        print()
        print("✅ 高亮投影比例正常（< 5%）")

    print()
    print(f"分析结果已保存到: {save_dir}")

    return {
        'total': num_check,
        'abnormal': num_abnormal,
        'abnormal_rate': abnormal_rate,
        'samples': bright_samples[:20]  # 保留前 20 个
    }


def visualize_brightness_distribution(results, threshold, bright_area_ratio, save_dir):
    """可视化亮度分布"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 白色面积比例分布
    ax = axes[0, 0]
    bright_ratios = [r['bright_ratio'] for r in results]
    ax.hist(bright_ratios, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(bright_area_ratio, color='red', linestyle='--',
               label=f'Threshold={bright_area_ratio*100:.0f}%')
    ax.set_xlabel('Bright Area Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bright Area Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 均值分布
    ax = axes[0, 1]
    means = [r['mean'] for r in results]
    ax.hist(means, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('Mean Value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Projection Mean')
    ax.grid(True, alpha=0.3)

    # 3. 标准差分布
    ax = axes[1, 0]
    stds = [r['std'] for r in results]
    ax.hist(stds, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Projection Std')
    ax.grid(True, alpha=0.3)

    # 4. 白色面积 vs 标准差散点图
    ax = axes[1, 1]
    abnormal = [r for r in results if r['is_abnormal']]
    normal = [r for r in results if not r['is_abnormal']]

    if normal:
        ax.scatter([r['std'] for r in normal],
                  [r['bright_ratio'] for r in normal],
                  alpha=0.5, s=20, label='Normal', color='blue')
    if abnormal:
        ax.scatter([r['std'] for r in abnormal],
                  [r['bright_ratio'] for r in abnormal],
                  alpha=0.7, s=50, label='Abnormal', color='red', marker='x')

    ax.axhline(bright_area_ratio, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Bright Area Ratio')
    ax.set_title('Bright Area vs Std')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = save_dir / "brightness_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 分布图已保存: {output_path}")


def save_projection_with_stats(proj, stats, threshold, name, save_dir):
    """保存投影及其统计信息"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 原始投影
    ax = axes[0]
    im = ax.imshow(proj.cpu().numpy(), cmap='gray')
    ax.set_title(f'{name}\nOriginal Projection', color='red')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 高亮掩码
    ax = axes[1]
    mask = (proj > threshold).cpu().numpy()
    im = ax.imshow(mask, cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'Bright Mask (>{threshold:.3f})\n'
                 f'Coverage: {stats["bright_ratio"]*100:.1f}%')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 统计信息
    ax = axes[2]
    ax.axis('off')
    info_text = f"""
统计信息:
  Index: {stats['index']}

  Min:   {stats['min']:.4f}
  Max:   {stats['max']:.4f}
  Mean:  {stats['mean']:.4f}
  Std:   {stats['std']:.4f}

  Bright Ratio: {stats['bright_ratio']*100:.1f}%

  Axis Angle:
    [{stats['axis_angle'][0]:.3f},
     {stats['axis_angle'][1]:.3f},
     {stats['axis_angle'][2]:.3f}]

❌ 异常: 白色面积过大
"""
    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center')

    output_path = save_dir / f"{name}.png"
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_report(results, threshold, bright_area_ratio, save_dir):
    """生成检测报告"""

    report_path = save_dir / "detection_report.txt"

    abnormal = [r for r in results if r['is_abnormal']]

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("高亮投影检测报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"检测参数:\n")
        f.write(f"  亮度阈值: {threshold:.4f}\n")
        f.write(f"  白色面积阈值: {bright_area_ratio*100:.0f}%\n\n")

        f.write(f"检测结果:\n")
        f.write(f"  总样本数: {len(results)}\n")
        f.write(f"  异常投影: {len(abnormal)} ({len(abnormal)/len(results)*100:.2f}%)\n")
        f.write(f"  正常投影: {len(results)-len(abnormal)} ({(1-len(abnormal)/len(results))*100:.2f}%)\n\n")

        if abnormal:
            f.write("异常样本列表:\n")
            f.write("-" * 70 + "\n")
            for sample in abnormal:
                f.write(f"  Index: {sample['index']}\n")
                f.write(f"    白色面积: {sample['bright_ratio']*100:.1f}%\n")
                f.write(f"    均值/标准差: {sample['mean']:.3f} ± {sample['std']:.3f}\n")
                f.write(f"    Axis angle: [{sample['axis_angle'][0]:.3f}, "
                       f"{sample['axis_angle'][1]:.3f}, {sample['axis_angle'][2]:.3f}]\n")
                f.write("\n")

    print(f"  ✅ 检测报告已保存: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="高亮投影检测工具")
    parser.add_argument('--cs', type=str, required=True,
                        help="CryoSPARC .cs 文件路径")
    parser.add_argument('--volume', type=str, required=True,
                        help="参考体积路径")
    parser.add_argument('--threshold-percentile', type=float, default=95,
                        help="亮度阈值百分位 (默认 95)")
    parser.add_argument('--bright-ratio', type=float, default=0.5,
                        help="白色面积比例阈值 (默认 0.5 即 50%%)")
    parser.add_argument('--num-check', type=int, default=1000,
                        help="检查的样本数 (默认 1000)")
    parser.add_argument('--save-dir', type=str, default='bright_projection_analysis',
                        help="保存目录")
    args = parser.parse_args()

    detect_bright_projections(
        cs_path=args.cs,
        volume_path=args.volume,
        threshold_percentile=args.threshold_percentile,
        bright_area_ratio=args.bright_ratio,
        num_check=args.num_check,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
