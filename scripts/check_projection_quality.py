#!/usr/bin/env python3
"""
投影质量检查工具：检测和诊断异常的投影图像

问题：TensorBoard 中某些投影是"一片白光"而不是清晰的蛋白结构
原因分析：
1. 参考体积质量问题（空体积、未归一化）
2. 投影算法数值溢出（CTF、插值）
3. 姿态参数异常（导致投影到空白区域）
4. 归一化问题（值范围异常）

这些异常投影会导致：
- 激活值爆炸
- 梯度爆炸
- 数值溢出 → NaN
"""

import torch
import numpy as np
import mrcfile
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class ProjectionQualityChecker:
    """投影质量检查器"""

    def __init__(self, save_dir="projection_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def check_volume_quality(self, vol_path: str) -> Dict:
        """检查参考体积的质量"""
        print(f"检查参考体积: {vol_path}")

        with mrcfile.open(vol_path, mode='r', permissive=True) as mrc:
            vol = np.asarray(mrc.data, dtype=np.float32)

        result = {
            'path': vol_path,
            'shape': vol.shape,
            'dtype': vol.dtype,
            'min': vol.min(),
            'max': vol.max(),
            'mean': vol.mean(),
            'std': vol.std(),
            'has_nan': np.isnan(vol).any(),
            'has_inf': np.isinf(vol).any(),
            'num_zeros': (vol == 0).sum(),
            'num_nonzeros': (vol != 0).sum(),
        }

        # 判断质量
        issues = []

        if result['has_nan']:
            issues.append("❌ 体积包含 NaN")

        if result['has_inf']:
            issues.append("❌ 体积包含 Inf")

        if result['num_nonzeros'] < vol.size * 0.01:
            issues.append("⚠️  体积几乎全是零（>99% 零值）")

        if abs(result['mean']) > 100 or result['std'] > 100:
            issues.append("⚠️  体积值范围异常（未归一化？）")

        if result['std'] < 0.01:
            issues.append("⚠️  体积几乎没有变化（标准差过小）")

        result['issues'] = issues
        result['quality'] = 'good' if len(issues) == 0 else 'bad'

        return result

    def visualize_volume_slices(self, vol_path: str, output_path: str = None):
        """可视化体积的切片"""
        with mrcfile.open(vol_path, mode='r', permissive=True) as mrc:
            vol = np.asarray(mrc.data, dtype=np.float32)

        D = vol.shape[0]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Volume Slices: {Path(vol_path).name}')

        # XY 切片（不同 Z）
        for i, z in enumerate([D//4, D//2, 3*D//4]):
            ax = axes[0, i]
            slice_data = vol[z, :, :]
            im = ax.imshow(slice_data, cmap='gray')
            ax.set_title(f'XY slice at Z={z}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # 其他方向
        ax = axes[1, 0]
        im = ax.imshow(vol[:, D//2, :], cmap='gray')
        ax.set_title(f'XZ slice at Y={D//2}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 1]
        im = ax.imshow(vol[:, :, D//2], cmap='gray')
        ax.set_title(f'YZ slice at X={D//2}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # 统计直方图
        ax = axes[1, 2]
        ax.hist(vol.flatten(), bins=100, alpha=0.7)
        ax.set_title('Value Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is None:
            output_path = self.save_dir / f"{Path(vol_path).stem}_slices.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 切片可视化已保存: {output_path}")

    def check_projection_quality(self, proj: torch.Tensor, threshold_std=0.01) -> Dict:
        """
        检查单个投影的质量

        Args:
            proj: [H, W] 或 [1, H, W] 投影图像
            threshold_std: 标准差阈值，低于此值认为是"白光"

        Returns:
            质量检查结果
        """
        if proj.dim() == 3:
            proj = proj.squeeze(0)

        result = {
            'shape': proj.shape,
            'min': proj.min().item(),
            'max': proj.max().item(),
            'mean': proj.mean().item(),
            'std': proj.std().item(),
            'has_nan': torch.isnan(proj).any().item(),
            'has_inf': torch.isinf(proj).any().item(),
        }

        issues = []

        if result['has_nan']:
            issues.append("NaN")

        if result['has_inf']:
            issues.append("Inf")

        if result['std'] < threshold_std:
            issues.append("几乎无变化（白光/黑屏）")

        if abs(result['mean']) > 100:
            issues.append("均值异常大")

        if result['std'] > 100:
            issues.append("标准差异常大")

        result['issues'] = issues
        result['is_valid'] = len(issues) == 0

        return result

    def test_projections_from_volume(
        self,
        vol_path: str,
        num_test: int = 100,
        save_samples: bool = True
    ) -> Dict:
        """
        从参考体积生成测试投影，检查质量

        Args:
            vol_path: 参考体积路径
            num_test: 测试投影数量
            save_samples: 是否保存样本图像

        Returns:
            统计结果
        """
        print(f"\n测试从 {vol_path} 生成投影...")

        # 加载体积
        with mrcfile.open(vol_path, mode='r', permissive=True) as mrc:
            vol_np = np.asarray(mrc.data, dtype=np.float32)

        vol = torch.from_numpy(vol_np)

        # 生成随机姿态
        axis_angles = torch.randn(num_test, 3) * 0.5  # 随机旋转

        # 投影
        from siamese.data.projection import project_fourier_slice_from_axis_angle

        print(f"生成 {num_test} 个投影...")
        projections = project_fourier_slice_from_axis_angle(vol, axis_angles)

        # 检查每个投影
        valid_count = 0
        invalid_count = 0
        invalid_samples = []

        for i in range(num_test):
            proj = projections[i]
            result = self.check_projection_quality(proj)

            if result['is_valid']:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_samples.append({
                    'index': i,
                    'axis_angle': axis_angles[i].tolist(),
                    'issues': result['issues'],
                    'stats': {k: result[k] for k in ['min', 'max', 'mean', 'std']}
                })

                # 保存前几个异常样本
                if save_samples and len(invalid_samples) <= 10:
                    self.save_projection_comparison(
                        proj,
                        f"invalid_proj_{i}",
                        result
                    )

        summary = {
            'total': num_test,
            'valid': valid_count,
            'invalid': invalid_count,
            'invalid_rate': invalid_count / num_test,
            'invalid_samples': invalid_samples[:10]  # 只保留前 10 个
        }

        print(f"\n投影质量统计:")
        print(f"  总数: {summary['total']}")
        print(f"  有效: {summary['valid']} ({valid_count/num_test*100:.1f}%)")
        print(f"  异常: {summary['invalid']} ({invalid_count/num_test*100:.1f}%)")

        if invalid_samples:
            print(f"\n前 5 个异常样本:")
            for sample in invalid_samples[:5]:
                print(f"  #{sample['index']}: {', '.join(sample['issues'])}")
                print(f"    Stats: min={sample['stats']['min']:.3f}, "
                      f"max={sample['stats']['max']:.3f}, "
                      f"std={sample['stats']['std']:.3f}")

        return summary

    def save_projection_comparison(self, proj: torch.Tensor, name: str, stats: Dict):
        """保存投影图像及其统计信息"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 投影图像
        ax = axes[0]
        im = ax.imshow(proj.cpu().numpy(), cmap='gray')
        ax.set_title(f'{name}\n{"INVALID" if not stats["is_valid"] else "VALID"}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # 统计信息
        ax = axes[1]
        ax.axis('off')
        info_text = f"""
统计信息:
  Min: {stats['min']:.3f}
  Max: {stats['max']:.3f}
  Mean: {stats['mean']:.3f}
  Std: {stats['std']:.3f}

问题:
"""
        if stats['issues']:
            for issue in stats['issues']:
                info_text += f"  ❌ {issue}\n"
        else:
            info_text += "  ✅ 无问题\n"

        ax.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                verticalalignment='center')

        output_path = self.save_dir / f"{name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="投影质量检查工具")
    parser.add_argument('--volume', type=str, required=True,
                        help="参考体积路径 (.mrc)")
    parser.add_argument('--num-test', type=int, default=100,
                        help="测试投影数量")
    parser.add_argument('--visualize', action='store_true',
                        help="可视化体积切片")
    args = parser.parse_args()

    checker = ProjectionQualityChecker()

    print("=" * 70)
    print("投影质量检查工具")
    print("=" * 70)
    print()

    # 1. 检查体积质量
    vol_result = checker.check_volume_quality(args.volume)

    print(f"体积形状: {vol_result['shape']}")
    print(f"值范围: [{vol_result['min']:.3f}, {vol_result['max']:.3f}]")
    print(f"均值/标准差: {vol_result['mean']:.3f} ± {vol_result['std']:.3f}")
    print(f"非零值: {vol_result['num_nonzeros']} / {vol_result['num_nonzeros'] + vol_result['num_zeros']}")
    print(f"质量: {vol_result['quality']}")

    if vol_result['issues']:
        print("\n发现问题:")
        for issue in vol_result['issues']:
            print(f"  {issue}")
    else:
        print("\n✅ 体积质量正常")

    print()

    # 2. 可视化体积
    if args.visualize:
        print("生成体积切片可视化...")
        checker.visualize_volume_slices(args.volume)
        print()

    # 3. 测试投影
    proj_summary = checker.test_projections_from_volume(
        args.volume,
        num_test=args.num_test,
        save_samples=True
    )

    print()
    print("=" * 70)

    # 4. 总结和建议
    if proj_summary['invalid_rate'] > 0.1:
        print("⚠️  警告: 超过 10% 的投影异常！")
        print("\n可能的原因和解决方案:")
        print("  1. 参考体积质量问题")
        print("     → 重新导出体积，确保正确归一化")
        print("  2. 投影算法数值溢出")
        print("     → 在投影后添加值裁剪: torch.clamp(proj, -10, 10)")
        print("  3. 需要在训练中过滤异常投影")
        print("     → 使用 ProjectionQualityChecker 过滤")
    else:
        print("✅ 投影质量正常")

    print()
    print(f"分析结果已保存到: {checker.save_dir}")


if __name__ == '__main__':
    main()
