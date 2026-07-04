#!/usr/bin/env python3
"""
NaN 调试工具：监控训练过程中的数值稳定性

功能：
1. 实时监控训练日志，检测 NaN
2. 保存导致 NaN 的 batch 信息
3. 检查梯度、loss、activation 是否有异常
4. 生成诊断报告
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime


def parse_log_for_nan(log_path: Path) -> dict:
    """
    解析日志文件，查找 NaN 出现的位置

    Returns:
        {
            'has_nan': bool,
            'first_nan_line': int,
            'context': [str],  # NaN 前后的日志行
            'last_good_metrics': dict
        }
    """
    result = {
        'has_nan': False,
        'first_nan_line': None,
        'context': [],
        'last_good_metrics': {}
    }

    if not log_path.exists():
        print(f"错误: 日志文件不存在: {log_path}")
        return result

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 查找 NaN
    for i, line in enumerate(lines):
        if 'nan' in line.lower():
            result['has_nan'] = True
            result['first_nan_line'] = i + 1
            # 获取上下文（前后 20 行）
            start = max(0, i - 20)
            end = min(len(lines), i + 20)
            result['context'] = lines[start:end]
            break

    # 查找最后一个正常的 loss 值
    loss_pattern = re.compile(r'loss[:\s=]+([\d.]+)', re.IGNORECASE)
    for line in reversed(lines[:result['first_nan_line']] if result['has_nan'] else lines):
        match = loss_pattern.search(line)
        if match:
            result['last_good_metrics']['loss'] = float(match.group(1))
            result['last_good_metrics']['line'] = line.strip()
            break

    return result


def check_config(config_path: Path) -> dict:
    """检查配置文件中可能导致 NaN 的设置"""
    import yaml

    issues = []

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 检查学习率
    lr = config.get('learning_rate', config.get('lr', None))
    if lr and lr > 0.01:
        issues.append(f"⚠️  学习率过大: {lr} (建议 < 0.01)")

    # 检查温度参数
    temp = config.get('temperature', 0.07)
    if temp < 0.01:
        issues.append(f"⚠️  温度参数过小: {temp} (可能导致 exp 溢出)")

    # 检查 batch size
    batch_size = config.get('batch_size', None)
    if batch_size and batch_size < 8:
        issues.append(f"⚠️  Batch size 过小: {batch_size} (统计不稳定)")

    return {
        'config': config,
        'issues': issues
    }


def check_deepspeed_config(ds_config_path: Path) -> dict:
    """检查 DeepSpeed 配置"""
    import json

    issues = []

    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)

    # 检查 FP16 配置
    fp16 = ds_config.get('fp16', {})
    if fp16.get('enabled', False):
        loss_scale = fp16.get('loss_scale', 0)
        if loss_scale == 0:  # 动态 loss scale
            initial_scale = fp16.get('initial_scale_power', 32)
            if initial_scale > 16:
                issues.append(f"⚠️  FP16 initial_scale_power 过大: {initial_scale}")

        # 检查梯度裁剪
        if 'gradient_clipping' not in ds_config:
            issues.append("⚠️  未启用梯度裁剪，可能导致梯度爆炸")

    return {
        'config': ds_config,
        'issues': issues
    }


def generate_diagnostic_script() -> str:
    """生成用于追踪 NaN 的调试代码"""
    return '''
# 在训练脚本中添加以下代码来追踪 NaN

import torch
import numpy as np

def check_nan_inf(tensor, name="tensor"):
    """检查张量中是否有 NaN 或 Inf"""
    if torch.isnan(tensor).any():
        print(f"❌ {name} contains NaN!")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️  {name} contains Inf!")
        return True
    return False

# 在训练循环中添加：
def train_step_with_nan_check(batch, model, ...):
    # 1. 检查输入数据
    particle, proj, axisang, _ = batch
    if check_nan_inf(particle, "particle"):
        print(f"NaN in input particle at indices: {torch.where(torch.isnan(particle))}")
        torch.save(particle, "debug_nan_particle.pt")
        raise ValueError("NaN in input data")

    if check_nan_inf(proj, "proj"):
        print(f"NaN in input proj at indices: {torch.where(torch.isnan(proj))}")
        torch.save(proj, "debug_nan_proj.pt")
        raise ValueError("NaN in projection")

    # 2. 前向传播
    outputs = model(particle, proj)

    # 3. 检查输出
    for key, value in outputs.items():
        if check_nan_inf(value, f"output_{key}"):
            torch.save({
                'particle': particle,
                'proj': proj,
                'outputs': outputs
            }, f"debug_nan_batch_{key}.pt")
            raise ValueError(f"NaN in model output: {key}")

    # 4. 计算 loss
    loss = compute_loss(outputs, ...)

    # 5. 检查 loss
    if check_nan_inf(loss, "loss"):
        torch.save({
            'batch': batch,
            'outputs': outputs,
            'loss': loss
        }, "debug_nan_loss.pt")
        raise ValueError("NaN in loss")

    # 6. 反向传播前检查梯度
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and check_nan_inf(param.grad, f"grad_{name}"):
            torch.save({
                'name': name,
                'param': param,
                'grad': param.grad
            }, f"debug_nan_grad_{name.replace('.', '_')}.pt")
            raise ValueError(f"NaN in gradient: {name}")

    return loss
'''


def main():
    parser = argparse.ArgumentParser(description="NaN 调试工具")
    parser.add_argument('--log', type=str, help="训练日志文件路径")
    parser.add_argument('--config', type=str, default='configs/proposer_ribosome_multi.yaml',
                        help="训练配置文件")
    parser.add_argument('--ds-config', type=str, default='configs/ds_config.json',
                        help="DeepSpeed 配置文件")
    parser.add_argument('--generate-patch', action='store_true',
                        help="生成调试代码补丁")
    args = parser.parse_args()

    print("=" * 70)
    print("NaN 诊断工具")
    print("=" * 70)
    print()

    # 1. 检查日志
    if args.log:
        log_path = Path(args.log)
        print(f"[1] 分析日志文件: {log_path}")
        result = parse_log_for_nan(log_path)

        if result['has_nan']:
            print(f"❌ 检测到 NaN！位置: 第 {result['first_nan_line']} 行")
            print()
            print("上下文:")
            for line in result['context']:
                print(f"  {line.rstrip()}")
            print()

            if result['last_good_metrics']:
                print("最后正常的指标:")
                print(f"  Loss: {result['last_good_metrics'].get('loss', 'N/A')}")
                print(f"  行: {result['last_good_metrics'].get('line', 'N/A')}")
        else:
            print("✅ 未检测到 NaN")
        print()

    # 2. 检查配置
    config_path = Path(args.config)
    if config_path.exists():
        print(f"[2] 检查训练配置: {config_path}")
        config_result = check_config(config_path)

        if config_result['issues']:
            print("发现潜在问题:")
            for issue in config_result['issues']:
                print(f"  {issue}")
        else:
            print("✅ 配置正常")
        print()

    # 3. 检查 DeepSpeed 配置
    ds_config_path = Path(args.ds_config)
    if ds_config_path.exists():
        print(f"[3] 检查 DeepSpeed 配置: {ds_config_path}")
        ds_result = check_deepspeed_config(ds_config_path)

        if ds_result['issues']:
            print("发现潜在问题:")
            for issue in ds_result['issues']:
                print(f"  {issue}")
        else:
            print("✅ DeepSpeed 配置正常")
        print()

    # 4. 常见原因分析
    print("[4] 常见 NaN 原因")
    print("  1. 学习率过大 → 梯度爆炸")
    print("  2. FP16 精度溢出 → loss_scale 设置不当")
    print("  3. 除零错误 → 检查归一化层、温度参数")
    print("  4. Log(0) 或 sqrt(负数) → 检查 loss 计算")
    print("  5. 输入数据异常 → 检查数据预处理")
    print("  6. 梯度累积过多 → 降低 gradient_accumulation_steps")
    print()

    # 5. 建议的修复方案
    print("[5] 建议的修复方案")
    print("  1. 添加梯度裁剪: gradient_clipping = 1.0")
    print("  2. 降低学习率: lr = 1e-4 或更小")
    print("  3. 调整 FP16 loss_scale: initial_scale_power = 12")
    print("  4. 增加数值稳定性: 在 log/sqrt 前加 eps=1e-8")
    print("  5. 检查输入: 添加 assert not torch.isnan(x).any()")
    print()

    # 6. 生成调试代码
    if args.generate_patch:
        print("[6] 生成调试代码")
        patch_file = Path("debug_nan_patch.py")
        with open(patch_file, 'w') as f:
            f.write(generate_diagnostic_script())
        print(f"✅ 已生成: {patch_file}")
        print("  将其中的代码添加到训练脚本中即可追踪 NaN")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
