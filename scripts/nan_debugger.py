"""
在训练脚本中添加 NaN 检测和调试功能

将此代码集成到 train_proposer_ds.py 中，可以：
1. 实时检测 NaN/Inf
2. 保存导致 NaN 的具体 batch
3. 记录详细的调试信息
"""

import torch
import numpy as np
from pathlib import Path


class NaNDebugger:
    """NaN 调试器：追踪训练过程中的数值异常"""

    def __init__(self, save_dir="debug_nan", enabled=True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.enabled = enabled
        self.nan_count = 0

    def check_tensor(self, tensor, name, step=None, save_on_nan=True):
        """
        检查张量是否包含 NaN 或 Inf

        Args:
            tensor: 要检查的张量
            name: 张量名称
            step: 当前步数
            save_on_nan: 是否在发现 NaN 时保存

        Returns:
            dict: {'has_nan': bool, 'has_inf': bool, 'stats': dict}
        """
        if not self.enabled:
            return {'has_nan': False, 'has_inf': False}

        result = {
            'has_nan': False,
            'has_inf': False,
            'stats': {}
        }

        # 转换为张量
        if isinstance(tensor, (list, tuple)):
            tensor = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in tensor])
        elif not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)

        # 检查 NaN
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        result['has_nan'] = has_nan
        result['has_inf'] = has_inf

        # 统计信息
        with torch.no_grad():
            result['stats'] = {
                'min': tensor.min().item() if tensor.numel() > 0 else None,
                'max': tensor.max().item() if tensor.numel() > 0 else None,
                'mean': tensor.float().mean().item() if tensor.numel() > 0 else None,
                'std': tensor.float().std().item() if tensor.numel() > 1 else None,
            }

        # 发现异常
        if has_nan or has_inf:
            self.nan_count += 1
            msg = f"{'NaN' if has_nan else 'Inf'} detected in {name}"
            if step is not None:
                msg += f" at step {step}"

            print(f"\n{'='*70}")
            print(f"❌ {msg}")
            print(f"   Shape: {tensor.shape}")
            print(f"   Stats: {result['stats']}")

            if has_nan:
                nan_indices = torch.where(torch.isnan(tensor))
                print(f"   NaN位置（前10个）: {[idx[:10].tolist() for idx in nan_indices]}")

            if has_inf:
                inf_indices = torch.where(torch.isinf(tensor))
                print(f"   Inf位置（前10个）: {[idx[:10].tolist() for idx in inf_indices]}")

            print(f"{'='*70}\n")

            # 保存调试信息
            if save_on_nan:
                self.save_debug_info(tensor, name, step, result)

        return result

    def save_debug_info(self, tensor, name, step, result):
        """保存导致 NaN 的张量"""
        filename = self.save_dir / f"nan_{self.nan_count}_{name}_step{step}.pt"

        try:
            torch.save({
                'tensor': tensor.cpu(),
                'name': name,
                'step': step,
                'stats': result['stats'],
                'shape': tensor.shape,
                'dtype': tensor.dtype,
            }, filename)
            print(f"   💾 已保存调试信息: {filename}")
        except Exception as e:
            print(f"   ⚠️  保存失败: {e}")

    def check_batch(self, batch, step=None):
        """检查整个 batch 的数据"""
        particle, proj, axisang, _ = batch

        results = {
            'particle': self.check_tensor(particle, 'input_particle', step, save_on_nan=True),
            'proj': self.check_tensor(proj, 'input_proj', step, save_on_nan=True),
            'axisang': self.check_tensor(axisang, 'input_axisang', step, save_on_nan=True),
        }

        has_nan = any(r['has_nan'] for r in results.values())
        has_inf = any(r['has_inf'] for r in results.values())

        return has_nan, has_inf, results

    def check_model_outputs(self, outputs, step=None):
        """检查模型输出"""
        results = {}

        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    results[key] = self.check_tensor(value, f'output_{key}', step)
        elif torch.is_tensor(outputs):
            results['output'] = self.check_tensor(outputs, 'output', step)

        has_nan = any(r['has_nan'] for r in results.values())
        has_inf = any(r['has_inf'] for r in results.values())

        return has_nan, has_inf, results

    def check_gradients(self, model, step=None, max_check=10):
        """检查模型梯度"""
        results = {}
        checked = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                results[name] = self.check_tensor(
                    param.grad,
                    f'grad_{name}',
                    step,
                    save_on_nan=True
                )

                checked += 1
                if checked >= max_check:
                    break

        has_nan = any(r['has_nan'] for r in results.values())
        has_inf = any(r['has_inf'] for r in results.values())

        return has_nan, has_inf, results


# ============================================================================
# 在训练循环中集成 NaNDebugger
# ============================================================================

"""
使用示例：

# 1. 在训练脚本开头初始化
debugger = NaNDebugger(save_dir="debug_nan", enabled=True)

# 2. 在训练循环中检查
for batch_idx, batch in enumerate(train_loader):
    # 检查输入数据
    has_nan, has_inf, _ = debugger.check_batch(batch, step=batch_idx)
    if has_nan or has_inf:
        print(f"❌ Batch {batch_idx} 的输入数据有异常，跳过")
        continue

    # 前向传播
    outputs = model(batch)

    # 检查输出
    has_nan, has_inf, _ = debugger.check_model_outputs(outputs, step=batch_idx)
    if has_nan or has_inf:
        print(f"❌ Batch {batch_idx} 的输出有异常")
        # 保存完整的 batch
        torch.save({
            'batch': batch,
            'outputs': outputs,
            'batch_idx': batch_idx
        }, f"debug_nan/nan_batch_{batch_idx}.pt")
        continue

    # 计算 loss
    loss = compute_loss(outputs, ...)

    # 检查 loss
    loss_check = debugger.check_tensor(loss, 'loss', step=batch_idx)
    if loss_check['has_nan'] or loss_check['has_inf']:
        print(f"❌ Loss 是 NaN/Inf，停止训练")
        break

    # 反向传播
    loss.backward()

    # 检查梯度（每 N 步检查一次）
    if batch_idx % 10 == 0:
        has_nan, has_inf, _ = debugger.check_gradients(model, step=batch_idx)
        if has_nan or has_inf:
            print(f"⚠️  梯度中发现 NaN/Inf")
"""


# ============================================================================
# 额外的数值稳定性工具
# ============================================================================

def safe_log(x, eps=1e-8):
    """安全的 log 操作，避免 log(0)"""
    return torch.log(torch.clamp(x, min=eps))


def safe_sqrt(x, eps=1e-8):
    """安全的 sqrt 操作，避免 sqrt(负数)"""
    return torch.sqrt(torch.clamp(x, min=eps))


def safe_div(numerator, denominator, eps=1e-8):
    """安全的除法，避免除零"""
    return numerator / torch.clamp(denominator, min=eps)


def clip_gradients(model, max_norm=1.0):
    """裁剪梯度，防止梯度爆炸"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def check_model_weights(model):
    """检查模型权重是否正常"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ 参数 {name} 包含 NaN")
            return False
        if torch.isinf(param).any():
            print(f"⚠️  参数 {name} 包含 Inf")
            return False
    return True
