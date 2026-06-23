"""InfoNCE loss 测试。"""

import torch
import pytest
from siamese.losses.infonce import InfoNCELoss


def test_infonce_loss_shape():
    """测试 loss 输出为标量。"""
    loss_fn = InfoNCELoss(temperature=0.07)
    z_mic = torch.randn(8, 128)
    z_proj = torch.randn(8, 128)
    loss = loss_fn(z_mic, z_proj)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


def test_infonce_loss_positive():
    """测试 loss 值为正。"""
    loss_fn = InfoNCELoss(temperature=0.07)
    z_mic = torch.randn(8, 128)
    z_proj = torch.randn(8, 128)
    loss = loss_fn(z_mic, z_proj)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


def test_infonce_loss_perfect_match():
    """测试当 mic 和 proj embedding 相同时 loss 最小。"""
    loss_fn = InfoNCELoss(temperature=0.07)
    z = torch.randn(8, 128)
    z = torch.nn.functional.normalize(z, dim=-1)
    loss_same = loss_fn(z, z)

    # 打乱 proj (破坏配对)
    z_shuffled = z[torch.randperm(8)]
    loss_shuffled = loss_fn(z, z_shuffled)

    assert loss_same.item() < loss_shuffled.item(), \
        "Loss should be lower when embeddings are aligned"


def test_infonce_loss_with_hard_negatives():
    """测试带 hard negatives 的 loss。"""
    # TODO: 后续实现 hard negative mining 后完善此测试
    pass