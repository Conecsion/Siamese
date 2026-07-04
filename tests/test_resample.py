"""Proposer 输入重采样 + 分桶测试 (design §6.4)。"""

import torch
from siamese.data.resample import (
    fourier_resample, pad_or_crop, choose_bucket, resample_to_working_ps,
)


def test_fourier_resample_shape_and_scale():
    img = torch.randn(2, 200, 200)
    out = fourier_resample(img, 100)
    assert out.shape == (2, 100, 100)
    # 降采样保持总体密度尺度 (均值量级不爆炸)
    assert out.isfinite().all()


def test_fourier_resample_identity():
    img = torch.randn(64, 64)
    assert torch.allclose(fourier_resample(img, 64), img, atol=1e-5)


def test_pad_or_crop():
    img = torch.ones(50, 50)
    assert pad_or_crop(img, 80).shape == (80, 80)
    assert pad_or_crop(img, 30).shape == (30, 30)
    # pad 居中: 中心仍是 1, 边缘是 0
    padded = pad_or_crop(img, 80)
    assert padded[40, 40] == 1.0 and padded[0, 0] == 0.0


def test_choose_bucket():
    assert choose_bucket(84) == 128
    assert choose_bucket(201) == 256
    assert choose_bucket(64) == 64
    assert choose_bucket(999) == 256   # 超最大桶取最大


def test_resample_to_working_ps():
    # 模拟 ribosome: 300 box, 1.34 Å/pix -> 工作 ps 2.0
    img = torch.randn(300, 300)
    out, bucket, ps_actual = resample_to_working_ps(img, 1.34, 2.0)
    assert out.shape == (bucket, bucket)
    # 工作 ps 应接近 2.0 (取整误差)
    assert abs(ps_actual - 2.0) < 0.1
    # 物理视场守恒: 300*1.34 ≈ resampled_size * ps_actual
    resampled_size = round(300 * 1.34 / 2.0)
    assert abs(300 * 1.34 - resampled_size * ps_actual) < 1.0
