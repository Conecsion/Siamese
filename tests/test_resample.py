"""Fourier binning 重采样测试 (无混叠降采样)。"""

import numpy as np
import torch
from siamese.data.resample import (
    fourier_crop_2d, fourier_crop_3d, fourier_pad_2d,
    resample_to_working_ps,
)


def test_fourier_crop_2d_shape():
    img = torch.randn(200, 200)
    out = fourier_crop_2d(img, 100)
    assert out.shape == (100, 100)
    assert torch.is_tensor(out)
    assert out.isfinite().all()


def test_fourier_crop_2d_identity():
    img = torch.randn(64, 64)
    out = fourier_crop_2d(img, 64)
    assert torch.allclose(out, img, atol=1e-4)


def test_fourier_crop_2d_numpy():
    img = np.random.randn(128, 128).astype(np.float32)
    out = fourier_crop_2d(img, 64)
    assert isinstance(out, np.ndarray)
    assert out.shape == (64, 64)


def test_fourier_crop_3d_shape():
    vol = np.random.randn(64, 64, 64).astype(np.float32)
    out = fourier_crop_3d(vol, 32)
    assert out.shape == (32, 32, 32)
    assert np.isfinite(out).all()


def test_fourier_pad_2d_shape():
    img = torch.randn(50, 50)
    out = fourier_pad_2d(img, 80)
    assert out.shape == (80, 80)


def test_resample_to_working_ps():
    img = torch.randn(300, 300)
    out, bucket, ps_actual = resample_to_working_ps(img, 1.34, 2.0)
    assert out.shape == (bucket, bucket)
    # 分桶后实际 ps 会偏离目标（因为桶是离散的）
    assert 0.5 < ps_actual < 4.0
    # 只验证输出形状是合法的桶尺寸
    assert bucket in (64, 128, 256, 384, 512)
