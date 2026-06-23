"""CTF 生成函数的测试。"""

import torch
import pytest
from siamese.utils.ctf import compute_ctf


def test_compute_ctf_shape_and_dtype():
    """测试 CTF 输出的形状和数据类型。"""
    D = 128
    ctf = compute_ctf(
        image_size=D,
        pixel_size=1.0,       # Å/pixel
        defocus=2.0,          # μm
        cs=2.7,               # mm, 球差系数
        voltage=300.0,        # kV
        amplitude_contrast=0.1,
    )
    # 输出应为复数 CTF，形状 [D, D]
    assert ctf.shape == (D, D), f"Expected shape ({D}, {D}), got {ctf.shape}"
    assert ctf.dtype in (torch.complex64, torch.complex128), \
        f"Expected complex dtype, got {ctf.dtype}"


def test_compute_ctf_batch_shape():
    """测试批量 CTF 输出形状。"""
    N = 4
    D = 64
    ctf = compute_ctf(
        image_size=D,
        pixel_size=1.0,
        defocus=torch.tensor([1.0, 1.5, 2.0, 2.5]),  # [N]
        cs=2.7,
        voltage=300.0,
        amplitude_contrast=0.1,
    )
    assert ctf.shape == (N, D, D), f"Expected shape ({N}, {D}, {D}), got {ctf.shape}"


def test_compute_ctf_values_in_range():
    """测试 CTF 值在 [-1, 1] 范围内。"""
    D = 64
    ctf = compute_ctf(
        image_size=D,
        pixel_size=1.0,
        defocus=2.0,
        cs=2.7,
        voltage=300.0,
        amplitude_contrast=0.1,
    )
    assert torch.all(ctf.real >= -1.0) and torch.all(ctf.real <= 1.0), \
        "CTF values should be in [-1, 1]"
    assert torch.all(ctf.imag.abs() < 1e-6), \
        "CTF should be nearly real-valued (imag ~ 0)"