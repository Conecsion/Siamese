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

def test_compute_ctf_2d_matches_pyem():
    """projection.py 的 _compute_ctf_2d 应与 pyem.ctf.eval_ctf 吻合 (符号回归测试)。

    防止振幅对比度项符号错误 (sin/cos 异号) 退化重现。
    pyem 作为权威 oracle (与 RELION/EMAN2 数值一致)。
    """
    import sys, os
    import numpy as np
    pyem_dir = os.path.join(os.path.dirname(__file__), "..", "pyem")
    if pyem_dir not in sys.path:
        sys.path.insert(0, pyem_dir)
    try:
        from pyem import ctf as pyem_ctf
    except ImportError:
        pytest.skip("pyem 不可用")
    from siamese.data.projection import _compute_ctf_2d

    D = 128
    psize, df, kv, cs, ac, ph = 1.5, 18000.0, 300.0, 2.7, 0.1, 0.3
    k = torch.arange(-D // 2, D // 2, dtype=torch.float32)
    kx, ky = torch.meshgrid(k, k, indexing="xy")
    proj = _compute_ctf_2d(kx, ky, D, psize, df, df, 0.0, kv, cs, ac, ph, -1.0).numpy()

    freq = (np.arange(-D // 2, D // 2) / (D * psize)).astype(np.float32)
    sx, sy = np.meshgrid(freq, freq)
    s = np.sqrt(sx**2 + sy**2); a = np.arctan2(sy, sx)
    pv = pyem_ctf.eval_ctf(s, a, def1=df, def2=df, angast=0.0,
                           phase=np.rad2deg(ph), kv=kv, ac=ac, cs=cs, bf=0, lp=0)

    # 符号正确性: 中心值应 +ac (不是 -ac), 整体差异远小于符号 bug 的 ~2.0
    assert proj[D // 2, D // 2] > 0, "DC 处 CTF 应为正 (振幅对比度符号)"
    assert np.abs(proj - pv).max() < 1e-3, \
        f"与 pyem 偏差过大: {np.abs(proj - pv).max()}"
