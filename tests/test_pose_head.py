"""PoseProposer (design §6.4) 形状 + 先验 + 限幅测试。"""

import torch
from siamese.models.encoder import SiameseEncoder, TwoTowerEncoder
from siamese.models.pose_head import PoseProposer


def _tiny_proposer(C=32):
    enc = TwoTowerEncoder(
        SiameseEncoder(image_size=64, embedding_dim=C,
                       convnext_depths=(1, 1, 1, 1), convnext_dims=(16, 32, 64, 128)),
        SiameseEncoder(image_size=64, embedding_dim=C,
                       convnext_depths=(1, 1, 1, 1), convnext_dims=(16, 32, 64, 128)),
    )
    return PoseProposer(enc, use_residual=True, use_shift=True, max_residual=0.15, max_shift=20.0)


def test_proposer_shapes_and_prior():
    prop = _tiny_proposer(C=32)
    G, C = 100, 32
    prop.set_gallery(
        torch.nn.functional.normalize(torch.randn(G, C), dim=1),
        torch.randn(G, 3),
    )
    out = prop(torch.randn(4, 1, 64, 64), top_m=16)
    assert out.topk_idx.shape == (4, 16)
    assert out.prior.shape == (4, 16)
    # 先验是概率分布: 行和为 1, 非负
    assert torch.allclose(out.prior.sum(1), torch.ones(4), atol=1e-4)
    assert (out.prior >= 0).all()
    assert out.residual.shape == (4, 16, 3)
    assert out.shift.shape == (4, 2)


def test_proposer_clamps():
    prop = _tiny_proposer(C=32)
    prop.set_gallery(torch.nn.functional.normalize(torch.randn(50, 32), dim=1), torch.randn(50, 3))
    out = prop(torch.randn(4, 1, 64, 64), top_m=10)
    assert out.residual.abs().max() <= 0.15 + 1e-4   # 残差限幅
    assert out.shift.abs().max() <= 20.0 + 1e-3      # 位移限幅


def test_proposer_no_heads():
    """可关闭残差/位移头 (纯检索模式)。"""
    enc = TwoTowerEncoder(
        SiameseEncoder(image_size=64, embedding_dim=32,
                       convnext_depths=(1, 1, 1, 1), convnext_dims=(16, 32, 64, 128)),
        SiameseEncoder(image_size=64, embedding_dim=32,
                       convnext_depths=(1, 1, 1, 1), convnext_dims=(16, 32, 64, 128)),
    )
    prop = PoseProposer(enc, use_residual=False, use_shift=False)
    prop.set_gallery(torch.nn.functional.normalize(torch.randn(50, 32), dim=1), torch.randn(50, 3))
    out = prop(torch.randn(2, 1, 64, 64), top_m=8)
    assert out.residual is None and out.shift is None
    assert out.topk_idx.shape == (2, 8)
