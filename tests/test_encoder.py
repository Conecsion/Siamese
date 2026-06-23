"""SiameseEncoder 测试。"""

import torch
import pytest
from siamese.models.encoder import SiameseEncoder


@pytest.fixture
def encoder():
    """创建默认编码器实例。"""
    return SiameseEncoder(
        backbone_name="convnext_tiny",
        image_size=128,
        real_in_channels=1,
        freq_in_channels=2,
        embedding_dim=128,
    )


def test_encoder_output_shape(encoder):
    """测试编码器输出形状。"""
    N = 4
    D = 128
    x = torch.randn(N, 1, D, D)  # 实空间输入
    emb = encoder(x)
    assert emb.shape == (N, 128), f"Expected (4, 128), got {emb.shape}"


def test_encoder_l2_normalized(encoder):
    """测试编码器输出 L2 归一化。"""
    x = torch.randn(8, 1, 128, 128)
    emb = encoder(x)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), \
        f"Expected L2 norm ≈ 1.0, got {norms}"


def test_encoder_consistency(encoder):
    """测试同一输入产生相同 embedding。"""
    x = torch.randn(1, 1, 128, 128)
    encoder.eval()
    with torch.no_grad():
        emb1 = encoder(x)
        emb2 = encoder(x)
    assert torch.allclose(emb1, emb2, atol=1e-6), \
        "Same input should produce same embedding in eval mode"


def test_encoder_different_inputs_different_embeddings(encoder):
    """测试不同输入产生不同 embedding。"""
    x1 = torch.randn(2, 1, 128, 128)
    x2 = torch.randn(2, 1, 128, 128)
    encoder.eval()
    with torch.no_grad():
        emb1 = encoder(x1)
        emb2 = encoder(x2)
    # 不同输入应有不同 embedding
    assert not torch.allclose(emb1, emb2, atol=1e-3), \
        "Different inputs should produce different embeddings"