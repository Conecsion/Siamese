"""检索和评估指标测试。"""

import numpy as np
import torch
from siamese.eval.retrieval import build_faiss_index, retrieve_topk
from siamese.eval.metrics import compute_accuracy_at_k


def test_build_and_retrieve():
    """测试建索引和检索基本功能。"""
    N, D = 100, 128
    # 创建随机 L2 归一化 embedding
    emb = torch.randn(N, D)
    emb = torch.nn.functional.normalize(emb, dim=-1)

    index = build_faiss_index(emb.numpy())

    # 用自身查询，top-1 应该返回自身
    distances, indices = retrieve_topk(emb, index, k=5)
    assert indices.shape == (N, 5), f"Expected ({N}, 5), got {indices.shape}"
    # top-1 应该 = 自身 (余弦相似度最高)
    assert np.all(indices[:, 0] == np.arange(N)), "Top-1 should be self"


def test_compute_accuracy():
    """测试准确率计算。"""
    # 模拟: 5 个 query, 检索 top-10, ground truth 在位置 0, 2, 4, 6, 8
    retrieved = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # gt=0, top-1 命中
        [5, 6, 2, 3, 4, 0, 1, 7, 8, 9],  # gt=2, top-3 命中
        [9, 4, 8, 7, 6, 5, 3, 2, 1, 0],  # gt=4, top-2 命中
        [3, 1, 5, 9, 2, 0, 6, 7, 8, 4],  # gt=6, top-7 命中
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  # gt=8, 未命中
    ])
    gt = np.array([0, 2, 4, 6, 8])

    acc = compute_accuracy_at_k(retrieved, gt, k_values=[1, 5, 10])
    assert acc["top-1"] == 0.2, f"Expected 0.2, got {acc['top-1']}"
    assert acc["top-5"] == 0.6, f"Expected 0.6, got {acc['top-5']}"
    assert acc["top-10"] == 0.8, f"Expected 0.8, got {acc['top-10']}"