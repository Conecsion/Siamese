"""
FAISS 检索模块。

对 clean proj 建索引，对 noisy mic 查询 top-k 匹配。
"""

from typing import Tuple

import faiss
import numpy as np
import torch


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    构建 FAISS 内积索引 (用于余弦相似度检索)。

    参数:
        embeddings: 形状 [N, D] 的 L2 归一化 embedding (numpy)

    返回:
        index: FAISS IndexFlatIP 索引
    """
    D = embeddings.shape[1]
    index = faiss.IndexFlatIP(D)  # 内积 = 余弦相似度 (因为 L2 归一化)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve_topk(
    query_embeddings: torch.Tensor,
    index: faiss.IndexFlatIP,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    检索 top-k 最相似 proj。

    参数:
        query_embeddings: 形状 [N, D] 的 mic embedding (torch tensor, L2 归一化)
        index: 已构建的 FAISS 索引
        k: 返回的 top-k 数量

    返回:
        distances: 形状 [N, k] 的余弦相似度 (numpy)
        indices: 形状 [N, k] 的 proj 索引 (numpy)
    """
    query_np = query_embeddings.detach().cpu().numpy().astype(np.float32)
    distances, indices = index.search(query_np, k)
    return distances, indices