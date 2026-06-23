"""
评估指标计算。

包括 top-k 准确率、检索结果可视化等。
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_accuracy_at_k(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
) -> dict:
    """
    计算 top-k 准确率。

    参数:
        retrieved_indices: 形状 [N, max_k] 的检索结果索引
        ground_truth_indices: 形状 [N] 的 ground truth proj 索引
        k_values: 要计算的 k 值列表

    返回:
        accuracies: dict, {f"top-{k}": accuracy}
    """
    N = len(ground_truth_indices)
    accuracies = {}

    for k in k_values:
        if k > retrieved_indices.shape[1]:
            continue
        # 检查 ground truth 是否在 top-k 中
        top_k = retrieved_indices[:, :k]  # [N, k]
        matches = np.any(top_k == ground_truth_indices[:, None], axis=1)  # [N]
        acc = matches.mean()
        accuracies[f"top-{k}"] = acc

    return accuracies


def plot_retrieval_results(
    query_mic: torch.Tensor,
    retrieved_projs: torch.Tensor,
    ground_truth: torch.Tensor,
    similarities: np.ndarray,
    k: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    可视化检索结果。

    展示 query mic + top-k retrieved projs + ground truth，每张图下方标注相似度。

    参数:
        query_mic: 形状 [D, D] 的 query image
        retrieved_projs: 形状 [k, D, D] 的检索结果
        ground_truth: 形状 [D, D] 的 ground truth proj
        similarities: 形状 [k] 的余弦相似度
        k: 展示的 top-k 数量
        save_path: 保存路径 (可选)
    """
    fig, axes = plt.subplots(1, k + 2, figsize=(3 * (k + 2), 3))

    # Query mic
    axes[0].imshow(query_mic.cpu().numpy(), cmap="gray")
    axes[0].set_title("Query (mic)")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(ground_truth.cpu().numpy(), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Top-k retrieved
    for i in range(k):
        if i < len(retrieved_projs):
            axes[i + 2].imshow(retrieved_projs[i].cpu().numpy(), cmap="gray")
            is_gt = "✓" if similarities[i] == similarities.max() else ""
            axes[i + 2].set_title(f"Top-{i + 1}: {similarities[i]:.3f} {is_gt}")
        axes[i + 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved retrieval visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tsne(
    mic_embeddings: torch.Tensor,
    proj_embeddings: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    t-SNE 可视化 embedding 空间。

    参数:
        mic_embeddings: 形状 [N, D] 的 mic embedding
        proj_embeddings: 形状 [M, D] 的 proj embedding
        labels: 形状 [N] 的 mic 对应的 proj 标签 (可选)
        save_path: 保存路径 (可选)
    """
    from sklearn.manifold import TSNE

    all_emb = torch.cat([mic_embeddings, proj_embeddings], dim=0).cpu().numpy()
    n_mic = len(mic_embeddings)
    n_proj = len(proj_embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_mic + n_proj - 1))
    emb_2d = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_2d[:n_mic, 0], emb_2d[:n_mic, 1], c="blue", label="mic", alpha=0.6, s=10)
    ax.scatter(emb_2d[n_mic:, 0], emb_2d[n_mic:, 1], c="red", label="proj", alpha=0.6, s=10)
    ax.legend()
    ax.set_title("t-SNE of Embedding Space")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()