#!/usr/bin/env python
"""
评估训练好的 Siamese 编码器的检索性能。

用法:
    python scripts/eval.py --checkpoint checkpoints/best.pt --data-dir data/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from siamese.training.config import TrainConfig
from siamese.models.encoder import SiameseEncoder
from siamese.data.dataset import MicProjDataset
from siamese.data.transforms import PreprocessTransform
from siamese.eval.retrieval import build_faiss_index, retrieve_topk
from siamese.eval.metrics import compute_accuracy_at_k, plot_retrieval_results, plot_tsne


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Siamese encoder retrieval performance."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory.")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image size.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device.")
    parser.add_argument("--k", type=int, default=20,
                        help="Top-k for retrieval.")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory for evaluation outputs.")
    parser.add_argument("--num-visualize", type=int, default=5,
                        help="Number of retrieval examples to visualize.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Epoch: {checkpoint['epoch'] + 1}, Best val loss: {checkpoint['best_val_loss']:.4f}")

    # 创建模型
    model = SiameseEncoder(
        backbone_name=config.backbone,
        image_size=config.image_size,
        real_in_channels=config.real_in_channels,
        freq_in_channels=config.freq_in_channels,
        embedding_dim=config.embedding_dim,
        convnext_depths=tuple(config.convnext_depths),
        convnext_dims=tuple(config.convnext_dims),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 加载数据
    test_dataset = MicProjDataset(
        data_dir=args.data_dir,
        split="test",
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 加载所有 proj (用于建索引) 和 pairs (用于获取 ground truth)
    projs = torch.load(Path(args.data_dir) / "projs.pt", weights_only=True)  # [N_proj, D, D]
    pairs = torch.load(Path(args.data_dir) / "pairs.pt", weights_only=True)  # [M, 2]
    transform = PreprocessTransform(normalize=True)

    print(f"Encoding {len(projs)} projs for index...")
    proj_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(projs), args.batch_size)):
            # 取出一个 batch 的原始 proj 图像，逐个 transform 后堆叠
            batch_raw = projs[i:i + args.batch_size]                               # [B, D, D]
            batch = torch.stack([transform(p) for p in batch_raw]).to(device)      # [B, 1, D, D]
            emb = model(batch)                                                     # [B, embedding_dim]
            proj_embeddings.append(emb.cpu())
    proj_embeddings = torch.cat(proj_embeddings, dim=0)  # [N_proj, embedding_dim]

    # 建 FAISS 索引
    index = build_faiss_index(proj_embeddings.numpy())
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 编码所有 test mics，同时记录原始 mic 图像用于可视化
    print(f"Encoding {len(test_dataset)} test mics...")
    mic_embeddings: list[torch.Tensor] = []
    mic_images: list[torch.Tensor] = []
    with torch.no_grad():
        for mic, proj in tqdm(test_loader):
            mic = mic.to(device)                # [B, 1, D, D]
            emb = model(mic)                    # [B, embedding_dim]
            mic_embeddings.append(emb.cpu())
            mic_images.append(mic.cpu())        # 保存带 channel 维度的原始图像
    mic_embeddings = torch.cat(mic_embeddings, dim=0)  # [N_test, embedding_dim]
    mic_images = torch.cat(mic_images, dim=0)           # [N_test, 1, D, D]

    # 从 pairs 获取 ground truth: test_dataset.indices 是全局 pair 索引，
    # pairs[global_idx, 0] 是 proj 索引
    test_indices = test_dataset.indices
    gt_indices = pairs[test_indices, 0].numpy()  # [N_test], 每个 test mic 对应的 proj 索引

    # 检索
    print(f"Retrieving top-{args.k}...")
    distances, indices = retrieve_topk(mic_embeddings, index, k=args.k)

    # 计算指标
    N = len(gt_indices)
    random_baseline = 1.0 / len(projs)
    acc = compute_accuracy_at_k(indices, gt_indices, k_values=[1, 5, 10, 20, 50, 100])

    print(f"\n{'='*50}")
    print(f"Evaluation Results (N_test={N}, N_proj={len(projs)})")
    print(f"Random baseline: {random_baseline:.6f}")
    print(f"{'='*50}")
    for k, v in acc.items():
        ratio = v / random_baseline if random_baseline > 0 else float("inf")
        print(f"  {k}: {v:.4f} ({ratio:.1f}x random)")
    print(f"{'='*50}")

    # 保存数值结果
    results = {
        "accuracies": {k: float(v) for k, v in acc.items()},
        "random_baseline": float(random_baseline),
        "n_test": N,
        "n_proj": len(projs),
        "checkpoint": str(args.checkpoint),
    }
    with open(output_dir / "results.yaml", "w") as f:
        yaml.dump(results, f)

    # 可视化 t-SNE
    print("Generating t-SNE visualization...")
    plot_tsne(
        mic_embeddings,
        proj_embeddings,
        save_path=str(output_dir / "tsne.png"),
    )

    # 可视化检索结果 (前几个样本)
    print(f"Generating retrieval examples...")
    for i in range(min(args.num_visualize, N)):
        plot_retrieval_results(
            query_mic=mic_images[i, 0],                  # [D, D] 去掉 channel 维度
            retrieved_projs=projs[indices[i]],           # [k, D, D] 原始 proj 图像
            ground_truth=projs[gt_indices[i]],           # [D, D] ground truth proj
            similarities=distances[i],                    # [k] 余弦相似度
            k=min(args.k, 10),
            save_path=str(output_dir / f"retrieval_{i}.png"),
        )

    print(f"\nEvaluation results saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()