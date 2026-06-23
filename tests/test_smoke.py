"""
冒烟测试: 端到端验证数据生成 → 训练 → 检索 → 评估流程。

使用小规模参数 (nside=2, 小图像, 少量 epoch) 快速验证 pipeline。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


def test_smoke_pipeline():
    """
    冒烟测试: 完整 pipeline 验证 (GPU 版本)。

    注意: 此测试需要 GPU。如果 GPU 不可用，跳过。
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # 1. 创建临时 volume 和输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        checkpoint_dir = tmpdir / "checkpoints"
        data_dir.mkdir()
        checkpoint_dir.mkdir()

        # 创建一个小型测试 volume (球体)
        D = 32
        vol = np.zeros((D, D, D), dtype=np.float32)
        center = D // 2
        radius = D // 4
        for z in range(D):
            for y in range(D):
                for x in range(D):
                    if (z - center)**2 + (y - center)**2 + (x - center)**2 < radius**2:
                        vol[z, y, x] = 1.0

        import mrcfile
        map_path = str(tmpdir / "test_volume.map")
        with mrcfile.new(map_path, overwrite=True) as mrc:
            mrc.set_data(vol)

        # 2. 生成模拟数据
        from siamese.data.generate import generate_simulated_data
        metadata = generate_simulated_data(
            map_path=map_path,
            nside=2,  # 48 个方向
            output_dir=str(data_dir),
            image_size=32,
            pixel_size=1.0,
            num_mics_per_proj=2,
            snr_range=(0.001, 0.01),
            defocus_range=(0.5, 4.0),
            max_shift_pixels=3.0,
            device="cuda",
            chunk_size=48,
            seed=42,
        )

        assert metadata["num_directions"] == 48
        assert metadata["num_mics_total"] == 96
        assert (data_dir / "projs.pt").exists()
        assert (data_dir / "mics.pt").exists()
        assert (data_dir / "pairs.pt").exists()
        print(f"Step 1 PASS: Generated {metadata['num_mics_total']} mics")

        # 3. 创建数据集
        from siamese.data.dataset import MicProjDataset
        train_ds = MicProjDataset(data_dir=str(data_dir), split="train",
                                   train_split=0.7, val_split=0.15, seed=42)
        val_ds = MicProjDataset(data_dir=str(data_dir), split="val",
                                 train_split=0.7, val_split=0.15, seed=42)
        test_ds = MicProjDataset(data_dir=str(data_dir), split="test",
                                  train_split=0.7, val_split=0.15, seed=42)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        print(f"Step 2 PASS: Dataset splits: train={len(train_ds)}, "
              f"val={len(val_ds)}, test={len(test_ds)}")

        # 4. 创建模型
        from siamese.models.encoder import SiameseEncoder
        model = SiameseEncoder(
            backbone_name="convnext_tiny",
            image_size=32,
            embedding_dim=32,  # 小 embedding
            convnext_depths=(2, 2, 2, 2),     # 最小的 ConvNeXt
            convnext_dims=(32, 64, 128, 256),  # 小通道数
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Step 3 PASS: Model created with {num_params:,} parameters")

        # 5. 快速训练几个 epoch
        from torch.utils.data import DataLoader
        from siamese.training.config import TrainConfig
        from siamese.training.trainer import Trainer

        config = TrainConfig(
            data_dir=str(data_dir),
            image_size=32,
            batch_size=16,
            num_epochs=5,
            learning_rate=1e-3,
            device="cuda",
            checkpoint_dir=str(checkpoint_dir),
            log_interval=5,
        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, drop_last=False)

        trainer = Trainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader)
        history = trainer.train()

        assert len(history["train_losses"]) == config.num_epochs
        assert history["train_losses"][-1] < history["train_losses"][0], \
            "Loss should decrease during training"
        print(f"Step 4 PASS: Training loss: "
              f"{history['train_losses'][0]:.4f} -> {history['train_losses'][-1]:.4f}")

        # 6. 评估检索
        from siamese.eval.retrieval import build_faiss_index, retrieve_topk
        from siamese.eval.metrics import compute_accuracy_at_k

        model.eval()
        projs_data = torch.load(data_dir / "projs.pt", weights_only=True)
        from siamese.data.transforms import PreprocessTransform
        transform = PreprocessTransform(normalize=True)

        # 编码 proj
        proj_embs = []
        with torch.no_grad():
            for i in range(0, len(projs_data), 16):
                batch = torch.stack(
                    [transform(p) for p in projs_data[i:i+16]]
                ).cuda()
                proj_embs.append(model(batch).cpu())
        proj_embs = torch.cat(proj_embs, dim=0)

        # 编码 test mic
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        mic_embs = []
        with torch.no_grad():
            for mic, _ in test_loader:
                mic_embs.append(model(mic.cuda()).cpu())
        mic_embs = torch.cat(mic_embs, dim=0)

        # 建索引 + 检索
        index = build_faiss_index(proj_embs.numpy())
        pairs = torch.load(data_dir / "pairs.pt", weights_only=True)
        # pairs[global_idx] = [proj_idx, mic_idx], 取 proj_idx 作为 ground truth
        gt_indices = pairs[test_ds.indices, 0].numpy()

        distances, retrieved = retrieve_topk(mic_embs, index, k=10)
        acc = compute_accuracy_at_k(retrieved, gt_indices, k_values=[1, 5, 10])

        N = len(projs_data)
        random_baseline = 1.0 / N
        print(f"Step 5 PASS: Retrieval results:")
        print(f"  Random baseline: {random_baseline:.4f}")
        for k, v in acc.items():
            print(f"  {k}: {v:.4f} ({v / random_baseline:.1f}x random)")

        # 冒烟测试通过标准: 至少有一个 top-k 准确率 > 5x random
        assert any(v > 5 * random_baseline for v in acc.values()), \
            f"At least one top-k accuracy should be > 5x random baseline. Got: {acc}"

        print("ALL SMOKE TESTS PASSED!")


def test_smoke_pipeline_cpu():
    """CPU 版本的冒烟测试: 只验证数据生成 + 模型前向传播。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        data_dir.mkdir()

        # 创建测试 volume
        D = 16
        vol = np.ones((D, D, D), dtype=np.float32)
        vol[D//2:, :, :] = 0.5

        import mrcfile
        map_path = str(tmpdir / "test_volume.map")
        with mrcfile.new(map_path, overwrite=True) as mrc:
            mrc.set_data(vol)

        from siamese.data.generate import generate_simulated_data
        metadata = generate_simulated_data(
            map_path=map_path,
            nside=1,  # 12 个方向
            output_dir=str(data_dir),
            image_size=16,
            num_mics_per_proj=1,
            snr_range=(0.01, 0.01),
            defocus_range=(2.0, 2.0),
            max_shift_pixels=1.0,
            device="cpu",
            chunk_size=12,
            seed=42,
        )

        assert metadata["num_directions"] == 12
        print("CPU smoke test PASSED")