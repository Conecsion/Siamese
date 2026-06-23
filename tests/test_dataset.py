"""MicProjDataset 测试。"""

import torch
import tempfile
import pytest
from pathlib import Path
from siamese.data.dataset import MicProjDataset


@pytest.fixture
def temp_data_dir():
    """创建临时数据目录用于测试。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        D = 32
        N = 20
        M = 40  # 2 mics per proj

        projs = torch.randn(N, D, D)
        mics = torch.randn(M, D, D)
        axisang = torch.randn(N, 3)
        # pairs: [proj_idx, mic_idx]
        pairs = torch.zeros(M, 2, dtype=torch.long)
        for i in range(N):
            for j in range(2):
                pairs[i * 2 + j, 0] = i
                pairs[i * 2 + j, 1] = i * 2 + j

        torch.save(projs, tmpdir / "projs.pt")
        torch.save(mics, tmpdir / "mics.pt")
        torch.save(axisang, tmpdir / "axisang.pt")
        torch.save(pairs, tmpdir / "pairs.pt")
        yield tmpdir


def test_dataset_len(temp_data_dir):
    """测试数据集长度。"""
    dataset = MicProjDataset(data_dir=str(temp_data_dir), split="train", train_split=0.7, val_split=0.15)
    # 40 mics, 0.7 train = 28
    assert len(dataset) == 28, f"Expected 28, got {len(dataset)}"


def test_dataset_getitem_shape(temp_data_dir):
    """测试 __getitem__ 返回形状。"""
    dataset = MicProjDataset(data_dir=str(temp_data_dir), split="train", train_split=0.7, val_split=0.15)
    mic, proj = dataset[0]
    # mic: [1, D, D], proj: [1, D, D]
    assert mic.shape == (1, 32, 32), f"Expected (1, 32, 32), got {mic.shape}"
    assert proj.shape == (1, 32, 32), f"Expected (1, 32, 32), got {proj.shape}"


def test_dataset_splits_disjoint(temp_data_dir):
    """测试 train/val/test 不重叠。"""
    train_ds = MicProjDataset(data_dir=str(temp_data_dir), split="train", train_split=0.7, val_split=0.15)
    val_ds = MicProjDataset(data_dir=str(temp_data_dir), split="val", train_split=0.7, val_split=0.15)
    test_ds = MicProjDataset(data_dir=str(temp_data_dir), split="test", train_split=0.7, val_split=0.15)

    assert len(train_ds) + len(val_ds) + len(test_ds) == 40, "Splits should cover all samples"
    assert len(train_ds) == 28, f"Expected 28 train, got {len(train_ds)}"
    assert len(val_ds) == 6, f"Expected 6 val, got {len(val_ds)}"
    assert len(test_ds) == 6, f"Expected 6 test, got {len(test_ds)}"