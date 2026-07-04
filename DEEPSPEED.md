# 多卡 DeepSpeed 训练指南

## 硬件配置

- **推荐**: 4 x RTX 3090 (24GB VRAM each)
- **最低**: 2 x GPU (16GB+ VRAM each)

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/Siamese.git
cd Siamese

# 安装依赖 (包含 DeepSpeed)
pip install -e ".[dev]"

# 验证 DeepSpeed 安装
ds_report
```

### 2. 准备数据

按照 `DEPLOYMENT.md` 解压训练数据包：

```bash
tar -xzf siamese_training_data.tar.gz
# 数据应在 data/ 目录下
```

### 3. 启动 4 卡训练

```bash
# 使用 DeepSpeed 启动器
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json
```

### 4. 监控训练

```bash
# TensorBoard (在另一个终端)
tensorboard --logdir checkpoints_proposer_multi/tensorboard --port 6006 --bind_all

# 访问 http://YOUR_SERVER_IP:6006
```

## DeepSpeed 配置说明

`configs/ds_config.json` 关键参数：

- **train_batch_size**: 64 (全局 batch size，4卡每卡16)
- **train_micro_batch_size_per_gpu**: 16 (每卡 batch size)
- **gradient_accumulation_steps**: 1 (无梯度累积)
- **zero_optimization.stage**: 2 (优化器+梯度分片)
- **fp16.enabled**: true (混合精度训练)
- **gradient_clipping**: 1.0 (梯度裁剪)

### 调整 batch size

如果显存不足 (OOM)：

```json
{
  "train_batch_size": 32,           // 全局: 32
  "train_micro_batch_size_per_gpu": 8,  // 每卡: 8
  "gradient_accumulation_steps": 1
}
```

如果显存充足，想加速：

```json
{
  "train_batch_size": 128,          // 全局: 128
  "train_micro_batch_size_per_gpu": 32, // 每卡: 32
  "gradient_accumulation_steps": 1
}
```

### ZeRO 优化等级

- **Stage 1**: 仅优化器分片 (省显存最少)
- **Stage 2**: 优化器+梯度分片 (推荐，省显存适中)
- **Stage 3**: 优化器+梯度+模型分片 (省显存最多，但通信开销大)

当前使用 Stage 2。如果仍然 OOM，改为 Stage 3：

```json
{
  "zero_optimization": {
    "stage": 3,
    ...
  }
}
```

## 训练性能

### 4 x RTX 3090 预期性能

- **数据集**: 37,710 颗粒
- **Batch size**: 64 (全局)
- **Epochs**: 80
- **预计时间**: ~1.5-2 小时 (相比单卡 ~8-10 小时)
- **加速比**: ~4-5x

### 吞吐量

- 单卡: ~150-200 samples/sec
- 4卡: ~600-800 samples/sec

## Checkpoint 管理

DeepSpeed 保存 checkpoint 格式：

```
checkpoints_proposer_multi/
├── best/
│   ├── mp_rank_00_model_states.pt  (rank 0 模型)
│   ├── mp_rank_01_model_states.pt  (rank 1 模型)
│   ├── mp_rank_02_model_states.pt  (rank 2 模型)
│   ├── mp_rank_03_model_states.pt  (rank 3 模型)
│   └── zero_pp_rank_0_mp_rank_00_optim_states.pt (优化器状态)
```

### 恢复训练

DeepSpeed 自动支持断点续训：

```bash
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json
# DeepSpeed 会自动检测并加载最新的 checkpoint
```

### 转换为单模型文件 (推理用)

```python
import torch
from siamese.models.pose_head import PoseProposer
from siamese.models.encoder import TwoTowerEncoder

# 加载 DeepSpeed checkpoint
checkpoint = torch.load("checkpoints_proposer_multi/best/mp_rank_00_model_states.pt")

# 提取模型权重
encoder = TwoTowerEncoder(backbone_name="convnext_tiny", embedding_dim=128)
proposer = PoseProposer(encoder=encoder, embedding_dim=128)
proposer.load_state_dict(checkpoint['module'])

# 保存为标准 PyTorch 模型
torch.save(proposer.state_dict(), "proposer_best.pt")
```

## 故障排除

### 1. NCCL 错误 (通信失败)

```bash
# 设置 NCCL 调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 2. OOM (显存不足)

- 降低 `train_micro_batch_size_per_gpu` (16 → 8 → 4)
- 启用 ZeRO Stage 3
- 启用 CPU offload (牺牲速度)

### 3. 多卡不均衡 (某卡利用率低)

检查数据加载是否成为瓶颈：

```bash
# 增加 DataLoader workers
num_workers=8  # 在 train_proposer_ds.py 里改
```

### 4. 训练速度慢

- 检查网络带宽 (多卡通信)
- 启用 `overlap_comm: true` (已默认开启)
- 使用 NVLink (如果硬件支持)

## 单卡训练 (向后兼容)

如果只有单卡，使用原始脚本：

```bash
python scripts/train_proposer.py --config configs/proposer_ribosome_multi.yaml
```

## 性能优化建议

1. **数据缓存**: 确保 `cache_samples: true` (配置文件中)
2. **持久化 workers**: PyTorch DataLoader `persistent_workers=True`
3. **混合精度**: 已默认启用 FP16
4. **梯度 checkpoint**: 如果显存不足，可在模型中启用 (牺牲 ~30% 速度)

## 远程训练工作流

```bash
# 1. SSH 到远程服务器
ssh user@remote-server

# 2. 启动 tmux (防止断线)
tmux new -s siamese_train

# 3. 激活环境并启动训练
cd Siamese
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json

# 4. Detach tmux (Ctrl+B, D)

# 5. 重新连接
tmux attach -t siamese_train
```
