#!/bin/bash
# 快速启动脚本 - 最简单的启动方式
# 使用方法: bash quick_start.sh

cd /data/shaodi/Siamese
export PATH=$HOME/.local/bin:$PATH
source .venv/bin/activate

# 清理旧进程
pkill -9 -f train_proposer_ds 2>/dev/null
pkill -9 -f tensorboard 2>/dev/null

# 创建日志目录
mkdir -p logs

# 启动 TensorBoard
echo "启动 TensorBoard (http://106:6006)..."
nohup tensorboard --logdir checkpoints_proposer_multi/tensorboard --port 6006 --bind_all > logs/tensorboard.log 2>&1 &

# 启动训练
echo "启动训练..."
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed_config configs/ds_config.json \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
