#!/bin/bash
# 远程训练启动脚本 - 使用 tmux + DeepSpeed

set -e

cd /data/shaodi/Siamese
export PATH=$HOME/.local/bin:$PATH

# 激活虚拟环境
source .venv/bin/activate

# 验证环境
echo "=== 环境验证 ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# 清理旧进程
pkill -9 -f train_proposer_ds 2>/dev/null || true

# 启动 DeepSpeed 训练
echo ""
echo "=== 启动 4 卡训练 ==="
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed_config configs/ds_config.json \
    2>&1 | tee train.log

echo ""
echo "训练完成或中断"
