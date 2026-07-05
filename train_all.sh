#!/bin/bash
# 一键训练脚本 - 自动启动 DeepSpeed 训练 + TensorBoard
# 使用方法: bash train_all.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Siamese Cryo-EM 训练启动器 - 4 x RTX 3090                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 进入项目目录
cd /data/shaodi/Siamese
export PATH=$HOME/.local/bin:$PATH

# 激活虚拟环境
echo -e "${YELLOW}[1/6]${NC} 激活虚拟环境..."
source .venv/bin/activate

# 验证环境
echo -e "${YELLOW}[2/6]${NC} 验证环境..."
echo "  Python: $(python --version)"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  PyTorch: $TORCH_VERSION"
echo "  CUDA: $CUDA_AVAILABLE"
echo "  GPUs: $GPU_COUNT"

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo -e "${RED}错误: CUDA 不可用！${NC}"
    exit 1
fi

if [ "$GPU_COUNT" != "4" ]; then
    echo -e "${YELLOW}警告: 检测到 $GPU_COUNT 个 GPU（预期 4 个）${NC}"
fi

# 清理旧进程
echo -e "${YELLOW}[3/6]${NC} 清理旧进程..."
pkill -9 -f train_proposer_ds 2>/dev/null && echo "  已清理旧训练进程" || echo "  无旧进程"
pkill -9 -f tensorboard 2>/dev/null && echo "  已清理旧 TensorBoard" || echo "  无旧 TensorBoard"

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}.log"
TB_LOG="$LOG_DIR/tensorboard_${TIMESTAMP}.log"

# 启动 TensorBoard
echo -e "${YELLOW}[4/6]${NC} 启动 TensorBoard..."
TB_PORT=6006
nohup tensorboard \
    --logdir checkpoints_proposer_multi/tensorboard \
    --port $TB_PORT \
    --bind_all \
    --reload_interval 30 \
    > $TB_LOG 2>&1 &
TB_PID=$!
echo "  TensorBoard PID: $TB_PID"
echo "  访问地址: http://106:$TB_PORT"
echo "  日志: $TB_LOG"

# 等待 TensorBoard 启动
sleep 3

# 启动训练
echo -e "${YELLOW}[5/6]${NC} 启动 DeepSpeed 训练..."
echo "  配置: configs/proposer_ribosome_multi.yaml"
echo "  DeepSpeed: configs/ds_config.json"
echo "  日志: $TRAIN_LOG"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}训练已启动！按 Ctrl+C 可以安全中断训练（模型会自动保存）${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 启动训练并实时显示日志
# 注意：GPU 0 被显示服务器占用，只使用 GPU 1-3
CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --num_gpus=3 \
    scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed_config configs/ds_config.json \
    2>&1 | tee $TRAIN_LOG

# 训练结束
echo ""
echo -e "${YELLOW}[6/6]${NC} 训练完成或中断"
echo "  训练日志: $TRAIN_LOG"
echo "  TensorBoard: 仍在运行（PID: $TB_PID）"
echo ""
echo -e "${BLUE}停止 TensorBoard: kill $TB_PID${NC}"
echo -e "${BLUE}查看训练日志: tail -f $TRAIN_LOG${NC}"
echo -e "${BLUE}查看最佳模型: ls -lh checkpoints_proposer_multi/best/${NC}"
echo ""
