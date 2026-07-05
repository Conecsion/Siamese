#!/bin/bash
# 快速诊断训练状态

echo "========================================================================"
echo "训练状态诊断"
echo "========================================================================"
echo ""

# 1. GPU 状态
echo "1. GPU 状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "2. 训练进程:"
ps aux | grep train_proposer_ds | grep -v grep | wc -l
echo "   进程详情:"
ps aux | grep train_proposer_ds | grep -v grep | head -5

echo ""
echo "3. 最新日志时间:"
ls -lt logs/train_*.log | head -1

echo ""
echo "4. 日志最后内容 (最后 30 行):"
ls -t logs/train_*.log | head -1 | xargs tail -30

echo ""
echo "5. 检查是否有 Epoch 输出:"
ls -t logs/train_*.log | head -1 | xargs grep -i "epoch" | tail -5

echo ""
echo "6. 检查是否有错误:"
ls -t logs/train_*.log | head -1 | xargs grep -i "error\|exception\|traceback" | tail -10

echo ""
echo "========================================================================"
