#!/bin/bash
# tmux 一键训练脚本 - 自动创建多窗格监控环境
# 使用方法: bash train_tmux.sh

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SESSION_NAME="siamese_train"

# 检查 tmux 是否安装
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux 未安装！请先安装: sudo apt install tmux"
    exit 1
fi

# 检查会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "会话 '$SESSION_NAME' 已存在"
    echo "选项:"
    echo "  1. 连接到现有会话: tmux attach -t $SESSION_NAME"
    echo "  2. 杀死旧会话并重新创建: tmux kill-session -t $SESSION_NAME && bash $0"
    exit 1
fi

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          创建 tmux 训练环境 - 多窗格监控                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 创建 tmux 会话并设置多窗格
tmux new-session -d -s $SESSION_NAME -n main

# 窗格 0: 训练主进程
tmux send-keys -t $SESSION_NAME:main.0 "cd /data/shaodi/Siamese" C-m
tmux send-keys -t $SESSION_NAME:main.0 "export PATH=\$HOME/.local/bin:\$PATH" C-m
tmux send-keys -t $SESSION_NAME:main.0 "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME:main.0 "clear" C-m
tmux send-keys -t $SESSION_NAME:main.0 "bash train_all.sh" C-m

# 分割窗格: GPU 监控（上下分屏）
tmux split-window -t $SESSION_NAME:main -v -l 30%
tmux send-keys -t $SESSION_NAME:main.1 "watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | column -t -s,'" C-m

# 分割窗格: 训练日志监控（右侧）
tmux split-window -t $SESSION_NAME:main.0 -h -l 50%
tmux send-keys -t $SESSION_NAME:main.2 "cd /data/shaodi/Siamese" C-m
tmux send-keys -t $SESSION_NAME:main.2 "sleep 5 && tail -f logs/train_*.log 2>/dev/null || echo '等待训练日志...' && sleep 999999" C-m

# 设置窗格布局
tmux select-layout -t $SESSION_NAME:main tiled

# 聚焦到训练窗格
tmux select-pane -t $SESSION_NAME:main.0

echo ""
echo -e "${GREEN}✅ tmux 会话已创建！${NC}"
echo ""
echo "连接到会话:"
echo -e "  ${BLUE}tmux attach -t $SESSION_NAME${NC}"
echo ""
echo "窗格布局:"
echo "  ┌─────────────────┬─────────────────┐"
echo "  │                 │                 │"
echo "  │   训练主进程    │   训练日志      │"
echo "  │                 │                 │"
echo "  ├─────────────────┴─────────────────┤"
echo "  │          GPU 监控 (nvidia-smi)    │"
echo "  └───────────────────────────────────┘"
echo ""
echo "快捷键:"
echo "  Ctrl+B + 方向键  - 切换窗格"
echo "  Ctrl+B + D       - 分离会话（训练继续运行）"
echo "  Ctrl+B + [       - 滚动查看历史（q 退出）"
echo "  Ctrl+C           - 停止训练"
echo ""
echo "重新连接:"
echo -e "  ${BLUE}ssh shaodi@106${NC}"
echo -e "  ${BLUE}tmux attach -t $SESSION_NAME${NC}"
echo ""
echo "TensorBoard:"
echo -e "  ${BLUE}http://106:6006${NC}"
echo ""
