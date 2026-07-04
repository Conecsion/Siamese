#!/bin/bash
# 远程训练管理脚本 - 在本地执行，控制远程训练
# 使用方法: bash remote_train.sh [start|stop|status|logs|restart]

set -e

# 配置
REMOTE_USER="shaodi"
REMOTE_HOST="106"
REMOTE_DIR="/data/shaodi/Siamese"
SESSION_NAME="siamese_train"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 帮助信息
function show_help() {
    echo "远程训练管理脚本"
    echo ""
    echo "用法: bash remote_train.sh <command>"
    echo ""
    echo "命令:"
    echo "  start    - 启动训练 (tmux 后台)"
    echo "  stop     - 停止训练"
    echo "  restart  - 重启训练"
    echo "  status   - 查看训练状态"
    echo "  logs     - 查看最新日志"
    echo "  gpu      - 查看 GPU 使用"
    echo "  attach   - 连接到 tmux 会话"
    echo "  pull     - 拉取最新代码"
    echo ""
    echo "示例:"
    echo "  bash remote_train.sh start    # 启动训练"
    echo "  bash remote_train.sh logs     # 查看日志"
    echo "  bash remote_train.sh stop     # 停止训练"
}

# 启动训练
function start_training() {
    echo -e "${BLUE}[1/3]${NC} 拉取最新代码..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && git pull"

    echo -e "${BLUE}[2/3]${NC} 检查现有会话..."
    if ssh ${REMOTE_USER}@${REMOTE_HOST} "tmux has-session -t ${SESSION_NAME} 2>/dev/null"; then
        echo -e "${YELLOW}警告: 训练会话已存在${NC}"
        echo "选项: 1) 连接 2) 重启 3) 取消"
        read -p "选择 [1/2/3]: " choice
        case $choice in
            1) attach_session; return;;
            2) restart_training; return;;
            3) echo "已取消"; return;;
            *) echo "无效选择"; return;;
        esac
    fi

    echo -e "${BLUE}[3/3]${NC} 启动 tmux 训练环境..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && bash train_tmux.sh"

    echo ""
    echo -e "${GREEN}✅ 训练已启动！${NC}"
    echo ""
    echo "查看状态: bash $0 status"
    echo "查看日志: bash $0 logs"
    echo "连接会话: bash $0 attach"
    echo "TensorBoard: http://${REMOTE_HOST}:6006"
}

# 停止训练
function stop_training() {
    echo -e "${YELLOW}停止训练...${NC}"

    # 检查会话是否存在
    if ! ssh ${REMOTE_USER}@${REMOTE_HOST} "tmux has-session -t ${SESSION_NAME} 2>/dev/null"; then
        echo -e "${RED}错误: 训练会话不存在${NC}"
        return 1
    fi

    # 杀死 tmux 会话
    ssh ${REMOTE_USER}@${REMOTE_HOST} "tmux kill-session -t ${SESSION_NAME} 2>/dev/null"

    # 清理进程
    ssh ${REMOTE_USER}@${REMOTE_HOST} "pkill -9 -f train_proposer_ds 2>/dev/null || true"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "pkill -9 -f tensorboard 2>/dev/null || true"

    echo -e "${GREEN}✅ 训练已停止${NC}"
}

# 重启训练
function restart_training() {
    echo -e "${YELLOW}重启训练...${NC}"
    stop_training
    sleep 2
    start_training
}

# 查看状态
function show_status() {
    echo -e "${BLUE}=== 训练状态 ===${NC}"

    # tmux 会话
    echo ""
    echo "tmux 会话:"
    if ssh ${REMOTE_USER}@${REMOTE_HOST} "tmux has-session -t ${SESSION_NAME} 2>/dev/null"; then
        echo -e "  ${GREEN}✓${NC} ${SESSION_NAME} (运行中)"
    else
        echo -e "  ${RED}✗${NC} ${SESSION_NAME} (未运行)"
    fi

    # 训练进程
    echo ""
    echo "训练进程:"
    TRAIN_COUNT=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "ps aux | grep train_proposer_ds | grep -v grep | wc -l")
    if [ "$TRAIN_COUNT" -gt "0" ]; then
        echo -e "  ${GREEN}✓${NC} $TRAIN_COUNT 个进程运行中"
    else
        echo -e "  ${RED}✗${NC} 无训练进程"
    fi

    # TensorBoard
    echo ""
    echo "TensorBoard:"
    TB_COUNT=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "ps aux | grep tensorboard | grep -v grep | wc -l")
    if [ "$TB_COUNT" -gt "0" ]; then
        echo -e "  ${GREEN}✓${NC} TensorBoard 运行中 (http://${REMOTE_HOST}:6006)"
    else
        echo -e "  ${RED}✗${NC} TensorBoard 未运行"
    fi

    # GPU
    echo ""
    echo "GPU 状态:"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf \"  GPU %s: %s%% | %s/%s MB\n\", \$1, \$2, \$3, \$4}'"

    # 最新日志
    echo ""
    echo "最新日志 (最后 10 行):"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && tail -10 logs/train_*.log 2>/dev/null | tail -10" || echo "  无日志文件"
}

# 查看日志
function show_logs() {
    echo -e "${BLUE}=== 实时日志 (Ctrl+C 退出) ===${NC}"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && tail -f logs/train_*.log 2>/dev/null"
}

# 查看 GPU
function show_gpu() {
    echo -e "${BLUE}=== GPU 监控 (Ctrl+C 退出) ===${NC}"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "watch -n 2 nvidia-smi"
}

# 连接会话
function attach_session() {
    echo -e "${BLUE}连接到 tmux 会话...${NC}"
    echo "提示: Ctrl+B 然后按 D 可以分离会话"
    echo ""
    ssh -t ${REMOTE_USER}@${REMOTE_HOST} "tmux attach -t ${SESSION_NAME}"
}

# 拉取代码
function pull_code() {
    echo -e "${BLUE}拉取最新代码...${NC}"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && git pull"
    echo -e "${GREEN}✅ 代码已更新${NC}"
}

# 主逻辑
case "${1:-help}" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    restart)
        restart_training
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    gpu)
        show_gpu
        ;;
    attach)
        attach_session
        ;;
    pull)
        pull_code
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${RED}错误: 未知命令 '$1'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
