#!/bin/bash
# NaN 实时监控脚本
# 监控训练日志，检测到 NaN 时自动触发诊断

REMOTE_USER="shaodi"
REMOTE_HOST="106"
REMOTE_DIR="/data/shaodi/Siamese"

echo "🔍 启动 NaN 实时监控..."
echo "监控远程服务器: ${REMOTE_USER}@${REMOTE_HOST}"
echo ""

# 获取最新的训练日志
LOG_FILE=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "ls -t ${REMOTE_DIR}/logs/train_*.log | head -1")

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "📄 监控日志: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo ""

# 实时监控日志
ssh ${REMOTE_USER}@${REMOTE_HOST} "tail -f ${LOG_FILE}" | while read line; do
    # 检测 NaN
    if echo "$line" | grep -iq "nan\|inf"; then
        echo ""
        echo "🚨🚨🚨 检测到 NaN/Inf！ 🚨🚨🚨"
        echo "时间: $(date)"
        echo "内容: $line"
        echo ""

        # 触发诊断
        echo "正在运行诊断工具..."
        ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR} && python3 scripts/debug_nan.py --log ${LOG_FILE}"

        # 保存快照
        SNAPSHOT_DIR="nan_snapshots/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$SNAPSHOT_DIR"

        echo ""
        echo "💾 保存诊断快照到: $SNAPSHOT_DIR"

        # 下载日志
        scp ${REMOTE_USER}@${REMOTE_HOST}:${LOG_FILE} "$SNAPSHOT_DIR/"

        # 下载配置
        scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/configs/proposer_ribosome_multi.yaml "$SNAPSHOT_DIR/"
        scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/configs/ds_config.json "$SNAPSHOT_DIR/"

        # 获取 GPU 状态
        ssh ${REMOTE_USER}@${REMOTE_HOST} "nvidia-smi" > "$SNAPSHOT_DIR/gpu_status.txt"

        # 获取进程状态
        ssh ${REMOTE_USER}@${REMOTE_HOST} "ps aux | grep train_proposer" > "$SNAPSHOT_DIR/process_status.txt"

        echo "✅ 快照已保存"
        echo ""
        echo "📋 建议的修复步骤："
        echo "  1. 检查学习率是否过大"
        echo "  2. 启用梯度裁剪"
        echo "  3. 调整 FP16 loss_scale"
        echo "  4. 检查输入数据"
        echo ""

        # 询问是否停止训练
        echo "是否停止训练并应用修复？(y/n)"
        read -t 30 -n 1 response
        if [ "$response" = "y" ]; then
            echo ""
            echo "停止训练..."
            ssh ${REMOTE_USER}@${REMOTE_HOST} "tmux kill-session -t siamese_train"
            echo "训练已停止。请应用修复后重新启动。"
            exit 0
        fi
    fi

    # 显示包含 loss/epoch 的行
    if echo "$line" | grep -iq "loss\|epoch\|recall"; then
        echo "$line"
    fi
done
