# 远程服务器 tmux 训练指南

## 快速启动

### 1. SSH 连接到远程服务器
```bash
ssh shaodi@106
cd /data/shaodi/Siamese
```

### 2. 启动 tmux 会话
```bash
# 创建新会话
tmux new -s siamese_train

# 或者恢复已有会话
tmux attach -t siamese_train
```

### 3. 启动训练
```bash
bash start_training.sh
```

### 4. 分离 tmux（保持训练运行）
按键：`Ctrl+B` 然后按 `D`

### 5. 重新连接查看进度
```bash
ssh shaodi@106
tmux attach -t siamese_train
```

## tmux 常用命令

| 操作 | 命令 |
|------|------|
| 创建会话 | `tmux new -s <name>` |
| 列出会话 | `tmux ls` |
| 连接会话 | `tmux attach -t <name>` |
| 分离会话 | `Ctrl+B` + `D` |
| 杀死会话 | `tmux kill-session -t <name>` |
| 滚动查看 | `Ctrl+B` + `[` (方向键滚动，`q` 退出) |

## 多窗口监控

在 tmux 中分屏监控：

```bash
# 1. 启动训练（窗口1）
bash start_training.sh

# 2. 水平分屏（Ctrl+B + "）
Ctrl+B, "

# 3. 切换到下方窗格（Ctrl+B + ↓）
Ctrl+B, ↓

# 4. 监控 GPU（窗口2）
watch -n 1 nvidia-smi

# 5. 再分屏监控日志（Ctrl+B + "）
Ctrl+B, "
tail -f train.log
```

## 训练状态检查

### 检查进程
```bash
ps aux | grep train_proposer_ds
```

### 查看最新日志
```bash
tail -100 train.log | grep -E '(epoch|loss|Recall)'
```

### 查看 GPU 使用
```bash
nvidia-smi
```

### 查看 TensorBoard
```bash
# 在 tmux 新窗口启动（Ctrl+B + C）
source .venv/bin/activate
tensorboard --logdir checkpoints_proposer_multi/tensorboard --port 6006 --bind_all

# 访问: http://106:6006
```

## 故障排除

### 训练意外停止
```bash
# 查看最后的错误
tail -200 train.log | grep -A 10 -B 5 -E '(Error|ERROR|Traceback)'
```

### 重启训练
```bash
# 在 tmux 会话中
bash start_training.sh
```

### 清理后重启
```bash
pkill -9 -f train_proposer_ds
rm -rf checkpoints_proposer_multi/
bash start_training.sh
```

## 本地 → 远程工作流

### 1. 本地编辑代码
```bash
cd /Data/Work/Siamese
# 编辑文件...
git add <files>
git commit -m "..."
git push origin main
```

### 2. 远程拉取更新
```bash
# 在 tmux 会话外执行
ssh shaodi@106 "cd /data/shaodi/Siamese && git pull"
```

### 3. 重启训练（如需要）
```bash
# 连接到 tmux 会话
tmux attach -t siamese_train

# Ctrl+C 停止当前训练
# 重新启动
bash start_training.sh
```

## 预期训练时间

- **数据集**: 37,710 颗粒
- **Batch size**: 64 (全局)
- **Epochs**: 80
- **预计时间**: 1.5-2 小时（4 x RTX 3090）
- **Checkpoint**: 每 5 epoch 评估，最佳模型自动保存

## 完成后

训练完成后，最佳模型保存在：
```
checkpoints_proposer_multi/best/
├── mp_rank_00_model_states.pt
├── mp_rank_01_model_states.pt
├── mp_rank_02_model_states.pt
└── mp_rank_03_model_states.pt
```

下载到本地：
```bash
scp -r shaodi@106:/data/shaodi/Siamese/checkpoints_proposer_multi/best ./
```
