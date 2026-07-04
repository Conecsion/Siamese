# 远程训练快速指南

## 🚀 一键启动训练

在**本地**执行：

```bash
cd /Data/Work/Siamese
bash remote_train.sh start
```

## 📊 管理命令

| 命令 | 说明 |
|------|------|
| `bash remote_train.sh start` | 启动训练 |
| `bash remote_train.sh stop` | 停止训练 |
| `bash remote_train.sh restart` | 重启训练 |
| `bash remote_train.sh status` | 查看状态 |
| `bash remote_train.sh logs` | 实时日志 |
| `bash remote_train.sh gpu` | GPU 监控 |
| `bash remote_train.sh attach` | 连接 tmux |
| `bash remote_train.sh pull` | 拉取代码 |

## 📺 监控训练

### 方式 1: 本地查看状态
```bash
bash remote_train.sh status
```

### 方式 2: 实时日志
```bash
bash remote_train.sh logs
```

### 方式 3: TensorBoard
浏览器打开: http://106:6006

### 方式 4: 连接 tmux（推荐）
```bash
bash remote_train.sh attach
# 按 Ctrl+B 然后 D 分离会话
```

## 🔄 开发工作流

### 1. 本地修改代码
```bash
cd /Data/Work/Siamese
# 编辑文件...
git add <files>
git commit -m "..."
git push origin main
```

### 2. 重启远程训练
```bash
bash remote_train.sh restart
```

## 📁 文件结构

远程服务器 (`shaodi@106:/data/shaodi/Siamese/`):
```
├── logs/                          # 训练日志
│   ├── train_YYYYMMDD_HHMMSS.log # 训练日志
│   └── tensorboard_*.log          # TensorBoard 日志
├── checkpoints_proposer_multi/    # 训练 checkpoint
│   ├── best/                      # 最佳模型
│   └── tensorboard/               # TensorBoard 数据
├── train_all.sh                   # 前台训练脚本
├── train_tmux.sh                  # tmux 训练脚本
└── quick_start.sh                 # 快速启动脚本
```

## 💾 下载模型

训练完成后，下载最佳模型：

```bash
scp -r shaodi@106:/data/shaodi/Siamese/checkpoints_proposer_multi/best ./model_best
```

## 🐛 故障排除

### 训练意外停止
```bash
# 查看错误日志
bash remote_train.sh logs | grep -i error

# 查看完整日志
ssh shaodi@106 "cat /data/shaodi/Siamese/logs/train_*.log"
```

### GPU 不可用
```bash
bash remote_train.sh gpu
```

### 重置环境
```bash
ssh shaodi@106
cd /data/shaodi/Siamese
rm -rf checkpoints_proposer_multi/
bash remote_train.sh restart
```

## ⏱️ 训练时间

- **数据集**: 37,710 颗粒
- **配置**: 4 x RTX 3090, Batch 64
- **预计时间**: 1.5-2 小时 (80 epoch)
- **Checkpoint**: 每 5 epoch 评估

## 🎯 训练完成后

模型保存在: `checkpoints_proposer_multi/best/`

包含 4 个文件（DeepSpeed 多卡格式）:
- `mp_rank_00_model_states.pt`
- `mp_rank_01_model_states.pt`
- `mp_rank_02_model_states.pt`
- `mp_rank_03_model_states.pt`

## 📚 详细文档

- [TMUX_GUIDE.md](TMUX_GUIDE.md) - tmux 使用指南
- [DEEPSPEED.md](docs/DEEPSPEED.md) - DeepSpeed 配置
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - 完整部署指南
- [CLAUDE.md](CLAUDE.md) - 项目架构

## 💡 提示

- 训练在 tmux 中运行，SSH 断开不影响训练
- TensorBoard 自动启动，端口 6006
- 日志自动保存，带时间戳
- 最佳模型自动保存到 `best/` 目录
