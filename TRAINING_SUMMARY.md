# Siamese 训练部署完成总结

## ✅ 当前状态

### 训练正在运行
- **状态**: ✅ 运行中
- **进程数**: 6 个（DeepSpeed launcher + 4 GPU workers + 主进程）
- **GPU 利用率**: GPU 1/2/3 = 100%, GPU 0 使用中
- **显存使用**: 4-7 GB / 24 GB per GPU
- **TensorBoard**: http://106:6006
- **tmux 会话**: siamese_train

### 训练配置
- **模型**: PoseProposer with TwoTowerEncoder
- **参数量**: 112,208,901
- **数据集**: 37,710 训练样本 + 4,714 验证样本
  - J8_3.00A: 16,108 train / 2,014 val
  - J20_3.68A: 16,416 train / 2,052 val
  - J31_4.16A: 5,186 train / 648 val
- **Gallery**: 9,216 个方向（HEALPix n_side=16）
- **硬件**: 4 x RTX 3090 (24GB each)
- **训练时长**: 预计 1.5-2 小时 (80 epoch)

---

## 🔧 已解决的问题

### 1. 代码错误修复
- ✅ TwoTowerEncoder 初始化参数错误 → 需要两个独立的 SiameseEncoder
- ✅ PoseProposer 初始化参数错误 → 不需要 embedding_dim
- ✅ set_gallery() 缺少 gallery_aa 参数 → 添加第二个参数
- ✅ project_volume 导入错误 → 使用 project_fourier_slice_from_axis_angle
- ✅ CUDA 多进程初始化错误 → 设置 multiprocessing.set_start_method('spawn')
- ✅ FP16 类型不匹配 → 将输入转换为模型 dtype
- ✅ 通道维度不匹配 → 添加 unsqueeze(1) 通道维度

### 2. 数据路径问题
- ✅ `.cs` 文件路径不正确 → 创建 fix_cs_paths.py 脚本修复
- ✅ project_dir 配置导致路径重复 → 设置为 "."
- ✅ 路径截断问题 → 修复 NumPy dtype 以支持完整路径长度

### 3. 环境配置
- ✅ 远程服务器 Python 环境 → uv + Python 3.11 + PyTorch 2.4.1
- ✅ DeepSpeed 安装 → 支持 4 卡分布式训练
- ✅ 数据传输 → 打包并上传 47.1 GB 训练数据

---

## 📁 项目结构

```
/Data/Work/Siamese/                    (本地)
/data/shaodi/Siamese/                  (远程)
├── scripts/
│   ├── train_proposer_ds.py          ✅ DeepSpeed 训练脚本
│   └── fix_cs_paths.py                ✅ .cs 文件路径修复脚本
├── configs/
│   ├── proposer_ribosome_multi.yaml   ✅ 训练配置
│   └── ds_config.json                 ✅ DeepSpeed 配置
├── train_all.sh                       ✅ 前台训练启动脚本
├── train_tmux.sh                      ✅ tmux 训练启动脚本
├── quick_start.sh                     ✅ 快速启动脚本
├── remote_train.sh                    ✅ 远程管理脚本（本地执行）
├── logs/                              ✅ 训练日志目录
├── checkpoints_proposer_multi/        ✅ 模型 checkpoint
│   ├── best/                          → 最佳模型
│   └── tensorboard/                   → TensorBoard 数据
└── data/
    ├── cs_processed/                  ✅ CryoSPARC 导出数据
    └── particles/                     ✅ 颗粒图像数据
```

---

## 🎯 使用指南

### 本地管理远程训练

```bash
# 在本地执行这些命令
cd /Data/Work/Siamese

# 查看训练状态
bash remote_train.sh status

# 实时查看日志
bash remote_train.sh logs

# 查看 GPU 使用
bash remote_train.sh gpu

# 连接到 tmux 会话
bash remote_train.sh attach

# 停止训练
bash remote_train.sh stop

# 重启训练
bash remote_train.sh restart
```

### 远程操作

```bash
# SSH 连接
ssh shaodi@106

# 连接到 tmux
tmux attach -t siamese_train

# 分离会话（训练继续）
Ctrl+B 然后 D

# 查看日志
tail -f /data/shaodi/Siamese/logs/train_*.log
```

### TensorBoard

浏览器访问: **http://106:6006**

---

## 📊 监控指标

### 训练日志位置
- 路径: `/data/shaodi/Siamese/logs/train_YYYYMMDD_HHMMSS.log`
- 自动保存，带时间戳

### 关键指标
- **Loss**: 总损失 = CE loss + NCE loss
- **Recall@K**: K=1, 5, 10, 50 的检索召回率
- **评估频率**: 每 5 epoch

### GPU 状态
- GPU 0: DeepSpeed 主进程 + 部分计算
- GPU 1/2/3: 训练 worker，利用率应接近 100%
- 正常温度: 60-80°C
- 正常显存: 4-7 GB / 24 GB

---

## 💾 模型保存

### Checkpoint 位置
```
checkpoints_proposer_multi/
├── best/                          # 最佳模型（自动保存）
│   ├── mp_rank_00_model_states.pt
│   ├── mp_rank_01_model_states.pt
│   ├── mp_rank_02_model_states.pt
│   └── mp_rank_03_model_states.pt
└── epoch_XX/                      # 定期 checkpoint
```

### 下载模型到本地

```bash
# 下载最佳模型
scp -r shaodi@106:/data/shaodi/Siamese/checkpoints_proposer_multi/best ./model_best

# 下载特定 epoch
scp -r shaodi@106:/data/shaodi/Siamese/checkpoints_proposer_multi/epoch_50 ./
```

---

## 🔄 开发工作流

### 1. 本地修改代码

```bash
cd /Data/Work/Siamese
# 编辑文件...
git add <files>
git commit -m "description"
git push origin main
```

### 2. 远程更新并重启

```bash
bash remote_train.sh restart
```

### 3. 监控训练

```bash
bash remote_train.sh status
bash remote_train.sh logs
```

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| [REMOTE_TRAINING.md](REMOTE_TRAINING.md) | 远程训练快速参考 |
| [TMUX_GUIDE.md](TMUX_GUIDE.md) | tmux 详细使用指南 |
| [DEEPSPEED.md](docs/DEEPSPEED.md) | DeepSpeed 配置说明 |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | 完整部署指南 |
| [CLAUDE.md](CLAUDE.md) | 项目架构文档 |

---

## 🐛 故障排除

### 训练意外停止

```bash
# 查看错误日志
bash remote_train.sh logs | grep -i error

# 查看完整日志
ssh shaodi@106 "cat /data/shaodi/Siamese/logs/train_*.log"

# 检查 GPU
bash remote_train.sh gpu
```

### 路径问题

```bash
# 重新修复 .cs 文件路径
ssh shaodi@106
cd /data/shaodi/Siamese
python3 scripts/fix_cs_paths.py
```

### 重置环境

```bash
ssh shaodi@106
cd /data/shaodi/Siamese
rm -rf checkpoints_proposer_multi/
bash remote_train.sh restart
```

---

## ⏱️ 时间线

- **00:00-02:00**: 数据加载和初始化
- **02:00-05:00**: Epoch 1-10 (验证配置)
- **05:00-60:00**: Epoch 10-50 (主要训练)
- **60:00-90:00**: Epoch 50-80 (收敛)
- **总时长**: 约 1.5-2 小时

---

## 🎓 训练完成后

1. ✅ 下载最佳模型到本地
2. ✅ 查看 TensorBoard 曲线
3. ✅ 评估模型性能（Recall@K）
4. ✅ 进行推理测试

---

## 📞 联系信息

- **项目**: Siamese 摊饼识别 (Amortized Pose Estimation)
- **远程服务器**: shaodi@106
- **本地路径**: /Data/Work/Siamese
- **远程路径**: /data/shaodi/Siamese
- **TensorBoard**: http://106:6006

---

生成时间: 2026-07-04
状态: ✅ 训练运行中
