# GitHub 和远程训练部署指南

## 当前状态

✅ 代码已提交到 git (2 commits)
✅ 训练数据已打包: `siamese_training_data.tar.gz` (8.7GB)
✅ 远程训练配置已创建

## 下一步操作

### 1. 推送代码到 GitHub

首先在 GitHub 创建仓库：
1. 访问 https://github.com/new
2. Repository name: `Siamese`
3. Description: `Amortized pose estimation for cryo-EM with Siamese networks`
4. 选择 **Private**
5. **不要**初始化 README/LICENSE/.gitignore
6. 点击 "Create repository"

然后推送代码：
```bash
cd /Data/Work/Siamese

# 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/Siamese.git

# 推送代码
git push -u origin main
```

### 2. 上传训练数据到远程服务器

#### 方案 A: scp 直接传输

```bash
# 传输到远程服务器（替换 YOUR_SERVER）
scp siamese_training_data.tar.gz YOUR_SERVER:~/

# 传输 SHA256 校验文件
scp siamese_training_data.tar.gz.sha256 YOUR_SERVER:~/
```

#### 方案 B: 通过云存储中转（推荐，如果网络较慢）

```bash
# 上传到 Google Drive / Dropbox / AWS S3 等
# 然后在远程服务器下载
```

### 3. 远程服务器设置

SSH 到远程服务器后：

```bash
# 1. 克隆代码
git clone https://github.com/YOUR_USERNAME/Siamese.git
cd Siamese

# 2. 验证并解压数据
sha256sum -c ~/siamese_training_data.tar.gz.sha256
tar -xzf ~/siamese_training_data.tar.gz

# 3. 移动数据到代码目录
mv data/ .
mv proposer_ribosome_multi_remote.yaml configs/
mv README_REMOTE.md .

# 4. 安装依赖
pip install -e ".[dev]"
pip install tensorboard

# 5. 启动训练（后台运行）
nohup python scripts/train_proposer.py \
    --config configs/proposer_ribosome_multi_remote.yaml \
    > train.log 2>&1 &

# 6. 启动 TensorBoard
nohup tensorboard \
    --logdir checkpoints_proposer_multi/tensorboard \
    --port 6006 \
    --bind_all \
    > tensorboard.log 2>&1 &

# 7. 查看训练进度
tail -f train.log

# 8. 访问 TensorBoard
# 浏览器打开: http://YOUR_SERVER_IP:6006
```

## 训练配置

- **数据集**: J8 (16108) + J20 (16416) + J31 (5186) = 37,710 颗粒
- **Epochs**: 80（约 2.5-3 小时）
- **Batch size**: 16
- **GPU 显存需求**: ~12GB
- **Gallery**: 9216 朝向 (HEALPix nside=8, inplane=12)

## 预期结果

- **Recall@10**: > 0.8
- **Recall@50**: > 0.95
- **Loss**: 收敛到 ~9-10（InfoNCE + Gallery CE）

## 文件清单

### 本地（已准备好）
- `siamese_training_data.tar.gz` (8.7GB) - 训练数据包
- `siamese_training_data.tar.gz.sha256` - SHA256 校验和
- Git 仓库（待推送到 GitHub）

### 远程（需要部署）
1. 从 GitHub 克隆代码仓库
2. 上传并解压 `siamese_training_data.tar.gz`
3. 按照上述步骤设置并启动训练

## 故障排除

### OOM 错误
降低 batch size：
```yaml
batch_size: 8  # 或 12
```

### 数据路径错误
确认目录结构：
```
Siamese/
├── data/
│   ├── cs_processed/
│   └── particles/
├── configs/
│   └── proposer_ribosome_multi_remote.yaml
└── scripts/
    └── train_proposer.py
```

### 远程仓库认证
使用 Personal Access Token (PAT)：
```bash
git remote set-url origin https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/Siamese.git
```

## 当前本地训练

本地训练仍在运行：
```bash
# 查看进度
tail -f /tmp/proposer_multi.log

# TensorBoard
http://localhost:6006
```
