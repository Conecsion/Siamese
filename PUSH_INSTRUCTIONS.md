# GitHub 推送说明

## 当前状态

✅ 代码已提交到本地 Git (5 commits)
✅ DeepSpeed 多卡训练代码已完成
✅ 所有测试通过 (28 passed)
❌ 需要配置 GitHub 认证

## 推送方式（三选一）

### 方式 1: SSH 推送 (推荐)

1. 添加 SSH 公钥到 GitHub:
   - 访问: https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴以下公钥:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB1s0KH3p8rcCoq7wZNtFyBrdK/mHkW5YU/aLuX7uEiz shaodi
```

2. 推送代码:

```bash
cd /Data/Work/Siamese
git remote set-url origin git@github.com:Conecsion/Siamese.git
git push -u origin main
```

### 方式 2: Personal Access Token (HTTPS)

1. 生成 Token:
   - 访问: https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - 勾选 `repo` 权限
   - 生成并复制 token

2. 推送代码:

```bash
cd /Data/Work/Siamese
git remote set-url origin https://YOUR_TOKEN@github.com/Conecsion/Siamese.git
git push -u origin main
```

### 方式 3: 手动推送（浏览器认证）

```bash
cd /Data/Work/Siamese
! git push -u origin main
```

在命令前加 `!` 会弹出浏览器认证窗口。

## 推送内容

### 提交记录 (5 commits)

```
8bbd10f feat: add DeepSpeed multi-GPU training support
74e3e2b docs: add deployment guide for GitHub and remote training
8e73fb7 docs: add data packing scripts and remote training config
a62b45d feat: implement amortized pose estimation (proposer)
e31b8a9 feat: optimize encoder with configurable stem_stride
```

### 新增文件

- `scripts/train_proposer_ds.py` - DeepSpeed 分布式训练脚本
- `configs/ds_config.json` - DeepSpeed 配置 (ZeRO-2 + FP16)
- `DEEPSPEED.md` - 多卡训练完整文档
- `DATA_MANIFEST.md` - 训练数据清单
- `DEPLOYMENT.md` - 远程部署指南
- `pack_training_data.sh` - 数据打包脚本

### 修改文件

- `pyproject.toml` - 添加 deepspeed>=0.14.0 依赖
- `scripts/train_proposer.py` - 修复 NaN loss (显式归一化 + 梯度裁剪)
- `siamese/models/pose_head.py` - 修复 Pyright 类型错误
- `tests/test_resample.py` - 更新为新的 Fourier binning API

## 验证推送成功

推送后访问: https://github.com/Conecsion/Siamese

检查:
- ✅ 5 个新提交已出现
- ✅ `DEEPSPEED.md` 文档可见
- ✅ `scripts/train_proposer_ds.py` 存在
- ✅ `configs/ds_config.json` 存在

## 远程训练准备清单

推送成功后，在远程服务器 (4 x RTX 3090) 执行:

```bash
# 1. 克隆仓库
git clone https://github.com/Conecsion/Siamese.git
cd Siamese

# 2. 上传并解压数据 (44GB)
scp siamese_training_data.tar.gz remote-server:~/
tar -xzf ~/siamese_training_data.tar.gz

# 3. 安装依赖
pip install -e ".[dev]"
pip install deepspeed tensorboard

# 4. 启动 4 卡训练
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json

# 5. 启动 TensorBoard
tensorboard --logdir checkpoints_proposer_multi/tensorboard --port 6006 --bind_all
```

## 预期性能 (4 x RTX 3090)

- **训练时间**: 1.5-2 小时 (80 epochs)
- **吞吐量**: 600-800 samples/sec
- **加速比**: 4-5x (相比单卡)
- **显存使用**: ~12-16GB per GPU

## 故障排除

如果推送失败，可以:

1. 使用 Claude Code 的 `!` 前缀命令 (会弹出浏览器认证)
2. 联系我继续协助配置 SSH/Token
3. 手动在终端外执行 `git push`
