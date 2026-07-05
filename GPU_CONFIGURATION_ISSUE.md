# GPU 配置问题和解决方案

## 🔍 问题描述

在 4 x RTX 3090 服务器上训练时，发现：
- GPU 0 利用率始终为 0%
- GPU 1-2 利用率 100%
- GPU 3 利用率 0%

## 🎯 根本原因

### GPU 0 问题
**原因**: GPU 0 被显示服务器占用

```
GPU 0: Disp.A = On (连接显示器)
进程: Xorg (桌面环境)
```

虽然 DeepSpeed 在 GPU 0 上分配了显存，但实际计算无法进行。

### GPU 3 问题
**原因**: 只启动了 2 个训练进程

```bash
# 检查进程数
ps aux | grep train_proposer_ds.py | grep -v grep
# 结果: 5 个进程 (1 主进程 + 1 launcher + 3 worker)
```

但实际只有 2 个 worker 在 GPU 1-2 上运行，GPU 3 未被使用。

## ✅ 已应用的修复

### 1. 排除 GPU 0

**修改文件**: `train_all.sh`

```bash
# 之前
deepspeed --num_gpus=4 \
    scripts/train_proposer_ds.py \
    ...

# 修改后
CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --num_gpus=3 \
    scripts/train_proposer_ds.py \
    ...
```

### 2. 调整 Batch Size

**修改文件**: `configs/ds_config.json`

```json
{
  "train_batch_size": 48,  // 从 64 改为 48 (16*3)
  "train_micro_batch_size_per_gpu": 16,
  "gradient_accumulation_steps": 1
}
```

### 3. 降低学习率和优化配置

**修改文件**: `configs/proposer_ribosome_multi.yaml`, `configs/ds_config.json`

```yaml
lr: 0.0001               # 从 0.0002 降低
warmup_steps: 1000       # 从 500 增加
```

```json
{
  "optimizer": {"lr": 0.0001},
  "scheduler": {
    "warmup_max_lr": 0.0001,
    "warmup_num_steps": 1000
  },
  "fp16": {
    "initial_scale_power": 12  // 从 16 降低
  }
}
```

## 🔄 当前状态

### GPU 使用情况

```
GPU 0: 0% (显示服务器占用，不可用)
GPU 1: 100% ✅ (正在训练)
GPU 2: 100% ✅ (正在训练)
GPU 3: 0% ❌ (未被使用)
```

### 训练进程

```
主进程: 1
Launcher: 1
Workers: 3 (但只有 2 个活跃)
```

## 🐛 待解决: GPU 3 未使用

### 可能原因

1. **DeepSpeed 启动问题**
   - 指定 `--num_gpus=3` 但只启动了 2 个
   - 可能是初始化延迟

2. **CUDA 设备映射问题**
   - `CUDA_VISIBLE_DEVICES=1,2,3` 映射为本地 `0,1,2`
   - 但第 3 个设备未被分配

3. **训练脚本问题**
   - 可能在某处硬编码了 GPU 数量

### 调试步骤

#### 1. 检查 DeepSpeed 日志

```bash
ssh shaodi@106 "cd /data/shaodi/Siamese && grep -i 'world_size\|num_gpus\|rank' logs/train_*.log | tail -20"
```

预期应该看到：
```
world_size=3
num_local_procs=3
global_rank_mapping={'localhost': [0, 1, 2]}
```

#### 2. 检查所有训练进程的 GPU 绑定

```bash
ssh shaodi@106 "nvidia-smi pmon -c 1 | grep python"
```

#### 3. 查看 GPU 3 的详细信息

```bash
ssh shaodi@106 "nvidia-smi -i 3"
```

### 临时解决方案

如果 GPU 3 持续无法使用，可以：

**方案 A: 只使用 GPU 1-2**
```bash
# train_all.sh
CUDA_VISIBLE_DEVICES=1,2 deepspeed --num_gpus=2 \
    scripts/train_proposer_ds.py \
    ...
```

```json
// ds_config.json
{
  "train_batch_size": 32,  // 16*2
  ...
}
```

**方案 B: 使用 GPU 0-1（如果可以关闭桌面环境）**
```bash
# 停止桌面环境
sudo systemctl stop gdm  # 或 lightdm/sddm

# 使用 GPU 0-1
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 \
    scripts/train_proposer_ds.py \
    ...
```

**方案 C: 明确指定 GPU (推荐尝试)**
```bash
# train_all.sh
deepspeed --include localhost:1,2,3 \
    scripts/train_proposer_ds.py \
    ...
```

## 📊 性能影响

### 当前配置 (2 GPU)
- 有效 GPU: 2 (GPU 1-2)
- Batch size: 48 (理论), 32 (实际)
- 训练速度: ~66% 的预期速度

### 理想配置 (3 GPU)
- 有效 GPU: 3 (GPU 1-2-3)
- Batch size: 48
- 训练速度: 100% (相对于 3 GPU)

## 🔧 推荐行动

### 立即尝试

1. **检查是否真的需要 3 个 GPU**
   
   由于现在只有 2 个 GPU 在工作，先确认训练是否正常进行：
   
   ```bash
   bash remote_train.sh logs
   ```
   
   如果看到正常的 epoch 输出和 loss 下降，说明 2 GPU 训练正常。

2. **尝试方案 C: 明确指定 GPU**
   
   修改 `train_all.sh`:
   ```bash
   deepspeed --include localhost:1,2,3 \
       scripts/train_proposer_ds.py \
       ...
   ```

3. **如果方案 C 失败，降级到 2 GPU (方案 A)**
   
   这样可以确保训练稳定，只是速度慢一些。

### 后续优化

1. **分析为什么 GPU 3 未被使用**
   - 查看完整的 DeepSpeed 启动日志
   - 检查是否有错误信息被隐藏

2. **考虑禁用桌面环境，使用 GPU 0**
   - 如果这是专用训练服务器
   - 可以获得 3-4 GPU 的完整性能

## 📝 相关文件

- `train_all.sh` - 训练启动脚本
- `configs/ds_config.json` - DeepSpeed 配置
- `configs/proposer_ribosome_multi.yaml` - 训练配置
- `logs/train_*.log` - 训练日志

## 🎓 经验教训

1. **GPU 0 常被显示服务器占用**
   - 训练服务器应禁用桌面环境
   - 或者明确排除 GPU 0

2. **DeepSpeed 的 GPU 分配需要验证**
   - 不要只看进程数
   - 要用 `nvidia-smi` 实际确认利用率

3. **Batch size 必须与 GPU 数量匹配**
   - `train_batch_size = micro_batch * num_gpus * accumulation_steps`

---

生成时间: 2026-07-05
状态: 部分解决 (GPU 1-2 正常，GPU 3 待调试)
