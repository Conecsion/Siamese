# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

**Phase A: Amortized Pose Estimation（当前阶段）**

构建 Siamese 对比学习框架，将真实冷冻电镜颗粒（mic）和参考 volume 的投影（proj）映射到共同的嵌入空间，用于快速朝向估计（pose proposal）。

**两阶段设计**：

1. **Proposer（本阶段）**：快速筛选 top-K 候选朝向（粗筛，~10-50 候选）
2. **Forward Model（未来）**：在候选内精确优化 pose 参数（精修）

**当前训练目标**：

- 输入：真实颗粒（含噪声、CTF 调制）
- 输出：在预定义 gallery（9216 个 HEALPix 朝向）中检索最接近的朝向
- 评估指标：Recall@K（真实朝向的最近 gallery 点是否在预测 top-K 内）

**数据**：

- 核糖体真实数据（cryoSPARC 导出）：J8/J20/J31 三套数据集，共 37710 颗粒
- Reference volume：核糖体 3D 重建（.mrc 格式）
- 在线投影：训练时根据真实朝向从 volume 实时生成配对投影

## 常用命令

```bash
# 安装
uv pip install -e ".[dev]"

# 生成训练数据 (nside=8 约 768 方向) - 合成数据，已弃用
python scripts/generate_data.py emd_19110.map 8 --output-dir data/

# 训练 - Proposer (单卡)
python scripts/train_proposer.py --config configs/proposer_ribosome_multi.yaml

# 训练 - Proposer (多卡 DeepSpeed，推荐)
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json

# 监控训练
tensorboard --logdir checkpoints_proposer_multi/tensorboard --port 6006

# 评估
python scripts/eval.py --checkpoint checkpoints/best.pt --data-dir data/

# 运行所有测试
python -m pytest tests/

# 运行单个测试
python -m pytest tests/test_encoder.py -v

# 检查显存
nvidia-smi
```

## 架构

### 模型：`siamese/models/`

**核心模型演进**：

1. `SiameseEncoder`（Phase 0）：单塔共享编码器，mic 和 proj 共用相同权重
2. `TwoTowerEncoder`（当前）：双塔独立编码器，mic 塔和 proj 塔各自学习领域特定特征
3. `PoseProposer`（当前）：包装 TwoTowerEncoder，增加 gallery 检索 + 先验分布 + 残差头

**TwoTowerEncoder** (`encoder.py`):

- **双塔架构**：`mic_encoder` 和 `proj_encoder` 独立 backbone + fusion head
- **双分支**：每个塔内部包含实空间分支 + 频域分支（FFT 后取实/虚部共 2 通道）
- **Backbone**：ConvNeXt（默认 `convnext_tiny`），支持 `stem_stride`（4/2/1）和 `share_backbone`
- **融合**：FusionHead（concat + MLP + L2 归一化）或 CrossAttentionFusion
- 输入：`[N, 1, D, D]`，输出：`[N, embedding_dim]`

**PoseProposer** (`pose_head.py`):

- Gallery 检索：`z_mic @ gallery_emb.T` → top-M 候选朝向
- 先验分布 π_θ：softmax over 检索分数（零参数）
- 残差头：MLP 输出 SO(3) 切空间残差 `[N, 3]`（未来用于精修）
- 位移头：MLP 输出粗略 shift `[N, 2]`（像素，tanh 限幅）

### 训练：两种模式

在远程服务器上训练：ssh shaodi@106, 该服务器有4张RTX 3090. 使用tmux等工具将训练进程放在后台以防ssh断开
远程服务器项目位置:/data/shaodi/Siamese，
尽量在本地编辑代码，本地push后在远程服务器pull，再开始训练

**单卡训练** (`scripts/train_proposer.py`):

- 标准 PyTorch 训练循环
- 混合精度：`torch.amp.autocast` + `GradScaler`
- 双损失：`w_ce * loss_ce + w_nce * loss_nce`
  - `loss_ce`: Gallery 软标签分类（高斯核，σ=7°）
  - `loss_nce`: InfoNCE 对比学习（temperature=0.07）
- 梯度裁剪：`clip_grad_norm_(max_norm=1.0)`
- NaN 检测：跳过异常 batch，避免污染权重

**多卡训练** (`scripts/train_proposer_ds.py`):

- DeepSpeed 分布式训练（ZeRO Stage 2 + FP16）
- 配置：`configs/ds_config.json`
- 启动：`deepspeed --num_gpus=4 scripts/train_proposer_ds.py --config ... --deepspeed ...`
- 特性：
  - ZeRO-2：优化器 + 梯度分片（每卡存储 1/N）
  - 分布式采样器：自动数据分片 + shuffle
  - Gallery 广播：主进程构建，`dist.broadcast()` 到所有卡
  - 仅主进程（rank 0）评估、保存 checkpoint、写 TensorBoard
- Checkpoint：多文件格式 `mp_rank_XX_model_states.pt`，恢复时 DeepSpeed 自动加载
- 性能（4×3090）：600-800 samples/sec，1.5-2h（80 epochs）
- 详细文档：见 `DEEPSPEED.md`

### 数据管线：`siamese/data/`

1. **投影** (`projection.py`)：傅里叶切片定理 (Fourier Slice Theorem)，GPU batch 向量化
   - 默认 Kaiser-Bessel gridding + 实空间去卷积预补偿（`pfac=2` 过采样），高频精度与 cryoSPARC 一致
   - 旋转矩阵遵循 **pyem 约定**（`axis_angle_to_matrix` 输出 = `Rotation.from_rotvec().as_matrix().T`）
   - CTF 模拟：Contrast Transfer Function，支持 particle_sign（暗场/亮场）
   - `trilinear` 备用插值高频失真明显（>8Å），仅作对比/回退

2. **方向采样** (`orientations.py`)：HEALPix 均匀格点（`healpy`）或均匀随机 SO(3)

3. **真实数据加载** (`cryosparc.py`)：`CryoSparcParticleDataset`
   - 输入：cryoSPARC 导出的 `.cs` 文件 + 颗粒图像目录 + reference MRC volume
   - 每个样本返回：`(particle [1,D,D], projection [1,D,D], axis_angle [3], ctf_params)`
   - 在线投影：根据存储的真实朝向，从 reference volume 实时生成配对投影
   - CTF 应用：`apply_ctf_to_proj=True` 时对投影应用 CTF（匹配颗粒的调制）
   - Fourier binning：颗粒和投影重采样到 `working_ps`（默认 2.0 Å/pix）
   - Volume 降采样：>260³ 自动降到 256³（节省 gallery 投影显存）
   - 数据缓存：`enable_cache()` 后首次访问时缓存到显存，后续 epoch 零开销

4. **重采样** (`resample.py`)：Fourier binning（无混叠降采样）
   - `fourier_crop_2d/3d`：频域裁剪，避免空间插值的混叠
   - `fourier_pad_2d`：频域补零升采样
   - `resample_to_working_ps`：重采样到目标 pixel size + 分桶（64/128/256/384/512）

### 损失函数：`siamese/losses/`

**OrientationAwareInfoNCELoss** (`infonce.py`):

- InfoNCE 对比学习损失，temperature=0.07
- 正样本：同一颗粒的 mic-proj 配对
- 负样本：batch 内其他颗粒
- 输入：`z_mic [N, C]`, `z_proj [N, C]`, `axis_angle [N, 3]`（用于计算朝向距离权重）

**GalleryClassificationLoss** (`gallery_ce.py`):

- Gallery 软标签分类损失
- 标签：高斯核 `exp(-angular_dist² / (2σ²))`，σ=7°（HEALPix 网格分辨率）
- 将检索问题转化为 9216-way 软标签分类
- 训练目标 = 检索目标（端到端优化）

### 评估：`siamese/eval/`

- **Recall@M**：真实朝向的最近 gallery 点是否在网络预测 top-M 内
- **t-SNE**：embedding 空间可视化
- **FAISS 检索**：`IndexFlatIP` 暴力检索 Top-K（embedding 已 L2 归一化，内积等价余弦相似度）

## 核心约定

### Fourier Slice Projection

**必须使用** `projection.py` 的 `project_fourier_slice`（默认）或 `project_volume`（封装）：

- Kaiser-Bessel gridding（`pfac=2` 过采样）+ 实空间去卷积预补偿
- 高频精度与 cryoSPARC 一致（实测 ~0.001 相对误差，<8Å 分辨率）
- GPU batch 向量化（~100x 快于 CPU trilinear）

**禁止**：

- ❌ 不使用 `scipy.ndimage` 或 `torch.nn.functional.grid_sample` 直接插值 volume（高频严重失真）
- ❌ 不直接调用 `scipy.spatial.transform.Rotation`（旋转约定与 pyem 不一致，需转置）

### CTF 模拟

**CTF 参数来源**：cryoSPARC `.cs` 文件（`ctf/` 字段组）

- `defocus_U`, `defocus_V`, `defocus_angle`: 散焦参数
- `cs_mm`: 球差
- `accel_kv`: 加速电压
- `amp_contrast`: 振幅对比度

**CTF 应用位置**：

- 颗粒（mic）：已在 cryoSPARC 处理时应用，不再重复
- 投影（proj）：`apply_ctf_to_proj=True` 时应用（匹配颗粒的调制）

**particle_sign**：默认 `-1.0`（暗场约定，蛋白呈暗色）

### Fourier Binning（重采样）

**必须使用** `resample.py` 的 Fourier 域重采样方法：

- 降采样：`fourier_crop_2d/3d`（频域裁剪，无混叠）
- 升采样：`fourier_pad_2d`（频域补零）

**自动应用**：

- `CryoSparcParticleDataset` 初始化时，>260³ 的 volume 自动降采样到 256³（节省 gallery 投影显存）
- 颗粒图像同步降采样（匹配 volume 降采样倍率），确保投影和颗粒的 pixel size 一致
- 降采样后 pixel size 相应放大：`new_psize = old_psize × (old_size / new_size)`

**禁止**：

- ❌ 不使用 `scipy.ndimage.zoom` 等空间插值方法降采样
- ❌ 不直接修改 `.cs` 文件的 `blob/psize_A`（必须同步修改真实图像文件）

### 其他约定

- **旋转约定**：pyem/cryoSPARC 轴角约定，与 `scipy.spatial.transform.Rotation.from_rotvec` 互为转置关系
- **位移方向**：`project_fourier_slice` 内部取反（`-shift`），与 cryoSPARC 存储格式一致
- **Nyquist 半径**：补零网格上截断半径为 `N/2`（`N = pfac * D`），而非 `D/2`
- **CTF particle_sign**：默认 `-1.0`（暗场约定）
- **Pixel size 一致性**：投影和颗粒重采样到 `working_ps` 前，必须确保原始 pixel size 一致

## DeepSpeed 多卡训练

详见 `DEEPSPEED.md`。关键点：

**启动命令**：

```bash
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json
```

**配置调整**（`ds_config.json`）：

- `train_batch_size`: 全局 batch size（4 卡 = 64）
- `train_micro_batch_size_per_gpu`: 每卡 batch size（16）
- `zero_optimization.stage`: 2（推荐）或 3（显存不足时）

**Checkpoint 恢复**：

```bash
# DeepSpeed 自动检测最新 checkpoint 并恢复
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json
```

**转换为单文件**（推理用）：

```python
import torch
checkpoint = torch.load("checkpoints_proposer_multi/best/mp_rank_00_model_states.pt")
proposer.load_state_dict(checkpoint['module'])
torch.save(proposer.state_dict(), "proposer_best.pt")
```

## 远程部署

详见 `DEPLOYMENT.md` 和 `PUSH_INSTRUCTIONS.md`。

**快速开始**：

```bash
# 1. 克隆仓库
git clone https://github.com/Conecsion/Siamese.git
cd Siamese

# 2. 上传并解压数据（44GB）
scp siamese_training_data.tar.gz remote-server:~/
tar -xzf ~/siamese_training_data.tar.gz

# 3. 安装依赖
pip install -e ".[dev]"

# 4. 启动训练
deepspeed --num_gpus=4 scripts/train_proposer_ds.py \
    --config configs/proposer_ribosome_multi.yaml \
    --deepspeed configs/ds_config.json
```

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 单个测试文件
python -m pytest tests/test_encoder.py -v

# 端到端烟雾测试（1 batch 训练）
python -m pytest tests/test_smoke.py -v

# 覆盖率
python -m pytest tests/ --cov=siamese --cov-report=html
```

## 常见问题

### NaN Loss

**症状**：训练 5-10 epoch 后 loss 变成 NaN
**根因**：

1. `z_proj` 在 InfoNCE 中未显式归一化（AMP 下 float16 精度漂移）
2. 学习率过高（lr > 0.0002）
3. 梯度爆炸（无梯度裁剪）

**修复**（已应用）：

```python
# 显式归一化
z_proj = F.normalize(proposer.encoder.encode_proj(proj), dim=1)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(proposer.parameters(), max_norm=1.0)

# NaN 检测跳过
if not torch.isfinite(loss):
    continue
```

### OOM（显存不足）

**单卡**：

- 降低 batch size（16 → 8 → 4）
- 减小 `working_ps`（2.0 → 2.5 Å/pix，图像更小）
- 禁用数据缓存：`cache_samples: false`

**多卡 DeepSpeed**：

- 降低 `train_micro_batch_size_per_gpu`（16 → 8）
- 启用 ZeRO Stage 3（模型也分片）
- CPU offload（牺牲速度）

### Gallery 投影慢

**症状**：每个 epoch 开始时卡顿 10-30 秒
**原因**：9216 个朝向投影 + 编码耗时
**优化**（已应用）：

- Volume 降采样到 256³（自动）
- Batch 投影（128 per batch）
- 缓存 gallery embedding（每 5 epoch 重建一次）

### 多卡训练不均衡

**症状**：某些卡利用率低
**检查**：

```bash
# 实时监控
watch -n 1 nvidia-smi
```

**修复**：

- 增加 DataLoader `num_workers`（4 → 8）
- 检查网络带宽（多卡通信瓶颈）
- 启用 `overlap_comm: true`（已默认）
