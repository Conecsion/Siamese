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
- Reference volume：各数据集的 refine 输出 volume
- 训练时投影应用 CTF（匹配真实颗粒），gallery 投影不应用 CTF（快速编码）

## 硬件

- 本机: RTX 5070Ti
- 远程工作站: 4× RTX 3090，`ssh 106` 连接，工作目录 `/home/shaodi/Work/Siamese`
- **调用 GPU 前先检查显存** (`nvidia-smi`)，防止 OOM

## 常用命令

```bash
# 安装
uv pip install -e ".[dev]"

# 生成训练数据 (nside=8 约 768 方向)
python scripts/generate_data.py emd_19110.map 8 --output-dir data/

# 训练
python scripts/train.py --config configs/default.yaml

# 评估
python scripts/eval.py --checkpoint checkpoints/best.pt --data-dir data/

# 运行所有测试
python -m pytest tests/

# 运行单个测试
python -m pytest tests/test_encoder.py -v
```

## 架构

### 模型：`siamese/models/`

`SiameseEncoder` 是核心模型，对 mic 和 proj **共用同一个编码器**（Siamese 结构）：

- **双分支**：实空间分支（1 通道归一化图像）+ 频域分支（FFT 后取实/虚部共 2 通道）
- **Backbone**：ConvNeXt（默认 `convnext_tiny`），通过 `timm` 构建，支持可配置 `stem_stride` 和可选共享权重 `share_backbone`
- **融合**：两分支 GAP 后经 `FusionHead`（concat + MLP）或可选的 `CrossAttentionFusion` 输出 L2 归一化 embedding
- 输入 `[N, 1, D, D]`，输出 `[N, embedding_dim]`

### 数据管线：`siamese/data/`

1. **投影** (`projection.py`)：傅里叶切片定理 (Fourier Slice Theorem)，GPU batch 向量化
   - 默认 Kaiser-Bessel gridding + 实空间去卷积预补偿（`pfac=2` 过采样），高频精度与 cryoSPARC 一致
   - 旋转矩阵遵循 **pyem 约定**（`axis_angle_to_matrix` 输出 = `pyem.geom.aa2rot`，是 scipy `Rotation` 的转置）
   - 遇 CUDA OOM 自动减半 chunk 重试
   - `trilinear` 备用插值高频失真明显（>8Å），仅作对比/回退

2. **方向采样** (`orientations.py`)：HEALPix 均匀格点（`healpy`）或均匀随机 SO(3)

3. **数据生成** (`generate.py`)：输出 `projs.pt`/`mics.pt`/`axisang.pt`/`pairs.pt`
   - projs：无 CTF、无噪声的干净投影
   - mics：有 CTF + 随机位移 + 白噪声的模拟颗粒，SNR 约 0.001–0.01

4. **数据集** (`dataset.py`)：`MicProjDataset` 加载 .pt 文件，按比例划分 train/val/test，每个样本返回 `(mic [1,D,D], proj [1,D,D])`

### 训练：`siamese/training/`

- 损失：InfoNCE（`siamese/losses/infonce.py`），temperature=0.07
- 优化器：AdamW + CosineAnnealingWarmRestarts
- `Trainer` 自动保存 `best.pt`（val loss 最优）和每 50 epoch 一个 checkpoint

### 评估：`siamese/eval/`

- `siamese/eval/retrieval.py`：FAISS IndexFlatIP 暴力检索 Top-K（embedding 已 L2 归一化，内积等价于余弦相似度）
- `siamese/eval/metrics.py`：Recall@K、中位角误差

## 关键约定

### 降采样规则（重要！）

**所有降采样必须使用 Fourier binning 方法**（`siamese/data/resample.py`）：
- **2D 图像**：`fourier_crop_2d()` - FFT → 裁剪中心区域 → 逆 FFT
- **3D volume**：`fourier_crop_3d()` - 3D FFT → 裁剪中心立方体 → 逆 FFT
- **原因**：直接空间插值（如 `scipy.ndimage.zoom`）会产生高频混叠伪影（斜向条纹），Fourier binning 通过频域截断高频成分，确保无混叠

**当前应用**：
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
