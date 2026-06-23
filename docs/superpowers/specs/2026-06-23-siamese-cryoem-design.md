# Siamese Contrastive Encoder for Cryo-EM Particle Retrieval — Design Spec

**Date:** 2026-06-23
**Status:** Approved
**Goal:** 快速原型验证 — 用模拟数据跑通"数据生成 → 训练 → 检索 → 评估"完整链路

---

## 1. 问题定义

训练一个 Siamese/对比学习编码器，将低信噪比 cryo-EM particle 图像（mic）和对应的高信噪比投影（proj）映射到同一 embedding 空间。推理时，给定一张 mic，从大量 proj 中检索最匹配的 proj。

**冒烟测试目标：** top-1 准确率显著优于随机水平（> 5-10× random baseline）。

---

## 2. 数据生成与预处理

### 2.1 数据源

- 3D volume: `emd_19110.map`（通过 `mrcfile` 读取）
- 使用 `project.py` 中的 `healpix_project()` 从 volume 生成模拟 proj

### 2.2 生成流程

1. 从 `emd_19110.map` 读取 3D volume
2. 用 HEALPix 采样生成方向（nside=8, 768 个方向）
3. 对每个方向用 `project()` 生成 clean proj [D, D]
4. 保存 proj + 对应轴角（axis-angle）
5. 对每个 proj 生成对应的 noisy mic：
   - 随机 in-plane shift（±5 pixels）
   - 随机 CTF 参数（defocus 0.5-4 μm）
   - 加噪声（SNR 范围 0.001-0.01）
6. 输出：`(mic, proj, axisang)` 三元组，每个 proj 生成 2 个不同噪声版本的 mic

### 2.3 预处理

- **GPU 预处理**：project 过程需要 GPU，提前一次性生成并存储到硬盘
- **存储格式**：`.pt` 或 `.npy` 文件
- **图像归一化**：输入模型前对所有图像做归一化（mean/std 或 min-max），因为不同数据亮度可能有较大差异
- **频域预处理**：对图像做 FFT2，取实部+虚部作为 2 通道输入

### 2.4 数据集类

- `MicProjDataset`：支持随机采样和 train/val/test split
- 按 HEALPix 方向半球划分，避免相邻方向泄漏

---

## 3. 模型架构

### 3.1 双分支编码器

```
输入: mic/proj 图像 [D, D]

实空间分支:
  [1, D, D] → Backbone → feature [C_real, D/32, D/32] → GlobalAvgPool → [C_real]

频域分支:
  FFT2 → [2, D, D] (实部+虚部)
  → Backbone (in_channels=2) → feature [C_freq, D/32, D/32] → GlobalAvgPool → [C_freq]

融合:
  concat([C_real], [C_freq]) → [C_real + C_freq]
  → Linear + BN + ReLU → [256]
  → Linear → [128]  (embedding, L2 normalized)
```

### 3.2 设计要点

- 两个分支共享同一 backbone 架构，但权重独立（输入通道数不同）
- 频域输入不做归一化到 [0,1]，保留原始数值范围
- Embedding 输出 L2 normalization，用于 InfoNCE 和 FAISS 余弦相似度检索
- 图像输入模型前和 FFT 前都做归一化

### 3.3 Backbone

- **默认**: ConvNeXt-Tiny（各 stage 通道数可配置）
- **预留接口**: ViT-Small / Swin-T（TODO）

### 3.4 可配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_size` | 128 | 128/192/256/512（TODO: 多尺寸） |
| `embedding_dim` | 128 | TODO: 后续可调 |
| `backbone` | ConvNeXt-Tiny | ConvNeXt/Swin-T/ViT |
| `real_in_channels` | 1 | 实空间分支输入通道 |
| `freq_in_channels` | 2 | 频域分支输入通道（实部+虚部） |

---

## 4. Loss 与训练

### 4.1 InfoNCE (NT-Xent)

```
给定 batch 中 N 对 (mic_i, proj_i)：

1. 编码: z_mic = encoder(mic_i), z_proj = encoder(proj_i)   [N, 128]
2. 计算相似度矩阵: S[i,j] = z_mic[i] · z_proj[j] / τ   [N, N]
   （τ 为 temperature，默认 0.07）
3. 对称 InfoNCE:
   loss_mic = CrossEntropy(S, labels=arange(N))  # 每行对角线为正样本
   loss_proj = CrossEntropy(S.T, labels=arange(N))
   loss = (loss_mic + loss_proj) / 2
```

### 4.2 训练配置

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingWarmRestarts |
| Batch size | 64-128（适配 5070 Ti 12GB） |
| Epochs | ~100-200 |
| Mixed precision | 可选（冒烟测试先不用） |
| Temperature (τ) | 0.07 |

### 4.3 负样本策略

- **冒烟测试**: batch 内其他样本作为负样本
- **TODO**: Hard negative mining（相邻 viewing direction、相似 silhouette、对称相关 projection）
- **TODO**: 代码结构预留 hard negative 接口

### 4.4 预留接口

- **TODO**: Hard negative sampling
- **TODO**: DeepSpeed 多卡训练支持
- **TODO**: 多尺寸模型（128/192/256/512）

---

## 5. 评估与检索

### 5.1 评估流程

1. **建索引**：对所有 clean proj 编码 → L2 normalized embeddings → FAISS IndexFlatIP
2. **查询**：对 noisy mic 编码 → 在 FAISS 中检索 top-k
3. **计算指标**：Top-1/5/10/20 准确率

### 5.2 评估指标

- 正确匹配 = 检索到的 proj 中是否包含该 mic 对应的 ground truth proj
- 随机基线 = 1/N（N = 索引中的 proj 数量）
- 冒烟测试通过标准：top-1 > 5-10× random baseline

### 5.3 可视化

- 训练曲线（loss）
- 检索结果展示（query mic + top-k retrieved projs）
- Embedding 空间 t-SNE 可视化

---

## 6. 项目结构

```
Siamese/
├── siamese/                    # Python 包（可 pip install）
│   ├── __init__.py
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   ├── encoder.py          # 双分支编码器
│   │   ├── backbone.py         # ConvNeXt/ViT/Swin backbone
│   │   └── fusion.py           # 融合层
│   ├── data/                   # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py          # MicProjDataset
│   │   ├── generate.py         # 模拟数据生成脚本
│   │   └── transforms.py       # 预处理变换
│   ├── losses/                 # 损失函数
│   │   ├── __init__.py
│   │   └── infonce.py          # InfoNCE loss
│   ├── training/               # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py          # 训练循环
│   │   └── config.py           # 配置类
│   ├── eval/                   # 评估
│   │   ├── __init__.py
│   │   ├── retrieval.py        # FAISS 检索
│   │   └── metrics.py          # 评估指标
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── ctf.py              # CTF 生成
│       └── fft.py              # FFT 工具
├── scripts/                    # 运行脚本
│   ├── generate_data.py        # 生成模拟数据
│   ├── train.py                # 训练脚本
│   └── eval.py                 # 评估脚本
├── configs/                    # 配置文件
│   └── default.yaml
├── tests/                      # 测试
│   └── test_encoder.py
├── data/                       # 数据目录（gitignore）
├── checkpoints/                # 模型保存（gitignore）
├── pyproject.toml              # 项目配置
└── README.md
```

---

## 7. 实现顺序

1. **数据生成** — `generate_data.py` + `ctf.py`，从 volume 生成模拟数据
2. **模型** — 双分支编码器 + ConvNeXt backbone
3. **数据集** — `MicProjDataset` + 预处理
4. **Loss** — InfoNCE
5. **训练** — 训练循环 + 配置
6. **评估** — FAISS 检索 + 指标计算 + 可视化
7. **冒烟测试** — 端到端验证

---

## 8. 非目标（Out of Scope）

- 真实数据训练（后续阶段）
- 生产级推理服务
- 传统算法精修（top-k 重排）
- 多尺寸模型完整训练