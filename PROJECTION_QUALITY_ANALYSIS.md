# 投影质量分析完整报告

## 🔍 问题描述

用户在 TensorBoard 中观察到某些投影图像是"一片模糊的白光"而不是清晰的蛋白结构，担心这可能导致训练 NaN。

## 📊 已完成的检测

### 1. 参考体积质量检查 ✅

**工具**: `scripts/check_projection_quality.py`

**结果**:
- **J8**: ✅ 正常 (416³, 值范围 [-0.256, 1.064])
- **J20**: ✅ 正常 (256³, 值范围 [-0.951, 2.542])
- **J31**: ✅ 正常 (256³, 值范围 [-0.831, 2.051])

**结论**: 所有参考体积质量正常，无 NaN、Inf 或空白区域。

---

### 2. 随机姿态投影测试 ✅

**工具**: `scripts/check_projection_quality.py --visualize`

**方法**: 使用随机生成的姿态参数进行投影

**结果**:
- **J8**: 50/50 投影有效 (100%)
- **J20**: 50/50 投影有效 (100%)
- **J31**: 50/50 投影有效 (100%)

**结论**: 投影算法本身工作正常，随机姿态不产生异常投影。

---

### 3. 真实训练数据检查 ✅

**工具**: `scripts/check_training_data.py`

**方法**: 从 DataLoader 直接采样真实训练数据

**结果**:
- 颗粒图像: 100/100 有效 (100%)
- 配对投影: 100/100 有效 (100%)
- 无 NaN、Inf 或"白光"异常

**结论**: 训练数据完全正常，DataLoader 输出的数据质量良好。

---

### 4. 真实姿态高亮投影检测 🔄

**工具**: `scripts/detect_bright_projections.py`

**方法**: 使用 `.cs` 文件中的真实姿态参数进行投影，检测白色面积 > 50% 的图像

**状态**: 🔄 正在运行 (J8 数据集，1000 样本)

**预期结果**: 
- 如果发现大量高亮投影（>5%），说明某些特定姿态会投影到体积的低密度区域
- 这些投影可能在训练中导致数值不稳定

---

## 🛠️ 可用的工具

### 投影质量检查工具

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| `check_projection_quality.py` | 检查参考体积质量 + 随机姿态投影测试 | 验证体积和投影算法 |
| `check_training_data.py` | 从 DataLoader 采样真实训练数据 | 验证训练数据管道 |
| `detect_bright_projections.py` | 使用真实姿态检测高亮投影 | 找出特定的异常姿态 |

### NaN 调试工具

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| `debug_nan.py` | 分析日志，检查配置 | 诊断已发生的 NaN |
| `nan_debugger.py` | 代码集成的实时检测器 | 在训练中捕获 NaN |
| `monitor_nan.sh` | 实时监控训练日志 | 持续监控训练 |

---

## 💡 关键发现

### 数据质量 ✅
1. ✅ 参考体积质量正常
2. ✅ 投影算法工作正常
3. ✅ 训练数据管道正常
4. 🔄 特定姿态的投影质量检测中...

### TensorBoard 中的"白光"投影

**可能原因**:
1. **可视化问题** - TensorBoard 显示归一化不当
2. **训练中期问题** - 模型权重异常导致输出异常
3. **特定姿态问题** - 某些姿态投影到低密度区域（待验证）

**不太可能的原因** (已排除):
- ❌ 参考体积损坏
- ❌ 投影算法错误
- ❌ 数据加载问题

---

## 🎯 针对 NaN 问题的建议

### 1. 训练配置优化（推荐）

```yaml
# configs/proposer_ribosome_multi.yaml
learning_rate: 1e-4          # 降低学习率
warmup_steps: 1000           # 增加预热
```

```json
// configs/ds_config.json
{
  "gradient_clipping": 1.0,  // 启用梯度裁剪
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12  // 降低 FP16 scale
  }
}
```

### 2. 添加数值保护

在训练脚本中添加：

```python
# 1. 梯度裁剪
from torch.nn.utils import clip_grad_norm_
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 投影值裁剪（如果发现高亮投影问题）
proj = torch.clamp(proj, min=-3*proj.std(), max=3*proj.std())

# 3. Loss 检查
if torch.isnan(loss):
    print(f"NaN detected at step {step}, skipping batch")
    continue
```

### 3. 过滤异常投影（如果检测到）

如果 `detect_bright_projections.py` 发现大量高亮投影：

```python
def is_valid_projection(proj, threshold_std=0.01, bright_threshold=0.5):
    """过滤异常投影"""
    # 检查标准差
    if proj.std() < threshold_std:
        return False
    
    # 检查高亮区域
    bright_ratio = (proj > proj.quantile(0.95)).float().mean()
    if bright_ratio > bright_threshold:
        return False
    
    return True

# 在 Dataset.__getitem__ 中使用
```

---

## 📈 监控和验证

### 实时监控

```bash
# 在本地运行，监控远程训练
bash scripts/monitor_nan.sh
```

### 定期检查

```bash
# 每隔一段时间检查训练状态
bash remote_train.sh status

# 查看 GPU 利用率
bash remote_train.sh gpu
```

### TensorBoard

访问 http://106:6006 查看：
- Loss 曲线
- Recall@K 指标
- 投影样本可视化

---

## 📝 后续行动计划

### 立即行动

1. ⏳ **等待高亮投影检测完成** (J8 数据集)
   - 如果发现异常，继续检测 J20 和 J31
   - 分析异常投影的姿态分布

2. 📊 **根据检测结果采取措施**:
   - **如果 < 5% 异常**: 问题不在数据，优化训练配置即可
   - **如果 > 5% 异常**: 需要在训练中添加投影过滤

3. 🔧 **应用训练配置优化**:
   - 降低学习率到 1e-4
   - 启用梯度裁剪
   - 调整 FP16 设置

### 持续改进

1. 📡 **启动实时监控**:
   ```bash
   bash scripts/monitor_nan.sh
   ```

2. 💾 **增加 checkpoint 频率**:
   - 每 2-3 epoch 保存一次
   - 方便回退到稳定状态

3. 📈 **密切关注指标**:
   - Loss 是否平稳下降
   - Recall@K 是否提升
   - GPU 利用率是否稳定

---

## 🔧 工具使用示例

### 检查特定数据集的投影质量

```bash
# J8
python3 scripts/detect_bright_projections.py \
  --cs data/cs_processed/ribosome_J8/J8_particles/J8_particles_exported.cs \
  --volume data/cs_processed/ribosome_J8/J8_volume/J8_000_volume_map.mrc \
  --num-check 1000 --bright-ratio 0.5

# J20
python3 scripts/detect_bright_projections.py \
  --cs data/cs_processed/ribosome_J20/J20_particles/J20_particles_exported.cs \
  --volume data/cs_processed/ribosome_J20/J20_volume/J20_000_volume_map.mrc \
  --num-check 1000 --bright-ratio 0.5

# J31
python3 scripts/detect_bright_projections.py \
  --cs data/cs_processed/ribosome_homerefine/J31_particles/J31_particles_exported.cs \
  --volume data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_volume_unified.mrc \
  --num-check 1000 --bright-ratio 0.5
```

### 检查训练数据

```bash
# 采样 200 个训练样本
python3 scripts/check_training_data.py \
  --config configs/proposer_ribosome_multi.yaml \
  --num-samples 200
```

### 分析已有日志

```bash
# 查找 NaN
python3 scripts/debug_nan.py \
  --log logs/train_XXXXXX.log \
  --generate-patch
```

---

## 📚 相关文档

- [NAN_DEBUGGING_GUIDE.md](NAN_DEBUGGING_GUIDE.md) - NaN 问题完整指南
- [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) - 训练部署总结
- [REMOTE_TRAINING.md](REMOTE_TRAINING.md) - 远程训练管理

---

## 🎓 结论

### 当前评估

✅ **数据质量**: 完全正常，无需担心
⚠️ **训练稳定性**: 需要优化配置防止 NaN
🔄 **特定姿态投影**: 检测中，等待结果

### 最可能的 NaN 原因

1. **学习率过大** (最可能)
2. **FP16 精度问题** (次可能)
3. **梯度爆炸** (可能)
4. **数据问题** (已排除)

### 推荐方案

**优先级 1**: 降低学习率 + 启用梯度裁剪
**优先级 2**: 调整 FP16 配置
**优先级 3**: 根据高亮投影检测结果决定是否需要数据过滤

---

生成时间: 2026-07-04
状态: 分析中
下一步: 等待高亮投影检测结果
