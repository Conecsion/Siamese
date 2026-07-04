# NaN 问题诊断和修复指南

## 📊 当前状态

✅ **训练正在运行**
- 进程数: 6 个
- 最新日志: `logs/train_20260704_203039.log` (4.2K)
- 诊断结果: **未检测到 NaN**

---

## 🔍 NaN 检测工具

### 1. 实时监控工具

```bash
# 在本地运行，实时监控远程训练
bash scripts/monitor_nan.sh
```

**功能：**
- ✅ 实时监控训练日志
- ✅ 检测到 NaN 自动触发诊断
- ✅ 保存完整的诊断快照
- ✅ 交互式修复提示

### 2. 日志诊断工具

```bash
# 分析已有的训练日志
python3 scripts/debug_nan.py --log logs/train_XXXXXX.log

# 生成调试代码补丁
python3 scripts/debug_nan.py --log logs/train_XXXXXX.log --generate-patch
```

**功能：**
- ✅ 解析日志找出 NaN 位置
- ✅ 分析最后正常的指标
- ✅ 检查配置文件
- ✅ 提供修复建议

### 3. 代码集成调试器

在训练脚本中集成：

```python
from scripts.nan_debugger import NaNDebugger

# 初始化
debugger = NaNDebugger(save_dir="debug_nan", enabled=True)

# 在训练循环中检查
for batch_idx, batch in enumerate(train_loader):
    # 检查输入数据
    has_nan, has_inf, _ = debugger.check_batch(batch, step=batch_idx)
    if has_nan or has_inf:
        continue  # 跳过异常 batch
    
    # 前向传播
    outputs = model(batch)
    
    # 检查输出
    has_nan, has_inf, _ = debugger.check_model_outputs(outputs, step=batch_idx)
    
    # 计算 loss
    loss = compute_loss(outputs)
    
    # 检查 loss
    loss_check = debugger.check_tensor(loss, 'loss', step=batch_idx)
    if loss_check['has_nan']:
        break  # 停止训练
```

---

## 🐛 常见 NaN 原因

### 1. **学习率过大** → 梯度爆炸
**症状：**
- Loss 在前几个 epoch 就变成 NaN
- 梯度值非常大 (>1000)

**修复：**
```yaml
# configs/proposer_ribosome_multi.yaml
learning_rate: 1e-5  # 降低学习率（原来可能是 1e-3）
```

### 2. **FP16 精度溢出** → 数值超出范围
**症状：**
- 使用 FP16 训练时出现 NaN
- Loss scale 调整失败

**修复：**
```json
// configs/ds_config.json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 12,  // 降低初始 scale (原来是 16)
    "loss_scale_window": 100,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

### 3. **除零错误** → 归一化层问题
**症状：**
- NaN 出现在归一化操作后
- 某些 batch 的标准差为 0

**修复：**
```python
# 在归一化前添加 eps
normalized = (x - mean) / (std + 1e-8)

# 使用安全的数学操作
from scripts.nan_debugger import safe_div, safe_log, safe_sqrt

loss = safe_log(predictions)
```

### 4. **输入数据异常** → 数据预处理问题
**症状：**
- 某些特定样本导致 NaN
- NaN 出现在前向传播的第一层

**修复：**
```python
# 检查并过滤异常数据
def collate_fn_with_nan_check(batch):
    # 过滤包含 NaN 的样本
    filtered_batch = []
    for item in batch:
        particle, proj, axisang, idx = item
        if not (torch.isnan(particle).any() or 
                torch.isnan(proj).any() or 
                torch.isnan(axisang).any()):
            filtered_batch.append(item)
    
    if len(filtered_batch) == 0:
        return None
    
    return default_collate(filtered_batch)

# 使用过滤后的 collate_fn
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn_with_nan_check
)
```

### 5. **温度参数过小** → exp 溢出
**症状：**
- 在计算 softmax 时出现 NaN
- Temperature 参数非常小

**修复：**
```yaml
# configs/proposer_ribosome_multi.yaml
temperature: 0.07  # 确保不要太小 (>0.01)
```

### 6. **梯度累积过多** → 数值累积误差
**症状：**
- 使用梯度累积时出现 NaN
- Loss 逐渐增大然后突然 NaN

**修复：**
```json
// configs/ds_config.json
{
  "gradient_accumulation_steps": 2,  // 减少累积步数 (原来可能是 4)
  "gradient_clipping": 1.0  // 启用梯度裁剪
}
```

---

## 🔧 推荐的预防措施

### 1. 配置文件优化

```yaml
# configs/proposer_ribosome_multi.yaml
learning_rate: 1e-4          # 保守的学习率
warmup_steps: 500            # 学习率预热
max_grad_norm: 1.0           # 梯度裁剪
temperature: 0.07            # 合理的温度参数
```

```json
// configs/ds_config.json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 2,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12,
    "loss_scale_window": 100
  }
}
```

### 2. 在训练脚本中添加保护

```python
# 1. 梯度裁剪
from torch.nn.utils import clip_grad_norm_
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 检查模型权重
from scripts.nan_debugger import check_model_weights
if not check_model_weights(model):
    print("❌ 模型权重异常，停止训练")
    break

# 3. 安全的数学操作
from scripts.nan_debugger import safe_log, safe_sqrt, safe_div

# 4. 异常值过滤
loss = torch.clamp(loss, min=-100, max=100)
```

### 3. 数据预处理验证

```python
# 在数据加载器中验证
def validate_batch(batch):
    particle, proj, axisang, _ = batch
    
    # 检查范围
    assert particle.min() >= -10 and particle.max() <= 10, "Particle 值异常"
    assert proj.min() >= -10 and proj.max() <= 10, "Projection 值异常"
    
    # 检查 NaN
    assert not torch.isnan(particle).any(), "Particle 包含 NaN"
    assert not torch.isnan(proj).any(), "Projection 包含 NaN"
    
    return True
```

---

## 📝 NaN 出现后的处理流程

### 第 1 步：确认 NaN 位置

```bash
# 运行诊断工具
python3 scripts/debug_nan.py --log logs/train_XXXXXX.log
```

### 第 2 步：检查保存的调试文件

```python
# 加载保存的异常 batch
debug_data = torch.load('debug_nan/nan_1_loss_step123.pt')

print("张量形状:", debug_data['shape'])
print("统计信息:", debug_data['stats'])
print("NaN 位置:", torch.where(torch.isnan(debug_data['tensor'])))
```

### 第 3 步：应用修复

根据诊断结果，应用对应的修复方案（见上文"常见 NaN 原因"）

### 第 4 步：验证修复

```bash
# 重启训练
bash remote_train.sh restart

# 启动实时监控
bash scripts/monitor_nan.sh
```

### 第 5 步：如果仍然出现 NaN

```bash
# 1. 切换到更保守的配置
cp configs/proposer_ribosome_multi.yaml configs/proposer_ribosome_multi.yaml.backup

# 2. 降低学习率到 1e-5
# 3. 禁用 FP16（在 ds_config.json 中设置 "fp16": {"enabled": false}）
# 4. 减小 batch size
# 5. 启用更严格的梯度裁剪
```

---

## 🎯 快速修复清单

如果 loss 变成 NaN，按以下顺序尝试：

- [ ] **1. 降低学习率** → `lr = 1e-5`
- [ ] **2. 启用梯度裁剪** → `gradient_clipping = 1.0`
- [ ] **3. 调整 FP16 设置** → `initial_scale_power = 12`
- [ ] **4. 减少梯度累积** → `gradient_accumulation_steps = 2`
- [ ] **5. 检查输入数据** → 使用 `NaNDebugger`
- [ ] **6. 添加数值稳定性** → 使用 `safe_log`, `safe_div`
- [ ] **7. 如果都不行** → 禁用 FP16，使用 FP32 训练

---

## 📞 技术支持

如果问题仍未解决，请提供以下信息：

1. **诊断报告输出**
   ```bash
   python3 scripts/debug_nan.py --log logs/train_XXXXXX.log > nan_report.txt
   ```

2. **训练配置文件**
   - `configs/proposer_ribosome_multi.yaml`
   - `configs/ds_config.json`

3. **GPU 状态**
   ```bash
   nvidia-smi > gpu_status.txt
   ```

4. **完整的训练日志**
   ```bash
   scp shaodi@106:/data/shaodi/Siamese/logs/train_*.log ./
   ```

5. **保存的调试文件**（如果有）
   - `debug_nan/*.pt`

---

## 📚 相关文档

- [debug_nan.py](scripts/debug_nan.py) - NaN 诊断工具
- [nan_debugger.py](scripts/nan_debugger.py) - NaN 调试器类
- [monitor_nan.sh](scripts/monitor_nan.sh) - 实时监控脚本
- [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) - 训练部署总结

---

生成时间: 2026-07-04
状态: ✅ 工具就绪，监控中
