#!/bin/bash
# 打包所有训练数据
# 注意：总大小约 20-25GB，需要足够的磁盘空间

set -e

OUTPUT_DIR="/Data/Work/Siamese/training_data_pack"
ARCHIVE_NAME="siamese_training_data.tar.gz"

echo "=== 开始打包训练数据 ==="
echo "输出目录: $OUTPUT_DIR"
echo ""

# 创建打包目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# 1. 复制元数据和 volume
echo "1. 复制 .cs 元数据和 volume .mrc 文件..."
mkdir -p data/cs_processed
cp -r /Data/Work/Siamese/data/cs_processed/ribosome_J8/ data/cs_processed/
cp -r /Data/Work/Siamese/data/cs_processed/ribosome_J20/ data/cs_processed/
cp -r /Data/Work/Siamese/data/cs_processed/ribosome_homerefine/ data/cs_processed/
echo "✓ 元数据和 volume 已复制"

# 2. 复制颗粒图像
echo ""
echo "2. 复制颗粒图像文件（这会花费较长时间）..."

echo "  - J8 颗粒..."
mkdir -p data/particles/J8/J4/imported
cp /Data/cryoPPP/10406/CS-10406/J4/imported/*.mrc data/particles/J8/J4/imported/
echo "    J8: $(ls data/particles/J8/J4/imported/*.mrc | wc -l) 文件"

echo "  - J20 颗粒..."
mkdir -p data/particles/J20/J16/extract
cp /Data/cryoPPP/10002/CS-10002/J16/extract/*.mrc data/particles/J20/J16/extract/
echo "    J20: $(ls data/particles/J20/J16/extract/*.mrc | wc -l) 文件"

echo "  - J31 颗粒..."
mkdir -p data/particles/J31/J27/extract
cp /Data/Work/Siamese/data/cs_processed/ribosome_homerefine/J31_particles/J27/extract/*.mrc data/particles/J31/J27/extract/
echo "    J31: $(ls data/particles/J31/J27/extract/*.mrc | wc -l) 文件"

echo "✓ 颗粒图像已复制"

# 3. 复制配置文件和说明
echo ""
echo "3. 复制配置文件..."
cp /Data/Work/Siamese/configs/proposer_ribosome_multi.yaml .
cp /Data/Work/Siamese/DATA_MANIFEST.md .

# 创建远程训练说明
cat > REMOTE_SETUP.md << 'EOF'
# 远程训练设置说明

## 1. 解压数据

```bash
tar -xzf siamese_training_data.tar.gz
cd siamese_training_data
```

## 2. 修改配置文件

编辑 `proposer_ribosome_multi.yaml`，修改 `project_dir` 路径：

```yaml
datasets:
  - name: "J8_3.00A"
    cs_path: "data/cs_processed/ribosome_J8/J8_particles/J8_particles_exported.cs"
    reference_path: "data/cs_processed/ribosome_J8/J8_volume/J8_000_volume_map.mrc"
    project_dir: "data/particles/J8"  # 改为解压后的路径

  - name: "J20_3.68A"
    cs_path: "data/cs_processed/ribosome_J20/J20_particles/J20_particles_exported.cs"
    reference_path: "data/cs_processed/ribosome_J20/J20_volume/J20_000_volume_map.mrc"
    project_dir: "data/particles/J20"  # 改为解压后的路径

  - name: "J31_4.16A"
    cs_path: "data/cs_processed/ribosome_homerefine/J31_particles/J31_particles_exported.cs"
    reference_path: "data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_004_volume_map.mrc"
    project_dir: "data/particles/J31"  # 改为解压后的路径
```

## 3. 安装依赖

```bash
# 克隆代码仓库
git clone https://github.com/YOUR_USERNAME/Siamese.git
cd Siamese

# 安装依赖
uv pip install -e ".[dev]"
# 或
pip install -e ".[dev]"
```

## 4. 将数据移动到代码目录

```bash
# 假设数据在 ~/siamese_training_data，代码在 ~/Siamese
cp -r ~/siamese_training_data/data ~/Siamese/
cp ~/siamese_training_data/proposer_ribosome_multi.yaml ~/Siamese/configs/
```

## 5. 启动训练

```bash
cd ~/Siamese
python scripts/train_proposer.py --config configs/proposer_ribosome_multi.yaml
```

## 6. TensorBoard 监控

```bash
tensorboard --logdir checkpoints_proposer_multi/tensorboard --port 6006 --bind_all
```

访问 http://YOUR_SERVER_IP:6006
EOF

echo "✓ 配置文件已复制"

# 4. 显示目录结构和大小
echo ""
echo "=== 打包内容 ==="
du -sh data/
du -sh .

# 5. 打包压缩
echo ""
echo "4. 压缩打包（这会花费较长时间）..."
cd ..
tar -czf "$ARCHIVE_NAME" -C "$OUTPUT_DIR" .

echo ""
echo "=== 打包完成 ==="
echo "文件位置: $(pwd)/$ARCHIVE_NAME"
ls -lh "$ARCHIVE_NAME"
echo ""
echo "下一步："
echo "1. 将 $ARCHIVE_NAME 传输到远程服务器"
echo "2. 推送代码到 GitHub: git push -u origin main"
echo "3. 在远程服务器上按照 REMOTE_SETUP.md 设置"
