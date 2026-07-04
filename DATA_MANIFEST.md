# 训练数据清单

本文件列出训练所需的所有数据文件及其来源。

## 数据集

### J8 (3.00Å)
- **颗粒**: `data/cs_processed/ribosome_J8/J8_particles/J8_particles_exported.cs` (元数据)
- **颗粒图像**: `/Data/cryoPPP/10406/CS-10406/` (实际 .mrcs 文件，由 .cs 中 blob/path 索引)
- **Volume**: `data/cs_processed/ribosome_J8/J8_volume/J8_000_volume_map.mrc`
- **数量**: 16108 train + 2014 val

### J20 (3.68Å)
- **颗粒**: `data/cs_processed/ribosome_J20/J20_particles/J20_particles_exported.cs`
- **颗粒图像**: `/Data/cryoPPP/10002/CS-10002/`
- **Volume**: `data/cs_processed/ribosome_J20/J20_volume/J20_000_volume_map.mrc`
- **数量**: 16416 train + 2052 val

### J31 (4.16Å)
- **颗粒**: `data/cs_processed/ribosome_homerefine/J31_particles/J31_particles_exported.cs`
- **颗粒图像**: `data/cs_processed/ribosome_homerefine/J31_particles/` (本地路径)
- **Volume**: `data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_004_volume_map.mrc`
- **数量**: 5186 train + 648 val

## 打包说明

训练数据打包为 `siamese_training_data.tar.gz`，包含：
1. 所有 `.cs` 文件（颗粒元数据）
2. 所有 `.mrc` volume 文件
3. 所有 `.mrcs` 颗粒图像栈

解压后需要：
1. 保持 `data/cs_processed/` 目录结构
2. 对于 J8/J20，如果远程路径不同，需要修改配置文件中的 `project_dir`
3. 对于 J31，颗粒图像已在本地 `data/cs_processed/ribosome_homerefine/J31_particles/`

## 配置文件

`configs/proposer_ribosome_multi.yaml` - 多数据集训练配置

远程训练时需要调整的路径：
- J8 `project_dir`: 指向解压后的 J8 颗粒图像目录
- J20 `project_dir`: 指向解压后的 J20 颗粒图像目录
