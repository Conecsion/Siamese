#!/usr/bin/env python3
"""
修复 CryoSPARC .cs 文件中的路径，使其指向正确的数据位置
"""

import numpy as np
import shutil
from pathlib import Path

def fix_cs_paths(cs_path: str, prefix: str, backup: bool = True):
    """
    修复 .cs 文件中的 blob/path 字段

    Args:
        cs_path: .cs 文件路径
        prefix: 要添加的路径前缀 (如 'data/particles/J8/')
        backup: 是否备份原文件
    """
    cs_path = Path(cs_path)

    if not cs_path.exists():
        print(f"错误: 文件不存在 {cs_path}")
        return

    # 备份原文件
    if backup:
        backup_path = cs_path.with_suffix('.cs.backup')
        if not backup_path.exists():
            shutil.copy(cs_path, backup_path)
            print(f"✓ 已备份: {backup_path}")

    # 加载数据
    data = np.load(cs_path)

    if 'blob/path' not in data.dtype.names:
        print(f"警告: {cs_path} 中没有 'blob/path' 字段")
        return

    # 修复路径
    original_paths = data['blob/path'].copy()
    fixed_count = 0

    # 首先计算所有新路径并找到最大长度
    new_paths = []
    for path in original_paths:
        path_str = path.decode() if isinstance(path, bytes) else str(path)
        # 移除可能的前缀字符（如 '>'）
        path_str = path_str.lstrip('>')
        # 添加正确的前缀
        new_path = prefix + path_str
        new_paths.append(new_path)
        fixed_count += 1

    # 计算需要的最大长度
    max_len = max(len(p) for p in new_paths)
    print(f"  最大路径长度: {max_len}")

    # 创建新的 dtype，为 blob/path 字段分配足够空间
    new_dtype = []
    for name in data.dtype.names:
        if name == 'blob/path':
            new_dtype.append((name, f'S{max_len}'))
        else:
            new_dtype.append((name, data.dtype.fields[name][0]))

    # 创建新的结构化数组
    new_data = np.empty(len(data), dtype=new_dtype)
    for field in data.dtype.names:
        if field == 'blob/path':
            # 转换为字节数组
            new_data['blob/path'] = np.array([p.encode() for p in new_paths], dtype=f'S{max_len}')
        else:
            new_data[field] = data[field]

    # 保存修复后的文件
    # 注意: .cs 文件就是 .npy 格式，但需要确保保存时不添加 .npy 扩展名
    output_path = str(cs_path).replace('.cs', '_fixed.cs')
    np.save(output_path, new_data)

    # 删除旧文件并重命名
    cs_path.unlink()
    Path(output_path + '.npy').rename(cs_path)

    print(f"✓ 修复完成: {cs_path}")
    print(f"  修复路径数: {fixed_count}")
    if len(original_paths) > 0:
        old_path = original_paths[0].decode() if isinstance(original_paths[0], bytes) else str(original_paths[0])
        print(f"  示例: {old_path} → {new_paths[0]}")


def main():
    # 定义需要修复的文件和对应的路径前缀
    fixes = [
        {
            'name': 'J8',
            'cs_path': 'data/cs_processed/ribosome_J8/J8_particles/J8_particles_exported.cs',
            'prefix': 'data/particles/J8/'
        },
        {
            'name': 'J20',
            'cs_path': 'data/cs_processed/ribosome_J20/J20_particles/J20_particles_exported.cs',
            'prefix': 'data/particles/J20/'
        },
        {
            'name': 'J31',
            'cs_path': 'data/cs_processed/ribosome_homerefine/J31_particles/J31_particles_exported.cs',
            'prefix': 'data/particles/J31/'
        }
    ]

    print("=" * 70)
    print("修复 CryoSPARC .cs 文件路径")
    print("=" * 70)
    print()

    for fix in fixes:
        print(f"[{fix['name']}]")
        fix_cs_paths(fix['cs_path'], fix['prefix'])
        print()

    print("=" * 70)
    print("✅ 所有文件修复完成！")
    print()
    print("备份文件: *.cs.backup (如需恢复，重命名为 .cs)")
    print("=" * 70)


if __name__ == '__main__':
    main()
