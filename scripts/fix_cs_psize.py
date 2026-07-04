"""修复 .cs 文件的 blob/psize_A 字段。"""

import numpy as np
from pathlib import Path


def update_cs_psize_correct(cs_path, new_psize):
    """正确修改 .cs 文件的 blob/psize_A。"""
    cs = np.load(cs_path)
    old_psize = cs['blob/psize_A'][0]

    # 创建新数组，复制所有数据
    cs_new = cs.copy()
    cs_new['blob/psize_A'] = new_psize

    # 保存（覆盖原文件）
    np.save(cs_path, cs_new)

    # 验证
    cs_check = np.load(cs_path)
    actual_psize = cs_check['blob/psize_A'][0]

    print(f"{Path(cs_path).name}: {old_psize:.3f} -> {actual_psize:.3f}Å {'✓' if abs(actual_psize - new_psize) < 0.01 else '✗'}")


if __name__ == "__main__":
    TARGET_PSIZE = 1.34

    cs_files = [
        "data/cs_processed/ribosome_J8/J8_particles/J8_particles_exported.cs",
        "data/cs_processed/ribosome_J20/J20_particles/J20_particles_exported.cs",
        "data/cs_processed/ribosome_homerefine/J31_particles/J31_particles_exported.cs",
    ]

    for cs_path in cs_files:
        update_cs_psize_correct(cs_path, TARGET_PSIZE)
