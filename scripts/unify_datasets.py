"""
统一三套数据的 pixel size 和 box size（方案 A: 降采样）。

步骤:
1. Volume: crop 去空白 -> resample 到统一 psize -> crop/pad 到统一 box
2. .cs 文件: 修改 blob/psize_A 为统一 psize
3. Particles: 保持原始文件不动，但 .cs 记录的 psize 改为统一值

关键: 投影和颗粒的 pixel size 必须一致，避免物理尺寸不匹配。
"""

import mrcfile
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path


def crop_center(vol, new_size):
    """中心 crop 到 new_size³。"""
    if vol.shape[0] == new_size:
        return vol
    start = (vol.shape[0] - new_size) // 2
    return vol[start:start+new_size, start:start+new_size, start:start+new_size]


def pad_center(vol, new_size):
    """pad 到 new_size³，填充均值。"""
    if vol.shape[0] == new_size:
        return vol
    pad_w = (new_size - vol.shape[0]) // 2
    padded = np.pad(vol, pad_w, mode='constant', constant_values=vol.mean())
    return padded[:new_size, :new_size, :new_size]


def process_volume(input_path, output_path, crop_size, target_psize, target_box):
    """
    处理 volume: crop -> resample -> crop/pad。

    参数:
        input_path: 输入 mrc
        output_path: 输出 mrc
        crop_size: 第一步 crop 尺寸（去空白）
        target_psize: 目标 pixel size (Å)
        target_box: 最终 box size
    """
    with mrcfile.open(input_path, permissive=True) as mrc:
        vol = mrc.data.copy()
        orig_psize = float(mrc.voxel_size.x)
        orig_box = vol.shape[0]

        print(f"原始: {orig_box}³ @ {orig_psize:.3f}Å = {orig_box*orig_psize:.1f}Å")

        # Step 1: crop 去空白
        if crop_size < orig_box:
            vol = crop_center(vol, crop_size)
            print(f"Crop: {crop_size}³ @ {orig_psize:.3f}Å = {crop_size*orig_psize:.1f}Å")

        # Step 2: resample 到目标 pixel size
        scale = orig_psize / target_psize
        new_box = int(np.round(vol.shape[0] * scale))
        vol = zoom(vol, scale, order=1)
        print(f"Resample: {new_box}³ @ {target_psize:.3f}Å = {new_box*target_psize:.1f}Å")

        # Step 3: crop/pad 到目标 box
        if new_box > target_box:
            vol = crop_center(vol, target_box)
            print(f"Crop: {target_box}³")
        elif new_box < target_box:
            vol = pad_center(vol, target_box)
            print(f"Pad: {target_box}³")

        print(f"最终: {vol.shape} @ {target_psize:.3f}Å\n")

        # 保存
        with mrcfile.new(output_path, overwrite=True) as out_mrc:
            out_mrc.set_data(vol.astype(np.float32))
            out_mrc.voxel_size = target_psize


def update_cs_psize(cs_path, new_psize):
    """修改 .cs 文件的 blob/psize_A 为新 pixel size。"""
    cs = np.load(cs_path)
    # 创建新结构（修改 blob/psize_A）
    old_psize = cs['blob/psize_A'][0]
    cs['blob/psize_A'] = new_psize
    np.save(cs_path, cs)
    print(f"Updated {Path(cs_path).name}: blob/psize_A {old_psize:.3f} -> {new_psize:.3f}Å")


if __name__ == "__main__":
    # 目标: 统一到 1.34Å (J31 原生) + 256³ box
    TARGET_PSIZE = 1.34
    TARGET_BOX = 256

    tasks = [
        # (name, input_vol, output_vol, crop_size, cs_path)
        ("J8",
         "data/cs_processed/ribosome_J8/J8_volume/J8_000_volume_map.mrc",
         "data/cs_processed/ribosome_J8/J8_volume/J8_volume_processed.mrc",
         390,  # crop 到 390³ 保留信号（414Å信号 / 1.07Å ≈ 387px，留余量）
         "data/cs_processed/ribosome_J8/J8_particles/J8_particles_exported.cs"),

        ("J20",
         "data/cs_processed/ribosome_J20/J20_volume/J20_000_volume_map.mrc",
         "data/cs_processed/ribosome_J20/J20_volume/J20_volume_processed.mrc",
         220,  # crop 到 220³（347Å信号 / 1.77Å ≈ 196px，留余量）
         "data/cs_processed/ribosome_J20/J20_particles/J20_particles_exported.cs"),

        ("J31",
         "data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_004_volume_map.mrc",
         "data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_volume_processed.mrc",
         280,  # crop 到 280³（365Å信号 / 1.34Å ≈ 272px，留余量）
         "data/cs_processed/ribosome_homerefine/J31_particles/J31_particles_exported.cs"),
    ]

    for name, inp_vol, out_vol, crop_sz, cs_path in tasks:
        print(f"=== {name} ===")
        process_volume(inp_vol, out_vol, crop_sz, TARGET_PSIZE, TARGET_BOX)
        update_cs_psize(cs_path, TARGET_PSIZE)
        print()
