"""预处理多数据集的 volume：统一 pixel size + crop 到最小 box size。"""

import mrcfile
import numpy as np
import torch
from scipy.ndimage import zoom


def resample_and_crop_volume(input_mrc: str, output_mrc: str,
                              target_psize: float, target_box: int):
    """
    将 volume 重采样到目标 pixel size，然后中心 crop 到目标 box size。

    参数:
        input_mrc: 输入 mrc 文件路径
        output_mrc: 输出 mrc 文件路径
        target_psize: 目标像素大小 (Å)
        target_box: 目标 box size (立方体边长)
    """
    with mrcfile.open(input_mrc, permissive=True) as mrc:
        vol = mrc.data.copy()
        orig_psize = float(mrc.voxel_size.x)
        orig_box = vol.shape[0]

        print(f"输入: {orig_box}³ @ {orig_psize:.2f}Å = {orig_box*orig_psize:.1f}Å 物理尺寸")

        # Step 1: 重采样到目标 pixel size
        scale = orig_psize / target_psize
        new_box = int(np.round(orig_box * scale))
        vol_resampled = zoom(vol, scale, order=1)  # trilinear
        print(f"重采样: {new_box}³ @ {target_psize:.2f}Å = {new_box*target_psize:.1f}Å")

        # Step 2: 中心 crop/pad 到目标 box size
        if new_box > target_box:
            # crop
            start = (new_box - target_box) // 2
            vol_final = vol_resampled[start:start+target_box,
                                     start:start+target_box,
                                     start:start+target_box]
            print(f"中心 crop: {target_box}³")
        elif new_box < target_box:
            # pad with mean
            pad_w = (target_box - new_box) // 2
            vol_final = np.pad(vol_resampled, pad_w, mode='constant',
                              constant_values=vol_resampled.mean())
            vol_final = vol_final[:target_box, :target_box, :target_box]
            print(f"pad: {target_box}³")
        else:
            vol_final = vol_resampled

        print(f"输出: {vol_final.shape} @ {target_psize:.2f}Å\n")

        # 保存
        with mrcfile.new(output_mrc, overwrite=True) as out_mrc:
            out_mrc.set_data(vol_final.astype(np.float32))
            out_mrc.voxel_size = target_psize


if __name__ == "__main__":
    # 目标: 统一到 1.34Å (J31 原生) + 256³ box (最小的)
    TARGET_PSIZE = 1.34
    TARGET_BOX = 256

    tasks = [
        ("J8", "data/cs_processed/ribosome_J8/J8_volume/J8_000_volume_map.mrc",
               "data/cs_processed/ribosome_J8/J8_volume/J8_volume_unified.mrc"),
        ("J20", "data/cs_processed/ribosome_J20/J20_volume/J20_000_volume_map.mrc",
                "data/cs_processed/ribosome_J20/J20_volume/J20_volume_unified.mrc"),
        ("J31", "data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_004_volume_map.mrc",
                "data/cs_processed/ribosome_homerefine/J31_volume/J31/J31_volume_unified.mrc"),
    ]

    for name, inp, out in tasks:
        print(f"=== {name} ===")
        resample_and_crop_volume(inp, out, TARGET_PSIZE, TARGET_BOX)
