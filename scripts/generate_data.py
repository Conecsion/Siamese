#!/usr/bin/env python
"""
生成模拟训练数据。

用法:
    python scripts/generate_data.py emd_19110.map 8 --output-dir data/ --image-size 128
    python scripts/generate_data.py emd_19110.map 8 --export-format cryosparc --output-dir data/
"""

import argparse

from siamese.data.generate import generate_simulated_data


def main():
    p = argparse.ArgumentParser(description="Generate simulated mic-proj data from 3D volume.")
    p.add_argument("map_path", type=str)
    p.add_argument("nside", type=int, default=8, help="HEALPix nside (N=12*nside^2).")
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--orientation-mode", choices=["healpix", "uniform"], default="healpix")
    p.add_argument("--n-inplane", type=int, default=1)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--pixel-size", type=float, default=1.0)
    p.add_argument("--num-mics-per-proj", type=int, default=2)
    p.add_argument("--snr-min", type=float, default=0.001)
    p.add_argument("--snr-max", type=float, default=0.01)
    p.add_argument("--defocus-min", type=float, default=0.5, help="μm")
    p.add_argument("--defocus-max", type=float, default=4.0, help="μm")
    p.add_argument("--max-shift", type=float, default=5.0, help="pixels")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk-size", type=int, default=None, help="batch分块; None=自动")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-format", choices=["none", "cryosparc", "relion"], default="none")
    a = p.parse_args()

    generate_simulated_data(
        map_path=a.map_path, nside=a.nside, output_dir=a.output_dir,
        orientation_mode=a.orientation_mode, n_inplane=a.n_inplane,
        image_size=a.image_size, pixel_size=a.pixel_size,
        num_mics_per_proj=a.num_mics_per_proj,
        snr_range=(a.snr_min, a.snr_max), defocus_range=(a.defocus_min, a.defocus_max),
        max_shift_pixels=a.max_shift, device=a.device,
        chunk_size=a.chunk_size, seed=a.seed, export_format=a.export_format,
    )


if __name__ == "__main__":
    main()
