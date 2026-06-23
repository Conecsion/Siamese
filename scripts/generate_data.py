#!/usr/bin/env python
"""
生成模拟训练数据。

用法:
    python scripts/generate_data.py emd_19110.map 8 --output-dir data/ --image-size 128
"""

import argparse
from pathlib import Path

from siamese.data.generate import generate_simulated_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated mic-proj paired data from 3D volume."
    )
    parser.add_argument("map_path", type=str, help="Path to .map 3D volume file.")
    parser.add_argument("nside", type=int, default=8,
                        help="HEALPix nside (directions = 12 * nside^2).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for generated data.")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image size D (square).")
    parser.add_argument("--pixel-size", type=float, default=1.0,
                        help="Pixel size in Angstrom.")
    parser.add_argument("--num-mics-per-proj", type=int, default=2,
                        help="Number of noisy mics per clean proj.")
    parser.add_argument("--snr-min", type=float, default=0.001,
                        help="Minimum SNR for noise.")
    parser.add_argument("--snr-max", type=float, default=0.01,
                        help="Maximum SNR for noise.")
    parser.add_argument("--defocus-min", type=float, default=0.5,
                        help="Minimum defocus in um.")
    parser.add_argument("--defocus-max", type=float, default=4.0,
                        help="Maximum defocus in um.")
    parser.add_argument("--max-shift", type=float, default=5.0,
                        help="Maximum in-plane shift in pixels.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device.")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="HEALPix projection chunk size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    generate_simulated_data(
        map_path=args.map_path,
        nside=args.nside,
        output_dir=args.output_dir,
        image_size=args.image_size,
        pixel_size=args.pixel_size,
        num_mics_per_proj=args.num_mics_per_proj,
        snr_range=(args.snr_min, args.snr_max),
        defocus_range=(args.defocus_min, args.defocus_max),
        max_shift_pixels=args.max_shift,
        device=args.device,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()