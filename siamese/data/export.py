"""
可选导出适配器: 将轴角 + mrcs 导出为 cryoSPARC (.cs) 或 RELION (.star) 格式。

均为纯写函数, 不依赖 cryoSPARC 可执行文件; 轴角到 ZYZ 欧拉角转换使用 pyem.geom。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def write_cryosparc_cs(
    axis_angles: np.ndarray,   # [N, 3] float32, 轴角向量 (弧度)
    cs_path: Path,
    mrcs_path: Path,
    pixel_size: float,
    box_size: int | None = None,
    voltage: float = 300.0,
    cs_mm: float = 2.7,
    amp_contrast: float = 0.07,
) -> None:
    """写 cryoSPARC .cs 格式 metadata (pose 存轴角, 即 cryoSPARC 原生约定)。"""
    N = len(axis_angles)
    if box_size is None:
        import mrcfile  # type: ignore[import-untyped]
        with mrcfile.open(str(mrcs_path), permissive=True) as mrc:
            box_size = mrc.data.shape[1]
    mrcs_name = mrcs_path.name.encode("ascii")[:31]

    dtype = np.dtype([
        ("uid", "<u8"), ("blob/path", "S31"), ("blob/idx", "<u4"),
        ("blob/shape", "<u4", (2,)), ("blob/psize_A", "<f4"),
        ("blob/sign", "<f4"), ("blob/import_sig", "<u8"),
        ("ctf/type", "S9"), ("ctf/exp_group_id", "<u4"),
        ("ctf/accel_kv", "<f4"), ("ctf/cs_mm", "<f4"),
        ("ctf/amp_contrast", "<f4"), ("ctf/df1_A", "<f4"),
        ("ctf/df2_A", "<f4"), ("ctf/df_angle_rad", "<f4"),
        ("ctf/phase_shift_rad", "<f4"), ("ctf/scale", "<f4"),
        ("ctf/scale_const", "<f4"), ("ctf/shift_A", "<f4", (2,)),
        ("ctf/tilt_A", "<f4", (2,)), ("ctf/trefoil_A", "<f4", (2,)),
        ("ctf/tetra_A", "<f4", (4,)), ("ctf/anisomag", "<f4", (4,)),
        ("ctf/bfactor", "<f4"), ("alignments3D/split", "<u4"),
        ("alignments3D/shift", "<f4", (2,)), ("alignments3D/pose", "<f4", (3,)),
        ("alignments3D/psize_A", "<f4"), ("alignments3D/error", "<f4"),
        ("alignments3D/error_min", "<f4"), ("alignments3D/resid_pow", "<f4"),
        ("alignments3D/slice_pow", "<f4"), ("alignments3D/image_pow", "<f4"),
        ("alignments3D/cross_cor", "<f4"), ("alignments3D/alpha", "<f4"),
        ("alignments3D/alpha_min", "<f4"), ("alignments3D/weight", "<f4"),
        ("alignments3D/pose_ess", "<f4"), ("alignments3D/shift_ess", "<f4"),
        ("alignments3D/class_posterior", "<f4"), ("alignments3D/class", "<u4"),
        ("alignments3D/class_ess", "<f4"),
    ])
    d = np.zeros(N, dtype=dtype)
    d["uid"] = np.arange(N, dtype=np.uint64)
    d["blob/path"] = np.full(N, mrcs_name, dtype="S31")
    d["blob/idx"] = np.arange(N, dtype=np.uint32)
    d["blob/shape"] = np.tile(np.array([box_size, box_size], dtype=np.uint32), (N, 1))
    d["blob/psize_A"] = pixel_size
    d["blob/sign"] = 1.0
    d["ctf/type"] = b"imported"
    d["ctf/accel_kv"] = voltage
    d["ctf/cs_mm"] = cs_mm
    d["ctf/amp_contrast"] = amp_contrast
    d["ctf/scale"] = 1.0; d["ctf/scale_const"] = 1.0
    d["ctf/anisomag"] = [1.0, 0.0, 0.0, 1.0]
    d["alignments3D/pose"] = axis_angles.astype(np.float32)
    d["alignments3D/psize_A"] = pixel_size
    d["alignments3D/error"] = 1.0; d["alignments3D/error_min"] = 1.0
    d["alignments3D/cross_cor"] = 1.0; d["alignments3D/weight"] = 1.0
    d["alignments3D/class_posterior"] = 1.0
    np.save(str(cs_path), d)
    npy = Path(str(cs_path) + ".npy")
    if npy.exists():
        npy.rename(cs_path)
    print(f"Wrote {cs_path}")


def write_relion_star(
    axis_angles: np.ndarray,   # [N, 3] float32, 轴角向量 (弧度)
    star_path: Path,
    mrcs_path: Path,
    pixel_size: float,
    voltage: float = 300.0,
    cs_mm: float = 2.7,
    amp_contrast: float = 0.1,
) -> None:
    """写 RELION .star 格式 (ZYZ 欧拉角由 pyem.geom 转换)。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pyem"))
    from pyem import geom  # type: ignore[import-untyped]

    N = len(axis_angles)
    mrcs_name = mrcs_path.name

    lines = [
        "", "data_optics", "", "loop_",
        "_rlnVoltage #1 ", "_rlnImagePixelSize #2 ",
        "_rlnSphericalAberration #3 ", "_rlnAmplitudeContrast #4 ",
        "_rlnOpticsGroup #5 ", "_rlnImageSize #6 ",
        "_rlnImageDimensionality #7 ", "_rlnOpticsGroupName #8 ",
    ]
    import mrcfile  # type: ignore[import-untyped]
    with mrcfile.open(str(mrcs_path), permissive=True) as mrc:
        sz = mrc.data.shape[1]
    lines.append(
        f"{voltage:.6f} {pixel_size:.6f} {cs_mm:.6f} {amp_contrast:.6f} 1 {sz} 2 opticsGroup1"
    )
    lines += [
        "", "data_particles", "", "loop_",
        "_rlnImageName #1 ", "_rlnAngleRot #2 ", "_rlnAngleTilt #3 ",
        "_rlnAnglePsi #4 ", "_rlnOriginXAngst #5 ", "_rlnOriginYAngst #6 ",
        "_rlnDefocusU #7 ", "_rlnDefocusV #8 ", "_rlnDefocusAngle #9 ",
        "_rlnPhaseShift #10 ", "_rlnCtfBfactor #11 ",
        "_rlnOpticsGroup #12 ", "_rlnClassNumber #13 ",
    ]
    for i, aa in enumerate(axis_angles):
        euler = np.asarray(geom.rot2euler(geom.expmap(aa.astype(np.float64)))).ravel()
        rot, tilt, psi = np.rad2deg(euler)
        lines.append(
            f"{i+1:06d}@{mrcs_name} {rot:.6f} {tilt:.6f} {psi:.6f} "
            f"0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1 1"
        )
    with open(star_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {star_path}")
