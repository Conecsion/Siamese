"""
cryoSPARC .cs 真实颗粒数据集 (摊销 pose 估计 Phase A)。

读取 cryoSPARC refine job 的产物:
  - 颗粒堆栈 (.mrcs, 由 .cs 的 blob/path + blob/idx 索引)
  - pose label   (alignments3D/pose  = 轴角向量, 弧度; expmap 输入)
  - shift label  (alignments3D/shift = 像素, 符号 +1, 不交换 xy)
  - CTF 参数     (ctf/df1_A, ctf/df2_A, ...)
  - reference volume (.mrc), 用于按 pose+CTF 生成配对投影

每个样本返回 (particle, proj, axisang, shift):
  particle [1,D,D] — 真实含噪颗粒 (MicEncoder 输入)
  proj     [1,D,D] — 该颗粒 pose 下 CTF 调制的干净投影 (ProjEncoder 输入)
  axisang  [3]     — pose 轴角 (弧度)
  shift    [2]     — 平面内位移 (像素)

约定 (data/simulate_test/reverse_cs 已实测确认, corr 0.99976):
  pose 直接喂 project_fourier_slice_from_axis_angle; shift 同样直传 (函数内部取负)。

双塔训练动机 (design §8.3): 两塔都看 CTF 调制数据, 域差距缩到只剩噪声。
proj 只加 CTF、不加噪。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import mrcfile
import numpy as np
import torch
from torch.utils.data import Dataset

from siamese.data.transforms import PreprocessTransform


def _load_cs(cs_path: Path) -> np.ndarray:
    """加载 cryoSPARC .cs 文件 (numpy 结构化数组)。"""
    return np.load(str(cs_path))


def _resolve_particle_path(blob_path: str, project_dir: Path) -> Path:
    """
    解析颗粒 .mrcs 的绝对路径。

    blob/path 是相对 cryoSPARC 项目根目录的相对路径。
    cryoSPARC 导出的 .cs 有时在路径前加 '>' 前缀 (gridfs 标记), 需剥离。
    """
    return project_dir / blob_path.lstrip(">")


class CryoSparcParticleDataset(Dataset):
    """
    cryoSPARC 真实颗粒 + 配对 CTF 投影数据集。

    懒加载: 颗粒图像按需从 .mrcs mmap 读取; reference 投影在 __getitem__
    内即时生成 (单颗粒投影开销小; 大批量预生成见 generate 脚本)。

    单蛋白单类场景: 不涉及 3D 类别分类, 仅 pose (轴角 + 位移)。
    """

    def __init__(
        self,
        cs_path: str,
        reference_path: str,
        project_dir: str,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        normalize: bool = True,
        apply_ctf_to_proj: bool = True,
        working_ps: float | None = None,
        bucket: int | None = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        参数:
            cs_path:          .cs 文件路径
            reference_path:   reference volume (.mrc) 路径
            project_dir:      cryoSPARC 项目根目录 (解析 blob/path)
            split:            "train" | "val" | "test"
            train_split:      训练集比例
            val_split:        验证集比例 (test = 1 - train - val)
            normalize:        是否对图像做 per-image 归一化
            apply_ctf_to_proj: 投影是否施加 CTF (双塔同域, 默认 True)
            device:           投影计算设备 ("cpu" / "cuda")
            seed:             划分随机种子
        """
        self.cs_path = Path(cs_path)
        self.project_dir = Path(project_dir)
        self.normalize = normalize
        self.apply_ctf_to_proj = apply_ctf_to_proj
        self.device = torch.device(device)

        self.cs = _load_cs(self.cs_path)
        self.transform = PreprocessTransform(normalize=normalize)
        self.working_ps = working_ps   # proposer 工作 ps (None=不重采样, 用原生)
        self.fixed_bucket = bucket      # 固定桶尺寸 (None=自动选)

        # reference volume [D,D,D]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mrcfile.open(reference_path, permissive=True) as m:
                vol = np.asarray(m.data, dtype=np.float32).copy()

        # 降采样大 volume 节省显存 (proposer gallery 投影)
        # 规则: >260³ 的降到 256³, pixel size 相应放大
        # 使用 Fourier binning（无混叠）
        self.downsample_factor = 1.0
        if vol.shape[0] > 260:
            from siamese.data.resample import fourier_crop_3d
            old_size = vol.shape[0]
            new_size = 256
            vol = fourier_crop_3d(vol, new_size)
            self.downsample_factor = old_size / new_size
            print(f"Volume downsampled: {old_size}³ -> {new_size}³, "
                  f"factor {self.downsample_factor:.2f}x (Fourier binning)")
        else:
            print(f"Volume kept: {vol.shape[0]}³ (no downsampling needed)")

        self.vol = torch.from_numpy(vol).to(self.device)  # [D,D,D]
        self.D_vol = vol.shape[0]
        self.image_size = int(self.cs[0]["blob/shape"][0])
        self.psize = float(self.cs[0]["blob/psize_A"])

        # 投影的有效 pixel size (考虑 volume 降采样)
        self.vol_psize = self.psize * self.downsample_factor

        # 划分: 按颗粒顺序 shuffle 后切分
        M = len(self.cs)
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(M, generator=g).tolist()
        tr_end = int(M * train_split)
        va_end = int(M * (train_split + val_split))
        if split == "train":
            self.indices = perm[:tr_end]
        elif split == "val":
            self.indices = perm[tr_end:va_end]
        else:  # test
            self.indices = perm[va_end:]

        # mmap 缓存: {abs_path: memmap}
        self._mmap_cache: dict[str, np.memmap] = {}
        # 样本缓存: {idx: (particle[1,B,B], proj[1,B,B], ps_work)}
        # 投影开销大且确定 (固定 pose+CTF+reference), 缓存后 epoch 变成纯编码器计算。
        # reference 更新时 (EM 外循环) 需 clear_cache()。
        self._sample_cache: dict[int, tuple] = {}
        self._use_cache = False

    def clear_cache(self) -> None:
        """清空样本缓存 (reference 更新后调用)。"""
        self._sample_cache.clear()

    def enable_cache(self) -> None:
        """开启样本缓存 (首次访问后驻留内存)。"""
        self._use_cache = True

    def __len__(self) -> int:
        return len(self.indices)

    def _read_particle(self, global_idx: int) -> np.ndarray:
        """从 .mrcs mmap 读取单颗粒图像 [D,D] float32。"""
        p = self.cs[global_idx]
        path = str(_resolve_particle_path(p["blob/path"].decode(), self.project_dir))
        if path not in self._mmap_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mrc = mrcfile.mmap(path, mode="r", permissive=True)
            self._mmap_cache[path] = mrc.data
        stack = self._mmap_cache[path]
        return np.asarray(stack[int(p["blob/idx"])], dtype=np.float32).copy()

    def _project(self, global_idx: int) -> torch.Tensor:
        """
        按颗粒 pose + CTF 生成配对干净投影 [D,D] (无噪声)。

        复用 project_fourier_slice_from_axis_angle, 约定与 reverse_cs 实测一致:
        pose 直传, shift 直传 (函数内部取负)。
        """
        from siamese.data.projection import project_fourier_slice_from_axis_angle

        p = self.cs[global_idx]
        aa = torch.from_numpy(
            np.asarray(p["alignments3D/pose"], dtype=np.float32)
        ).to(self.device).unsqueeze(0)  # [1,3]
        shift = torch.from_numpy(
            np.asarray(p["alignments3D/shift"], dtype=np.float32)
        ).to(self.device).unsqueeze(0)  # [1,2]

        ctf_kwargs = {}
        if self.apply_ctf_to_proj:
            ctf_kwargs = dict(
                apply_ctf=True,
                psize=self.vol_psize,  # 使用降采样后的 pixel size
                ctf_voltage=float(p["ctf/accel_kv"]),
                ctf_cs=float(p["ctf/cs_mm"]),
                ctf_amp_contrast=float(p["ctf/amp_contrast"]),
                ctf_df_u=float(p["ctf/df1_A"]),
                ctf_df_v=float(p["ctf/df2_A"]),
                ctf_df_angle=float(p["ctf/df_angle_rad"]),
                ctf_phase_shift=float(p["ctf/phase_shift_rad"]),
                ctf_particle_sign=float(p["blob/sign"]),
            )

        with torch.no_grad():
            proj = project_fourier_slice_from_axis_angle(
                self.vol, aa, shifts=shift, pfac=2, normalize=True,
                noise_model="none", **ctf_kwargs,
            )[0]  # [D_vol, D_vol]

        # 裁剪到颗粒尺寸 (若 reference 与颗粒 box 不同)
        if self.D_vol != self.image_size:
            m = (self.D_vol - self.image_size) // 2
            proj = proj[m : m + self.image_size, m : m + self.image_size]
        return proj  # [D, D]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            particle [1,B,B] — 真实含噪颗粒 (重采样到工作 ps 并落桶, 或原生)
            proj     [1,B,B] — pose+CTF 调制的干净投影 (同样处理)
            axisang  [3]     — pose 轴角 (弧度)
            shift    [2]     — 位移 (像素, 原生 ps 下)
            ps_work  []      — 重采样后实际像素大小 (Å, 标量; FiLM 条件用)
        """
        gi = self.indices[idx]
        p = self.cs[gi]
        axisang = torch.from_numpy(np.asarray(p["alignments3D/pose"], dtype=np.float32))
        shift = torch.from_numpy(np.asarray(p["alignments3D/shift"], dtype=np.float32))

        if self._use_cache and idx in self._sample_cache:
            particle, proj, ps_work = self._sample_cache[idx]
            return particle, proj, axisang, shift, ps_work

        particle = torch.from_numpy(self._read_particle(gi))  # [D,D]
        proj = self._project(gi).cpu()                         # [D,D]

        # 颗粒同步降采样（匹配 volume 降采样）
        # 使用 Fourier binning（无混叠）
        particle_psize = self.psize
        if self.downsample_factor > 1.0:
            from siamese.data.resample import fourier_crop_2d
            old_size = particle.shape[0]
            new_size = int(np.round(old_size / self.downsample_factor))
            particle = fourier_crop_2d(particle, new_size)
            particle_psize = self.psize * self.downsample_factor

        # 重采样到 working_ps（现在颗粒和投影的 pixel size 一致）
        ps_work = particle_psize
        if self.working_ps is not None:
            from siamese.data.resample import resample_to_working_ps, DEFAULT_BUCKETS
            buckets = (self.fixed_bucket,) if self.fixed_bucket else DEFAULT_BUCKETS
            particle, _, ps_work = resample_to_working_ps(
                particle, particle_psize, self.working_ps, buckets)
            proj, _, _ = resample_to_working_ps(
                proj, self.vol_psize, self.working_ps, buckets)  # 使用 vol_psize

        particle = self.transform(particle)  # [1,B,B]
        proj = self.transform(proj)          # [1,B,B]
        ps_work_t = torch.tensor(ps_work, dtype=torch.float32)

        if self._use_cache:
            self._sample_cache[idx] = (particle, proj, ps_work_t)
        return particle, proj, axisang, shift, ps_work_t


