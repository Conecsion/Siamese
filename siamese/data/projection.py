"""
傅里叶切片定理投影模块 (GPU batch-vectorized)。

使用 GPU 加速的傅里叶切片定理 (Fourier Slice Theorem) 从 3D volume
生成 2D 投影图像。与 cryoSPARC 的投影算法完全一致。

GPU 加速:
    整条流水线 (旋转频率坐标、KB/三线性插值、CTF、位移相位、IFFT、裁切)
    在 batch 维 B 上向量化, 不再逐颗粒 Python 循环。KB gridding 的 125 个
    邻域偏移在所有颗粒上一次性 gather (总计 125 次 kernel, 而非 125*B 次)。
    显存由 project_fourier_slice 的 chunk_size 自动分块控制 (见 _auto_chunk_size),
    遇 OOM 自动减半重试。数值与逐颗粒版本一致 (同 KB 核、同 n_half=N/2)。

核心原理:
    2D 投影的 2D 傅里叶变换 = 3D volume 傅里叶变换的中央切片

实现细节 (与 cryoSPARC 匹配):
    - 过采样因子 pfac=2 (投影前将 volume 补零到 2 倍)
    - 旋转矩阵使用 pyem 约定 (axis_angle_to_matrix 输出，无需额外转置)
    - 位移方向取反 (-shift)
    - 傅里叶域插值: 可选 Kaiser-Bessel gridding (默认) 或三线性 (备用)
    - 超出 Nyquist 半径的傅里叶分量置零
    - FFT 使用 norm="backward" (默认) 保证来回值域不变
    - 位移在傅里叶域通过相位斜坡实现

插值方法 (method 参数):
    - "kaiser_bessel" (默认): 傅里叶域 Kaiser-Bessel 窗核 gridding 插值,
      并在实空间对 volume 做去卷积预补偿 (deapodization / gridding 预补偿)。
      这是 cryoSPARC / RELION 使用的高精度方案, 高频精度远优于三线性,
      可使 homo-reconstruct-only 接近 Nyquist 分辨率。
    - "trilinear" (备用): 原有的傅里叶域三线性插值 (order=1)。速度快但高频
      精度不足 (~8 Å 以上失真), 保留以便对比与回退。

Gridding 原理 (forward NUFFT / type-2):
    1. 实空间预补偿: vol /= c(x), c 为 KB 核的傅里叶变换 (apodization 函数);
    2. 补零过采样 + 3D FFT;
    3. 在旋转后的切片频率坐标处用 KB 核做卷积式插值。
    步骤 1 与步骤 3 的核互为精确傅里叶对, 二者抵消后得到无伪影的切片。
"""

from typing import Optional, Union

import torch
from torch import Tensor


def axis_angle_to_matrix(
    axis_angle: Tensor,
) -> Tensor:
    """
    将轴角表示转换为旋转矩阵 (Rodrigues 公式)。

    与 pyem.geom.aa2rot 输出完全一致 (兼容 cryoSPARC 约定)。

    注意: pyem 的约定与 scipy.spatial.transform.Rotation.from_rotvec
    返回的矩阵互为转置关系。本函数遵循 pyem 约定。

    参数:
        axis_angle: 形状 [B, 3] 或 [3] 的轴角向量。
                   方向表示旋转轴，幅值表示旋转角度 (弧度)。

    返回:
        rot_matrix: 形状 [B, 3, 3] 或 [3, 3] 的旋转矩阵。
                   与 pyem.geom.aa2rot 输出一致。
    """
    squeeze = False
    if axis_angle.ndim == 1:
        squeeze = True
        axis_angle = axis_angle.unsqueeze(0)  # [1, 3]

    B = axis_angle.shape[0]
    device = axis_angle.device
    dtype = axis_angle.dtype

    # 旋转角度 theta = |axis_angle|
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)  # [B, 1]

    # 处理 theta ≈ 0 的情况 (返回单位矩阵)
    small_angle = theta < 1e-16
    theta = torch.where(small_angle, torch.ones_like(theta), theta)

    # 归一化旋转轴
    w = axis_angle / theta  # [B, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    # 反对称矩阵 K (pyem 约定，与标准 Rodrigues 互为负矩阵):
    # K = [[0,  wz, -wy], [-wz, 0,  wx], [wy, -wx, 0]]
    # 注意: 这与 scipy.spatial.transform.Rotation 的 K 矩阵符号相反。
    #       因此 pyem 的 aa2rot 返回 scipy 旋转矩阵的转置 (逆矩阵)。
    zeros = torch.zeros(B, device=device, dtype=dtype)
    K = torch.stack(
        [
            torch.stack([zeros, wz, -wy], dim=-1),
            torch.stack([-wz, zeros, wx], dim=-1),
            torch.stack([wy, -wx, zeros], dim=-1),
        ],
        dim=1,
    )  # [B, 3, 3]

    # Rodrigues: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    sin_t = torch.sin(theta).unsqueeze(-1)  # [B, 1, 1]
    cos_t = torch.cos(theta).unsqueeze(-1)  # [B, 1, 1]

    I = (
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    )  # [B, 3, 3]
    K2 = K @ K  # [B, 3, 3]

    R = I + sin_t * K + (1 - cos_t) * K2  # [B, 3, 3]

    # 处理小角度情况: theta ≈ 0 时返回单位矩阵
    R = torch.where(
        small_angle.unsqueeze(-1).expand(-1, 3, 3),
        I,
        R,
    )

    if squeeze:
        R = R.squeeze(0)
    return R


def _freq_to_idx(k: Tensor, N: int) -> Tensor:
    """
    将频率坐标转换为数组索引。

    频率 k 在 [-N/2, N/2) 范围内，
    对应的数组索引为: k if k >= 0 else k + N。
    """
    return torch.where(k >= 0, k, k + N)


def _trilinear_interpolate_wrap(
    data_real: Tensor,  # [D, H, W]
    data_imag: Tensor,  # [D, H, W]
    x: Tensor,  # [M], 坐标 (0-indexed, 可以是浮点数)
    y: Tensor,  # [M]
    z: Tensor,  # [M]
) -> tuple[Tensor, Tensor]:
    """
    在 3D 数组上进行三线性插值，使用 wrap 模式处理边界。

    参数:
        data_real: 形状 [D, H, W] 的实部数据。
        data_imag: 形状 [D, H, W] 的虚部数据。
        x: 形状 [M] 的 x 坐标 (0-indexed, 浮点数)。
        y: 形状 [M] 的 y 坐标 (0-indexed, 浮点数)。
        z: 形状 [M] 的 z 坐标 (0-indexed, 浮点数)。

    返回:
        (real_vals, imag_vals): 各形状 [M] 的插值结果。
    """
    D, H, W = data_real.shape

    # 确保坐标在 [0, size-1] 范围内 (wrap)
    x = x % W
    y = y % H
    z = z % D

    # 整数部分和分数部分
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()

    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    z1 = (z0 + 1) % D

    ax = x - x0.float()
    ay = y - y0.float()
    az = z - z0.float()

    # 三线性插值权重
    w000 = (1 - ax) * (1 - ay) * (1 - az)
    w001 = ax * (1 - ay) * (1 - az)
    w010 = (1 - ax) * ay * (1 - az)
    w011 = ax * ay * (1 - az)
    w100 = (1 - ax) * (1 - ay) * az
    w101 = ax * (1 - ay) * az
    w110 = (1 - ax) * ay * az
    w111 = ax * ay * az

    # 实部插值
    real_vals = (
        data_real[z0, y0, x0] * w000
        + data_real[z0, y0, x1] * w001
        + data_real[z0, y1, x0] * w010
        + data_real[z0, y1, x1] * w011
        + data_real[z1, y0, x0] * w100
        + data_real[z1, y0, x1] * w101
        + data_real[z1, y1, x0] * w110
        + data_real[z1, y1, x1] * w111
    )

    # 虚部插值
    imag_vals = (
        data_imag[z0, y0, x0] * w000
        + data_imag[z0, y0, x1] * w001
        + data_imag[z0, y1, x0] * w010
        + data_imag[z0, y1, x1] * w011
        + data_imag[z1, y0, x0] * w100
        + data_imag[z1, y0, x1] * w101
        + data_imag[z1, y1, x0] * w110
        + data_imag[z1, y1, x1] * w111
    )

    return real_vals, imag_vals


# ============================================================================
# Kaiser-Bessel gridding 插值 (高精度方案)
# ============================================================================
#
# 默认核参数遵循常见的 cryo-EM gridding 约定:
#   - 核半宽 W = 1.5 (核全宽 2W+1 = 4 个采样点/维)
#   - 形状参数 beta 由 W 与过采样因子决定 (Beatty et al. 2005 公式)
#
# Kaiser-Bessel 核 (实空间为核, 频率域插值时作为卷积窗):
#   kb(u) = I0( beta * sqrt(1 - (u/W)^2) ) / I0(beta),  |u| <= W
# 其傅里叶变换 (用于实空间去卷积预补偿, deapodization):
#   c(x) ∝ sinh(sqrt(beta^2 - (pi*W*x)^2)) / sqrt(beta^2 - (pi*W*x)^2)
# 其中 x 为归一化到 [-0.5, 0.5) 的实空间坐标 (cycles/sample)。


def _kb_beta(width: float, pfac: float) -> float:
    """
    由核半宽与过采样因子计算 Kaiser-Bessel 形状参数 beta。

    采用 Beatty et al. (2005), IEEE TMI 的经验公式:
        beta = pi * sqrt( (2*W/pfac * (pfac - 0.5))^2 - 0.8 )

    参数:
        width: 核半宽 W (单位: 采样点)。
        pfac: 过采样因子。

    返回:
        beta: Kaiser-Bessel 形状参数 (标量 float)。
    """
    import math

    val = (2.0 * width / pfac * (pfac - 0.5)) ** 2 - 0.8
    # 数值保护: val 可能在极端参数下为负
    val = max(val, 1e-6)
    return math.pi * math.sqrt(val)


def _kb_kernel(u: Tensor, width: float, beta: float) -> Tensor:
    """
    Kaiser-Bessel 窗核 kb(u), 用于频率域卷积式插值。

    kb(u) = I0( beta * sqrt(1 - (u/W)^2) ) / I0(beta),  |u| <= W, 否则 0

    参数:
        u: 任意形状的偏移量 (频率域采样点单位, = 实际频率间距/1)。
        width: 核半宽 W。
        beta: 形状参数。

    返回:
        与 u 同形状的核权重 (未归一化求和, 调用方需按权重和归一化)。
    """
    # 1 - (u/W)^2, 在窗外为负 -> 置零
    t = 1.0 - (u / width) ** 2
    inside = t > 0
    t_clamped = torch.clamp(t, min=0.0)
    arg = beta * torch.sqrt(t_clamped)
    # torch.special.i0e(x) = exp(-|x|) * I0(x); I0(x) = i0e(x) * exp(|x|)
    i0_arg = torch.special.i0(arg)
    i0_beta = torch.special.i0(torch.tensor(beta, dtype=u.dtype, device=u.device))
    k = i0_arg / i0_beta
    return torch.where(inside, k, torch.zeros_like(k))


def _kb_apodization_1d(
    n: int, width: float, beta: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    计算 1D Kaiser-Bessel 核的傅里叶变换 (apodization 函数 c(x)),
    用于实空间去卷积预补偿。

    c(x) = sinh(q) / q,  q = sqrt( (pi*W*x)^2 - beta^2 ) 为虚时取 sin 形式。
    实现中统一用复数 sqrt 处理 (pi*W*x)^2 - beta^2 的正负, 取实部。

    参数:
        n: volume 边长 (补零后的网格大小 N_pad)。
        width: 核半宽 W。
        beta: 形状参数。
        device, dtype: 输出张量的设备与精度。

    返回:
        apod: 形状 [n] 的 1D apodization 函数, 已归一化到中心值为 1。
    """
    # 实空间归一化坐标 x ∈ [-0.5, 0.5)  (cycles/sample)
    x = torch.arange(n, device=device, dtype=dtype) - n // 2
    x = x / n  # [-0.5, 0.5)
    # KB 核 I0(beta*sqrt(1-(u/W)^2)) (|u|<=W) 的连续傅里叶变换:
    #   c(x) ∝ sinh(q)/q,  q = sqrt( beta^2 - (2*pi*W*x)^2 )
    # 低频 (|x| 小) 时 a>0 -> q 实数 -> sinh(q)/q;
    # 高频时 a<0 -> q 虚数 -> sin(|q|)/|q|; 用复数 sqrt 统一处理。
    a = beta**2 - (2.0 * torch.pi * width * x) ** 2
    q = torch.sqrt(torch.complex(a, torch.zeros_like(a)))
    # sinh(q)/q, 当 q->0 时极限为 1
    eps = 1e-12
    apod = torch.where(
        q.abs() < eps,
        torch.ones_like(q),
        torch.sinh(q) / q,
    )
    apod = apod.real
    # 归一化到中心值 (x=0) 为 1, 保证去卷积不改变整体亮度
    center = apod[n // 2]
    return apod / center


def _kb_apodization_3d(
    n: int, width: float, beta: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    3D 可分离 Kaiser-Bessel apodization 函数 (外积), 形状 [n, n, n]。
    """
    c1 = _kb_apodization_1d(n, width, beta, device, dtype)  # [n]
    # 外积: c(x,y,z) = c(x)*c(y)*c(z)
    c3 = c1[:, None, None] * c1[None, :, None] * c1[None, None, :]
    return c3  # [n, n, n]


def _kaiser_bessel_interpolate(
    data_real: Tensor,  # [N, N, N]
    data_imag: Tensor,  # [N, N, N]
    x: Tensor,  # 任意形状 (如 [M] 或 [B, N, N]), 频率坐标 (0-indexed, 浮点)
    y: Tensor,  # 同 x 形状
    z: Tensor,  # 同 x 形状
    width: float,
    beta: float,
    normalize_weights: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    在 3D 频率网格上用 Kaiser-Bessel 核做 gridding 插值 (wrap 边界)。

    对每个查询点, 在其周围 (2*ceil(W)+1)^3 邻域内按 KB 核加权求和。

    查询坐标 x/y/z 可为任意同形状张量 (如逐点 [M] 或稠密 batch [B,N,N]);
    gather 与累加全部按元素广播, 返回与 x 同形状的结果。

    重要: 标准 type-2 gridding (forward NUFFT) **不**对核权重和做归一化 ——
    频率域的 KB 卷积恰好等价于实空间乘以 apodization 函数 c(x), 而第 1b 步的
    实空间预除 c(x) 正好精确抵消它。若再除以权重和会破坏这一精确对消关系。
    normalize_weights 仅为对比/调试保留 (默认 False)。

    参数:
        data_real / data_imag: 形状 [N, N, N] 的频率域实/虚部。
        x, y, z: 同形状的查询频率坐标 (0-indexed, 浮点, 可超界由 wrap 处理)。
        width: 核半宽 W。
        beta: KB 形状参数。
        normalize_weights: 是否按邻域核权重和归一化 (默认 False, 即标准 gridding)。

    返回:
        (real_vals, imag_vals): 各与 x 同形状的插值结果。
    """
    N = data_real.shape[-1]
    device = x.device
    dtype = x.dtype

    # 邻域偏移范围: [-r, ..., r], r = ceil(W)
    import math

    r = int(math.ceil(width))
    offsets = torch.arange(-r, r + 1, device=device, dtype=torch.long)  # [2r+1]

    # 核归一化常数: 使核在整数网格上的权重和为 1 (DC 增益归一),
    # 与 apodization 中心值=1 的约定配套, 保证整体亮度不变。
    # 对一维核在 [-r, r] 整数偏移 (相对最近格点 0 偏移) 求和近似为常数, 这里用
    # 连续意义下的归一: 取一个细网格估计 1D 核积分。
    u_fine = torch.linspace(-width, width, 4097, device=device, dtype=dtype)
    ksum_1d = _kb_kernel(u_fine, width, beta).sum() * (2.0 * width / 4096.0)  # ∫kb du
    norm_const = ksum_1d**3  # 3D 可分离

    # 基准整数格点 (floor)
    x0 = torch.floor(x).long()  # 同 x 形状
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()

    acc_real = torch.zeros_like(x)
    acc_imag = torch.zeros_like(x)
    acc_w = torch.zeros_like(x)

    # 三重循环遍历邻域 (r=2 时共 125 次), 每次在所有查询点上一次性 gather
    for dz in offsets:
        gz = z0 + dz
        wz = _kb_kernel((z - gz.to(dtype)), width, beta)  # 同 x 形状
        gz = gz % N
        for dy in offsets:
            gy = y0 + dy
            wy = _kb_kernel((y - gy.to(dtype)), width, beta)
            gy = gy % N
            for dx in offsets:
                gx = x0 + dx
                wx = _kb_kernel((x - gx.to(dtype)), width, beta)
                gx = gx % N
                w = wx * wy * wz
                acc_real += data_real[gz, gy, gx] * w
                acc_imag += data_imag[gz, gy, gx] * w
                acc_w += w

    if normalize_weights:
        # 备用/调试: 归一化卷积 (会与 deapodization 双重补偿, 一般不用)
        acc_w = torch.where(acc_w.abs() < 1e-12, torch.ones_like(acc_w), acc_w)
        return acc_real / acc_w, acc_imag / acc_w

    # 标准 gridding: 仅除以核的 DC 增益常数 (使 apodization 中心=1 时整体亮度守恒)
    return acc_real / norm_const, acc_imag / norm_const


def _compute_ctf_2d(
    kx_grid: Tensor,  # [N, N], 补零傅里叶索引单位
    ky_grid: Tensor,  # [N, N]
    N: int,  # 补零后 box 大小
    psize: float,  # Å/pixel
    df_u: Union[float, Tensor],  # 散焦 U (Å), 欠焦为正; 标量或 [B] 向量
    df_v: Union[float, Tensor],  # 散焦 V (Å); 标量或 [B] 向量
    df_angle: float,  # 散焦角 (弧度)
    voltage_kv: float,  # 加速电压 (kV)
    cs_mm: float,  # 球差系数 (mm)
    amp_contrast: float,  # 振幅对比度 (0–1)
    phase_shift: float,  # 附加相位偏移 (弧度)
    particle_sign: float,  # 颗粒符号 (+1 或 -1)
) -> Tensor:
    """
    在补零傅里叶网格上计算 2D CTF。

    CTF 公式 (与 cryoSPARC/RELION 一致):
        gamma(k, phi) = pi * lambda * df(phi) * k^2
                       - 0.5 * pi * lambda^3 * Cs * k^4
                       + phase_shift
        CTF = particle_sign * (-sqrt(1-Q^2) * sin(gamma) + Q * cos(gamma))

    其中:
        lambda: 相对论电子波长 (Å), lambda = 12.2643/sqrt(V_eV*(1+0.978e-6*V_eV))
        df(phi) = 0.5*(df_u + df_v + (df_u-df_v)*cos(2*(phi-df_angle)))
        Q: 振幅对比度
        k: 物理空间频率 (1/Å) = sqrt(kx^2+ky^2), kx = kx_grid/(N*psize)

    参数:
        kx_grid, ky_grid: 形状 [N, N], 补零网格的傅里叶索引 (整数范围 -N/2..N/2)。
        N: 补零后的 box 大小 (= pfac * D)。
        psize: 像素大小 (Å)。
        df_u/df_v: 散焦 (Å)。标量 -> 返回 [N, N]; [B] 向量 -> 广播返回 [B, N, N]。
        其余参数见上方注释。

    返回:
        ctf: 形状 [N, N] (标量散焦) 或 [B, N, N] (向量散焦) 的 float32 CTF。
    """
    import math

    # --- 相对论电子波长 (Å) ---
    V_eV = voltage_kv * 1e3
    lam = 12.2643247 / math.sqrt(V_eV * (1.0 + 0.978466e-6 * V_eV))
    cs_A = cs_mm * 1e7  # mm -> Å

    # --- 物理频率 (1/Å) ---
    kx = kx_grid / (N * psize)  # [N, N]
    ky = ky_grid / (N * psize)  # [N, N]
    k2 = kx**2 + ky**2  # [N, N]
    phi = torch.atan2(ky, kx)  # 方位角 [N, N]

    # --- 散焦: 标量 -> [N,N]; [B] 向量 -> [B,1,1] 广播为 [B,N,N] ---
    df_u_is_t = isinstance(df_u, Tensor)
    df_v_is_t = isinstance(df_v, Tensor)
    if df_u_is_t or df_v_is_t:
        # 至少一个是 batch 向量 -> 统一广播到 [B, N, N]
        ref = df_u if df_u_is_t else df_v
        dev, dt = ref.device, ref.dtype  # type: ignore[union-attr]
        dfu = df_u.to(dev, dt).reshape(-1, 1, 1) if df_u_is_t else torch.as_tensor(
            float(df_u), device=dev, dtype=dt
        )
        dfv = df_v.to(dev, dt).reshape(-1, 1, 1) if df_v_is_t else torch.as_tensor(
            float(df_v), device=dev, dtype=dt
        )
        k2 = k2.unsqueeze(0)  # [1, N, N]
        phi = phi.unsqueeze(0)  # [1, N, N]
    else:
        dfu = float(df_u)
        dfv = float(df_v)

    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2.0 * (phi - df_angle)))

    # --- CTF 相位 ---
    gamma = (
        torch.pi * lam * df * k2
        - 0.5 * torch.pi * (lam**3) * cs_A * (k2**2)
        + phase_shift
    )

    w1 = math.sqrt(1.0 - amp_contrast**2)
    # CTF 两项必须同号 (与 RELION ctf.cpp / EMAN2 / pyem ctf.py 一致, 验证差 <1e-14)。
    # 之前误写 +amp_contrast*cos (异号), 等价注入 ~2*asin(Q) 虚假相移, Thon 环错位。
    ctf = particle_sign * (-w1 * torch.sin(gamma) - amp_contrast * torch.cos(gamma))
    return ctf.to(dtype=torch.float32)  # [N, N] 或 [B, N, N]


def _auto_chunk_size(N: int, B: int, device: torch.device) -> int:
    """
    按可用显存自动估算 batch 分块大小 (稠密 [bc, N, N] 计算的峰值显存控制)。

    KB gridding 循环中每颗粒持有若干 [N, N] 中间量 (坐标、索引、累加器、
    gather 临时量), 经验上峰值约 120 * N^2 字节/颗粒。取空闲显存的 60%
    作为预算 (3D FFT 体已占用, 用 mem_get_info 的 free 值即扣除后余量)。

    CPU 或无法查询时回退到一个保守的固定值。

    参数:
        N: 补零后网格边长 (= pfac * D)。
        B: 总颗粒数。
        device: 计算设备。

    返回:
        chunk: 每块颗粒数 (>=1, <=B)。
    """
    if device.type != "cuda":
        return min(B, 64)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return min(B, 64)
    bytes_per_particle = 120.0 * N * N
    budget = 0.6 * free_bytes
    chunk = int(budget / max(bytes_per_particle, 1.0))
    return max(1, min(B, chunk))


def project_fourier_slice(
    vol: Tensor,
    rot_matrices: Tensor,
    shifts: Optional[Tensor] = None,
    pfac: int = 2,  # padding factor
    method: str = "kaiser_bessel",
    kb_width: float = 1.5,
    kb_beta: Optional[float] = None,
    chunk_size: Optional[int] = None,  # batch 分块大小; None=按显存自动选
    # CTF 参数 (apply_ctf=False 时全部忽略)
    apply_ctf: bool = False,
    psize: float = 1.0,  # 像素大小 (Å/pixel), 计算物理频率用
    ctf_voltage: float = 300.0,  # 加速电压 (kV)
    ctf_cs: float = 2.7,  # 球差系数 (mm)
    ctf_amp_contrast: float = 0.07,  # 振幅对比度
    ctf_df_u: Union[float, Tensor] = 20000.0,  # 散焦 U (Å), 或 per-particle Tensor[B]
    ctf_df_v: Union[float, Tensor] = 20000.0,  # 散焦 V (Å), 或 per-particle Tensor[B]
    ctf_df_angle: float = 0.0,  # 散焦角 (弧度)
    ctf_phase_shift: float = 0.0,  # 附加相位偏移 (弧度)
    ctf_particle_sign: float = -1.0,  # 颗粒符号 (+1 或 -1)
) -> Tensor:
    """
    使用傅里叶切片定理从 3D volume 生成 2D 投影。

    与 cryoSPARC 的投影算法一致 (kaiser_bessel 方法时为高精度匹配)。

    参数:
        vol: 形状 [D, D, D] 的 3D volume。
        rot_matrices: 形状 [B, 3, 3] 的旋转矩阵 (pyem 约定，即 axis_angle_to_matrix 的输出)。
                      无需额外转置。
        shifts: 形状 [B, 2] 的平面内位移 (像素单位)。
                如果为 None，则不施加位移。
                注意: 位移方向与 cryoSPARC 中存储的 shift 方向相反 (取负)。
        pfac: 过采样因子，默认 2。volume 在 FFT 前会补零到 pfac * D。
        method: 频率域插值方法。
                - "kaiser_bessel" (默认): KB 核 gridding + 实空间去卷积预补偿,
                  高频精度高, 推荐用于重建。
                - "trilinear": 原有三线性插值, 速度快但高频失真, 作为备用。
        kb_width: Kaiser-Bessel 核半宽 W (采样点), 仅 method="kaiser_bessel" 时使用。
                  默认 1.5 (核全宽 4 点/维)。
        kb_beta: KB 形状参数。None 时由 kb_width 与 pfac 自动计算 (Beatty 2005)。
        chunk_size: batch 维分块大小。None 时按可用显存自动选 (见 _auto_chunk_size);
                    遇 CUDA OOM 会自动减半重试。
        apply_ctf: 是否在傅里叶切片上乘以 CTF (默认 False)。
                   启用时在步骤 8 (构建 2D 切片) 之后、步骤 9 (位移) 之前乘以 CTF。
        psize: 像素大小 (Å/pixel), 计算 CTF 物理频率所需, 默认 1.34。
        ctf_voltage: 加速电压 (kV), 默认 300。
        ctf_cs: 球差系数 (mm), 默认 2.7。
        ctf_amp_contrast: 振幅对比度 Q (0–1), 默认 0.07。
        ctf_df_u/v: 散焦 U/V (Å, 欠焦为正), 默认 20000。
        ctf_df_angle: 散焦角 (弧度), 默认 0。
        ctf_phase_shift: 附加相位偏移 (弧度), 默认 0。
        ctf_particle_sign: 颗粒符号 (+1 或 -1), 默认 -1。

    返回:
        projections: 形状 [B, D, D] 的 2D 投影图像。
    """
    assert method in ("kaiser_bessel", "trilinear"), (
        f"method 必须为 'kaiser_bessel' 或 'trilinear'，实际为 {method!r}"
    )

    D = vol.shape[-1]
    assert vol.shape == (D, D, D), f"vol 形状必须为 (D, D, D)，实际为 {vol.shape}"
    B = rot_matrices.shape[0]
    assert rot_matrices.shape == (B, 3, 3), (
        f"rot_matrices 形状必须为 (B, 3, 3)，实际为 {rot_matrices.shape}"
    )
    if shifts is not None:
        assert shifts.shape == (B, 2), (
            f"shifts 形状必须为 (B, 2)，实际为 {shifts.shape}"
        )

    device = vol.device
    dtype = vol.dtype

    # --- KB 参数 (仅 kaiser_bessel 用) ---
    use_kb = method == "kaiser_bessel"
    if use_kb and kb_beta is None:
        kb_beta = _kb_beta(kb_width, float(pfac))

    # --- 1. 补零 (oversampling) ---
    N_pad = int(D * pfac)
    pad_before = (N_pad - D) // 2
    pad_after = N_pad - D - pad_before
    vol_padded = torch.nn.functional.pad(
        vol,
        (pad_before, pad_after, pad_before, pad_after, pad_before, pad_after),
        mode="constant",
        value=0.0,
    )  # [N_pad, N_pad, N_pad]

    # --- 1b. Gridding 预补偿 (deapodization): 实空间除以 KB 核的傅里叶变换 ---
    # 该步骤与第 7 步频率域 KB 卷积互为精确傅里叶对, 抵消后得到无伪影切片。
    if use_kb:
        apod3d = _kb_apodization_3d(
            N_pad, kb_width, kb_beta, device, dtype
        )  # [N,N,N], 中心=1
        # 避免除以过小值放大数值噪声 (padding 边缘处 apod 可能极小; 那里 vol=0, 不受影响)
        safe_apod = torch.where(apod3d.abs() < 1e-3, torch.ones_like(apod3d), apod3d)
        vol_padded = vol_padded / safe_apod

    # --- 2. ifftshift + 3D FFT ---
    vol_shifted = torch.fft.ifftshift(vol_padded)
    vol_fft = torch.fft.fftn(vol_shifted, dim=(-3, -2, -1), norm="backward")
    # vol_fft: [N_pad, N_pad, N_pad] complex

    N = N_pad
    vol_fft_real = vol_fft.real.contiguous()  # [N, N, N]
    vol_fft_imag = vol_fft.imag.contiguous()  # [N, N, N]
    del vol_fft, vol_shifted, vol_padded

    # --- 3. 构建频率网格 ---
    k = torch.arange(-N // 2, N // 2, device=device, dtype=dtype)  # [N]
    kx_grid, ky_grid = torch.meshgrid(k, k, indexing="xy")  # 各 [N, N]

    # --- 3.5. CTF 预处理 ---
    # ctf_df_u/v 为标量 -> 所有颗粒共用一张 CTF (循环外算一次);
    # 为 per-particle 向量 Tensor[B] -> 每块按散焦切片向量化计算 [bc, N, N]。
    _df_per_particle = apply_ctf and (
        isinstance(ctf_df_u, Tensor) or isinstance(ctf_df_v, Tensor)
    )
    ctf_2d_shared: Optional[Tensor] = None
    if apply_ctf and not _df_per_particle:
        ctf_2d_shared = _compute_ctf_2d(
            kx_grid, ky_grid, N, psize,
            float(ctf_df_u), float(ctf_df_v), ctf_df_angle,  # type: ignore[arg-type]
            ctf_voltage, ctf_cs, ctf_amp_contrast,
            ctf_phase_shift, ctf_particle_sign,
        )  # [N, N]

    # Nyquist 截断半径 (补零后频率索引单位)。
    # 关键: 实空间补零 D->N=pfac*D 等价于傅里叶域加密采样, 物理频率范围不变,
    # 因此物理 Nyquist 落在补零网格索引 N/2 处, 而非 D/2 (否则高频失真)。
    n_half = N / 2.0

    def _project_chunk(
        rc: Tensor, sc: Optional[Tensor], dfu_c, dfv_c
    ) -> Tensor:
        """处理一块旋转矩阵 rc:[bc,3,3] -> 投影 [bc,D,D] (全程向量化)。"""
        bc = rc.shape[0]
        # 旋转后频率坐标 [bc, N, N] (R 列向量, 与逐颗粒版一致)
        kx_o = rc[:, 0, 0].view(bc, 1, 1) * kx_grid + rc[:, 1, 0].view(bc, 1, 1) * ky_grid
        ky_o = rc[:, 0, 1].view(bc, 1, 1) * kx_grid + rc[:, 1, 1].view(bc, 1, 1) * ky_grid
        kz_o = rc[:, 0, 2].view(bc, 1, 1) * kx_grid + rc[:, 1, 2].view(bc, 1, 1) * ky_grid

        # --- Nyquist 掩膜 [bc, N, N] ---
        radius = torch.sqrt(kx_o**2 + ky_o**2 + kz_o**2)
        mask = radius < n_half

        # --- 频率 -> 索引 (稠密全网格; 超界 wrap, mask 外随后置零) ---
        idx_x = _freq_to_idx(kx_o, N)
        idx_y = _freq_to_idx(ky_o, N)
        idx_z = _freq_to_idx(kz_o, N)
        del kx_o, ky_o, kz_o, radius

        # --- 频率域插值 [bc, N, N] ---
        if use_kb:
            real_vals, imag_vals = _kaiser_bessel_interpolate(
                vol_fft_real, vol_fft_imag, idx_x, idx_y, idx_z,
                width=kb_width, beta=kb_beta,
            )
        else:
            real_vals, imag_vals = _trilinear_interpolate_wrap(
                vol_fft_real, vol_fft_imag, idx_x, idx_y, idx_z,
            )
        del idx_x, idx_y, idx_z

        # --- 构建 2D 傅里叶切片 (mask 外置零, 与 masked-select 等价) ---
        maskf = mask.to(real_vals.dtype)
        slice_ft = torch.complex(real_vals * maskf, imag_vals * maskf)
        del real_vals, imag_vals, mask, maskf

        # --- 乘以 CTF (可选) ---
        if apply_ctf:
            if _df_per_particle:
                ctf_c = _compute_ctf_2d(
                    kx_grid, ky_grid, N, psize,
                    dfu_c, dfv_c, ctf_df_angle,
                    ctf_voltage, ctf_cs, ctf_amp_contrast,
                    ctf_phase_shift, ctf_particle_sign,
                )  # [bc, N, N]
                slice_ft = slice_ft * ctf_c
            else:
                slice_ft = slice_ft * ctf_2d_shared  # [N, N] 广播

        # --- 相位斜坡 (位移) [bc, N, N] ---
        if sc is not None:
            sx = sc[:, 0].view(bc, 1, 1)
            sy = sc[:, 1].view(bc, 1, 1)
            phase = -2.0 * torch.pi * (kx_grid * sx + ky_grid * sy) / N
            slice_ft = slice_ft * torch.exp(1j * phase)

        # --- 2D IFFT [bc, N, N] ---
        slice_ft = torch.fft.ifftshift(slice_ft, dim=(-2, -1))
        proj_full = torch.fft.ifft2(slice_ft, dim=(-2, -1), norm="backward").real
        proj_full = torch.fft.fftshift(proj_full, dim=(-2, -1))

        # --- 裁切回原始大小 [bc, D, D] (clone 以释放 proj_full) ---
        start = (N - D) // 2
        return proj_full[:, start : start + D, start : start + D].clone()

    # --- 驱动: 按 chunk 分块, 遇 CUDA OOM 自动减半重试 ---
    cs = chunk_size if chunk_size is not None else _auto_chunk_size(N, B, device)
    cs = max(1, min(cs, B))
    out_chunks = []
    b0 = 0
    while b0 < B:
        bc = min(cs, B - b0)
        rc = rot_matrices[b0 : b0 + bc]
        sc = shifts[b0 : b0 + bc] if shifts is not None else None
        dfu_c = (
            ctf_df_u[b0 : b0 + bc]
            if (_df_per_particle and isinstance(ctf_df_u, Tensor))
            else ctf_df_u
        )
        dfv_c = (
            ctf_df_v[b0 : b0 + bc]
            if (_df_per_particle and isinstance(ctf_df_v, Tensor))
            else ctf_df_v
        )
        try:
            out_chunks.append(_project_chunk(rc, sc, dfu_c, dfv_c))
            b0 += bc
        except torch.cuda.OutOfMemoryError:
            if cs == 1:
                raise
            torch.cuda.empty_cache()
            cs = max(1, cs // 2)

    return torch.cat(out_chunks, dim=0)  # [B, D, D]


def project_fourier_slice_from_axis_angle(
    vol: Tensor,
    axis_angles: Tensor,
    shifts: Optional[Tensor] = None,
    pfac: int = 2,  # padding factor
    normalize: bool = False,
    method: str = "kaiser_bessel",
    kb_width: float = 1.5,
    kb_beta: Optional[float] = None,
    chunk_size: Optional[int] = None,  # batch 分块大小; None=按显存自动选
    # CTF 参数 (apply_ctf=False 时全部忽略)
    apply_ctf: bool = False,
    psize: float = 1.34,
    ctf_voltage: float = 300.0,
    ctf_cs: float = 2.7,
    ctf_amp_contrast: float = 0.07,
    ctf_df_u: Union[float, Tensor] = 20000.0,
    ctf_df_v: Union[float, Tensor] = 20000.0,
    ctf_df_angle: float = 0.0,
    ctf_phase_shift: float = 0.0,
    ctf_particle_sign: float = -1.0,
    # 噪声参数
    noise_model: str = "none",  # "white" 或 "none"
    snr: float = 0.05,          # 信噪比; noise_var = var(无噪声投影) / snr
) -> Tensor:
    """
    从轴角表示和位移直接生成投影的便捷函数。

    自动处理旋转矩阵转换和位移符号翻转，
    与 cryoSPARC 的投影结果一致 (method="kaiser_bessel" 时为高精度匹配)。

    参数:
        vol: 形状 [D, D, D] 的 3D volume。
        axis_angles: 形状 [B, 3] 的轴角向量 (弧度)。
        shifts: 形状 [B, 2] 的平面内位移 (像素单位)。
                如果为 None，则不施加位移。
        pfac: 过采样因子，默认 2。
        normalize: 是否对输出做逐图零均值 (cryoSPARC 约定)。
        method: 频率域插值方法, "kaiser_bessel" (默认, 高精度) 或 "trilinear" (备用)。
        kb_width: Kaiser-Bessel 核半宽 W, 仅 KB 方法使用。
        kb_beta: KB 形状参数, None 时自动计算。
        apply_ctf: 是否施加 CTF, 默认 False。
        psize .. ctf_particle_sign: CTF 参数, 见 project_fourier_slice 文档。

    返回:
        projections: 形状 [B, D, D] 的 2D 投影图像。
    """
    # 1. 轴角 → 旋转矩阵 (pyem 约定)
    rot_matrices = axis_angle_to_matrix(axis_angles)  # [B, 3, 3]

    # 2. 位移取反 (cryoSPARC 约定)
    shifts_neg = -shifts if shifts is not None else None

    # 3. 投影 (含可选 CTF)
    proj = project_fourier_slice(
        vol,
        rot_matrices,
        shifts_neg,
        pfac,
        method=method,
        kb_width=kb_width,
        kb_beta=kb_beta,
        chunk_size=chunk_size,
        apply_ctf=apply_ctf,
        psize=psize,
        ctf_voltage=ctf_voltage,
        ctf_cs=ctf_cs,
        ctf_amp_contrast=ctf_amp_contrast,
        ctf_df_u=ctf_df_u,
        ctf_df_v=ctf_df_v,
        ctf_df_angle=ctf_df_angle,
        ctf_phase_shift=ctf_phase_shift,
        ctf_particle_sign=ctf_particle_sign,
    )

    # 4. 逐图零均值归一化 (可选)
    if normalize:
        mean = proj.mean(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
        proj = proj - mean

    # 5. 白噪声 (可选)
    # noise_var = var(无噪声投影所有像素) / snr; 每像素 IID N(0, noise_var)
    if noise_model == "white":
        signal_var = proj.var()
        noise_std = torch.sqrt(signal_var / snr)
        proj = proj + torch.randn_like(proj) * noise_std

    return proj

