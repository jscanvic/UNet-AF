"""Metal (Apple Silicon / PyTorch MPS) implementation of `circular_filtered_lrelu`.

The kernels are located in `circular_filtered_lrelu.metal`.

Separable (1D) filters, which is what the UNet-AF layers use, run through a single
fused kernel that keeps the upsampled intermediate in threadgroup memory.
Non-separable (2D) filters use a simpler two-pass path with a float32 intermediate.
Gradients w.r.t. `x` and `b` are supported; gradients w.r.t. the filters are not,
matching the CUDA implementation.

The same kernels also run StyleGAN3's zero-padded `filtered_lrelu` (see
`unet_af.ops.stylegan3.filtered_lrelu.filtered_lrelu_mps`) through `_run(zero_pad=True)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from unet_af.ops.mps_utils import DTYPE_NAMES as _DTYPE_NAMES
from unet_af.ops.mps_utils import compile_library, dummy_tensor, params

_METAL_SOURCE = Path(__file__).with_name("circular_filtered_lrelu.metal")

# Keep in sync with the enums and constants in the .metal file.
_TG_DIM = 16
_BUF0_CAP = 1856
_BUF1_CAP = 1280
_AXIS_CAP = 48
_TILE_CANDIDATES = (16, 12, 8, 6, 4, 2, 1)
_FLT_MAX = 3.4028234663852886e38
_SQRT2 = math.sqrt(2)
_MODE_FWD, _MODE_FWD_MASK, _MODE_BWD = 0, 1, 2


# ----------------------------------------------------------------------------
# Filter handling.


@dataclass(frozen=True)
class _Filter:
    """Correlation weights plus left offset along each axis.

    `w` is either a `(wx, wy)` pair of 1D tensors (separable) or a 2D
    `[taps_y, taps_x]` tensor (non-separable).
    """

    w: tuple[torch.Tensor, torch.Tensor] | torch.Tensor
    offset: tuple[int, int]  # (x, y)

    @property
    def separable(self) -> bool:
        return isinstance(self.w, tuple)

    @property
    def taps(self) -> tuple[int, int]:
        if isinstance(self.w, tuple):
            return self.w[0].numel(), self.w[1].numel()
        return self.w.shape[1], self.w.shape[0]

    def as_full(self) -> _Filter:
        if isinstance(self.w, tuple):
            wx, wy = self.w
            return _Filter(torch.outer(wy, wx), self.offset)
        return self

    def adjoint(self) -> _Filter:
        """Filter for the transposed operation: reversed weights, right offset."""
        tx, ty = self.taps
        offset = (tx - 1 - self.offset[0], ty - 1 - self.offset[1])
        if isinstance(self.w, tuple):
            wx, wy = self.w
            return _Filter((wx.flip(0), wy.flip(0)), offset)
        return _Filter(self.w.flip([0, 1]), offset)

    def packed(self) -> torch.Tensor:
        if isinstance(self.w, tuple):
            return torch.cat(self.w)
        return self.w.reshape(-1)


def _make_filter(f: torch.Tensor | None, device, flip_filter: bool) -> _Filter:
    """Convert a user filter to correlation weights, matching `upfirdn2d` semantics."""
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=device)
    f = f.to(device=device, dtype=torch.float32)
    w: tuple[torch.Tensor, torch.Tensor] | torch.Tensor
    if f.ndim == 1:
        wf = f if flip_filter else f.flip(0)
        w = (wf, wf)
    elif f.shape == (1, 1):
        w = (f.reshape(1), torch.ones(1, dtype=f.dtype, device=device))
    else:
        w = f if flip_filter else f.flip([0, 1])
    tx, ty = _Filter(w, (0, 0)).taps
    return _Filter(w, ((tx - 1) // 2, (ty - 1) // 2))


# ----------------------------------------------------------------------------
# Kernel launch.


def _plan_tile(Wo, Ho, up, down, fu: _Filter, fd: _Filter):
    """Pick the largest output tile whose working set fits the threadgroup buffers."""
    (fux, fuy), (fdx, fdy) = fu.taps, fd.taps
    (odx, ody) = fd.offset
    for t in _TILE_CANDIDATES:
        tox, toy = min(t, Wo), min(t, Ho)
        znx = max(down * (tox - 1) + fdx, odx + down * tox)
        zny = max(down * (toy - 1) + fdy, ody + down * toy)
        inx = (znx + fux - 2) // up + 1
        iny = (zny + fuy - 2) // up + 1
        buf0 = max(inx * iny, znx * zny)
        buf1 = max(iny * znx, zny * tox)
        fits_axes = max(znx, zny, inx, iny) <= _AXIS_CAP
        if fits_axes and buf0 <= _BUF0_CAP and buf1 <= _BUF1_CAP:
            return tox, toy, znx, zny, inx, iny
    return None


def _run(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    fu: _Filter,
    fd: _Filter,
    up: int,
    down: int,
    scale: float,
    gain_out: float,
    slope: float,
    clamp: float,
    mode: int,
    mask: torch.Tensor | None,
    *,
    out_size: tuple[int, int] | None = None,
    z_size: tuple[int, int] | None = None,
    zero_pad: bool = False,
) -> torch.Tensor:
    """One pass of: bias, upsample, filter fu, activation/gating, filter fd, downsample.

    By default the boundary is circular, the intermediate has size `in * up` and the
    output `in * up // down`. With `zero_pad`, samples outside the input read as zero
    and the caller gives the intermediate extent `z_size` (the mask layout) and the
    output size `out_size`, both as (height, width).
    """
    x = x.contiguous()
    N, C, H, W = x.shape
    Ho, Wo = out_size if out_size is not None else (H * up // down, W * up // down)
    Lzy, Lzx = z_size if z_size is not None else (H * up, W * up)
    y = torch.empty([N, C, Ho, Wo], dtype=x.dtype, device=x.device)
    if y.numel() == 0:
        return y

    lib = compile_library(_METAL_SOURCE)
    tname = _DTYPE_NAMES[x.dtype]
    has_bias = bias is not None
    bias_t = bias.contiguous() if has_bias else x
    mask_t = mask if mask is not None else dummy_tensor(x.device)
    floats = (float(scale), float(gain_out), float(slope), float(clamp))

    plan = (
        _plan_tile(Wo, Ho, up, down, fu, fd) if fu.separable and fd.separable else None
    )
    if plan is None:
        fu, fd = fu.as_full(), fd.as_full()
        tiles = (0,) * 6
    else:
        tiles = plan

    # Order matches the P_* enum in the .metal file.
    ints = (W, H, up, down, *fu.taps, *fu.offset, *fd.taps, *fd.offset, *tiles)
    ints += (C, int(has_bias), mode, Wo, Ho, Lzx, Lzy, int(zero_pad))
    ip, fp = params(ints, floats, x.device)
    filt = torch.cat([fu.packed(), fd.packed()]).contiguous()

    if plan is not None:
        kern = getattr(lib, f"cflrelu_sep_{tname}_{mode}")
        tox, toy = plan[:2]
        ntx, nty = math.ceil(Wo / tox), math.ceil(Ho / toy)
        kern(
            x,
            bias_t,
            filt,
            mask_t,
            y,
            ip,
            fp,
            threads=[ntx * _TG_DIM, nty * _TG_DIM, N * C],
            group_size=[_TG_DIM, _TG_DIM, 1],
        )
        return y

    z = torch.empty([N * C, Lzy, Lzx], dtype=torch.float32, device=x.device)
    getattr(lib, f"cflrelu_up2d_{tname}")(
        x,
        bias_t,
        filt,
        mask_t,
        z,
        ip,
        fp,
        threads=[Lzx, Lzy, N * C],
        group_size=[16, 16, 1],
    )
    getattr(lib, f"cflrelu_down2d_{tname}")(
        z,
        filt,
        y,
        ip,
        fp,
        threads=[Wo, Ho, N * C],
        group_size=[16, 16, 1],
    )
    return y


# ----------------------------------------------------------------------------
# Public API.


def mps_supported(x, fu, fd, b, up, down) -> bool:
    """Whether the Metal path can handle these arguments (otherwise use the reference)."""
    if not (isinstance(x, torch.Tensor) and x.device.type == "mps" and x.ndim == 4):
        return False
    if x.dtype not in _DTYPE_NAMES:
        return False
    if b is not None and b.dtype != x.dtype:
        return False
    for f in (fu, fd):
        if f is not None and (f.requires_grad or not 1 <= f.ndim <= 2):
            return False
    H, W = x.shape[2:]
    return (H * up) % down == 0 and (W * up) % down == 0


class _CircularFilteredLReluMPS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fu, fd, b, up, down, gain, slope, clamp, flip_filter):
        fu_f = _make_filter(fu, x.device, flip_filter)
        fd_f = _make_filter(fd, x.device, flip_filter)
        scale = up * up * gain
        clamp_v = _FLT_MAX if clamp is None else float(clamp)

        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[3]
        N, C, H, W = x.shape
        mask = None
        if need_grad:
            mask = torch.empty(
                [N * C, H * up, W * up], dtype=torch.uint8, device=x.device
            )
        mode = _MODE_FWD_MASK if need_grad else _MODE_FWD
        y = _run(x, b, fu_f, fd_f, up, down, scale, 1.0, slope, clamp_v, mode, mask)

        if need_grad:
            ctx.save_for_backward(mask)
            ctx.filters = (fu_f, fd_f)
            ctx.cfg = (up, down, scale, slope, clamp_v)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        (mask,) = ctx.saved_tensors
        fu_f, fd_f = ctx.filters
        up, down, scale, slope, clamp_v = ctx.cfg
        # Transpose of the forward: swap roles of the filters and of up/down.
        dx = _run(
            dy,
            None,
            fd_f.adjoint(),
            fu_f.adjoint(),
            down,
            up,
            scale,
            1.0,
            slope,
            clamp_v,
            _MODE_BWD,
            mask,
        )
        db = dx.sum([0, 2, 3]) if ctx.needs_input_grad[3] else None
        if not ctx.needs_input_grad[0]:
            dx = None
        return dx, None, None, db, None, None, None, None, None, None


def circular_filtered_lrelu_mps(
    x,
    fu=None,
    fd=None,
    b=None,
    up=1,
    down=1,
    gain=_SQRT2,
    slope=0.2,
    clamp=None,
    flip_filter=False,
):
    """Metal implementation of `circular_filtered_lrelu`. See that function for docs."""
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    if b is not None:
        assert b.ndim == 1 and b.shape[0] == x.shape[1]
    return _CircularFilteredLReluMPS.apply(
        x,
        fu,
        fd,
        b,
        up,
        down,
        float(gain),
        float(slope),
        None if clamp is None else float(clamp),
        bool(flip_filter),
    )
