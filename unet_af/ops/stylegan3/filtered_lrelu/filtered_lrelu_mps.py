"""Metal (Apple Silicon / PyTorch MPS) implementation of `filtered_lrelu`.

Runs the fused kernels from `circular_filtered_lrelu_mps` with zero padding instead
of circular wrapping. Supports float32, float16 and bfloat16, and gradients w.r.t.
`x` and `b`; gradients w.r.t. the filters are not supported, matching the CUDA
implementation.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from unet_af.ops.circular_filtered_lrelu.circular_filtered_lrelu_mps import (
    _FLT_MAX,
    _MODE_BWD,
    _MODE_FWD,
    _MODE_FWD_MASK,
    _make_filter,
    _run,
)
from unet_af.ops.mps_utils import DTYPE_NAMES

_SQRT2 = math.sqrt(2)


def _filter_taps(f: torch.Tensor | None) -> tuple[int, int]:
    """(taps_x, taps_y), counting a 1D filter along both axes."""
    if f is None:
        return 1, 1
    return f.shape[-1], f.shape[0]


def _sizes(H, W, taps_u, taps_d, up, down, padding):
    """Intermediate extent and output size, both as (height, width)."""
    px0, px1, py0, py1 = padding
    (fux, fuy), (fdx, fdy) = taps_u, taps_d
    lzx = W * up + px0 + px1 - fux + 1
    lzy = H * up + py0 + py1 - fuy + 1
    wo = (lzx - fdx + down) // down
    ho = (lzy - fdy + down) // down
    return (lzy, lzx), (ho, wo)


def mps_supported(x, fu, fd, b, up, down, padding) -> bool:
    """Whether the Metal path can handle these arguments (otherwise use the reference)."""
    if not (isinstance(x, torch.Tensor) and x.device.type == "mps" and x.ndim == 4):
        return False
    if x.dtype not in DTYPE_NAMES:
        return False
    if b is not None and b.dtype != x.dtype:
        return False
    for f in (fu, fd):
        if f is not None and (f.requires_grad or not 1 <= f.ndim <= 2):
            return False
    H, W = x.shape[2:]
    taps_d = _filter_taps(fd)
    (lzy, lzx), (ho, wo) = _sizes(H, W, _filter_taps(fu), taps_d, up, down, padding)
    return lzx >= taps_d[0] and lzy >= taps_d[1] and ho >= 1 and wo >= 1


class _FilteredLReluMPS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fu, fd, b, up, down, padding, gain, slope, clamp, flip_filter):
        px0, _, py0, _ = padding
        # Forward offsets: fu reads the padded image starting at -pad0, fd starts at 0.
        fu_f = dataclasses.replace(
            _make_filter(fu, x.device, flip_filter), offset=(px0, py0)
        )
        fd_f = dataclasses.replace(
            _make_filter(fd, x.device, flip_filter), offset=(0, 0)
        )
        N, C, H, W = x.shape
        z_size, out_size = _sizes(H, W, fu_f.taps, fd_f.taps, up, down, padding)
        scale = up * up * gain
        clamp_v = _FLT_MAX if clamp is None else float(clamp)

        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[3]
        mask = None
        if need_grad:
            mask = torch.empty([N * C, *z_size], dtype=torch.uint8, device=x.device)
        mode = _MODE_FWD_MASK if need_grad else _MODE_FWD
        y = _run(
            x,
            b,
            fu_f,
            fd_f,
            up,
            down,
            scale,
            1.0,
            slope,
            clamp_v,
            mode,
            mask,
            out_size=out_size,
            z_size=z_size,
            zero_pad=True,
        )

        if need_grad:
            ctx.save_for_backward(mask)
            ctx.filters = (fu_f, fd_f)
            ctx.cfg = (up, down, scale, slope, clamp_v, (H, W), z_size)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        (mask,) = ctx.saved_tensors
        fu_f, fd_f = ctx.filters
        up, down, scale, slope, clamp_v, in_size, z_size = ctx.cfg
        # Transpose of the forward: swap roles of the filters and of up/down. The
        # intermediate grid (and so the mask layout) is the same as in the forward.
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
            out_size=in_size,
            z_size=z_size,
            zero_pad=True,
        )
        db = dx.sum([0, 2, 3]) if ctx.needs_input_grad[3] else None
        if not ctx.needs_input_grad[0]:
            dx = None
        return dx, None, None, db, None, None, None, None, None, None, None


def filtered_lrelu_mps(
    x,
    fu=None,
    fd=None,
    b=None,
    up=1,
    down=1,
    padding=(0, 0, 0, 0),
    gain=_SQRT2,
    slope=0.2,
    clamp=None,
    flip_filter=False,
):
    """Metal implementation of `filtered_lrelu`. `padding` is `(px0, px1, py0, py1)`."""
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    if b is not None:
        assert b.ndim == 1 and b.shape[0] == x.shape[1]
    return _FilteredLReluMPS.apply(
        x,
        fu,
        fd,
        b,
        up,
        down,
        tuple(padding),
        float(gain),
        float(slope),
        None if clamp is None else float(clamp),
        bool(flip_filter),
    )
