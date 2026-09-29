"""Metal (PyTorch MPS) implementation of the upfirdn2d CUDA plugin interface.

`upfirdn2d()` routes MPS tensors through the same autograd functions as CUDA
(`_upfirdn2d_cuda`), which only needs the plugin's `upfirdn2d` entry point. The
backward pass is another upfirdn2d call, so gradients of any order work.
"""

from __future__ import annotations

from pathlib import Path

import torch

from unet_af.ops.mps_utils import DTYPE_NAMES, compile_library, params

_METAL_SOURCE = Path(__file__).with_name("upfirdn2d.metal")


def mps_supported(x, f) -> bool:
    """Whether the Metal path can handle these arguments (otherwise use the reference)."""
    if not (isinstance(x, torch.Tensor) and x.device.type == "mps" and x.ndim == 4):
        return False
    if x.dtype not in DTYPE_NAMES:
        return False
    return f is None or not f.requires_grad


def upfirdn2d(
    x: torch.Tensor,
    f: torch.Tensor,
    upx: int,
    upy: int,
    downx: int,
    downy: int,
    padx0: int,
    padx1: int,
    pady0: int,
    pady1: int,
    flip: bool,
    gain: float,
) -> torch.Tensor:
    """Same contract as the CUDA plugin: `f` is a 2D float32 filter."""
    assert x.ndim == 4 and f.ndim == 2
    assert upx >= 1 and upy >= 1 and downx >= 1 and downy >= 1
    x = x.contiguous()
    N, C, H, W = x.shape
    fh, fw = f.shape
    Wo = (W * upx + padx0 + padx1 - fw + downx) // downx
    Ho = (H * upy + pady0 + pady1 - fh + downy) // downy
    assert Wo >= 1 and Ho >= 1, "output must be at least 1x1"
    y = torch.empty([N, C, Ho, Wo], dtype=x.dtype, device=x.device)
    if y.numel() == 0:
        return y

    w = f.to(device=x.device, dtype=torch.float32)
    if not flip:
        w = w.flip([0, 1])  # convolution -> correlation weights
    ints = (W, H, upx, upy, downx, downy, padx0, pady0, fw, fh, Wo, Ho)
    ip, fp = params(ints, (float(gain),), x.device)
    kern = getattr(compile_library(_METAL_SOURCE), f"upfirdn2d_{DTYPE_NAMES[x.dtype]}")
    kern(
        x,
        w.contiguous(),
        y,
        ip,
        fp,
        threads=[Wo, Ho, N * C],
        group_size=[16, 16, 1],
    )
    return y
