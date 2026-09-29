"""Metal (PyTorch MPS) implementation of the bias_act CUDA plugin interface.

`bias_act()` routes MPS tensors through the same autograd functions as CUDA
(`_bias_act_cuda`), which only needs the plugin's `bias_act` entry point, so first
and second order gradients work as they do on CUDA.
"""

from __future__ import annotations

from pathlib import Path

import torch

from unet_af.ops.mps_utils import DTYPE_NAMES, compile_library, params

_METAL_SOURCE = Path(__file__).with_name("bias_act.metal")
_INT_MAX = 2**31 - 1


def mps_supported(x, b) -> bool:
    """Whether the Metal path can handle these arguments (otherwise use the reference)."""
    if not (isinstance(x, torch.Tensor) and x.device.type == "mps"):
        return False
    if x.dtype not in DTYPE_NAMES or x.numel() > _INT_MAX:
        return False
    return b is None or b.dtype == x.dtype


def _is_dense(t: torch.Tensor) -> bool:
    """Non-overlapping and dense: the strides are a permutation of a contiguous layout."""
    expected = 1
    for stride, size in sorted(
        (s, n) for n, s in zip(t.shape, t.stride(), strict=True) if n > 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _same_layout(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape and the same strides along every dimension of size >= 2."""
    if a.shape != b.shape:
        return False
    return all(
        n < 2 or sa == sb
        for n, sa, sb in zip(a.shape, a.stride(), b.stride(), strict=True)
    )


def bias_act(
    x: torch.Tensor,
    b: torch.Tensor,
    xref: torch.Tensor,
    yref: torch.Tensor,
    dy: torch.Tensor,
    grad: int,
    dim: int,
    act: int,
    alpha: float,
    gain: float,
    clamp: float,
) -> torch.Tensor:
    """Same contract as the CUDA plugin: empty tensors stand for missing inputs."""
    assert _is_dense(x), "x must be non-overlapping and dense"
    for t in (xref, yref, dy):
        assert t.numel() == 0 or _same_layout(t, x), "xref, yref and dy must match x"
    y = torch.empty_like(x)
    assert _same_layout(y, x)
    if x.numel() == 0:
        return y
    has_b = b.numel() > 0
    ints = (
        x.numel(),
        b.numel(),
        x.stride(dim) if has_b else 1,
        grad,
        int(has_b),
        int(xref.numel() > 0),
        int(yref.numel() > 0),
        int(dy.numel() > 0),
    )
    ip, fp = params(ints, (float(alpha), float(gain), float(clamp)), x.device)

    def buf(t: torch.Tensor) -> torch.Tensor:
        return t if t.numel() > 0 else x  # never read when absent

    kern = getattr(
        compile_library(_METAL_SOURCE), f"bias_act_{DTYPE_NAMES[x.dtype]}_{act}"
    )
    kern(
        x,
        buf(b),
        buf(xref),
        buf(yref),
        buf(dy),
        y,
        ip,
        fp,
        threads=[x.numel(), 1, 1],
        group_size=[256, 1, 1],
    )
    return y
