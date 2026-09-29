"""Shared helpers for the Metal (Apple Silicon / PyTorch MPS) kernels in `unet_af.ops`.

Kernels are JIT-compiled from `.metal` sources with `torch.mps.compile_shader`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import torch

# torch dtype -> Metal type name, as used in the kernel function names.
DTYPE_NAMES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}


@cache
def compile_library(source: Path):
    return torch.mps.compile_shader(source.read_text())


@cache
def dummy_tensor(device: torch.device) -> torch.Tensor:
    """Placeholder for optional buffer arguments the kernel does not read."""
    return torch.zeros(1, dtype=torch.uint8, device=device)


_param_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def params(ints: tuple[int, ...], floats: tuple[float, ...], device: torch.device):
    """Device-resident int32/float32 parameter buffers, cached because models reuse a few shapes.

    Python lists passed to `compile_shader` kernels arrive as int64, so small
    tensors are used instead.
    """
    key = (ints, floats, device)
    if (hit := _param_cache.get(key)) is not None:
        return hit
    if len(_param_cache) > 1024:
        _param_cache.clear()
    ip = torch.tensor(ints, dtype=torch.int32, device=device)
    fp = torch.tensor(floats, dtype=torch.float32, device=device)
    _param_cache[key] = (ip, fp)
    return ip, fp
