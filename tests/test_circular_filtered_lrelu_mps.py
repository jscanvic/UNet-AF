"""Checks the Metal circular_filtered_lrelu against the reference implementation.

Run with `pytest tests/test_circular_filtered_lrelu_mps.py` or directly as a script.
"""

import itertools
import random

import pytest
import torch

from unet_af.ops.circular_filtered_lrelu.circular_filtered_lrelu import (
    _circular_filtered_lrelu_ref,
    circular_filtered_lrelu,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

DEVICE = torch.device("mps")


def _filter(kind, gen):
    if kind is None:
        return None
    if isinstance(kind, int):
        return torch.randn(kind, generator=gen)
    return torch.randn(*kind, generator=gen)


# (up, down, fu, fd, H, W): filter kinds are None, taps (1D), or (h, w) (2D).
GEOMETRIES = [
    (2, 2, 12, 12, 16, 16),  # UNet-AF StyleGAN3-style filters
    (2, 2, 13, 13, 8, 8),  # UNet-AF fitted filters, deepest scale
    (2, 2, 13, 13, 3, 5),  # image smaller than the filter: multiple wraps
    (1, 1, 5, 7, 9, 11),
    (1, 1, None, None, 6, 7),
    (2, 1, 6, 3, 10, 6),
    (1, 2, 4, 8, 10, 12),
    (4, 2, 16, 8, 6, 6),
    (2, 4, 8, 16, 8, 12),
    (2, 2, 1, 1, 7, 5),
    (2, 2, (5, 4), (3, 6), 8, 10),  # non-separable path
    (2, 2, 12, (1, 1), 8, 8),
    (1, 1, (3, 3), 5, 5, 9),  # mixed separable and non-separable
]


def _cases():
    rng = random.Random(0)
    for geom, flip in itertools.product(GEOMETRIES, (False, True)):
        slope = rng.choice([0.0, 0.2])
        clamp = rng.choice([None, 0.7])
        gain = rng.choice([1.0, 2**0.5])
        bias = rng.choice([True, False])
        yield (*geom, flip, slope, clamp, gain, bias)


CASES = list(_cases())


@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_forward_and_backward_match_reference(case):
    up, down, fu_k, fd_k, H, W, flip, slope, clamp, gain, bias = case
    gen = torch.Generator().manual_seed(hash(case) & 0xFFFF)
    N, C = 2, 3
    x = torch.randn(N, C, H, W, generator=gen, dtype=torch.float64)
    b = torch.randn(C, generator=gen, dtype=torch.float64) if bias else None
    fu, fd = _filter(fu_k, gen), _filter(fd_k, gen)
    kwargs = dict(
        up=up, down=down, gain=gain, slope=slope, clamp=clamp, flip_filter=flip
    )

    def to_mps(t: torch.Tensor) -> torch.Tensor:
        return t.to(DEVICE, torch.float32)

    def to_mps_opt(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else to_mps(t)

    x_ref = x.clone().requires_grad_()
    b_ref = None if b is None else b.clone().requires_grad_()
    y_ref = _circular_filtered_lrelu_ref(x_ref, fu=fu, fd=fd, b=b_ref, **kwargs)
    dy = torch.randn(y_ref.shape, generator=gen, dtype=torch.float64)
    y_ref.backward(dy)

    x_m = to_mps(x).requires_grad_()
    b_m = None if b is None else to_mps(b).requires_grad_()
    y_m = circular_filtered_lrelu(
        x_m, fu=to_mps_opt(fu), fd=to_mps_opt(fd), b=b_m, impl="metal", **kwargs
    )
    y_m.backward(to_mps(dy))

    tol = 1e-4
    torch.testing.assert_close(y_m.cpu().double(), y_ref.detach(), rtol=tol, atol=tol)
    assert x_m.grad is not None and x_ref.grad is not None
    torch.testing.assert_close(x_m.grad.cpu().double(), x_ref.grad, rtol=tol, atol=tol)
    if b_m is not None and b_ref is not None:
        assert b_m.grad is not None and b_ref.grad is not None
        torch.testing.assert_close(
            b_m.grad.cpu().double(), b_ref.grad, rtol=tol, atol=tol
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision(dtype):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 4, 16, 16, generator=gen)
    fu = torch.randn(12, generator=gen) / 4
    b = torch.randn(4, generator=gen)
    kwargs = dict(up=2, down=2, gain=1.0, slope=0.0)
    y_ref = _circular_filtered_lrelu_ref(
        x.double(), fu=fu, fd=fu, b=b.double(), **kwargs
    )
    y_m = circular_filtered_lrelu(
        x.to(DEVICE, dtype),
        fu=fu.to(DEVICE),
        fd=fu.to(DEVICE),
        b=b.to(DEVICE, dtype),
        **kwargs,
    )
    assert y_m.dtype == dtype
    tol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(y_m.cpu().double(), y_ref, rtol=tol, atol=tol)


def test_integer_shift_equivariance():
    """Circularly shifting the input by 2 px shifts the output by 2 px when up == down."""
    x = torch.randn(1, 2, 32, 32, device=DEVICE)
    fu = torch.randn(12, device=DEVICE)
    shift = lambda t: t.roll((2, 2), dims=(2, 3))  # noqa: E731
    y1 = shift(circular_filtered_lrelu(x, fu=fu, fd=fu, up=2, down=2))
    y2 = circular_filtered_lrelu(shift(x), fu=fu, fd=fu, up=2, down=2)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-5)


def test_non_contiguous_and_empty():
    x = torch.randn(2, 16, 16, 3, device=DEVICE).permute(0, 3, 1, 2)
    fu = torch.randn(12, device=DEVICE)
    y = circular_filtered_lrelu(x, fu=fu, fd=fu, up=2, down=2)
    y_c = circular_filtered_lrelu(x.contiguous(), fu=fu, fd=fu, up=2, down=2)
    torch.testing.assert_close(y, y_c)
    empty = torch.randn(0, 3, 8, 8, device=DEVICE)
    y_empty = circular_filtered_lrelu(empty, fu=fu, fd=fu, up=2, down=2)
    assert y_empty.shape == (0, 3, 8, 8)


def test_filter_grad_falls_back_to_reference():
    x = torch.randn(1, 2, 8, 8, device=DEVICE)
    fu = torch.randn(6, device=DEVICE, requires_grad=True)
    with pytest.warns(RuntimeWarning, match="no Metal kernel"):
        y = circular_filtered_lrelu(x, fu=fu, fd=fu, up=2, down=2)
    y.sum().backward()
    assert fu.grad is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
