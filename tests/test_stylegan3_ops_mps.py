"""Checks the Metal versions of the StyleGAN3 ops against their reference implementations.

Covers `filtered_lrelu` (zero padding), `upfirdn2d` and `bias_act`.
Run with `pytest tests/test_stylegan3_ops_mps.py` or directly as a script.
"""

import itertools
import random
import zlib

import pytest
import torch

from unet_af.ops.stylegan3.bias_act.bias_act import (
    _bias_act_ref,
    activation_funcs,
    bias_act,
)
from unet_af.ops.stylegan3.filtered_lrelu.filtered_lrelu import (
    _filtered_lrelu_ref,
    filtered_lrelu,
)
from unet_af.ops.stylegan3.upfirdn2d.upfirdn2d import (
    _upfirdn2d_ref,
    downsample2d,
    filter2d,
    upfirdn2d,
    upsample2d,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)

DEVICE = torch.device("mps")
TOL = 1e-4


def _to_mps(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE, torch.float32)


def _to_mps_opt(t: torch.Tensor | None) -> torch.Tensor | None:
    return None if t is None else _to_mps(t)


def _close(actual: torch.Tensor, expected: torch.Tensor, tol: float = TOL) -> None:
    """Float32 (or lower) result vs float64 reference; atol scales with the output."""
    expected = expected.detach()
    atol = tol * max(1.0, expected.abs().max().item()) if expected.numel() else tol
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=tol, atol=atol)


def _filter(kind, gen):
    if kind is None:
        return None
    if isinstance(kind, int):
        return torch.randn(kind, generator=gen)
    return torch.randn(*kind, generator=gen)


# ----------------------------------------------------------------------------
# filtered_lrelu

# (up, down, fu, fd, H, W, padding): filter kinds are None, taps (1D), or (h, w) (2D).
FLRELU_GEOMETRIES = [
    (2, 2, 12, 12, 16, 16, 11),  # UNet-AF zero-padding ablation, StyleGAN3 filters
    (2, 2, 13, 13, 8, 8, 12),  # fitted filters, deepest scale
    (2, 2, 12, 12, 128, 128, 11),  # largest scale: many tiles
    (2, 2, 12, 12, 13, 21, 0),  # no padding
    (2, 2, 6, 6, 10, 10, [3, 2, 1, 4]),  # asymmetric padding
    (2, 2, 6, 6, 12, 12, [-2, 1, 3, -1]),  # cropping
    (1, 1, 5, 7, 9, 11, [4, 4, 6, 6]),
    (1, 1, None, None, 6, 7, 0),
    (2, 1, 6, 3, 10, 6, 5),
    (1, 2, 4, 8, 10, 12, 7),
    (4, 2, 16, 8, 6, 6, 20),
    (2, 4, 8, 16, 8, 12, 17),
    (2, 2, 1, 1, 7, 5, 0),
    (2, 2, (5, 4), (3, 6), 8, 10, [3, 5, 2, 4]),  # non-separable path
    (2, 2, 12, (1, 1), 8, 8, 11),
    (1, 1, (3, 3), 5, 5, 9, 3),  # mixed separable and non-separable
]


def _flrelu_cases():
    rng = random.Random(0)
    for geom, flip in itertools.product(FLRELU_GEOMETRIES, (False, True)):
        slope = rng.choice([0.0, 0.2])
        clamp = rng.choice([None, 0.7])
        gain = rng.choice([1.0, 2**0.5])
        bias = rng.choice([True, False])
        yield (*geom, flip, slope, clamp, gain, bias)


FLRELU_CASES = list(_flrelu_cases())


@pytest.mark.parametrize(
    "case", FLRELU_CASES, ids=[str(i) for i in range(len(FLRELU_CASES))]
)
def test_filtered_lrelu_matches_reference(case):
    up, down, fu_k, fd_k, H, W, padding, flip, slope, clamp, gain, bias = case
    gen = torch.Generator().manual_seed(zlib.crc32(repr(case).encode()))
    N, C = 2, 3
    x = torch.randn(N, C, H, W, generator=gen, dtype=torch.float64)
    b = torch.randn(C, generator=gen, dtype=torch.float64) if bias else None
    fu, fd = _filter(fu_k, gen), _filter(fd_k, gen)
    kwargs = dict(
        up=up,
        down=down,
        padding=padding,
        gain=gain,
        slope=slope,
        clamp=clamp,
        flip_filter=flip,
    )

    x_ref = x.clone().requires_grad_()
    b_ref = None if b is None else b.clone().requires_grad_()
    y_ref = _filtered_lrelu_ref(x_ref, fu=fu, fd=fd, b=b_ref, **kwargs)
    dy = torch.randn(y_ref.shape, generator=gen, dtype=torch.float64)
    y_ref.backward(dy)

    x_m = _to_mps(x).requires_grad_()
    b_m = None if b is None else _to_mps(b).requires_grad_()
    y_m = filtered_lrelu(
        x_m, fu=_to_mps_opt(fu), fd=_to_mps_opt(fd), b=b_m, impl="metal", **kwargs
    )
    y_m.backward(_to_mps(dy))

    assert y_m.shape == y_ref.shape
    _close(y_m, y_ref)
    assert x_m.grad is not None and x_ref.grad is not None
    _close(x_m.grad, x_ref.grad)
    if b_m is not None and b_ref is not None:
        assert b_m.grad is not None and b_ref.grad is not None
        _close(b_m.grad, b_ref.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_filtered_lrelu_low_precision(dtype):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 4, 16, 16, generator=gen)
    fu = torch.randn(12, generator=gen) / 4
    b = torch.randn(4, generator=gen)
    kwargs = dict(up=2, down=2, padding=11, gain=1.0, slope=0.0)
    y_ref = _filtered_lrelu_ref(x.double(), fu=fu, fd=fu, b=b.double(), **kwargs)
    y_m = filtered_lrelu(
        x.to(DEVICE, dtype),
        fu=fu.to(DEVICE),
        fd=fu.to(DEVICE),
        b=b.to(DEVICE, dtype),
        **kwargs,
    )
    assert y_m.dtype == dtype
    _close(y_m, y_ref, 2e-2 if dtype == torch.bfloat16 else 3e-3)


def test_filtered_lrelu_filter_grad_falls_back_to_reference():
    x = torch.randn(1, 2, 8, 8, device=DEVICE)
    fu = torch.randn(6, device=DEVICE, requires_grad=True)
    with pytest.warns(RuntimeWarning, match="no Metal kernel"):
        y = filtered_lrelu(x, fu=fu, fd=fu, up=2, down=2, padding=5)
    y.sum().backward()
    assert fu.grad is not None


# ----------------------------------------------------------------------------
# upfirdn2d

# (up, down, padding, filter kind, H, W)
UPFIRDN_CASES = [
    (1, 1, 0, None, 7, 9),
    (2, 1, [2, 1, 2, 1], 4, 8, 8),  # separable: two passes
    (1, 2, 3, 8, 12, 10),
    ([2, 3], [1, 2], [3, 1, 0, 4], (5, 4), 6, 7),
    (3, 2, [-1, 2, 4, -2], (6, 6), 9, 8),  # cropping
    (2, 2, 11, 12, 16, 16),
    (1, 1, 2, (1, 1), 5, 5),
    (4, 4, [6, 5, 6, 5], 12, 5, 7),
]


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize(
    "case", UPFIRDN_CASES, ids=[str(i) for i in range(len(UPFIRDN_CASES))]
)
def test_upfirdn2d_matches_reference(case, flip):
    up, down, padding, f_k, H, W = case
    gen = torch.Generator().manual_seed(zlib.crc32(repr(case).encode()))
    x = torch.randn(2, 3, H, W, generator=gen, dtype=torch.float64)
    f = _filter(f_k, gen)
    kwargs = dict(up=up, down=down, padding=padding, flip_filter=flip, gain=1.5)

    x_ref = x.clone().requires_grad_()
    y_ref = _upfirdn2d_ref(x_ref, f, **kwargs)
    dy = torch.randn(y_ref.shape, generator=gen, dtype=torch.float64)
    y_ref.backward(dy)

    x_m = _to_mps(x).requires_grad_()
    y_m = upfirdn2d(x_m, _to_mps_opt(f), impl="metal", **kwargs)
    y_m.backward(_to_mps(dy))

    assert y_m.shape == y_ref.shape
    _close(y_m, y_ref)
    assert x_m.grad is not None and x_ref.grad is not None
    _close(x_m.grad, x_ref.grad)


def test_upfirdn2d_wrappers_and_low_precision():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1, 2, 10, 12, generator=gen)
    f = torch.randn(8, generator=gen)
    for fn in (upsample2d, downsample2d, filter2d):
        _close(fn(_to_mps(x), _to_mps(f)), fn(x.double(), f, impl="ref"))
    y16 = upsample2d(x.to(DEVICE, torch.float16), _to_mps(f))
    assert y16.dtype == torch.float16
    y_ref = upsample2d(x.double(), f, impl="ref")
    # fp16 rounding of the input and output gives errors proportional to the output's scale.
    atol = 2e-3 * y_ref.abs().max().item()
    torch.testing.assert_close(y16.cpu().double(), y_ref, rtol=2e-3, atol=atol)


def test_upfirdn2d_second_order():
    """The backward pass is itself an upfirdn2d call, so it is differentiable."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(1, 2, 8, 8, generator=gen, dtype=torch.float64)
    w = torch.randn(1, 2, 16, 16, generator=gen, dtype=torch.float64)
    f = torch.randn(6, generator=gen)

    def grad_norm(x, w, f, impl):
        x = x.requires_grad_()
        y = upfirdn2d(x, f, up=2, padding=[3, 2, 3, 2], impl=impl)
        (dx,) = torch.autograd.grad((y * w).sum(), x, create_graph=True)
        return dx

    w_ref = w.clone().requires_grad_()
    grad_norm(x.clone(), w_ref, f, "ref").square().sum().backward()
    w_m = _to_mps(w).requires_grad_()
    grad_norm(_to_mps(x), w_m, _to_mps(f), "metal").square().sum().backward()
    assert w_m.grad is not None and w_ref.grad is not None
    _close(w_m.grad, w_ref.grad)


# ----------------------------------------------------------------------------
# bias_act


@pytest.mark.parametrize("clamp", [None, 0.5])
@pytest.mark.parametrize("act", list(activation_funcs))
def test_bias_act_matches_reference(act, clamp):
    gen = torch.Generator().manual_seed(4)
    x = 2 * torch.randn(2, 5, 6, 7, generator=gen, dtype=torch.float64)
    b = torch.randn(5, generator=gen, dtype=torch.float64)
    kwargs = dict(dim=1, act=act, clamp=clamp)

    x_ref, b_ref = x.clone().requires_grad_(), b.clone().requires_grad_()
    y_ref = _bias_act_ref(x_ref, b_ref, **kwargs)
    dy = torch.randn(y_ref.shape, generator=gen, dtype=torch.float64)
    y_ref.backward(dy)

    x_m, b_m = _to_mps(x).requires_grad_(), _to_mps(b).requires_grad_()
    y_m = bias_act(x_m, b_m, impl="metal", **kwargs)
    y_m.backward(_to_mps(dy))

    _close(y_m, y_ref)
    assert x_m.grad is not None and x_ref.grad is not None
    assert b_m.grad is not None and b_ref.grad is not None
    _close(x_m.grad, x_ref.grad)
    # Looser: each bias gradient sums 84 float32 terms.
    _close(b_m.grad, b_ref.grad, 1e-3)


@pytest.mark.parametrize(
    "act", [a for a, spec in activation_funcs.items() if spec.has_2nd_grad]
)
def test_bias_act_second_order(act):
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, 4, 5, generator=gen, dtype=torch.float64)
    b = torch.randn(4, generator=gen, dtype=torch.float64)
    w = torch.randn(3, 4, 5, generator=gen, dtype=torch.float64)

    def run(x, b, w, impl):
        x, b = x.requires_grad_(), b.requires_grad_()
        y = bias_act(x, b, dim=1, act=act, impl=impl)
        (dx,) = torch.autograd.grad((y * w).sum(), x, create_graph=True)
        dx.square().sum().backward()
        assert x.grad is not None and b.grad is not None
        return x.grad, b.grad

    gx_ref, gb_ref = run(x.clone(), b.clone(), w, "ref")
    gx_m, gb_m = run(_to_mps(x), _to_mps(b), _to_mps(w), "metal")
    _close(gx_m, gx_ref, 1e-3)
    _close(gb_m, gb_ref, 1e-3)


def test_bias_act_channels_last_and_low_precision():
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(2, 3, 4, 5, generator=gen)
    b = torch.randn(3, generator=gen)
    y_ref = _bias_act_ref(x.double(), b.double(), act="lrelu")
    x_cl = _to_mps(x).contiguous(memory_format=torch.channels_last)
    y_cl = bias_act(x_cl, _to_mps(b), act="lrelu")
    assert y_cl.is_contiguous(memory_format=torch.channels_last)
    _close(y_cl, y_ref)
    for dtype, tol in ((torch.float16, 3e-3), (torch.bfloat16, 2e-2)):
        y = bias_act(x.to(DEVICE, dtype), b.to(DEVICE, dtype), act="lrelu")
        assert y.dtype == dtype
        _close(y, y_ref, tol)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
