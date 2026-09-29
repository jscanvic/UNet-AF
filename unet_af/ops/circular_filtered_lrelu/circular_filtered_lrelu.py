# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import warnings

import numpy as np
import torch

from unet_af.ops.stylegan3 import bias_act, custom_ops, misc, upfirdn2d

# ----------------------------------------------------------------------------

_plugin = None


def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name="circular_filtered_lrelu_plugin",
            sources=[
                "circular_filtered_lrelu.cpp",
                "circular_filtered_lrelu_wr.cu",
                "circular_filtered_lrelu_rd.cu",
                "circular_filtered_lrelu_ns.cu",
            ],
            headers=["circular_filtered_lrelu.h", "circular_filtered_lrelu.cu"],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=["--use_fast_math", "--allow-unsupported-compiler"],
        )
    return True


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0]  # width, height


# ----------------------------------------------------------------------------


def circular_filtered_lrelu(
    x,
    fu=None,
    fd=None,
    b=None,
    up=1,
    down=1,
    gain=np.sqrt(2),
    slope=0.2,
    clamp=None,
    flip_filter=False,
    impl="cuda",
):
    r"""Filtered leaky ReLU with circular padding for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image circularly based on filter footprints.

    4. Convolve the image with the specified upsampling FIR filter (`fu`).

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Pad the image circularly based on downsampling filter footprint.

    9. Convolve the image with the specified downsampling FIR filter (`fd`).

    10. Downsample the image by keeping every Nth pixel (`down`).

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'`, `'cuda'` or `'metal'` (default: `'cuda'`).
                     On MPS tensors, `'cuda'` and `'metal'` both select the Metal kernels.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ["ref", "cuda", "metal"]
    if impl in ("cuda", "metal") and x.device.type == "mps":
        from .circular_filtered_lrelu_mps import (
            circular_filtered_lrelu_mps,
            mps_supported,
        )

        if mps_supported(x, fu, fd, b, up, down):
            return circular_filtered_lrelu_mps(
                x,
                fu=fu,
                fd=fd,
                b=b,
                up=up,
                down=down,
                gain=gain,
                slope=slope,
                clamp=clamp,
                flip_filter=flip_filter,
            )
        warnings.warn(
            "circular_filtered_lrelu called with parameters that have no Metal kernel, using reference fallback",
            RuntimeWarning,
            stacklevel=2,
        )
    if impl == "cuda" and x.device.type == "cuda" and _init():
        return _circular_filtered_lrelu_cuda(
            up=up,
            down=down,
            gain=gain,
            slope=slope,
            clamp=clamp,
            flip_filter=flip_filter,
        ).apply(x, fu, fd, b, None, 0, 0)
    return _circular_filtered_lrelu_ref(
        x,
        fu=fu,
        fd=fd,
        b=b,
        up=up,
        down=down,
        gain=gain,
        slope=slope,
        clamp=clamp,
        flip_filter=flip_filter,
    )


# ----------------------------------------------------------------------------


@misc.profiled_function
def _circular_filtered_lrelu_ref(
    x,
    fu=None,
    fd=None,
    b=None,
    up=1,
    down=1,
    gain=np.sqrt(2),
    slope=0.2,
    clamp=None,
    flip_filter=False,
):
    """Slow and memory-inefficient reference implementation of `circular_filtered_lrelu()` using
    existing `upfirdn2d()` and `bias_act()` ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if fu is None:
        fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    if fd is None:
        fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up) // down
    out_h = (in_h * up) // down

    # Compute using existing ops and circular padding.
    x = bias_act(x=x, b=b)  # Apply bias.

    # Upsample: insert up-1 zeros after each pixel.
    ones_1x1 = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    x = upfirdn2d(
        x=x,
        f=ones_1x1,
        up=up,
        padding=0,
        gain=up**2,
        flip_filter=flip_filter,
        impl="ref",
    )

    # Circular pad for fu
    px0_fu = (fu_w - 1) // 2
    px1_fu = (fu_w - 1) - px0_fu
    py0_fu = (fu_h - 1) // 2
    py1_fu = (fu_h - 1) - py0_fu
    x = torch.nn.functional.pad(x, [px0_fu, px1_fu, py0_fu, py1_fu], mode="circular")

    # Convolve with fu (using padding=0)
    x = upfirdn2d(x=x, f=fu, padding=0, flip_filter=flip_filter, impl="ref")

    # Bias, leaky ReLU, clamp
    x = bias_act(x=x, act="lrelu", alpha=slope, gain=gain, clamp=clamp)

    # Circular pad for fd
    px0_fd = (fd_w - 1) // 2
    px1_fd = (fd_w - 1) - px0_fd
    py0_fd = (fd_h - 1) // 2
    py1_fd = (fd_h - 1) - py0_fd
    x = torch.nn.functional.pad(x, [px0_fd, px1_fd, py0_fd, py1_fd], mode="circular")

    # Convolve with fd and downsample (using padding=0)
    x = upfirdn2d(x=x, f=fd, down=down, padding=0, flip_filter=flip_filter, impl="ref")

    # Check output shape & dtype.
    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x


# ----------------------------------------------------------------------------

_circular_filtered_lrelu_cuda_cache = dict()


def _circular_filtered_lrelu_cuda(
    up=1, down=1, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False
):
    """Fast CUDA implementation of `circular_filtered_lrelu()` using custom ops."""
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else "inf")

    # Lookup from cache.
    key = (up, down, gain, slope, clamp, flip_filter)
    if key in _circular_filtered_lrelu_cuda_cache:
        return _circular_filtered_lrelu_cuda_cache[key]

    # Forward op.
    class CircularFilteredLReluCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy):  # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels (faster than separable).
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale factor is 1.
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            # Missing sign input tensor.
            if si is None:
                si = torch.empty([0])

            # Missing bias tensor.
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)

            # Construct internal sign tensor only if gradients are needed.
            write_signs = (si.numel() == 0) and (x.requires_grad or b.requires_grad)

            # Warn if input storage strides are not in decreasing order due to e.g. channels-last layout.
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:], strict=False)):
                warnings.warn(
                    "low-performance memory layout detected in circular_filtered_lrelu input",
                    RuntimeWarning,
                )

            # Calculate padding parameters internally from filters
            fu_w, fu_h = _get_filter_size(fu)
            fd_w, fd_h = _get_filter_size(fd)
            px0 = (fu_w - 1) // 2 + (fd_w - 1) // 2
            px1 = (fu_w - 1) - (fu_w - 1) // 2 + (fd_w - 1) - (fd_w - 1) // 2
            py0 = (fu_h - 1) // 2 + (fd_h - 1) // 2
            py1 = (fu_h - 1) - (fu_h - 1) // 2 + (fd_h - 1) - (fd_h - 1) // 2

            # Call C++/Cuda plugin if datatype is supported.
            if x.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(
                    x.device
                ):
                    warnings.warn(
                        "circular_filtered_lrelu called with non-default cuda stream but concurrent execution is not supported",
                        RuntimeWarning,
                    )
                y, so, return_code = _plugin.filtered_lrelu(
                    x,
                    fu,
                    fd,
                    b,
                    si,
                    up,
                    down,
                    px0,
                    px1,
                    py0,
                    py1,
                    sx,
                    sy,
                    gain,
                    slope,
                    clamp,
                    flip_filter,
                    write_signs,
                )
            # y, so, return_code = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1

            # Fall back to generic implementation if no CUDA kernel found.
            if return_code < 0:
                warnings.warn(
                    "circular_filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback",
                    RuntimeWarning,
                )

                y = x.add(b.unsqueeze(-1).unsqueeze(-1))  # Add bias.

                # Upsample and convolve with fu using circular padding
                ones_1x1 = torch.ones([1, 1], dtype=torch.float32, device=x.device)
                y = upfirdn2d(
                    x=y,
                    f=ones_1x1,
                    up=up,
                    padding=0,
                    gain=up**2,
                    flip_filter=flip_filter,
                )
                px0_fu = (fu_w - 1) // 2
                px1_fu = (fu_w - 1) - px0_fu
                py0_fu = (fu_h - 1) // 2
                py1_fu = (fu_h - 1) - py0_fu
                y = torch.nn.functional.pad(
                    y, [px0_fu, px1_fu, py0_fu, py1_fu], mode="circular"
                )
                y = upfirdn2d(x=y, f=fu, padding=0, flip_filter=flip_filter)

                # Activation function and sign handling
                so = _plugin.filtered_lrelu_act_(
                    y, si, sx, sy, gain, slope, clamp, write_signs
                )  # Modifies y in-place.

                # Downsample and convolve with fd using circular padding
                px0_fd = (fd_w - 1) // 2
                px1_fd = (fd_w - 1) - px0_fd
                py0_fd = (fd_h - 1) // 2
                py1_fd = (fd_h - 1) - py0_fd
                y = torch.nn.functional.pad(
                    y, [px0_fd, px1_fd, py0_fd, py1_fd], mode="circular"
                )
                y = upfirdn2d(x=y, f=fd, down=down, padding=0, flip_filter=flip_filter)

            # Prepare for gradient computation.
            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx = None
            dfu = None
            assert not ctx.needs_input_grad[1]
            dfd = None
            assert not ctx.needs_input_grad[2]
            db = None
            dsi = None
            assert not ctx.needs_input_grad[4]
            dsx = None
            assert not ctx.needs_input_grad[5]
            dsy = None
            assert not ctx.needs_input_grad[6]

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                fu_w, fu_h = _get_filter_size(fu)
                fd_w, fd_h = _get_filter_size(fd)
                px0 = (fu_w - 1) // 2 + (fd_w - 1) // 2
                py0 = (fu_h - 1) // 2 + (fd_h - 1) // 2

                gg = gain * (up**2) / (down**2)
                ff = not flip_filter
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0] - 1) + py0
                dx = _circular_filtered_lrelu_cuda(
                    up=down, down=up, gain=gg, slope=slope, clamp=None, flip_filter=ff
                ).apply(dy, fd, fu, None, si, sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _circular_filtered_lrelu_cuda_cache[key] = CircularFilteredLReluCuda
    return CircularFilteredLReluCuda


# ----------------------------------------------------------------------------
