import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_af.ops import circular_filtered_lrelu
from unet_af.ops.stylegan3 import filtered_lrelu
from unet_af.io import load_filter_bank


def test_pad(model, L, modulo=16):
    """
    Pads the image to fit the model's expected image size.

    Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
    """
    h, w = L.size()[-2:]
    padding_bottom = int(np.ceil(h / modulo) * modulo - h)
    padding_right = int(np.ceil(w / modulo) * modulo - w)
    L = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From Mohan et al.

    "Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks"
    S. Mohan, Z. Kadkhodaie, E. P. Simoncelli, C. Fernandez-Granda
    Int'l. Conf. on Learning Representations (ICLR), Apr 2020.
    """

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, use_bias=False, affine=True
    ):
        super().__init__(num_features, eps, momentum)
        self.use_bias = use_bias
        self.affine = affine

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** 0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (
                            1 - self.momentum
                        ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** 0.5 + self.eps)
        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/layer_norm.py#L50
class LayerNorm_AF(nn.Module):
    """
    Alias-Free Layer Normalization

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param int in_channels: number of input channels.
    :param int out_channels: number of output channels.
    :param bool residual: if True, the output is the sum of the input and the denoised image.
    :param bool cat: if True, the network uses skip connections.
    :param int scales: number of scales in the network.
    :param bool rotation_equivariant: if True, the network is rotation-equivariant.

    :param Union[int, list, torch.Size] normalized_shape: Input shape from an expected input of size.
    :param float eps: A value added to the denominator for numerical stability. Default: 1e-6.
    :param bool bias: If set to False, the layer will not learn an additive bias. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-6, bias=True, type="AF"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = None
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        # In our previous implementation, self.u_dims was wrongly set to (1, 2, 3) but it is actually meant to be equal to 1
        # in the original implementation.
        # https://github.com/hmichaeli/alias_free_convnets/blob/7ef0a6eea3990c015d746d13a5cd79735809dd45/models/layer_norm.py#L16
        if type == "AF":
            self.u_dims = 1
        elif type == "CHW":
            self.u_dims = (1, 2, 3)
        else:
            raise ValueError(f"type={type} is not supported")
        self.s_dims = (1, 2, 3)

    def forward(self, x):
        """
        Forward pass for layer normalization.

        :param torch.Tensor x: Input tensor
        """
        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x
        if self.bias is not None:
            x = x + self.bias[:, None, None]
        return x


class LayerNormC(nn.Module):
    """
    LayerNorm over the channel dimension for (B, C, H, W) tensors.
    """

    def __init__(self, channels, eps=1e-5, bias=True):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=bias)

    def forward(self, x):
        # x: (B, C, H, W) -> permute to channels-last for nn.LayerNorm
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)


class KaiserSincReLU(nn.Module):
    """
    Wrapper around the Karras et al. filtered ReLU with optional circular padding.
    """

    def __init__(self, padding_mode: str = "zeros"):
        super().__init__()

        # Copied from fd (adapted for x2 up/downsampling)
        fu = torch.tensor(
            [
                0.0006092520197853446,
                0.005267132073640823,
                -0.018742641434073448,
                -0.04972728714346886,
                0.1221000999212265,
                0.4404934346675873,
                0.4404934346675873,
                0.1221000999212265,
                -0.04972728714346886,
                -0.018742641434073448,
                0.005267132073640823,
                0.0006092520197853446,
            ]
        )
        self.register_buffer("fu", fu, persistent=False)

        # Kaiser-Sinc with 12 taps and a target cut-off of 1/4 Hz
        fd = torch.tensor(
            [
                0.0006092520197853446,
                0.005267132073640823,
                -0.018742641434073448,
                -0.04972728714346886,
                0.1221000999212265,
                0.4404934346675873,
                0.4404934346675873,
                0.1221000999212265,
                -0.04972728714346886,
                -0.018742641434073448,
                0.005267132073640823,
                0.0006092520197853446,
            ]
        )
        self.register_buffer("fd", fd, persistent=False)

        self.padding = max(fu.numel(), fd.numel()) - 1

        self.up = 2
        self.down = 2

        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode == "circular":
            x = F.pad(
                x,
                (
                    math.floor(self.padding / 2),
                    math.ceil(self.padding / 2),
                    math.floor(self.padding / 2),
                    math.ceil(self.padding / 2),
                ),
                mode="circular",
            )
            padding = 0
        elif self.padding_mode == "zeros":
            padding = self.padding
        else:
            raise ValueError(f"Unexpected padding_mode={self.padding_mode}")

        return filtered_lrelu(
            x,
            up=self.up,
            down=self.down,
            fu=self.fu,
            fd=self.fd,
            padding=padding,
            gain=1.0,
            slope=0.0,
        )


class KaiserSincReLUv2(nn.Module):
    """
    Wrapper around the Karras et al. filtered ReLU with optional circular padding.
    """

    # NOTE: The cache capacity of 5 corresponds to the number of default scales in the UNet.
    def __init__(
        self,
        filters_kind: str = "fitted",
        padding_mode: str = "zeros",
        impl: str = "cuda",
        cache_capacity: int = 5,
    ):
        super().__init__()
        self.up = 2
        self.down = 2
        self.padding_mode = padding_mode
        self.impl = impl

        # Cache the filters
        get_filters = self.get_filters
        self.get_filters = lru_cache(maxsize=cache_capacity)(
            lambda *args, **kwargs: get_filters(
                *args, **{**kwargs, "filters_kind": filters_kind}
            )
        )

        # Preload cache
        preload_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preload_dtype = torch.float32
        for size in [128, 64, 32, 16, 8]:
            _ = self.get_filters(size, device=preload_device, dtype=preload_dtype)

    @classmethod
    def get_filters(
        cls,
        image_size: int,
        device: torch.device,
        dtype: torch.dtype,
        filters_kind: str = "fitted",
    ):
        if filters_kind == "fitted":
            filter_bank = load_filter_bank()
            data = filter_bank[image_size]
            fu, fd = data["fu"], data["fd"]
        elif filters_kind == "stylegan3":
            fu = torch.tensor(
                [
                    0.0006092520197853446,
                    0.005267132073640823,
                    -0.018742641434073448,
                    -0.04972728714346886,
                    0.1221000999212265,
                    0.4404934346675873,
                    0.4404934346675873,
                    0.1221000999212265,
                    -0.04972728714346886,
                    -0.018742641434073448,
                    0.005267132073640823,
                    0.0006092520197853446,
                ]
            )
            fd = torch.tensor(
                [
                    0.0006092520197853446,
                    0.005267132073640823,
                    -0.018742641434073448,
                    -0.04972728714346886,
                    0.1221000999212265,
                    0.4404934346675873,
                    0.4404934346675873,
                    0.1221000999212265,
                    -0.04972728714346886,
                    -0.018742641434073448,
                    0.005267132073640823,
                    0.0006092520197853446,
                ]
            )
        elif (
            filters_kind == "stylegan3:synthesis.L12_1044_32.down_filter"
        ):  # Windowed sinc, 1/4 Hz cut-off
            fu = torch.tensor(
                [
                    0.023374248296022415,
                    0.034449294209480286,
                    -0.05088069662451744,
                    -0.0786152333021164,
                    0.1396104395389557,
                    0.43206194043159485,
                    0.43206194043159485,
                    0.1396104395389557,
                    -0.0786152333021164,
                    -0.05088069662451744,
                    0.034449294209480286,
                    0.023374248296022415,
                ]
            )

            fd = torch.tensor(
                [
                    0.023374248296022415,
                    0.034449294209480286,
                    -0.05088069662451744,
                    -0.0786152333021164,
                    0.1396104395389557,
                    0.43206194043159485,
                    0.43206194043159485,
                    0.1396104395389557,
                    -0.0786152333021164,
                    -0.05088069662451744,
                    0.034449294209480286,
                    0.023374248296022415,
                ]
            )
        else:
            raise ValueError(
                f"filters_kind={filters_kind} is not supported; expected 'fitted' or 'stylegan3'"
            )

        fu, fd = fu.to(device=device, dtype=dtype), fd.to(device=device, dtype=dtype)
        return fu, fd

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, f"Input must be square (H == W), got {H}x{W}"

        fu, fd = self.get_filters(H, device=x.device, dtype=x.dtype)

        if self.padding_mode == "circular":
            return circular_filtered_lrelu(
                x,
                fu=fu,
                fd=fd,
                up=self.up,
                down=self.down,
                gain=1.0,
                slope=0.0,
                impl=self.impl,
            )
        elif self.padding_mode == "zeros":
            padding = fu.numel() - 1
            return filtered_lrelu.filtered_lrelu(
                x,
                up=self.up,
                down=self.down,
                fu=fu,
                fd=fd,
                padding=padding,
                gain=1.0,
                slope=0.0,
                impl=self.impl,
            )
        else:
            raise ValueError(f"Unexpected padding_mode={self.padding_mode}")


def create_1d_mask(N, cutoff, direction, device=None):
    low = int((N * cutoff) // 2)
    high = int(N - low)
    mask = torch.ones(N, device=device)
    mask[low + 1 : high] = 0

    # Set the value for the Nyquist frequency
    # See Michaeli et al. (2023) for details
    if N % 4 == 0:
        if direction == "upsampling":
            nyquist_value = 0.5
        elif direction == "downsampling":
            nyquist_value = 0.0
        else:
            raise ValueError(f"direction={direction} is not supported")

        mask[low] = nyquist_value
        mask[high] = nyquist_value

    return mask


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L31
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L5
def create_mask(shape, cutoff, mask_shape, direction):
    assert len(shape) == 2, "Only 2D low-pass filters are supported"
    if mask_shape == "square":
        lpfs = [create_1d_mask(N, cutoff, direction) for N in shape]
        # Thanks to separability, the 2d mask is the outer product of the 1d masks
        mask = lpfs[0][:, None] * lpfs[1][None, :]
    else:
        # disk-shaped masks were removed on purpose (breaking change).
        raise ValueError(
            f"mask_shape={mask_shape} is not supported; only 'square' is allowed"
        )
    return mask


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L73
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L45
class UnwindowedIdealFilter(nn.Module):
    def __init__(self, direction, cutoff, mask_shape, transform_mode="rfft"):
        super().__init__()
        self.cutoff = cutoff
        assert transform_mode in [
            "fft",
            "rfft",
        ], f"transform_mode={transform_mode} is not supported"
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == "fft" else torch.fft.rfft2
        self.itransform = (
            (lambda x, **kwargs: torch.real(torch.fft.ifft2(x)))
            if transform_mode == "fft"
            else torch.fft.irfft2
        )
        self.direction = direction
        self.mask_shape = mask_shape
        self.masks = {}

    def forward(self, x):
        # A tuple containing the shape of x used as a key
        # for caching the masks
        shape = x.shape[-2:]
        x_fft = self.transform(x)
        if shape not in self.masks:
            mask = create_mask(
                shape, self.cutoff, self.mask_shape, direction=self.direction
            )
            N = x.shape[-1]
            mask = mask[:, : int(N / 2 + 1)] if self.transform_mode == "rfft" else mask
            self.masks[shape] = mask
        mask = self.masks[shape]
        mask = mask.to(x.device)
        x_fft *= mask
        return self.itransform(x_fft, s=(x.shape[-2], x.shape[-1]))


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/ideal_lpf.py#L99
class UnwindowedIdealUpsampling(nn.Module):
    """
    A module that upsamples an input tensor in the frequency domain using FFT or RFFT.

    :param stride: int, the upsampling factor. Default is 2.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
    """

    def __init__(self, mask_shape, stride=2):
        super().__init__()
        self.up = stride
        self.filter = UnwindowedIdealFilter(
            direction="upsampling", cutoff=1 / stride, mask_shape=mask_shape
        )

    def forward(self, x):
        """
        Upsample the input tensor.

        :param x: torch.Tensor, the input tensor to be upsampled.
        :return: torch.Tensor, the upsampled output tensor.
        """
        batch_size, num_channels, in_height, in_width = x.shape
        # Interpolate x with zeros
        x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
        x = torch.nn.functional.pad(x, [0, self.up - 1, 0, 0, 0, self.up - 1])
        x = x.reshape(
            [batch_size, num_channels, in_height * self.up, in_width * self.up]
        )
        # Apply an ideal low-pass filter removing the aliasing introduced by the interpolation
        x = self.filter(x)
        # Renormalize the amplitude to preserve the energy
        x = x * (self.up**2)
        return x


# Adapted from
# https://github.com/adobe/antialiased-cnns/blob/b27a34a26f3ab039113d44d83c54d0428598ac9c/antialiased_cnns/blurpool.py#L13
class BlurPool(nn.Module):
    def __init__(self, mask_shape, stride=2):
        super().__init__()
        self.stride = stride

        self.filter = UnwindowedIdealFilter(
            direction="downsampling",
            cutoff=1 / self.stride,
            mask_shape=mask_shape,
        )

    def forward(self, x):
        x = self.filter(x)
        return x[:, :, :: self.stride, :: self.stride]


# A faster version of LayrNorm_AF
class FasterLayerNorm_AF(nn.Module):
    """
    Alias-Free Layer Normalization

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param int in_channels: number of input channels.
    :param int out_channels: number of output channels.
    :param bool residual: if True, the output is the sum of the input and the denoised image.
    :param bool cat: if True, the network uses skip connections.
    :param int scales: number of scales in the network.
    :param bool rotation_equivariant: if True, the network is rotation-equivariant.

    :param Union[int, list, torch.Size] normalized_shape: Input shape from an expected input of size.
    :param float eps: A value added to the denominator for numerical stability. Default: 1e-6.
    :param bool bias: If set to False, the layer will not learn an additive bias. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(normalized_shape).unsqueeze(-1).unsqueeze(-1)
        )
        self.bias = nn.Parameter(
            torch.zeros(normalized_shape).unsqueeze(-1).unsqueeze(-1)
        )
        self.eps = eps

    @torch.compile
    def forward(self, x):
        """
        Forward pass for layer normalization.

        :param torch.Tensor x: Input tensor
        """
        CHW = math.prod(x.shape[-3:])
        x = x.to(memory_format=torch.channels_last)
        u = x.mean(1, keepdim=True)
        x_bar = x - u
        s = torch.linalg.vector_norm(x_bar, dim=(1, 2, 3), keepdim=True).square() / CHW
        x = x_bar * torch.rsqrt(s + self.eps)
        x = self.weight * x
        return x + self.bias

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Accept older checkpoints that stored 1D parameters.
        for name in ("weight", "bias"):
            key = f"{prefix}{name}"
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            if tensor.ndim == 1 and tensor.shape[0] == self.weight.shape[0]:
                state_dict[key] = tensor[:, None, None]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/activation.py#L117
class PolyActPerChannel(nn.Module):
    """
    A module that applies a polynomial activation function per channel.

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param channels: int, the number of channels.
    """

    def __init__(self, channels, trainable: bool = True):
        super().__init__()
        self.channels = channels
        init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=trainable)

    def forward(self, x):
        """
        Apply the polynomial activation function to the input tensor.

        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor after applying the activation function.
        """
        if self.deg == 2:
            # maybe this is faster?
            res = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x**2)
        else:
            res = self.coef[:, 0] + self.coef[:, 1] * x
            for i in range(2, self.deg):
                res = res + self.coef[:, i] * (x**i)
        return res

    def __repr__(self):
        return f"PolyActPerChannel(channels={self.channels})"


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/activation.py#L187
class FilteredNonlinearity(nn.Module):
    """
    A module that applies an upsampled polynomial activation function per channel.

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param channels: int, the number of channels.
    :param up: int, the upsampling factor. Default is 2.
    :param transform_mode: str, the transform mode to use ('fft' or 'rfft'). Default is 'rfft'.
    :param rotation_equivariant: bool, whether to use a rotation-equivariant filter. Default is False.
    :param kwargs: additional keyword arguments for the polynomial activation function.
    """

    def __init__(
        self,
        down_mask_shape,
        channels=None,
        up=2,
        kind="gelu",
        up_mask_shape=None,
        **kwargs,
    ):
        super().__init__()
        self.up = up
        self.filter = UnwindowedIdealFilter(
            direction="downsampling",
            cutoff=1 / up,
            mask_shape=down_mask_shape,
        )

        if up_mask_shape is None:
            up_mask_shape = down_mask_shape
        self.upsample = UnwindowedIdealUpsampling(stride=up, mask_shape=up_mask_shape)

        if kind == "gelu":
            self.pact = torch.nn.GELU()
        elif kind == "relu":
            self.pact = torch.nn.ReLU()
        elif kind == "poly":
            self.pact = PolyActPerChannel(channels, **kwargs)
        else:
            raise ValueError(f"kind={kind} is not supported")

    def forward(self, x):
        """
        Apply the upsampled polynomial activation function per channel to the input tensor.

        :param x: torch.Tensor, the input tensor.
        :return: torch.Tensor, the output tensor after applying the activation function.
        """
        out = self.upsample(x)
        out = self.pact(out)
        out = self.filter(out)
        out = out[:, :, :: self.up, :: self.up]
        return out


class KaiserSincBlurPool(nn.Module):
    """Blur pooling using the Kaiser-Sinc filter with explicit padding and separable depthwise conv."""

    def __init__(self, padding_mode: str = "zeros"):
        super().__init__()
        # Kaiser-Sinc with 12 taps and a target cut-off of 1/4 Hz
        fd = torch.tensor(
            [
                0.0006092520197853446,
                0.005267132073640823,
                -0.018742641434073448,
                -0.04972728714346886,
                0.1221000999212265,
                0.4404934346675873,
                0.4404934346675873,
                0.1221000999212265,
                -0.04972728714346886,
                -0.018742641434073448,
                0.005267132073640823,
                0.0006092520197853446,
            ]
        )
        self.register_buffer("fd", fd, persistent=False)
        self.padding = fd.numel() - 1
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode == "circular":
            pad_mode = "circular"
        elif self.padding_mode == "zeros":
            pad_mode = "constant"
        else:
            raise ValueError(f"Unexpected padding_mode={self.padding_mode}")

        pad_left = math.floor(self.padding / 2)
        pad_right = math.ceil(self.padding / 2)
        # Pad spatial dims symmetrically before separable filtering.
        x = F.pad(x, (pad_left, pad_right, pad_left, pad_right), mode=pad_mode)

        channels = x.shape[1]
        # Separable depthwise filtering: vertical then horizontal.
        vert_kernel = self.fd.view(1, 1, -1, 1).expand(channels, -1, -1, -1)
        horiz_kernel = self.fd.view(1, 1, 1, -1).expand(channels, -1, -1, -1)
        x = F.conv2d(x, vert_kernel, groups=channels)
        x = F.conv2d(x, horiz_kernel, groups=channels)

        return x[:, :, ::2, ::2]


class KaiserSincUpsampling(nn.Module):
    """Upsample by 2 using the Kaiser-Sinc filter with explicit padding and separable depthwise conv."""

    def __init__(self, padding_mode: str = "zeros"):
        super().__init__()
        # Kaiser-Sinc with 12 taps and a target cut-off of 1/4 Hz
        fd = torch.tensor(
            [
                0.0006092520197853446,
                0.005267132073640823,
                -0.018742641434073448,
                -0.04972728714346886,
                0.1221000999212265,
                0.4404934346675873,
                0.4404934346675873,
                0.1221000999212265,
                -0.04972728714346886,
                -0.018742641434073448,
                0.005267132073640823,
                0.0006092520197853446,
            ]
        )
        self.register_buffer("fd", fd, persistent=False)
        self.padding = fd.numel() - 1
        self.padding_mode = padding_mode

    def forward(self, x):
        if self.padding_mode == "circular":
            pad_mode = "circular"
        elif self.padding_mode == "zeros":
            pad_mode = "constant"
        else:
            raise ValueError(f"Unexpected padding_mode={self.padding_mode}")

        # Zero-insert upsampling by factor 2.
        upsampled = torch.zeros(
            x.shape[0],
            x.shape[1],
            x.shape[2] * 2,
            x.shape[3] * 2,
            device=x.device,
            dtype=x.dtype,
        )
        upsampled[:, :, ::2, ::2] = x

        pad_left = math.floor(self.padding / 2)
        pad_right = math.ceil(self.padding / 2)
        upsampled = F.pad(
            upsampled, (pad_left, pad_right, pad_left, pad_right), mode=pad_mode
        )

        channels = upsampled.shape[1]
        vert_kernel = self.fd.view(1, 1, -1, 1).expand(channels, -1, -1, -1)
        horiz_kernel = self.fd.view(1, 1, 1, -1).expand(channels, -1, -1, -1)
        out = F.conv2d(upsampled, vert_kernel, groups=channels)
        out = F.conv2d(out, horiz_kernel, groups=channels)

        return out
