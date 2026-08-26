import deepinv as dinv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from unet_af.io import resolve_weights_path
from unet_af.layers import (
    BlurPool,
    FasterLayerNorm_AF,
    FilteredNonlinearity,
    KaiserSincBlurPool,
    KaiserSincReLU,
    KaiserSincReLUv2,
    KaiserSincUpsampling,
    LayerNorm_AF,
    LayerNormC,
    PolyActPerChannel,
    UnwindowedIdealUpsampling,
    test_pad,
)


def sequentialize_batches_decorator(func):
    import functools

    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        outs = [func(xi, *args, **kwargs) for xi in torch.split(x, 1, dim=0)]
        return torch.cat(outs, dim=0)

    return wrapper


class UNetAF(dinv.models.Denoiser):
    r"""
    U-Net convolutional denoiser.

    This network is a fully convolutional denoiser based on the U-Net architecture. The number of downsample steps
    can be controlled with the ``scales`` parameter. The number of trainable parameters increases with the number of
    scales.

    .. warning::
        When using the bias-free batch norm ``BFBatchNorm2d`` via ``batch_norm="biasfree"``, NaNs may be encountered
        during training, causing the whole training procedure to fail.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param bool residual: use a skip-connection between output and output.
    :param bool circular_padding: circular padding for the convolutional layers.
    :param bool cat: use skip-connections between intermediate levels.
    :param bool bias: use learnable biases.
    :param str norm: normalization mode. Supported values:
        - ``"layer_norm_AF"`` (default): alias-free LayerNorm implementation used in this file.
        - ``"faster_layer_norm_AF"``: compiled, faster alias-free LayerNorm.
        - ``"layer_norm_c"``: channels-first wrapper around ``nn.LayerNorm``.
        - ``"none"``: no normalization (identity).
        - ``"batch_norm"``: use ``torch.nn.BatchNorm2d``.
        - ``"instance_norm"``: use ``torch.nn.InstanceNorm2d``.
    :param int scales: Number of downsampling steps used in the U-Net. The options are 2,3,4 and 5.
        The number of trainable parameters increases with the scale.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        residual=True,
        cat=True,
        bias=True,
        norm: str = "layer_norm_AF",
        scales=5,
        pooling: str = "blur_pool",
        ksize=None,
        padding: str = "circular",
        act_padding: str | None = None,
        act_filters_kind: str = "fitted",
        activation: str = "filtered_gelu",
        trainable_act: bool = True,
        transpose: bool = False,
        upsampling: str = "filtered_upsampling",
        sequentialize_batches: bool = False,
        torch_compile: bool = False,
        second_scale_fft: bool = False,
    ):
        super().__init__()

        self.name = "UNetAF"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.compact = scales
        self.pooling = pooling
        self.activation = activation
        self.trainable_act = trainable_act
        self.transpose = transpose
        self.padding = padding
        self.act_padding = act_padding if act_padding is not None else padding
        self.act_filters_kind = act_filters_kind
        self.upsampling = upsampling
        self.sequentialize_batches = sequentialize_batches
        self.torch_compile = torch_compile

        # This implementation always uses the translation-equivariant filtered
        # operations (circular padding + frequency-domain filters). Rotation
        # equivariance options were removed (breaking change).
        filter_mask = "square"

        # Determine kernel size and padding size.
        if ksize is None:
            ksize = 3
        # Use floor(ksize/2) for padding size except for 'valid' -> use 0.
        pad_size = 0 if padding == "valid" else (ksize // 2)
        # Map 'valid' to 'zeros' for Conv2d.padding_mode; otherwise pass provided string through.
        conv_padding_mode = "zeros" if padding == "valid" else padding
        if self.transpose and padding not in ("zeros", "valid"):
            raise ValueError(
                "ConvTranspose2d supports only padding modes 'zeros' or 'valid' when transpose=True"
            )

        def conv(ch_in, ch_out, ksize, stride, padding, bias, transpose: bool = False):
            # Use Conv2d by default; ConvTranspose2d when transpose=True. Always forward padding_mode.
            padding_mode = conv_padding_mode
            layer_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
            return layer_cls(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
            )

        def norm_layer(channels, bias):
            # Factory returning the requested normalization layer.
            if norm == "layer_norm_AF":
                return LayerNorm_AF(channels, bias=bias, type="AF")
            elif norm == "faster_layer_norm_AF":
                return FasterLayerNorm_AF(channels)
            elif norm == "layer_norm_c":
                return LayerNormC(channels=channels, bias=bias)
            elif norm == "batch_norm":
                return nn.BatchNorm2d(channels)
            elif norm == "instance_norm":
                return nn.InstanceNorm2d(channels, affine=bias)
            elif norm == "none":
                return nn.Identity()
            else:
                raise ValueError(f"norm={norm} is not supported")

        def act(channels):
            if self.activation == "filtered_gelu":
                return FilteredNonlinearity(
                    down_mask_shape=filter_mask,
                    up_mask_shape=filter_mask,
                    kind="gelu",
                )
            elif self.activation == "filtered_relu":
                return FilteredNonlinearity(
                    down_mask_shape=filter_mask,
                    up_mask_shape=filter_mask,
                    kind="relu",
                )
            elif self.activation == "filtered_poly":
                return FilteredNonlinearity(
                    down_mask_shape=filter_mask,
                    up_mask_shape=filter_mask,
                    kind="poly",
                    channels=channels,
                    trainable=self.trainable_act,
                )
            elif self.activation == "kaiser_sinc_relu":
                return KaiserSincReLU(padding_mode=self.padding)
            elif self.activation == "kaiser_sinc_relu_v2":
                if second_scale_fft and channels == 128:
                    return FilteredNonlinearity(
                        down_mask_shape=filter_mask,
                        up_mask_shape=filter_mask,
                        kind="relu",
                    )
                return KaiserSincReLUv2(
                    filters_kind=self.act_filters_kind, padding_mode=self.act_padding
                )
            elif self.activation == "poly":
                return PolyActPerChannel(channels, trainable=self.trainable_act)
            elif self.activation == "relu":
                return nn.ReLU()
            elif self.activation == "gelu":
                return nn.GELU()
            elif self.activation == "none":  # for debugging purposes
                return nn.Identity()
            else:
                raise ValueError(f"activation={self.activation} is not supported")

        def pool():
            # Choose downsampling layer based on the requested pooling strategy.
            # Validate pooling value here (only place we validate).
            if self.pooling == "blur_pool":
                return BlurPool(stride=2, mask_shape=filter_mask)
            elif self.pooling == "kaiser_sinc_blur_pool":
                return KaiserSincBlurPool(padding_mode=self.padding)
            elif self.pooling == "max_pool":
                return nn.MaxPool2d(kernel_size=2, stride=2)
            elif self.pooling == "avg_pool":
                return nn.AvgPool2d(kernel_size=2, stride=2)
            elif self.pooling == "max_blur_pool":
                return nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    BlurPool(stride=2, mask_shape=filter_mask),
                )
            else:
                raise ValueError(
                    f"pooling={self.pooling} is not supported; supported: ['blur_pool', 'kaiser_sinc_blur_pool', 'max_pool', 'avg_pool', 'max_blur_pool']"
                )

        def upsample():
            if self.upsampling == "filtered_upsampling":
                return UnwindowedIdealUpsampling(stride=2, mask_shape=filter_mask)
            elif self.upsampling == "kaiser_sinc_upsampling":
                return KaiserSincUpsampling(padding_mode=self.padding)
            elif self.upsampling == "nn_upsample":
                return nn.Upsample(scale_factor=2)
            else:
                raise ValueError(
                    f"upsampling={self.upsampling} is not supported; supported: ['filtered_upsampling', 'kaiser_sinc_upsampling', 'nn_upsample']"
                )

        def conv_block(ch_in, ch_out, transpose_block: bool = False):
            # Include normalization layers only when norm != 'none'.
            if norm != "none":
                return nn.Sequential(
                    conv(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        ksize=ksize,
                        stride=1,
                        padding=pad_size,
                        bias=bias,
                        transpose=transpose_block,
                    ),
                    norm_layer(channels=ch_out, bias=bias),
                    act(ch_out),
                    conv(
                        ch_in=ch_out,
                        ch_out=ch_out,
                        ksize=ksize,
                        stride=1,
                        padding=pad_size,
                        bias=bias,
                        transpose=transpose_block,
                    ),
                    norm_layer(channels=ch_out, bias=bias),
                    act(ch_out),
                )
            else:
                return nn.Sequential(
                    conv(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        ksize=ksize,
                        stride=1,
                        padding=pad_size,
                        bias=bias,
                        transpose=transpose_block,
                    ),
                    act(ch_out),
                    conv(
                        ch_in=ch_out,
                        ch_out=ch_out,
                        ksize=ksize,
                        stride=1,
                        padding=pad_size,
                        bias=bias,
                        transpose=transpose_block,
                    ),
                    act(ch_out),
                )

        def up_conv(ch_in, ch_out):
            """
            Decoder path: upsample in frequency domain then apply a conv (or transposed conv
            when transpose=True) so that 'valid' padding (pad_size==0) does not shrink spatial dimensions.
            """
            if norm != "none":
                return nn.Sequential(
                    upsample(),
                    conv(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        ksize=ksize,
                        stride=1,
                        padding=pad_size,  # <-- use pad_size
                        bias=bias,
                        transpose=self.transpose,
                    ),
                    norm_layer(ch_out, bias),
                    act(ch_out),
                )
            else:
                return nn.Sequential(
                    upsample(),
                    conv(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        ksize=ksize,
                        stride=1,
                        padding=pad_size,  # <-- use pad_size
                        bias=bias,
                        transpose=self.transpose,
                    ),
                    act(ch_out),
                )

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Maxpool1 = pool()
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Maxpool2 = pool() if self.compact in [3, 4, 5] else None
        self.Conv3 = (
            conv_block(ch_in=128, ch_out=256) if self.compact in [3, 4, 5] else None
        )
        self.Maxpool3 = pool() if self.compact in [4, 5] else None
        self.Conv4 = (
            conv_block(ch_in=256, ch_out=512) if self.compact in [4, 5] else None
        )
        self.Maxpool4 = pool() if self.compact in [5] else None
        self.Conv5 = conv_block(ch_in=512, ch_out=1024) if self.compact in [5] else None

        self.Up5 = up_conv(ch_in=1024, ch_out=512) if self.compact in [5] else None
        self.Up_conv5 = (
            conv_block(ch_in=1024, ch_out=512, transpose_block=self.transpose)
            if self.compact in [5]
            else None
        )

        self.Up4 = up_conv(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        self.Up_conv4 = (
            conv_block(ch_in=512, ch_out=256, transpose_block=self.transpose)
            if self.compact in [4, 5]
            else None
        )

        self.Up3 = up_conv(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        self.Up_conv3 = (
            conv_block(ch_in=256, ch_out=128, transpose_block=self.transpose)
            if self.compact in [3, 4, 5]
            else None
        )

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, transpose_block=self.transpose)

        self.Conv_1x1 = conv(
            ch_in=64,
            ch_out=out_channels,
            ksize=1,
            stride=1,
            padding=0,
            bias=bias,
            transpose=self.transpose,
        )

        if self.compact == 5:
            self._forward = self.forward_standard
        if self.compact == 4:
            self._forward = self.forward_compact4
        if self.compact == 3:
            self._forward = self.forward_compact3
        if self.compact == 2:
            self._forward = self.forward_compact2

        if self.sequentialize_batches:
            self.forward = sequentialize_batches_decorator(self.forward)

        if self.torch_compile:
            self.forward = torch.compile(self.forward)

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
        # Accept Conv weights saved with in/out channels swapped (0<->1).
        for key, tensor in list(state_dict.items()):
            if not key.startswith(prefix) or not key.endswith(".weight"):
                continue
            module_path = key[len(prefix) : -len(".weight")]
            if not module_path:
                continue
            try:
                mod = self.get_submodule(module_path)
            except AttributeError:
                continue
            if not isinstance(mod, nn.Conv2d | nn.ConvTranspose2d):
                continue
            expected_shape = mod.weight.shape
            swapped_shape = torch.Size(
                [expected_shape[1], expected_shape[0], *expected_shape[2:]]
            )
            if tensor.shape == swapped_shape and tensor.shape != expected_shape:
                state_dict[key] = tensor.transpose(0, 1).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _concat(self, enc_feat, dec_feat, dim):
        if self.padding == "valid":
            _, _, h_enc, w_enc = enc_feat.shape
            _, _, h_dec, w_dec = dec_feat.shape
            dh = (h_enc - h_dec) // 2
            dw = (w_enc - w_dec) // 2
            enc_feat = enc_feat[..., dh : dh + h_dec, dw : dw + w_dec]
        return torch.cat((enc_feat, dec_feat), dim=dim)

    def _residual_add(self, x, dec_out):
        if self.padding == "valid":
            _, _, h_dec, w_dec = dec_out.shape
            _, _, h_x, w_x = x.shape
            dh = (h_x - h_dec) // 2
            dw = (w_x - w_dec) // 2
            pad = (dw, w_x - w_dec - dw, dh, h_x - h_dec - dh)
            dec_out = F.pad(dec_out, pad, mode="constant", value=0)
        return dec_out + x

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (self.compact - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def forward_standard(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        if self.cat:
            d5 = self._concat(x4, d5, cat_dim)
            d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if self.cat:
            d4 = self._concat(x3, d4, cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if self.cat:
            d3 = self._concat(x2, d3, cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = self._concat(x1, d2, cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        if self.residual and self.in_channels == self.out_channels:
            d1 = self._residual_add(x, d1)
        return d1

    def forward_compact4(self, x):
        # encoding path
        cat_dim = 1
        input = x

        x1 = self.Conv1(input)  # 1->64

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)  # 64->128

        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)  # 128->256

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)  # 256->512

        d4 = self.Up4(x4)  # 512->256
        if self.cat:
            d4 = self._concat(x3, d4, cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)  # 256->128
        if self.cat:
            d3 = self._concat(x2, d3, cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)  # 128->64
        if self.cat:
            d2 = self._concat(x1, d2, cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        if self.residual and self.in_channels == self.out_channels:
            d1 = self._residual_add(x, d1)
        return d1

    def forward_compact3(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        if self.cat:
            d3 = self._concat(x2, d3, cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = self._concat(x1, d2, cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        if self.residual and self.in_channels == self.out_channels:
            d1 = self._residual_add(x, d1)
        return d1

    def forward_compact2(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)

        d2 = self.Up2(x2)
        if self.cat:
            d2 = self._concat(x1, d2, cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        if self.residual and self.in_channels == self.out_channels:
            d1 = self._residual_add(x, d1)
        return d1


def unet_af(weights_path: str, *, fs: str = "local") -> dinv.models.Reconstructor:
    """
    Build the canonical UNetAF used in this project and optionally load a fixed state_dict.
    Either pass `weights` (a state_dict) or `weights_path` (path to a checkpoint). If neither is
    provided, the model is returned with random initialization.
    """
    denoiser = UNetAF()

    # Resolve checkpoint path via shared helper
    resolved_path = resolve_weights_path(weights_path, fs=fs) if weights_path else None

    model = dinv.models.ArtifactRemoval(backbone_net=denoiser, mode="direct")

    if resolved_path:
        ckpt = torch.load(resolved_path, weights_only=True, map_location="cpu")
        state_dict = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state_dict)

    return model
