import importlib.resources as resources
from pathlib import Path

import deepinv as dinv
import torch
import torch.nn as nn
import yaml

from unet_af.models.center_crop_reconstructor import CenterCropReconstructor
from unet_af.models.unet_af import UNetAF

from .io import resolve_weights_path


def load_model_from_config(
    config: dict,
    *,
    physics_padding: str | None = None,
    physics: dinv.physics.Physics | None = None,
    device: torch.device | str = torch.device("cpu"),
    weights_path: str | Path | None = None,
    weights_fs: str = "local",
    strict: bool = True,
    eval_mode: bool = False,
) -> nn.Module:  # noqa: B008
    # If using the ArtifactRemoval workflow we construct a UNet denoiser first.
    # Otherwise, for model_kind == "ram" we instantiate RAM directly.
    # New: model_kind choices: "unet", "ram", or "tv"
    model_kind = config.get("model_kind", "unet")
    # Read the required new key `unet_impl` (no legacy fallback)
    unet_impl = config.get("unet_impl")
    unet_af_ksize = config.get("unet_af_ksize", 3)
    weights_initialization = config.get("weights_initialization", "default")
    ram_init = config.get("ram_init", "pretrained")

    # Validate activation weights handling for UNetAF; ignored for other implementations but must be valid.
    act_weights_mode = config.get("unet_act_weights", "trainable")
    if act_weights_mode == "trainable":
        trainable_act = True
    elif act_weights_mode == "untrainable":
        trainable_act = False
    else:
        raise ValueError("unet_act_weights must be one of 'trainable' or 'untrainable'")

    if model_kind == "unet":
        # Pack UNet-specific parameters into a dictionary and use it
        # when constructing different UNet implementations.
        unet_params = {
            "in_channels": 3,
            "out_channels": 3,
            "scales": config["unet_scales"],
            "bias": config["unet_bias"],
            "residual": config["unet_residual"],
            "padding": config.get("unet_padding", "zeros"),
            "act_padding": config.get("unet_act_padding"),
            "act_filters_kind": config.get("unet_act_filters_kind"),
            "pooling": config.get("unet_pooling", "max_pool"),
            "upsampling": config.get("unet_upsampling", "filtered_upsampling"),
            "norm": config.get("unet_norm", "batch_norm"),
            "activation": config.get("unet_activation", "filtered_gelu"),
            "second_scale_fft": config.get("unet_second_scale_fft", False),
        }

        if unet_impl == "ronneberger":
            from unet_af.models.ronneberger import UNet

            # Consume and validate normalization
            norm = unet_params.pop("norm")
            if norm != "batch_norm":
                raise ValueError(
                    f"Ronneberger UNet only supports 'batch_norm' normalization (requested: {norm!r})"
                )
            activation = unet_params.pop("activation")
            if activation != "relu":
                raise ValueError(
                    f"Ronneberger UNet only supports 'relu' activation (requested: {activation!r})"
                )

            if unet_params.pop("scales") != 5:
                raise ValueError("Ronneberger UNet only supports 5 scales")

            if not unet_params.pop("bias"):
                raise ValueError("Ronneberger UNet implementation requires bias=True")

            if unet_params.pop("residual"):
                raise ValueError(
                    "Ronneberger UNet implementation does not support residual connections"
                )

            # Only zero padding is supported for Ronneberger UNet
            padding = unet_params.pop("padding")
            if padding != "zeros":
                raise ValueError(
                    f"Ronneberger UNet only supports 'zeros' padding (requested: {padding!r})"
                )

            act_padding = unet_params.pop("act_padding")
            if act_padding is not None:
                raise ValueError(
                    f"Ronneberger UNet does not support act_padding (requested: {act_padding!r})"
                )

            if (act_filters_kind := unet_params.pop("act_filters_kind")) is not None:
                raise ValueError(
                    f"Ronneberger UNet does not support act_filters_kind (requested: {act_filters_kind!r})"
                )

            # Only max_pool supported for Ronneberger
            pooling = unet_params.pop("pooling")
            if pooling != "max_pool":
                raise ValueError(
                    f"Ronneberger UNet only supports 'max_pool' pooling (requested: {pooling!r})"
                )
            upsampling = unet_params.pop("upsampling")
            if upsampling != "strided_conv":
                raise ValueError(
                    f"Ronneberger UNet only supports 'strided_conv' upsampling (requested: {upsampling!r})"
                )

            if second_scale_fft := unet_params.pop("second_scale_fft"):
                raise ValueError(
                    f"Ronneberger UNet does not support second_scale_fft (requested: {second_scale_fft!r})"
                )

            denoiser = UNet(
                n_channels=unet_params.pop("in_channels"),
                n_classes=unet_params.pop("out_channels"),
                bilinear=False,
            )
        elif unet_impl == "deepinv":
            # Consume and validate normalization
            norm = unet_params.pop("norm")
            if norm != "batch_norm":
                raise ValueError(
                    f"deepinv UNet only supports 'batch_norm' normalization (requested: {norm!r})"
                )
            activation = unet_params.pop("activation")
            if activation != "relu":
                raise ValueError(
                    f"deepinv UNet only supports 'relu' activation (requested: {activation!r})"
                )

            # Only zero padding is supported for the deepinv UNet wrapper
            padding = unet_params.pop("padding")
            if padding != "zeros":
                raise ValueError(
                    f"deepinv UNet only supports 'zeros' padding (requested: {padding!r})"
                )

            act_padding = unet_params.pop("act_padding")
            if act_padding is not None:
                raise ValueError(
                    f"deepinv UNet does not support act_padding (requested: {act_padding!r})"
                )

            if (act_filters_kind := unet_params.pop("act_filters_kind")) is not None:
                raise ValueError(
                    f"deepinv UNet does not support act_filters_kind (requested: {act_filters_kind!r})"
                )

            # deepinv wrapper supports only max_pool
            pooling = unet_params.pop("pooling")
            if pooling != "max_pool":
                raise ValueError(
                    f"deepinv UNet only supports 'max_pool' pooling (requested: {pooling!r})"
                )
            upsampling = unet_params.pop("upsampling")
            if upsampling != "nn_upsample":
                raise ValueError(
                    f"deepinv UNet only supports 'nn_upsample' upsampling (requested: {upsampling!r})"
                )

            if second_scale_fft := unet_params.pop("second_scale_fft"):
                raise ValueError(
                    f"deepinv UNet does not support second_scale_fft (requested: {second_scale_fft!r})"
                )

            denoiser = dinv.models.UNet(
                in_channels=unet_params.pop("in_channels"),
                out_channels=unet_params.pop("out_channels"),
                scales=unet_params.pop("scales"),
                bias=unet_params.pop("bias"),
                residual=unet_params.pop("residual"),
            )
        elif unet_impl == "equivariant":
            if second_scale_fft := unet_params.pop("second_scale_fft"):
                raise ValueError(
                    f"equivariant UNet does not support second_scale_fft (requested: {second_scale_fft!r})"
                )

            act_padding = unet_params.pop("act_padding")
            if act_padding is not None:
                raise ValueError(
                    f"equivariant UNet does not support act_padding (requested: {act_padding!r})"
                )

            if (act_filters_kind := unet_params.pop("act_filters_kind")) is not None:
                raise ValueError(
                    f"equivariant UNet does not support act_filters_kind (requested: {act_filters_kind!r})"
                )

            denoiser = UNetAF(
                in_channels=unet_params.pop("in_channels"),
                out_channels=unet_params.pop("out_channels"),
                scales=unet_params.pop("scales"),
                ksize=unet_af_ksize,
                bias=unet_params.pop("bias"),
                residual=unet_params.pop("residual"),
                norm=unet_params.pop("norm"),
                padding=unet_params.pop("padding"),
                pooling=unet_params.pop("pooling"),
                upsampling=unet_params.pop("upsampling"),
                activation=unet_params.pop("activation"),
                trainable_act=trainable_act,
            )
        elif unet_impl == "equivariant_fast":
            denoiser = UNetAF(
                in_channels=unet_params.pop("in_channels"),
                out_channels=unet_params.pop("out_channels"),
                scales=unet_params.pop("scales"),
                ksize=unet_af_ksize,
                bias=unet_params.pop("bias"),
                residual=unet_params.pop("residual"),
                norm=unet_params.pop("norm"),
                padding=unet_params.pop("padding"),
                act_padding=unet_params.pop("act_padding"),
                act_filters_kind=unet_params.pop("act_filters_kind"),
                pooling=unet_params.pop("pooling"),
                upsampling=unet_params.pop("upsampling"),
                activation=unet_params.pop("activation"),
                trainable_act=trainable_act,
                torch_compile=config.get("unet_torch_compile", False),
                sequentialize_batches=config.get("unet_sequentialize_batches", False),
                second_scale_fft=unet_params.pop("second_scale_fft"),
            )
        else:
            raise ValueError(f"Unknown unet_impl: {unet_impl}")

        # Ensure all expected UNet params were consumed by the chosen implementation.
        # If any keys remain, raise to avoid silently ignoring unsupported options.
        if unet_params:
            remaining = ", ".join(sorted(unet_params.keys()))
            raise ValueError(
                f"Unsupported/unconsumed unet_params detected after model construction: {remaining}"
            )

        # Optionally initialize UNet weights (kaiming)
        if weights_initialization == "kaiming":
            for m in denoiser.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
        else:
            assert (
                weights_initialization == "default"
            ), f"Unknown init kind: {weights_initialization}"

        # Wrap the denoiser into the standard ArtifactRemoval pipeline
        model = dinv.models.ArtifactRemoval(denoiser, mode="direct")
    elif model_kind == "ram":
        if ram_init == "pretrained":
            ram_pretrained = True
        elif ram_init == "default_random_init":
            ram_pretrained = False
        else:
            raise ValueError(
                "ram_init must be one of 'pretrained' or 'default_random_init'"
            )
        model = dinv.models.RAM(device=device, pretrained=ram_pretrained)
    elif model_kind == "tv":
        # TV baseline using DeepInverse PGD solver with TV prior.
        data_fidelity = dinv.optim.data_fidelity.L2()
        prior = dinv.optim.prior.TVPrior(n_it_max=50)

        model = dinv.optim.PGD(
            prior=prior,
            data_fidelity=data_fidelity,
            stepsize=1.0,
            lambda_reg=1e-2,
            early_stop=True,
            max_iter=300,
            verbose=False,
        )
    elif model_kind == "pnp_dpir":
        if physics is None:
            raise ValueError(
                "PnP-DPIR requires a physics object and none was provided."
            )
        try:
            noise_level_img = float(physics.noise_model.sigma)
        except Exception as e:
            raise ValueError(
                "pnp_dpir requires a GaussianNoise physics with a 'sigma' attribute"
            ) from e
        from deepinv.optim.dpir import get_DPIR_params

        sigma_denoiser, stepsize, max_iter = get_DPIR_params(
            noise_level_img, device=device
        )
        data_fidelity = dinv.optim.data_fidelity.L2()
        prior = dinv.optim.prior.PnP(
            denoiser=dinv.models.DRUNet(pretrained="download", device=device)
        )
        model = dinv.optim.HQS(
            prior=prior,
            data_fidelity=data_fidelity,
            stepsize=stepsize,
            sigma_denoiser=sigma_denoiser,
            early_stop=False,
            max_iter=max_iter,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}.")

    if physics_padding == "valid_v2":
        model = CenterCropReconstructor(model, crop_size=(128, 128))

    if weights_path is not None:
        resolved_path = resolve_weights_path(str(weights_path), fs=weights_fs)
        ckpt = torch.load(resolved_path, map_location=device)
        state_dict = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state_dict, strict=strict)

    model = model.to(device)

    if eval_mode:
        model.eval()

    return model


def load_full_test_config() -> dict:
    test_config_path = resources.files("unet_af").joinpath("test.yaml")
    full_test_config = yaml.safe_load(test_config_path.read_text())
    return full_test_config


def load_model_from_test_config(
    *,
    test_config_name: str,
    full_test_config: dict | None = None,
    config_dir: str | None = None,
    device: torch.device | str = torch.device("cpu"),
    weights_fs: str = "remote",
):
    if full_test_config is None:
        full_test_config = load_full_test_config()

    # Load defaults.overrides
    default_overrides = full_test_config.get("defaults", {}).get("overrides", {})

    # Load test-specific config
    test_config = full_test_config.get("tests", {}).get(test_config_name, {})

    # Load the base config associated to the test
    config_name = test_config["base"]

    # Load test-specific overrides
    specific_overrides = test_config.get("overrides", {})

    # Load the base config
    if config_dir is None:
        config_dir = resources.files("unet_af").joinpath("configs").resolve()

    config_path = config_dir / config_name
    config = yaml.safe_load(config_path.read_text()) or {}

    # Apply the overrides
    overrides = {**default_overrides, **specific_overrides}
    for key, value in overrides.items():
        config[key] = value

    weights_path = config["checkpoint_path"]
    model = load_model_from_config(
        config,
        device=device,
        weights_path=weights_path,
        weights_fs=weights_fs,
        eval_mode=True,
    )

    return model
