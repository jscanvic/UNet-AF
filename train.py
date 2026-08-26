import argparse
import importlib.resources as resources
import math
import os
import random
import time
from pathlib import Path

import deepinv as dinv
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from unet_af.config import load_model_from_config
from unet_af.dataset import DATASET_DIRS
from unet_af.metrics import EQUIV, ContinuousTranslation, compute_instability
from unet_af.mlops import MLFLOW_EXPERIMENT_NAME
from unet_af.trainer import ResampledMeasurements, Trainer


def load_config() -> dict:
    r"""
    Load the training parameters from the command line arguments and (possibly) from a config file.

    .. note::

        This function takes no argument and directly reads in the arguments passed to the program.

    :return: dict, the parsed arguments
    """
    config_parser = argparse.ArgumentParser(description="Config parser", add_help=False)
    config_parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(parents=[config_parser])
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--optimizer_cls",
        type=str,
        default="AdamW",
        help="Optimizer class",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--train_split_size", type=int, default=700, help="Number of training samples"
    )
    parser.add_argument(
        "--val_split_size", type=int, default=100, help="Number of validation samples"
    )

    # Make number of UNet scales configurable
    parser.add_argument(
        "--unet_scales",
        type=int,
        default=5,
        help="Number of scales in the UNet (integer)",
    )

    # Model configuration
    parser.add_argument(
        "--unet_impl",
        type=str,
        default="UNet",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--weights_initialization",
        type=str,
        default="default",
        help="Weights initialization scheme",
    )

    # --- New argument: model_kind ---
    parser.add_argument(
        "--model_kind",
        type=str,
        default="unet",
        help="Kind of model to use: 'unet' (ArtifactRemoval), 'ram' (Reconstruct Anything Model), 'tv' (TV prior baseline), or 'pnp_dpir' (PnP DPIR HQS solver)",
    )
    parser.add_argument(
        "--ram_init",
        type=str,
        default="pretrained",
        help="RAM initialization: 'pretrained' uses pretrained weights, 'default_random_init' disables pretrained weights",
    )

    parser.add_argument(
        "--unet_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use residual connections in the UNet (bool)",
    )
    parser.add_argument(
        "--unet_bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use bias in conv layers of the UNet (bool)",
    )

    # Alias-free UNet options (formerly Equivariant UNet)
    parser.add_argument(
        "--unet_af_ksize",
        type=int,
        default=3,
        help="Kernel size for UNetAF variant",
    )

    # UNet padding scheme
    parser.add_argument(
        "--unet_padding",
        type=str,
        default="zeros",
        help="Padding scheme to use for UNet implementations (e.g. 'zeros' or 'circular')",
    )
    parser.add_argument(
        "--unet_act_padding",
        type=str,
        default=None,
        help="Padding scheme to use for activation functions in UNet implementations (e.g. 'zeros' or 'circular'); if None, defaults to unet_padding",
    )
    parser.add_argument(
        "--unet_act_filters_kind",
        type=str,
        default=None,
        help="Filters kind to use in the filtered activations",
    )

    # UNet pooling selection: either max_pool or blur_pool
    parser.add_argument(
        "--unet_pooling",
        type=str,
        default="max_pool",
        help="Pooling scheme to use for UNet implementations ('max_pool' or 'blur_pool')",
    )
    # UNet upsampling selection
    parser.add_argument(
        "--unet_upsampling",
        type=str,
        default="filtered_upsampling",
        help="Upsampling scheme to use for UNet implementations ('filtered_upsampling', 'nn_upsample', or 'strided_conv' where supported)",
    )

    # Normalization selection for UNet variants
    parser.add_argument(
        "--unet_norm",
        type=str,
        default="batch_norm",
        help="Normalization layer type for UNet implementations (e.g. 'batch_norm' or 'layer_norm_AF')",
    )
    parser.add_argument(
        "--unet_activation",
        type=str,
        default="filtered_gelu",
        help="Activation to use for UNet implementations (e.g. 'relu' or 'filtered_gelu')",
    )
    parser.add_argument(
        "--unet_act_weights",
        type=str,
        default="trainable",
        help="For UNetAF: 'trainable' learns activation weights, 'untrainable' freezes them (ignored for other UNet implementations)",
    )
    parser.add_argument(
        "--unet_torch_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile for UNetAF (bool)",
    )
    parser.add_argument(
        "--unet_sequentialize_batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process batches sequentially for UNetAF (bool)",
    )
    # Runtime and IO
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (defaults to cuda if available)",
    )
    parser.add_argument(
        "--div2k-deblur-path",
        type=str,
        default=os.environ.get("DIV2K_DEBLUR_PATH", "datasets/DIV2K-Deblur"),
        help="Path to the Div2K-Deblur base directory, under which each physics modality's "
        "HDF5 dataset is stored in its own subdirectory (default: DIV2K_DEBLUR_PATH "
        "environment variable, or datasets/DIV2K-Deblur if unset)",
    )
    parser.add_argument(
        "--physics_padding",
        type=str,
        default="circular",
        help="Padding mode for physics: 'circular' (BlurFFT), 'valid' (Blur), or 'valid_v2' (large valid blur with center crop)",
    )
    parser.add_argument(
        "--physics_modality",
        type=str,
        default="deblurring",
        help="Physics modality: 'deblurring' or 'denoising'",
    )
    # Checkpointing: prefer explicit path and corresponding run id (no legacy 'ckpt' arg)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to initialize the model from",
    )
    parser.add_argument(
        "--checkpoint_run_id",
        type=str,
        default=None,
        help="MLflow run id corresponding to the checkpoint (required if --checkpoint_path is set)",
    )
    # MLflow on by default; allow --no-mlflow to turn it off
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable mlflow logging (default: enabled)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.environ.get("OUT_DIR", "./out"),
        help="Output directory",
    )
    parser.add_argument(
        "--eq_metric_log_conversion_time",
        type=str,
        default="early",
        help="When to convert EQUIV metric to log-scale: 'early', 'late', or 'never'",
    )
    parser.add_argument(
        "--eq_metric_backbone",
        type=str,
        default="psnr",
        help="Backbone metric for EQUIV: 'psnr' (MSE with optional log conversion) or 'snr' (SNR in dB)",
    )

    parser.add_argument(
        "--eq_metric_kind",
        type=str,
        default="basic",
        help="EQUIV computation kind: 'basic' uses direct output, 'residual' uses residue (default: basic)",
    )
    parser.add_argument(
        "--eq_metric_residual_mode",
        type=str,
        default="direct",
        help="Residual mode to use for EQUIV residual metrics; overrides backbone mode",
    )
    parser.add_argument(
        "--eq_translation_mode",
        type=str,
        default="circular",
        help="Translation mode for EQUIV metric: 'circular' or 'crop_shift'",
    )

    # New argument: prefix to prepend to MLflow run name and logged run name
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Base run name to use instead of the config filename (default: None)",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="",
        help="Prefix prepended to the MLflow run name and logged run name",
    )

    args_config, remaining = config_parser.parse_known_args()
    parser.set_defaults(**vars(args_config))

    config_filename = args_config.config

    if config_filename:
        print(f"Loading config {config_filename}")
        config_dir = resources.files("unet_af").joinpath("configs").resolve()
        cfg_path = config_dir / config_filename
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise ValueError(
                    "The provided config file must contain a YAML mapping (dict) of defaults"
                )
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    return vars(args)


if __name__ == "__main__":
    config = load_config()

    # Device and paths
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    physics_modality = config.get("physics_modality", "deblurring")
    physics_padding = config.get("physics_padding", "circular")
    base_root = Path(config["div2k_deblur_path"])
    if physics_modality == "deblurring":
        if physics_padding not in ("circular", "valid", "valid_v2"):
            raise ValueError(
                "physics_padding must be one of 'circular', 'valid', or 'valid_v2'"
            )
        dataset_key = physics_padding
    elif physics_modality == "denoising":
        dataset_key = "denoising"
    else:
        raise ValueError("physics_modality must be one of 'deblurring' or 'denoising'")
    dataset_path = base_root / DATASET_DIRS[dataset_key] / "dinv_dataset0.h5"

    # Split sizes
    train_split_size = config["train_split_size"]
    val_split_size = config["val_split_size"]

    # Model and architecture parameters
    # Number of scales for UNet variants
    batch_size = config["batch_size"]
    eq_metric_backbone = config.get("eq_metric_backbone", "psnr")
    eq_metric_log_conversion_time = config.get("eq_metric_log_conversion_time", "early")
    eq_metric_kind = config.get("eq_metric_kind", "basic")
    eq_metric_residual_mode = config.get("eq_metric_residual_mode")
    eq_translation_mode = config.get("eq_translation_mode", "circular")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Step 1. Load the dataset.
    print(f"Loading dataset from {dataset_path}...")
    trainval_dataset = dinv.datasets.HDF5Dataset(str(dataset_path), split="train")
    test_dataset = dinv.datasets.HDF5Dataset(str(dataset_path), split="test")

    # NOTE: It is meant to be a deterministic function.
    # Reproducibility is guaranteed as long as trainval_dataset is always the same.
    def get_splits(
        trainval_dataset,
        train_split_size,
        val_split_size,
        physics=None,
        resampling=True,
    ):
        # Split (deterministically) the training split into training and validation splits.
        generator = torch.Generator().manual_seed(0)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            trainval_dataset, [train_split_size, val_split_size], generator=generator
        )

        if resampling:
            assert physics is not None, "Physics must be provided for resampling"
            # NOTE: Resampling the measurements takes compute time but thanks to parallelization it does not hinder performance.
            # Indeed, the next batch is computed while the current one is being processed.
            train_dataset = ResampledMeasurements(train_dataset, physics)

        return train_dataset, eval_dataset

    if physics_modality == "deblurring":
        if physics_padding == "circular":
            physics = dinv.physics.BlurFFT(
                img_size=(3, 128, 128),
                filter=dinv.physics.blur.gaussian_blur(sigma=(1, 1)),
                noise_model=dinv.physics.GaussianNoise(sigma=0.01),
            )
        elif physics_padding == "valid":
            physics = dinv.physics.Blur(
                img_size=(3, 128, 128),
                filter=dinv.physics.blur.gaussian_blur(sigma=(1, 1)),
                noise_model=dinv.physics.GaussianNoise(sigma=0.01),
                padding="valid",
            )
        elif physics_padding == "valid_v2":
            physics = dinv.physics.Blur(
                img_size=(3, 256, 256),
                filter=dinv.physics.blur.gaussian_blur(sigma=(1, 1)),
                noise_model=dinv.physics.GaussianNoise(sigma=0.01),
                padding="valid",
            )
        else:
            raise ValueError(
                "physics_padding must be one of 'circular', 'valid', or 'valid_v2'"
            )
    elif physics_modality == "denoising":
        physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=0.1))
    else:
        raise ValueError("physics_modality must be one of 'deblurring' or 'denoising'")
    physics.to(device)

    train_dataset, eval_dataset = get_splits(
        trainval_dataset=trainval_dataset,
        train_split_size=train_split_size,
        val_split_size=val_split_size,
        physics=physics,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Enable loading next batches while the model is processing the current one.
        num_workers=1,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Step 2. Train the model.
    model = load_model_from_config(
        config, physics=physics, physics_padding=physics_padding, device=device
    )

    # Move model to device (works for both branches)
    model.to(device)

    optimizer_cls = config["optimizer_cls"]
    amsgrad = True
    if optimizer_cls == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            amsgrad=amsgrad,
        )
    elif optimizer_cls == "none":
        optimizer = None
    else:
        raise ValueError(f"Unknown optimizer class: {optimizer_cls}")
    scheduler = None

    if isinstance(physics, dinv.physics.Blur) and physics.padding == "valid":
        if physics_padding == "valid_v2":
            metrics_center_crop = (128, 128)
        else:
            metrics_center_crop = (
                128 - physics.filter.shape[-2] + 1,
                128 - physics.filter.shape[-1] + 1,
            )
        print(f"Setting crop size to {metrics_center_crop} for metrics")
    else:
        metrics_center_crop = None

    loss = [dinv.loss.SupLoss(metric=dinv.metric.MSE(center_crop=metrics_center_crop))]
    metrics = [
        dinv.metric.PSNR(center_crop=metrics_center_crop),
        dinv.metric.SSIM(center_crop=metrics_center_crop),
        dinv.metric.LPIPS(center_crop=metrics_center_crop, device=device),
    ]
    translation = ContinuousTranslation()
    if eq_metric_backbone == "snr":
        if eq_metric_log_conversion_time != "early":
            raise ValueError(
                "eq_metric_log_conversion_time must be 'early' when eq_metric_backbone is 'snr'"
            )
        metric = dinv.metric.SNR()
        db = False
    elif eq_metric_backbone == "psnr":
        if eq_metric_log_conversion_time not in ("early", "late", "never"):
            raise ValueError(
                "eq_metric_log_conversion_time must be one of 'early', 'late', or 'never'"
            )
        metric = dinv.metric.MSE()
        db = eq_metric_log_conversion_time == "early"
    else:
        raise ValueError("eq_metric_backbone must be one of 'psnr' or 'snr'")
    if eq_translation_mode == "circular":
        translation_mode = "circular"
    elif eq_translation_mode == "crop_shift":
        translation_mode = "crop_shift"
    else:
        raise ValueError("eq_translation_mode must be 'circular' or 'crop_shift'")

    if eq_metric_kind == "basic":
        eq_residue = False
    elif eq_metric_kind == "residual":
        eq_residue = True
    else:
        raise ValueError("eq_metric_kind must be 'basic' or 'residual'")

    eq_crop_size = 128 if physics_padding == "valid_v2" else None

    test_metrics = [
        EQUIV(
            translation=translation,
            metric=metric,
            n_samples=5,
            db=db,
            ignore_nyquist=True,
            residue=eq_residue,
            residual_mode=eq_metric_residual_mode,
            translation_mode=translation_mode,
            crop_size=eq_crop_size,
        )
    ]

    run_name = config.get("run_name") or config.get("config", "Run")
    run_name_prefix = config.get("run_name_prefix", "")  # new: allow empty prefix
    mlflow_vis = config["mlflow"]
    if mlflow_vis:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Start a new MLflow run to track this script
        mlops_run_name = f"{run_name_prefix}{run_name}"
        mlflow.start_run(run_name=mlops_run_name)

        # Log hyperparameters (skip Nones)
        mlops_config = {k: v for k, v in config.items() if v is not None}
        for key in ("GIT_COMMIT_ID", "SLURM_JOB_ID"):
            if key in os.environ:
                mlops_config[key] = os.environ[key]
        mlflow.log_params(mlops_config)

        # HOTFIX
        mlflow.end_run = lambda: None

    epochs = config["epochs"]
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is not None:
        assert (
            config.get("checkpoint_run_id") is not None
        ), "If checkpoint_path is specified, checkpoint_run_id must also be specified."

    trainer = Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=epochs,
        losses=loss,
        metrics=metrics,
        device=device,
        online_measurements=False,  # We put the augmentation in the dataset itself.
        ckpt_pretrained=checkpoint_path,
        wandb_vis=False,
        mlflow_vis=mlflow_vis,
        save_path=config["out_dir"],
        display_losses_eval=True,
        ckp_interval=50,
        no_learning_method="y",
    )

    _ = trainer.train()
    if epochs != 0:
        _ = trainer.load_best_model()

    test_save_path = f"{trainer.save_path}/test"
    os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    trainer.metrics += test_metrics
    metrics = trainer.test(
        test_dataloader, save_path=test_save_path, compare_no_learning=True
    )

    # Post-process the symmetric confidence interval in log-scale
    if eq_metric_log_conversion_time == "late":
        for mean_key, std_key in [
            ("EQUIV no learning", "EQUIV no learning_std"),
            ("EQUIV", "EQUIV_std"),
        ]:
            if mean_key in metrics and std_key in metrics:
                avg_mse = metrics[mean_key]
                std_mse = metrics[std_key]
                if avg_mse > 0:
                    if std_mse < avg_mse:
                        upper_mse = avg_mse + std_mse
                        lower_mse = avg_mse - std_mse
                        mean_log = -10.0 * math.log10(avg_mse)
                        # Upper and lower are swapped because the log transform is decreasing.
                        upper_log = -10.0 * math.log10(lower_mse)
                        lower_log = -10.0 * math.log10(upper_mse)
                        std_log = max(mean_log - lower_log, upper_log - mean_log)
                    else:
                        mean_log = -10.0 * math.log10(avg_mse)
                        std_log = float("inf")
                elif avg_mse == 0:
                    mean_log = float("inf")
                    std_log = float("nan")
                else:
                    mean_log = float("nan")
                    std_log = float("nan")
                metrics[mean_key] = mean_log
                metrics[std_key] = std_log

    # Measure inference speed
    torch.cuda.empty_cache()
    model.eval()

    # Warm-up pass
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            y = y.to(device, non_blocking=True)
            _ = model(y, physics)
    torch.cuda.synchronize()

    total_time = 0.0
    total_images = 0
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            batch_size = y.size(0)
            total_images += batch_size
            y = y.to(device, non_blocking=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(y, physics)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - start

    metrics["INFER_TIME"] = (total_time / total_images) * 1000
    metrics["FPS"] = total_images / total_time

    if mlflow_vis:
        run_id = mlflow.active_run().info.run_id
        metrics["STAB"] = compute_instability(run_id=run_id, source="mlflow")

        metrics["config_name"] = f"{run_name_prefix}{run_name}"

        mlflow.log_dict(metrics, "metrics_dict.json")
        mlflow.log_text(str(metrics), "metrics.txt")

        # metrics is a dict that can be understood as a row of a dataframe
        df = pd.DataFrame([metrics])
        mlflow.log_table(data=df, artifact_file="metrics_tbl.json")
