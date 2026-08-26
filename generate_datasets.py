import argparse
import os
import random
from pathlib import Path

import deepinv as dinv
import numpy as np
import torch

from unet_af.dataset import DATASET_DIRS
from unet_af.transforms import TRANSFORM, TRANSFORM_CROP_SHIFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the training datasets")
    parser.add_argument(
        "--div2k-root",
        type=str,
        default=os.environ.get("DIV2K_ROOT", "datasets/DIV2K"),
        help="Path to the DIV2K dataset root. Defaults to the DIV2K_ROOT env "
        "variable, or datasets/DIV2K if unset.",
    )
    parser.add_argument(
        "--div2k-deblur-path",
        type=str,
        default=os.environ.get("DIV2K_DEBLUR_PATH", "datasets/DIV2K-Deblur"),
        help="Path to the Div2K-Deblur base directory, under which each physics modality's "
        "HDF5 dataset is stored in its own subdirectory. Defaults to the DIV2K_DEBLUR_PATH "
        "env variable, or datasets/DIV2K-Deblur if unset.",
    )
    parser.add_argument(
        "--circular-source-path",
        type=str,
        default=os.environ.get("CIRCULAR_SOURCE_PATH"),
        help="Path to a pre-generated circular dataset to symlink instead of regenerating it. "
        "Defaults to the CIRCULAR_SOURCE_PATH env variable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Generating the datasets")

    base_root = Path(args.div2k_deblur_path)
    circular_source_path = Path(args.circular_source_path)

    div2k_root = args.div2k_root
    if not div2k_root:
        raise RuntimeError(
            "--div2k-root (or DIV2K_ROOT) must be set to generate Div2K-Deblur."
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    physics_configs = {
        "circular": lambda: dinv.physics.BlurFFT(
            img_size=(3, 128, 128),
            filter=dinv.physics.blur.gaussian_blur(sigma=(1, 1)),
            noise_model=dinv.physics.GaussianNoise(sigma=0.01),
        ),
        "valid": lambda: dinv.physics.Blur(
            img_size=(3, 128, 128),
            filter=dinv.physics.blur.gaussian_blur(sigma=(1, 1)),
            noise_model=dinv.physics.GaussianNoise(sigma=0.01),
            padding="valid",
        ),
        "valid_v2": lambda: dinv.physics.Blur(
            img_size=(3, 256, 256),
            filter=dinv.physics.blur.gaussian_blur(sigma=(1, 1)),
            noise_model=dinv.physics.GaussianNoise(sigma=0.01),
            padding="valid",
        ),
        "denoising": lambda: dinv.physics.Denoising(
            dinv.physics.GaussianNoise(sigma=0.1)
        ),
    }

    for physics_modality, physics_fn in physics_configs.items():
        dataset_root = base_root / DATASET_DIRS[physics_modality]
        dataset_root.mkdir(parents=True, exist_ok=True)

        if physics_modality == "circular" and circular_source_path.is_file():
            target_path = dataset_root / "dinv_dataset0.h5"
            if target_path.exists():
                print(
                    f"Circular dataset already present at {target_path}, skipping generation."
                )
            else:
                os.symlink(circular_source_path, target_path)
                print(
                    f"Symlinked circular dataset from {circular_source_path} to {target_path}"
                )
            continue

        physics = physics_fn()
        print(f"  Save directory: {dataset_root}")

        if physics_modality == "valid_v2":
            transform = TRANSFORM_CROP_SHIFT
        else:
            transform = TRANSFORM

        train_gt_dataset = dinv.datasets.DIV2K(
            root=div2k_root,
            mode="train",
            transform=transform,
            download=True,
        )
        test_gt_dataset = dinv.datasets.DIV2K(
            root=div2k_root,
            mode="val",
            transform=transform,
            download=True,
        )

        dinv.datasets.generate_dataset(
            train_dataset=train_gt_dataset,
            test_dataset=test_gt_dataset,
            physics=physics,
            save_dir=str(dataset_root),
            batch_size=1,
            device="cpu",
            show_progress_bar=True,
            overwrite_existing=False,
        )

    print("Dataset generation has finished")


if __name__ == "__main__":
    main()
