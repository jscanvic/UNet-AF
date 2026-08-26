import argparse
import os
from pathlib import Path

import deepinv as dinv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-install setup steps")
    parser.add_argument(
        "--div2k-root",
        type=str,
        default=os.environ.get("DIV2K_ROOT", "datasets/DIV2K"),
        help="Path to the DIV2K dataset root. Defaults to the DIV2K_ROOT env "
        "variable, or datasets/DIV2K if unset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting post-install steps...")

    # Step 1: Ensure datasets exist
    print("Step 1: Ensure DIV2K is available")
    div2k_root = args.div2k_root
    div2k_path = Path(div2k_root)
    div2k_exists = (
        div2k_path.exists() and div2k_path.is_dir() and any(div2k_path.iterdir())
    )

    if not div2k_exists:
        print(f"DIV2K not found at {div2k_root}; it will be downloaded now.")
        _ = dinv.datasets.DIV2K(root=div2k_root, mode="train", download=True)
        _ = dinv.datasets.DIV2K(root=div2k_root, mode="val", download=True)
    else:
        print(f"DIV2K found at {div2k_root}, skipping download.")

    # Step 2: Ensure metrics
    print("Step 2: Initializing LPIPS metric...")
    _ = dinv.loss.LPIPS(device="cpu")

    # Step 3: Ensure pre-trained weights
    print("Step 3: Loading RAM model with pretrained weights...")
    _ = dinv.models.RAM(device="cpu", pretrained=True)
    print("Step 3b: Loading DRUNet model with pretrained weights...")
    _ = dinv.models.DRUNet(pretrained="download", device="cpu")

    print("Post-install steps completed.")


if __name__ == "__main__":
    main()
