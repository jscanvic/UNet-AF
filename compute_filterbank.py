import argparse

import numpy as np
import torch
from skimage import data

from unet_af.devices import default_device
from unet_af.filters import compute_filter_bank
from unet_af.io import save_filter_bank


def plot_psnr_curves(all_train_psnrs, all_val_psnrs, optimizer_name):
    print("\nGenerating and saving PSNR curves plot...")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Train PSNR
    for imsize, psnrs in all_train_psnrs.items():
        label = f"{imsize}px" if isinstance(imsize, int) else str(imsize)
        ax1.plot(range(1, len(psnrs) + 1), psnrs, label=label)
    ax1.set_xlabel("Iteration / Evaluation")
    ax1.set_ylabel("Train PSNR (dB)")
    ax1.set_title("Training PSNR Curves")
    ax1.legend()
    ax1.grid(True)

    # Right plot: Eval PSNR
    for imsize, psnrs in all_val_psnrs.items():
        label = f"{imsize}px" if isinstance(imsize, int) else str(imsize)
        ax2.plot(range(1, len(psnrs) + 1), psnrs, label=label)
    ax2.set_xlabel("Iteration / Evaluation")
    ax2.set_ylabel("Eval PSNR (dB)")
    ax2.set_title("Evaluation PSNR Curves")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f"PSNR Optimization Curves ({optimizer_name.upper()})")
    plt.tight_layout()

    plot_filenames = [
        "optimization_curves_fitting.png",
        "optimization_curves_fitting.pdf",
        "optimization_curves_fitting.svg",
    ]
    for plot_filename in plot_filenames:
        plt.savefig(plot_filename)
    print(f"Saved PSNR curves to '{', '.join(plot_filenames)}'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Tune circular filtered LReLU filters to match target FilteredNonlinearity."
    )
    parser.add_argument(
        "--imsize",
        type=int,
        nargs="+",
        default=[128, 64, 32, 16, 8],
        help="Target image sizes to compute (default: [128, 64, 32, 16, 8]).",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=13,
        help="Number of taps for the filters (default: 2 * imsize).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "lbfgs"],
        default="lbfgs",
        help="Optimizer to use (default: lbfgs).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3 for Adam, 1.0 for LBFGS).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of optimization steps (default: 300 for Adam, 200 for LBFGS).",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Tune full asymmetric filters instead of symmetric half-filters.",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    symmetric = not args.asymmetric
    device = default_device()
    print(f"Using device: {device}")

    # Set defaults based on optimizer selection
    if args.optimizer == "lbfgs":
        lr = args.lr if args.lr is not None else 1.0
        steps = args.steps if args.steps is not None else 200
    else:
        lr = args.lr if args.lr is not None else 1e-3
        steps = args.steps if args.steps is not None else 300

    # Load natural images from scikit-image
    print("Loading natural images from scikit-image...")
    train_np = data.chelsea()  # cat image (300 x 451)
    val_np = data.astronaut()  # astronaut image (512 x 512)

    filter_bank, all_train_psnrs, all_val_psnrs = compute_filter_bank(
        resolution_list=args.imsize,
        taps_default=args.taps,
        optimizer_name=args.optimizer,
        lr=lr,
        steps=steps,
        symmetric=symmetric,
        device=device,
        train_np=train_np,
        val_np=val_np,
    )

    # Save filters to disk
    save_filter_bank(
        filter_bank, taps=args.taps, asymmetric=args.asymmetric, verbose=True
    )

    # Plot and save PSNR curves
    plot_psnr_curves(all_train_psnrs, all_val_psnrs, args.optimizer)


if __name__ == "__main__":
    main()
