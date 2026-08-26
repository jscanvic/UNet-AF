import torch
import torch.nn as nn
import torch.optim as optim

from unet_af.ops.circular_filtered_lrelu import circular_filtered_lrelu
from unet_af.layers import FilteredNonlinearity, create_1d_mask
from unet_af.metrics import compute_psnr


def get_full_filter(param, num_taps, symmetric=True):
    if not symmetric:
        return param
    if num_taps % 2 != 0:
        return torch.cat([param, param[:-1].flip(dims=[0])])
    else:
        N = num_taps // 2
        left = param[1:N].flip(dims=[0])
        right = param[1:N]
        return torch.cat([param[N : N + 1], left, param[0:1], right])


def compute_ideal_filter(size, cutoff=0.5, direction="upsampling", device=None):
    mask = create_1d_mask(size, cutoff, direction, device=device)
    h_idft = torch.fft.ifft(mask).real
    h_centered = torch.roll(h_idft, shifts=size // 2, dims=0)
    return h_centered


def optimize_filters(
    params,
    optimizer_type,
    lr,
    steps,
    compute_loss_fn,
    log_metrics_fn,
    print_progress_fn=None,
):
    if optimizer_type == "lbfgs":
        optimizer = optim.LBFGS(params, lr=lr, max_iter=steps)

        def closure():
            optimizer.zero_grad()
            loss = compute_loss_fn()
            loss.backward()
            with torch.no_grad():
                log_metrics_fn()
            return loss

        optimizer.step(closure)
    else:
        optimizer = optim.Adam(params, lr=lr)
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            loss = compute_loss_fn()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_metrics_fn()
            if print_progress_fn and (step % 50 == 0 or step == 1):
                print_progress_fn(step, loss)


def preprocess_image(img_np, size=128):
    # Convert numpy HWC image to [1, 3, size, size] PyTorch tensor in range [0, 1]
    img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
    h, w = img.shape[1:]
    min_dim = min(h, w)

    # Center crop to a square
    ch, cw = h // 2, w // 2
    img = img[
        :, ch - min_dim // 2 : ch + min_dim // 2, cw - min_dim // 2 : cw + min_dim // 2
    ]

    # Resize to size x size using bilinear interpolation
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    )
    img = img / 255.0  # Normalize to [0, 1]
    return img


def compute_filter_bank(
    resolution_list,
    taps_default,
    optimizer_name,
    lr,
    steps,
    symmetric,
    device,
    train_np,
    val_np,
):
    all_train_psnrs = {}
    all_val_psnrs = {}
    filter_bank = {}

    for resolution in resolution_list:
        print("\n==========================================")
        print(f"Processing image size: {resolution}x{resolution}")
        print("==========================================")

        train_psnrs = []
        val_psnrs = []

        taps = taps_default if taps_default is not None else resolution * 2

        # Use CenterCrop-Resize to target image size
        train_img = preprocess_image(train_np, size=resolution).to(device)
        val_img = preprocess_image(val_np, size=resolution).to(device)

        # Instantiate target FilteredNonlinearity with ReLU
        print("Instantiating target FilteredNonlinearity model...")
        target_net = FilteredNonlinearity(
            down_mask_shape="square", channels=3, up=2, kind="relu"
        ).to(device)

        # Compute base ideal filters
        ideal_size = resolution * 2
        print(f"Computing base ideal {ideal_size}-tap filters...")
        fu_ideal = compute_ideal_filter(
            size=ideal_size, cutoff=0.5, direction="upsampling", device=device
        )
        fd_ideal = compute_ideal_filter(
            size=ideal_size, cutoff=0.5, direction="downsampling", device=device
        )
        with torch.no_grad():
            target_out = target_net(train_img)
            val_target_out = target_net(val_img)

        # Crop base ideal filters without shifts
        center = resolution

        # Initialize symmetric half-filter parameters
        if not symmetric:
            if taps % 2 != 0:
                start = center - taps // 2
                end = center + taps // 2 + 1
            else:
                start = center - taps // 2
                end = center + taps // 2
            fu_half = nn.Parameter(fu_ideal[start:end].clone())
            fd_half = nn.Parameter(fd_ideal[start:end].clone())
        else:
            if taps % 2 != 0:
                half_taps = taps // 2 + 1
                start_half = center - (half_taps - 1)
                end_half = center + 1
                fu_half = nn.Parameter(fu_ideal[start_half:end_half].clone())
                fd_half = nn.Parameter(fd_ideal[start_half:end_half].clone())
            else:
                half_taps = taps // 2 + 1
                start_half = center - (half_taps - 1)
                end_half = center + 1
                fu_half = nn.Parameter(
                    fu_ideal[start_half:end_half].clone().flip(dims=[0])
                )
                fd_half = nn.Parameter(
                    fd_ideal[start_half:end_half].clone().flip(dims=[0])
                )

        with torch.no_grad():
            fu_init = get_full_filter(fu_half, taps, symmetric=symmetric)
            fd_init = get_full_filter(fd_half, taps, symmetric=symmetric)
            circular_out = circular_filtered_lrelu(
                train_img,
                fu=fu_init,
                fd=fd_init,
                up=2,
                down=2,
                slope=0.0,
                gain=1.0,
                impl="ref",
            )
            init_psnr = compute_psnr(circular_out, target_out, batched=False).item()
        print(f"Initial PSNR: {init_psnr:.2f} dB")

        # Optimize filter coefficients
        print(
            f"\nTuning {taps}-tap filter parameters using {optimizer_name.upper()}..."
        )
        loss_fn = nn.MSELoss()

        def compute_loss():
            fu = get_full_filter(fu_half, taps, symmetric=symmetric)
            fd = get_full_filter(fd_half, taps, symmetric=symmetric)
            circular_out = circular_filtered_lrelu(
                train_img, fu=fu, fd=fd, up=2, down=2, slope=0.0, gain=1.0, impl="ref"
            )
            return loss_fn(circular_out, target_out)

        def log_metrics():
            fu = get_full_filter(fu_half, taps, symmetric=symmetric)
            fd = get_full_filter(fd_half, taps, symmetric=symmetric)
            circular_out = circular_filtered_lrelu(
                train_img, fu=fu, fd=fd, up=2, down=2, slope=0.0, gain=1.0, impl="ref"
            )
            train_psnrs.append(
                compute_psnr(circular_out, target_out, batched=False).item()
            )

            val_circular_out = circular_filtered_lrelu(
                val_img, fu=fu, fd=fd, up=2, down=2, slope=0.0, gain=1.0, impl="ref"
            )
            val_psnrs.append(
                compute_psnr(val_circular_out, val_target_out, batched=False).item()
            )

        def print_progress(step, loss):
            print(
                f"Adam Step {step:3d}/{steps} | Train MSE: {loss.item():.2e} | Train PSNR: {train_psnrs[-1]:6.2f} dB | Val PSNR: {val_psnrs[-1]:6.2f} dB"
            )

        optimize_filters(
            params=[fu_half, fd_half],
            optimizer_type=optimizer_name,
            lr=lr,
            steps=steps,
            compute_loss_fn=compute_loss,
            log_metrics_fn=log_metrics,
            print_progress_fn=print_progress,
        )

        train_psnr = train_psnrs[-1]
        val_psnr = val_psnrs[-1]

        if optimizer_name == "lbfgs":
            print(
                f"L-BFGS optimization completed | Train PSNR: {train_psnr:.2f} dB | Val PSNR: {val_psnr:.2f} dB"
            )

        # Save final results and print optimized parameters
        print("\nOptimization completed successfully!")
        print("=" * 60)
        print(f"Final Train PSNR: {train_psnr:.2f} dB")
        print(f"Final Val PSNR:   {val_psnr:.2f} dB")
        print("=" * 60)

        all_train_psnrs[resolution] = train_psnrs
        all_val_psnrs[resolution] = val_psnrs

        # Gather filters to save at the end
        with torch.no_grad():
            fu_opt = get_full_filter(fu_half, taps, symmetric=symmetric).detach().cpu()
            fd_opt = get_full_filter(fd_half, taps, symmetric=symmetric).detach().cpu()
        filter_bank[resolution] = {"fu": fu_opt, "fd": fd_opt}

    return filter_bank, all_train_psnrs, all_val_psnrs
