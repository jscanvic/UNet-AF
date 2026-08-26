import torch


def compute_psnr(
    x_ref: torch.Tensor, x_rec: torch.Tensor, batched: bool
) -> torch.Tensor:
    dims = tuple(range(1, x_ref.dim())) if batched else None
    mse = (x_ref - x_rec).square().mean(dims)
    return torch.where(mse == 0.0, float("inf"), -10 * torch.log10(mse))
