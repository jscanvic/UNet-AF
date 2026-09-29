"""Device selection and synchronization for CUDA, Apple Silicon (MPS) and CPU."""

import torch


def default_device() -> torch.device:
    """CUDA if available, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: str | torch.device) -> None:
    """Wait for all queued work on `device` to finish (no-op on CPU)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: str | torch.device) -> None:
    """Release cached allocator memory on `device` (no-op on CPU)."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
