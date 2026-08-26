"""Compute training instability from a metric time series.

Instability is defined as the population standard deviation of the first-order
finite differences y_k = x_{k+1} - x_k of the Eval PSNR time series, truncated
at the last epoch achieving the best Eval PSNR.
"""

from __future__ import annotations

from typing import Literal

import torch

from unet_af.io import fetch_metric_series

def stab(metrics: torch.Tensor, *, better: str = "higher", db: bool = True) -> float:
    r"""
    Training stability

    The stability metric is the variance of the first-order finite differences of a metric (e.g., evaluation PSNR),
    taken between the first and best epoch, and reported in dB (higher is better).

    :param torch.Tensor metrics: 1D tensor of per-epoch metric values.
    :param str better: whether ``"higher"`` or ``"lower"`` values of ``metrics`` are better.
    :param bool db: if ``True``, report the metric in decibels.
    :return: (:class:`float`) stability metric.
    """

    if metrics.ndim != 1:
        raise ValueError("Metrics tensor must be 1D")

    # The best epoch is the last best-performing epoch.
    # NOTE: The argmax function featured in PyTorch returns the first index associated to the maximum value.
    match better:
        case "higher":
            best_idx = metrics.shape[-1] - 1 - metrics.flip(-1).argmax(dim=-1)
        case "lower":
            best_idx = metrics.shape[-1] - 1 - metrics.flip(-1).argmin(dim=-1)
        case _:
            raise ValueError(f"better must be 'higher' or 'lower', got {better!r}")

    # Retrieve the best index as a Python float
    best_idx = best_idx.item()

    # The stability metric is computed only up to the best epoch.
    metrics = metrics[: best_idx + 1]

    # Stable trainings are characterized by a smooth monotonic progression up to the best epoch. This is measured
    # by the variance of the first-order finite differences of the metric time series.
    grad = metrics[1:] - metrics[:-1]
    out = torch.var(grad, unbiased=False)

    # The stability metric typically takes values between 0 and 1, requiring heavy scientific notation when displayed.
    # Instead, we report the metric in decibels, which is more compact.
    if db:
        out = -10 * torch.log10(out)

    return out.item()


def compute_instability(
    *,
    run_id: str,
    metric_key: str = "Eval PSNR",
    source: Literal["mlflow", "huggingface"] = "huggingface",
) -> float:
    """Compute instability for a (possibly resumed) run chain in an experiment.

    Args:
        run_id: The latest run id (may be a continuation of a previous run).
        metric_key: Metric key to use (defaults to ``Eval PSNR``).
        source: Where to fetch the metric series from, ``"mlflow"`` (the
            private HPC tracking server) or ``"huggingface"`` (the public
            snapshot released alongside the model weights).
    Returns:
        Instability value as a float.
    Raises:
        ValueError: If runs are missing metrics or experiment/run mismatch.
        ImportError: If mlflow is not installed (only when source="mlflow").
    """

    psnr = fetch_metric_series(run_id, metric_key, source=source)
    return stab(psnr, better="higher", db=True)
