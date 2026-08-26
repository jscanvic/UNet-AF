import hashlib
import importlib.resources as resources
import os
import subprocess
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.datasets import load_sample_image

from unet_af.mlops import _fetch_metrics_table_mlflow

HF_REPO_ID = "jscanvic/unet-af"


def load_checkpoint_manifest() -> dict:
    """Load the friendly-filename -> remote path mapping from checkpoints.yaml."""
    checkpoints_path = os.environ["CHECKPOINTS_PATH"]
    manifest_path = resources.files("unet_af").joinpath("..", checkpoints_path)
    return yaml.safe_load(manifest_path.read_text()) or {}


def pull_file(
    remote_path: str, local_path: Path, host: str | None = None
) -> None:
    """Fetch a file from a remote host into local_path if not already cached."""
    if host is None:
        host = os.environ["REMOTE_HOSTNAME"]
    if not local_path.exists():
        remote = f"{host}:{remote_path}"
        subprocess.run(["rsync", remote, str(local_path)], check=True)


def compute_cache_path(remote_path: str) -> Path:
    dirname = hashlib.md5(remote_path.encode("utf-8")).hexdigest()[:8]
    cache_dir = Path(torch.hub.get_dir()) / dirname
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / Path(remote_path).name


def resolve_remote_image_path(path: str, fs: str = "local") -> Path:
    if fs == "local":
        local_path = Path(path).expanduser()
    elif fs == "remote":
        local_path = compute_cache_path(path)
        pull_file(path, local_path)
    else:
        raise ValueError(f"Unsupported fs='{fs}', use 'local' or 'remote'")
    if not local_path.exists():
        raise FileNotFoundError(f"Image not found at {local_path}")
    return local_path


def load_sklearn_image(name: str) -> Image.Image:
    name = name.lower().replace(".jpg", "")
    samples = {"china": "china.jpg", "flower": "flower.jpg"}
    if name not in samples:
        raise ValueError(
            f"Unknown sample image '{name}', choose from {list(samples.keys())}"
        )
    arr = load_sample_image(samples[name])
    return Image.fromarray(arr)


def load_image(path_or_name: str, fs: str = "local") -> Image.Image:
    if fs == "remote":
        local_path = resolve_remote_image_path(path_or_name, fs=fs)
        return Image.open(local_path).convert("RGB")
    if Path(path_or_name).exists():
        return Image.open(path_or_name).convert("RGB")
    return load_sklearn_image(path_or_name)


def resolve_weights_path(path: str, fs: str = "local") -> Path:
    """Resolve a checkpoint path locally, from a remote host, or from HuggingFace.

    Parameters
    ----------
    path: str
        For fs="local", a local filesystem path. For fs="remote", either a
        user-friendly filename registered in checkpoints.yaml (e.g.
        "Main_UNet-AF.pth.tar") or, as a fallback, a full path on the remote
        host. For fs="huggingface", a user-friendly filename, used verbatim
        as the filename in the HF_REPO_ID repo.
    fs: {"local", "remote", "huggingface"}
        Filesystem selector. When "remote", the file is rsynced into a
        torch hub cache directory if not already present. When
        "huggingface", the file is downloaded (and cached) via
        huggingface_hub.
    """
    if fs == "local":
        ckpt_path = Path(path).expanduser()
    elif fs == "remote":
        manifest = load_checkpoint_manifest()
        remote_path = manifest.get(path, path)
        ckpt_path = compute_cache_path(remote_path)
        pull_file(remote_path, ckpt_path)
    elif fs == "huggingface":
        ckpt_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=path))
    else:
        raise ValueError(
            f"fs={fs} is not supported; use 'local', 'remote', or 'huggingface'"
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    return ckpt_path


def _fetch_metrics_table_huggingface(run_id: str) -> pd.DataFrame:
    """Fetch the metrics table for a run from the public HuggingFace snapshot."""
    digest = hashlib.md5(run_id.encode("utf-8")).hexdigest()
    csv_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=f"runs/{digest}.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics table not found at {csv_path}")
    return pd.read_csv(csv_path, index_col="step")


def fetch_metrics_table(
    run_id: str, source: Literal["mlflow", "huggingface"] = "huggingface"
) -> pd.DataFrame:
    """Fetch a wide metrics table (one column per metric key) for a run.

    ``source="mlflow"`` reads the private HPC tracking server, walking the
    resumed-run chain. ``source="huggingface"`` reads the public snapshot
    released alongside the model weights.
    """
    if source == "mlflow":
        return _fetch_metrics_table_mlflow(run_id)
    if source == "huggingface":
        return _fetch_metrics_table_huggingface(run_id)
    raise ValueError(f"Unknown source '{source}'")


def fetch_metric_series(
    run_id: str,
    metric_key: str,
    source: Literal["mlflow", "huggingface"] = "huggingface",
) -> torch.Tensor:
    table = fetch_metrics_table(run_id, source=source)
    series = table[metric_key].dropna()
    series = series.tolist()
    return torch.tensor(series)


def load_filter_bank(*, taps: int = 13) -> dict:
    path = (
        resources.files("unet_af") / f"filters/filterbank_128-8px_{taps}taps.pt"
    )
    return torch.load(path, weights_only=True)


def save_filter_bank(
    filter_bank: dict,
    *,
    taps: int | None = 13,
    asymmetric: bool = False,
    verbose: bool = True,
) -> None:
    asym_suffix = "_asymmetric" if asymmetric else ""
    path = (
        resources.files("unet_af")
        / f"filters/filterbank_128-8px_{taps}taps{asym_suffix}.pt"
    )
    torch.save(filter_bank, path)
    if verbose:
        print(f"Filter bank saved to '{path}'")
