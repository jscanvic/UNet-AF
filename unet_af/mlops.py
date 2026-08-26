"""Fetch metric time series from MLflow, including resumed run chains.

When runs are resumed, we walk backwards through the linked runs via the
``checkpoint_run_id`` parameter and merge overlapping epochs by preferring
the most recent run's values.
"""

from __future__ import annotations

import os

import pandas as pd

try:  # pragma: no cover - dependency availability
    import mlflow
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient
except Exception as exc:  # pragma: no cover - surfaced at import time
    raise ImportError("mlflow is required to fetch metric series") from exc


_CHECKPOINT_PARAM = "checkpoint_run_id"
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "UNetAF")

# "step" is logged by the training script as a metric in its own right
# (redundant with the epoch index), a quirk that shouldn't leak into tables.
_IGNORED_METRIC_KEYS = {"step"}


def _fetch_metrics_table_mlflow(starting_run_id: str) -> pd.DataFrame:
    """Fetch every metric logged along a resumed-run chain into one table.

    Returned as a wide table indexed by ``step``, with one column per metric
    key. When runs overlap on a given (metric_key, step), the value from the
    most recent run is kept.
    """
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")

    client = MlflowClient()

    # Walk the chain of runs from latest to earliest using checkpoint_run_id.
    ordered_runs: list[tuple[str, dict[str, list[Metric]]]] = []
    seen_runs: set[str] = set()
    current = starting_run_id
    while current and current not in seen_runs:
        run = client.get_run(current)
        if run.info.experiment_id != experiment.experiment_id:
            raise ValueError(
                f"Run {current} is in experiment {run.info.experiment_id}, "
                f"expected {experiment.experiment_id}"
            )
        histories = {
            metric_key: client.get_metric_history(current, metric_key)
            for metric_key in run.data.metrics
            if metric_key not in _IGNORED_METRIC_KEYS
        }
        ordered_runs.append((current, histories))
        seen_runs.add(current)
        current = run.data.params.get(_CHECKPOINT_PARAM)

    # Merge histories, preferring newer runs when (metric_key, step) overlap.
    cell_values: dict[tuple[str, int], float] = {}
    for _run_id, histories in ordered_runs:
        for metric_key, history in histories.items():
            for m in history:
                step = int(m.step if m.step is not None else 0)
                cell = (metric_key, step)
                if cell in cell_values:
                    continue
                cell_values[cell] = float(m.value)

    long_table = pd.DataFrame(
        [
            {"metric_key": metric_key, "step": step, "value": value}
            for (metric_key, step), value in cell_values.items()
        ],
        columns=["metric_key", "step", "value"],
    )
    return (
        long_table.pivot(index="step", columns="metric_key", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
