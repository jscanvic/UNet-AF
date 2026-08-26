#!/usr/bin/env python3
"""Run tests defined in a single test.yaml sequentially.

Behavior:
- Loads a YAML file (default: unet_af/test.yaml) containing optional top-level
  `defaults` applied to every test, and a required `tests` mapping of test names
  to entries.
- For each test entry, optionally reads `base` (a training config filename) and an
  `overrides` mapping with keys/values to pass as extra CLI arguments to train.py.
- Runs `python train.py` sequentially for each test. If `base` is present, passes
  `-c <base>`; otherwise omits `-c`.
- Overrides are converted to command-line args of the form `--key value` or
  `--flag` for boolean true.

This script intentionally runs tests sequentially and does not provide a subset
selection; it executes every specified entry.
"""

from __future__ import annotations

import argparse
import fnmatch
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - informative error when yaml is missing
    print("PyYAML is required (pip install pyyaml)")
    raise


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    # Parse known args while keeping any trailing args to forward to train.py
    p = argparse.ArgumentParser(description="Run tests from a consolidated YAML file.")
    p.add_argument(
        "--test_config",
        type=str,
        default="unet_af/test.yaml",
        help="Path to the consolidated test YAML (default: unet_af/test.yaml)",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for running train.py (defaults to current interpreter)",
    )
    p.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Comma-separated list of test names to run; runs all when omitted",
    )
    args, remaining = p.parse_known_args()
    return args, remaining


def normalize_value(v: Any) -> str:
    """Convert a YAML value into a string suitable for a CLI argument value.

    Booleans are treated specially: True -> flag (no value) when used as a simple
    override; False is passed as "false" string so the called script can parse it.
    Lists and dicts are JSON-ish stringified (using repr) so they can be passed as a
    single argument. For most numeric/string scalars we convert to str().
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int | float | str):
        return str(v)
    # For complex types, use repr so it's a single token; caller can interpret
    return repr(v)


def overrides_to_args(overrides: dict[str, Any]) -> list[str]:
    """Turn a mapping of overrides into a flat list of CLI args.

    Example: {"epochs": 0, "model_kind": "tv", "use_flag": True}
    -> ["--epochs", "0", "--model_kind", "tv", "--use_flag"]

    Note: If a value is boolean True, we emit a flag without a value. For False we
    emit the value 'false'. This keeps the behavior explicit.
    """
    args: list[str] = []
    for k, v in overrides.items():
        key = f"--{k}"
        if isinstance(v, bool) and v is True:
            args.append(key)
        else:
            args.append(key)
            args.append(normalize_value(v))
    return args


def run_test_entry(
    py_executable: str,
    entry_name: str,
    entry: dict[str, Any],
    repo_dir: Path,
    default_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
) -> int:
    print(f"\n=== Running test: {entry_name} ===")

    cmd = [py_executable, "train.py"]

    base = entry.get("base")
    if base:
        # Pass -c <base>
        cmd.extend(["-c", str(base)])

    overrides: dict[str, Any] = {}
    if default_overrides:
        overrides.update(default_overrides)

    entry_overrides = entry.get("overrides") or {}
    if not isinstance(entry_overrides, dict):
        print(
            f"Warning: 'overrides' for test '{entry_name}' is not a mapping; skipping overrides"
        )
        entry_overrides = {}
    overrides.update(entry_overrides)

    args = overrides_to_args(overrides)
    cmd.extend(args)

    # Append any additional CLI arguments passed to test.py after known flags.
    if extra_args:
        cmd.extend(extra_args)

    # Run in the repo directory so relative paths inside train.py behave as expected
    cwd = str(repo_dir)

    print("Command:", " ".join(shlex.quote(x) for x in cmd))

    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


def main() -> int:
    args, passthrough_args = parse_args()
    config_path = Path(args.test_config)
    if not config_path.exists():
        print(f"Test config file not found: {config_path}")
        return 2

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        print("Top-level of test config must be a mapping")
        return 3

    defaults = cfg.get("defaults") or {}
    if defaults and not isinstance(defaults, dict):
        print("'defaults' must be a mapping when provided")
        return 3

    default_overrides = defaults.get("overrides") or {}
    if default_overrides and not isinstance(default_overrides, dict):
        print("'defaults.overrides' must be a mapping when provided")
        return 3

    tests_cfg = cfg.get("tests")
    if not isinstance(tests_cfg, dict):
        print("Config must include a 'tests' mapping of test-name -> entry")
        return 3

    repo_dir = Path(__file__).resolve().parent

    subset_names: list[str] | None = None
    if args.subset:
        subset_patterns = [
            name.strip() for name in args.subset.split(",") if name.strip()
        ]
        if not subset_patterns:
            print("Subset provided but no valid test names parsed; running nothing")
            return 0
        available_names = list(tests_cfg.keys())
        subset_names = []
        for pattern in subset_patterns:
            matches = [
                name for name in available_names if fnmatch.fnmatch(name, pattern)
            ]
            if not matches:
                print(f"No tests matched pattern '{pattern}'; skipping")
                continue
            for name in matches:
                if name not in subset_names:
                    subset_names.append(name)

    # Iterate over entries; by default follow YAML order. If subset is given, respect
    # the caller-provided order and warn on unknown names.
    entries_iter = (
        ((name, tests_cfg.get(name)) for name in subset_names)
        if subset_names is not None
        else tests_cfg.items()
    )

    for name, entry in entries_iter:
        if entry is None:
            print(f"Skipping test '{name}': not found in config")
            continue
        if not isinstance(entry, dict):
            print(f"Skipping test '{name}': entry is not a mapping")
            continue
        rc = run_test_entry(
            args.python, name, entry, repo_dir, default_overrides, passthrough_args
        )
        if rc != 0:
            print(
                f"Test '{name}' exited with return code {rc}; continuing to next test"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
