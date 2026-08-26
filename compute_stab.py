import importlib.resources as resources
import warnings

import yaml

from unet_af.metrics import compute_instability

config_path = resources.files("unet_af").joinpath("test.yaml").resolve()
with config_path.open("r") as fh:
    config = yaml.safe_load(fh) or {}

tests = config.get("tests", {}) or {}
name_to_test_key = {
    "UNet-Ronneberger": "Increment_1_Ronneberger",
    "UNet-DeepInverse": "Increment_2_DeepInverse",
    "Increment_3_Baseline": "Increment_3_Baseline",
    "Increment_4_CircularPadding": "Increment_4_CircularPadding",
    "Increment_5_FilteredGELU": "Increment_5_FilteredGELU",
    "Increment_6_FilteredUpconv": "Increment_6_FilteredUpconv",
    "UNet-AF": "Main_UNet-AF",
    "UNet-AFF": "Main_UNet-AFF",
}

for name, test_key in name_to_test_key.items():
    run_id = (
        (tests.get(test_key, {}) or {}).get("overrides", {}).get("checkpoint_run_id")
    )
    if not run_id or run_id == "...":
        warnings.warn(
            f"Run id for {name} ({test_key}) not found in {config_path}; skipping",
            stacklevel=2,
        )
        continue
    value = compute_instability(run_id=run_id)
    print(f"{name}: {value}")
