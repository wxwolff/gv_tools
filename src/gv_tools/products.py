"""Common product conventions and validation."""

from __future__ import annotations

import numpy as np
import xarray as xr

REQUIRED_GLOBAL_ATTRIBUTES = {
    "instrument_id",
    "instrument_type",
    "site_id",
    "processing_level",
    "processing_software",
    "processing_software_version",
    "processing_time",
    "source",
    "time_reference",
}


def validate_product(dataset: xr.Dataset) -> xr.Dataset:
    """Validate the minimum contract shared by all normalized GV products."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("product must be an xarray.Dataset")
    if "time" not in dataset.coords:
        raise ValueError("product must have a time coordinate")
    if not np.issubdtype(dataset.time.dtype, np.datetime64):
        raise TypeError("time coordinate must contain datetime64 values")
    if dataset.sizes.get("time", 0) == 0:
        raise ValueError("time coordinate must not be empty")
    if dataset.indexes["time"].has_duplicates:
        raise ValueError("time coordinate must not contain duplicates")
    if not dataset.indexes["time"].is_monotonic_increasing:
        raise ValueError("time coordinate must be sorted")
    missing = REQUIRED_GLOBAL_ATTRIBUTES.difference(dataset.attrs)
    if missing:
        raise ValueError(f"product is missing global attributes: {sorted(missing)}")
    if dataset.attrs["time_reference"] != "UTC":
        raise ValueError("normalized products must use UTC")
    for name, variable in dataset.data_vars.items():
        missing_variable_attrs = {"units", "long_name"}.difference(variable.attrs)
        if missing_variable_attrs:
            raise ValueError(
                f"variable {name!r} is missing attributes: {sorted(missing_variable_attrs)}"
            )
    return dataset
