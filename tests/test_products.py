import pandas as pd
import pytest
import xarray as xr

from gv_tools.products import validate_product


def product():
    return xr.Dataset(
        {"precipitation_rate": (
            "time", [1.0, 2.0], {"units": "mm h-1", "long_name": "Rain rate"}
        )},
        coords={"time": pd.date_range("2026-01-01", periods=2, freq="min")},
        attrs={
            "instrument_id": "apu01",
            "instrument_type": "disdrometer",
            "site_id": "WFF",
            "processing_level": "2",
            "processing_software": "gv_tools",
            "processing_software_version": "0.1.0",
            "processing_time": "2026-01-01T00:00:00+00:00",
            "source": "fixture",
            "time_reference": "UTC",
        },
    )


def test_valid_product_is_returned():
    dataset = product()
    assert validate_product(dataset) is dataset


def test_duplicate_times_are_rejected():
    dataset = product().assign_coords(
        time=pd.to_datetime(["2026-01-01", "2026-01-01"])
    )
    with pytest.raises(ValueError, match="duplicates"):
        validate_product(dataset)


def test_non_utc_product_is_rejected():
    dataset = product()
    dataset.attrs["time_reference"] = "local"
    with pytest.raises(ValueError, match="UTC"):
        validate_product(dataset)


def test_empty_time_coordinate_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_product(product().isel(time=slice(0, 0)))


def test_variable_metadata_is_required():
    dataset = product()
    dataset.precipitation_rate.attrs.pop("units")
    with pytest.raises(ValueError, match="precipitation_rate"):
        validate_product(dataset)
