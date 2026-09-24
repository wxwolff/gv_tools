import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gv_tools import InstrumentMetadata
from gv_tools.adapters import parsivel as module
from gv_tools.adapters.parsivel import ParsivelAdapter
from gv_tools.outputs import output_directory, plot_directory, write_product


def sample_product(monkeypatch):
    monkeypatch.setattr(module, "_backend", lambda: SimpleNamespace(__version__="1.0.0"))
    time = pd.date_range("2026-07-21", periods=2, freq="min")
    parameters = pd.DataFrame({"Rain": [1.0, 2.0]}, index=time)
    psd = pd.DataFrame([[1, 2], [3, 4]], index=time, columns=[0.1, 0.2])
    moments = pd.DataFrame(np.arange(4).reshape(2, 2), index=time, columns=["M0", "M1"])
    metadata = InstrumentMetadata("apu01", "disdrometer", "WFF")
    return ParsivelAdapter(metadata).normalize(parameters, psd, moments)


def test_output_directory_follows_convention(tmp_path, monkeypatch):
    product = sample_product(monkeypatch)
    assert output_directory(tmp_path, product) == tmp_path / "WFF/apu01/2026/07/21"


def test_plot_directory_follows_requested_date_convention(tmp_path):
    result = plot_directory(tmp_path, "Parsivel Parameters", "2026-08-10")
    assert result == tmp_path / "Plots/Parsivel_Parameters/2026/08"
    assert result.is_dir()


def test_csv_writer_creates_tables_and_manifest(tmp_path, monkeypatch):
    created = write_product(sample_product(monkeypatch), tmp_path, formats=("csv",))
    assert set(created) == {"parameters_csv", "dsd_csv", "moments_csv", "manifest"}
    manifest = json.loads(open(created["manifest"], encoding="utf-8").read())
    assert manifest["schema_version"] == "0.2"
    assert manifest["source_platform"] == "apu"
    assert manifest["time_coverage_start"] == "2026-07-21T00:00:00"
    assert len(manifest["files"]) == 3
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_netcdf_writer_creates_readable_product(tmp_path, monkeypatch):
    import xarray as xr

    created = write_product(
        sample_product(monkeypatch), tmp_path, formats=("netcdf",), engine="h5netcdf"
    )
    with xr.open_dataset(created["netcdf"], engine="h5netcdf") as dataset:
        assert dataset.attrs["instrument_id"] == "apu01"
        assert dataset.sizes["drop_diameter"] == 2
