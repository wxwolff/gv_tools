"""Readers for METEK Micro Rain Radar products."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import xarray as xr


_MRR2_VARIABLES = frozenset(
    {
        "MRR_H",
        "MRR_TF",
        "MRR_F",
        "MRR_D",
        "MRR_N",
        "MRR_K",
        "MRR_Capital_Z",
        "MRR_Small_z",
        "MRR_PIA",
        "MRR_RR",
        "MRR_LWC",
        "MRR_W",
    }
)
_MRRPRO_VARIABLES = frozenset(
    {
        "Za",
        "Z",
        "Zea",
        "Ze",
        "RR",
        "LWC",
        "PIA",
        "VEL",
        "WIDTH",
        "SNR",
        "spectrum_raw",
        "N",
        "D",
        "range",
        "time",
    }
)
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _open_zipped_netcdf(source: Path, **options: Any) -> xr.Dataset:
    """Open and eagerly load the single NetCDF payload in *source*."""
    try:
        with ZipFile(source) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
                and not member.filename.startswith("__MACOSX/")
                and member.filename.lower().endswith((".nc", ".cdf", ".netcdf"))
            ]
            if len(members) != 1:
                raise ValueError(
                    "an MRR ZIP archive must contain exactly one NetCDF file; "
                    f"found {len(members)}"
                )
            payload = archive.read(members[0])
    except BadZipFile as exc:
        raise ValueError(f"invalid MRR ZIP archive: {source}") from exc

    default_engine = "h5netcdf" if payload.startswith(_HDF5_SIGNATURE) else "scipy"
    engine = options.pop("engine", default_engine)
    try:
        with xr.open_dataset(BytesIO(payload), engine=engine, **options) as opened:
            return opened.load()
    except (ImportError, ModuleNotFoundError) as exc:
        if default_engine == "h5netcdf":
            raise ImportError(
                "zipped MRRPro ingest requires h5netcdf and h5py; install "
                "GV Tools with the 'netcdf' extra"
            ) from exc
        raise


def _identify_model(dataset: xr.Dataset) -> str:
    variables = frozenset(dataset.variables)
    if _MRR2_VARIABLES <= variables:
        return "MRR2"
    if _MRRPRO_VARIABLES <= variables:
        title = str(dataset.attrs.get("title", ""))
        conventions = str(dataset.attrs.get("Conventions", ""))
        if "MRR Pro" in title or conventions.casefold().startswith("cf/radial"):
            return "MRRPro"

    missing_mrr2 = sorted(_MRR2_VARIABLES.difference(variables))
    missing_mrrpro = sorted(_MRRPRO_VARIABLES.difference(variables))
    raise ValueError(
        "unsupported MRR product: file matches neither the MRR2 processed-data "
        f"schema (missing {', '.join(missing_mrr2[:3])}) nor the MRRPro "
        f"CF/Radial schema (missing {', '.join(missing_mrrpro[:3])})"
    )


def _decode_unix_time(dataset: xr.Dataset) -> xr.Dataset:
    """Decode the non-CF ``UNIX Time Stamp`` coordinate used by MRR2."""
    if "time" not in dataset.coords:
        raise ValueError("invalid MRR2 product: missing time coordinate")

    time = dataset.coords["time"]
    if str(time.attrs.get("units", "")).casefold() != "unix time stamp":
        return dataset

    values = np.asarray(time.values)
    fill_value = time.encoding.get("_FillValue", time.attrs.get("_FillValue", -9999))
    valid = values != fill_value
    decoded = np.full(values.shape, np.datetime64("NaT"), dtype="datetime64[s]")
    decoded[valid] = values[valid].astype("datetime64[s]")
    dataset = dataset.assign_coords(time=(time.dims, decoded.astype("datetime64[ns]")))
    dataset.coords["time"].attrs.update(
        standard_name="time", long_name="Time (UTC)", timezone="UTC"
    )
    return dataset


def read_mrr(file: str | Path, **open_dataset_options: Any) -> xr.Dataset:
    """Read an MRR2 or MRRPro NetCDF product into an xarray Dataset.

    ``file`` may name a NetCDF file or a ZIP archive containing exactly one
    NetCDF file. Options are forwarded to :func:`xarray.open_dataset`.
    MRR2's non-CF Unix timestamp is converted to ``datetime64[ns]`` in UTC;
    MRRPro's CF time coordinate is decoded by xarray. The detected instrument
    model is recorded in the dataset attribute ``mrr_model``.
    """
    source = Path(file).expanduser()
    if not source.is_absolute():
        raise ValueError(f"MRR input filename must be fully qualified: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"MRR input is not a file: {source}")

    if source.suffix.casefold() == ".zip":
        dataset = _open_zipped_netcdf(source, **open_dataset_options)
    else:
        dataset = xr.open_dataset(source, **open_dataset_options)

    try:
        model = _identify_model(dataset)
        if model == "MRR2":
            dataset = _decode_unix_time(dataset)
        dataset.attrs["mrr_model"] = model
        return dataset
    except Exception:
        dataset.close()
        raise
