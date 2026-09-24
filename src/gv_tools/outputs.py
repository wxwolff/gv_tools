"""Stable output paths, NetCDF/CSV writing, and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr

from .products import validate_product


@dataclass(frozen=True)
class ManifestFile:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ProductManifest:
    schema_version: str
    created_utc: str
    site_id: str
    instrument_id: str
    product_date: str
    processing_software: str
    processing_software_version: str
    source: str
    source_platform: str
    time_coverage_start: str
    time_coverage_end: str
    files: list[ManifestFile]


def plot_directory(root, plot_type: str, day, *, create=True) -> Path:
    """Return ``root/Plots/PlotType/YYYY/MM`` for a requested plot date."""
    name = plot_type.strip().replace(" ", "_")
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("plot_type must be one safe directory name")
    stamp = pd.Timestamp(day)
    target = Path(root).expanduser() / "Plots" / name / stamp.strftime("%Y/%m")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def output_directory(root, dataset: xr.Dataset, day=None) -> Path:
    """Return ``root/site/instrument/YYYY/MM/DD`` for a normalized product."""
    validate_product(dataset)
    stamp = pd.Timestamp(day if day is not None else dataset.time.values[0])
    return (
        Path(root).expanduser()
        / str(dataset.attrs["site_id"])
        / str(dataset.attrs["instrument_id"])
        / stamp.strftime("%Y/%m/%d")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace(staged: Path, destination: Path) -> None:
    """Atomically place a completely written file at its destination."""
    os.replace(staged, destination)


def write_product(dataset, root, *, formats=("netcdf",), day=None, engine=None):
    """Write a normalized product and JSON manifest; return created paths."""
    validate_product(dataset)
    requested = tuple(item.lower() for item in formats)
    unsupported = set(requested).difference({"netcdf", "csv"})
    if unsupported:
        raise ValueError(f"unsupported output format(s): {sorted(unsupported)}")
    target = output_directory(root, dataset, day)
    target.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(day if day is not None else dataset.time.values[0])
    stem = f"{dataset.attrs['site_id']}_{dataset.attrs['instrument_id']}_{stamp:%Y%m%d}"
    created: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix=".gv-tools-", dir=target) as staging_text:
        staging = Path(staging_text)
        staged: dict[str, tuple[Path, Path]] = {}
        for fmt in requested:
            if fmt == "netcdf":
                final = target / f"{stem}.nc"
                temporary = staging / final.name
                dataset.to_netcdf(temporary, engine=engine)
                staged[fmt] = temporary, final
            elif fmt == "csv":
                final = target / f"{stem}_parameters.csv"
                temporary = staging / final.name
                dataset.drop_dims([
                    dim for dim in ("drop_diameter", "moment_order")
                    if dim in dataset.dims
                ]).to_dataframe().to_csv(temporary)
                staged["parameters_csv"] = temporary, final
                if "drop_size_distribution" in dataset:
                    final = target / f"{stem}_dsd.csv"
                    temporary = staging / final.name
                    dataset.drop_size_distribution.to_pandas().to_csv(temporary)
                    staged["dsd_csv"] = temporary, final
                if "drop_size_moment" in dataset:
                    final = target / f"{stem}_moments.csv"
                    temporary = staging / final.name
                    dataset.drop_size_moment.to_pandas().to_csv(temporary)
                    staged["moments_csv"] = temporary, final
        for role, (temporary, final) in staged.items():
            _replace(temporary, final)
            created[role] = str(final)

    manifest_path = target / "manifest.json"
    manifest = ProductManifest(
        schema_version="0.2",
        created_utc=datetime.now(timezone.utc).isoformat(),
        site_id=str(dataset.attrs["site_id"]),
        instrument_id=str(dataset.attrs["instrument_id"]),
        product_date=stamp.strftime("%Y-%m-%d"),
        processing_software=str(dataset.attrs["processing_software"]),
        processing_software_version=str(dataset.attrs["processing_software_version"]),
        source=str(dataset.attrs["source"]),
        source_platform=str(dataset.attrs.get("source_platform", "unknown")),
        time_coverage_start=pd.Timestamp(dataset.time.values[0]).isoformat(),
        time_coverage_end=pd.Timestamp(dataset.time.values[-1]).isoformat(),
        files=[ManifestFile(role, Path(path).name, _sha256(Path(path)))
        for role, path in created.items()
        ],
    )
    handle, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=target)
    os.close(handle)
    temporary_manifest = Path(temporary_name)
    try:
        temporary_manifest.write_text(
            json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
        )
        _replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    created["manifest"] = str(manifest_path)
    return created
