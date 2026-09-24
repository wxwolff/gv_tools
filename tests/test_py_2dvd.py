"""Independent checks of parsing, SI equations, archive I/O and public plotting.

The synthetic 12-drop minute has an analytically known concentration and moments,
so it checks units and normalization rather than mirroring the implementation.
Archive fixtures exercise identical logical content in different containers.
Clock and empty-input cases cover selection/missing semantics. Figure inspection
checks the selected physical variables, colors, units and returned objects.
Real-file parity is recorded separately in VALIDATION.json; it is not a substitute
for these independent checks and does not prove the physical assumptions.
"""

import io
import tarfile
import zipfile

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gv_tools import py_2dvd as lite
from gv_tools.core.py_2dvd import prepare_drops
from gv_tools.io.py_2dvd import date_from_filename

NUMERIC = "1 0 0 1 0.5235987755982988 4 1 10000 1 1 1 1 1 2 3 4"


def write_input(tmp_path, content=None):
    path = tmp_path / "V23022.drops.txt"
    path.write_text(
        content if content is not None else "header\nheader\n" + (NUMERIC + "\n") * 12
    )
    return path


@pytest.mark.parametrize("extras", [[], ["rain"], ["rain", "1"]])
@pytest.mark.parametrize("extension", ["txt", "zip", "tgz", "tar.gz"])
def test_archive_clock_mapping(tmp_path, extras, extension):
    tokens = NUMERIC.split()
    clock_row = " ".join(["01:00:00.000"] + tokens[3:8] + extras + tokens[8:])
    content = "header\nheader\n" + clock_row + "\n"
    plain = write_input(tmp_path, content)
    if extension == "txt":
        path = plain
    elif extension == "zip":
        path = tmp_path / "drops.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("nested/V23022.drops.txt", content)
    else:
        path = tmp_path / ("drops." + extension)
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("nested/V23022.drops.txt")
            member.size = len(content.encode())
            archive.addfile(member, io.BytesIO(content.encode()))
    raw = lite.ingest_raw(path)
    assert (
        raw.attrs["source_data_sha256"]
        == lite.ingest_raw(plain).attrs["source_data_sha256"]
    )
    assert raw.camera_a_height_mm.iloc[0] == 1
    assert raw.camera_b_max_index.iloc[0] == 4
    for index, value in enumerate(extras):
        assert raw[f"source_extra_{index+1}"].iloc[0] == value
    assert not (tmp_path / "nested").exists()


def test_parser_errors_and_dates(tmp_path):
    with pytest.raises(ValueError):
        date_from_filename("V23366.drops.txt")
    assert str(date_from_filename("V24366.drops.txt").date()) == "2024-12-31"
    with pytest.raises(ValueError):
        date_from_filename("V23022.drops.txt", date="2023-01-23")
    with pytest.raises(ValueError, match="header"):
        lite.ingest_raw(write_input(tmp_path, "head\n"))
    for invalid in [
        NUMERIC.replace("1 0 0", "25 0 0", 1),
        NUMERIC + " 99",
        NUMERIC.replace("10000", "NaN"),
    ]:
        with pytest.raises(ValueError, match="Line 3"):
            lite.ingest_raw(write_input(tmp_path, "head\nhead\n" + invalid))
    archive_path = tmp_path / "many.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("V23022.drops.txt", "x")
        archive.writestr("V23023.drops.txt", "x")
    with pytest.raises(ValueError, match="exactly one"):
        lite.ingest_raw(archive_path)


def test_analytic_moments_and_dsd(tmp_path):
    raw = lite.ingest_raw(write_input(tmp_path))
    unchanged = raw.copy(deep=True)
    parameters, dsd = lite.calculate_products(raw)
    pd.testing.assert_frame_equal(raw, unchanged)
    assert "accepted" not in raw
    minute = parameters.sel(time="2023-01-22 01:00", velocity="measured")
    # Twelve 1-mm drops, each contributing 1/(0.01 m² * 4 m/s * 60 s).
    concentration = 12 / (0.01 * 4 * 60)
    np.testing.assert_allclose(minute.number_concentration, concentration)
    np.testing.assert_allclose(minute.reflectivity, concentration)
    np.testing.assert_allclose(
        minute.liquid_water_content, np.pi / 6 * 1e-3 * concentration
    )
    np.testing.assert_allclose(
        minute.rain_rate, 12 * np.pi / 6 * (1e-3) ** 3 / (0.01 * 60) * 3.6e6
    )
    np.testing.assert_allclose(minute.mass_weighted_diameter, 1)
    np.testing.assert_allclose(minute.diameter_std, 0)
    np.testing.assert_allclose(
        dsd.dsd.sel(time="2023-01-22 01:00", velocity="measured").sum() * 0.2,
        concentration,
    )
    np.testing.assert_array_equal(
        parameters.rain_rate.sel(velocity="measured"),
        parameters.rain_rate.sel(velocity="terminal"),
    )
    assert int(dsd.bin_drop_count.sum()) == 12
    assert int(parameters.output_valid.sum()) == 1


def test_clock_qc_and_empty(tmp_path):
    raw = lite.ingest_raw(
        write_input(
            tmp_path,
            "h\nh\n"
            + NUMERIC
            + "\n"
            + NUMERIC.replace("1 0 0", "0 59 59", 1)
            + "\n"
            + NUMERIC,
        )
    )
    assert prepare_drops(raw).accepted.tolist() == [True, False, False]
    empty = lite.ingest_raw(write_input(tmp_path, "h\nh\n"))
    parameters, dsd = lite.calculate_products(empty)
    assert np.isnan(parameters.rain_rate).all()
    assert parameters.drop_count.sum() == 0
    assert dsd.dsd.sum() == 0


def test_export_and_plots(tmp_path):
    parameters, dsd = lite.calculate_products(lite.ingest_raw(write_input(tmp_path)))
    paths = lite.save_products(parameters, dsd, tmp_path / "output")
    with xr.open_dataset(paths["measured_netcdf"]) as saved:
        xr.testing.assert_equal(saved.rain_rate, parameters.rain_rate.sel(velocity=["measured"]))
    fig = lite.plot_integral_parameters(parameters, output_dir=tmp_path / "output")
    assert len(fig.axes) == 8
    assert not any("Linear" in axis.get_title() for axis in fig.axes)
    assert fig.axes[3].lines[0].get_color() != fig.axes[3].lines[1].get_color()
    assert fig.axes[3].yaxis.label.get_fontsize() >= 15
    assert "2023-01-22" in fig._suptitle.get_text()
    spectrum = lite.plot_dsd(dsd, output_dir=tmp_path / "output")
    assert "Literature" in spectrum.axes[0].get_title()
    assert len(list((tmp_path / "output/Plots").rglob("*.png"))) == 2
    plt.close(fig)
    plt.close(spectrum)
    with pytest.raises(ValueError):
        lite.plot_dsd(dsd, velocity="rainnasa")


def test_separate_velocity_exports_and_plots(tmp_path):
    import json
    from gv_tools.cli_2dvd import plot_main
    parameters, dsd = lite.calculate_products(lite.ingest_raw(write_input(tmp_path)))
    paths = lite.save_products(parameters, dsd, tmp_path / "output")
    assert len(paths) == len(set(paths.values())) == 8
    for velocity in ("measured", "terminal"):
        with xr.open_dataset(paths[f"{velocity}_netcdf"]) as saved:
            assert saved.velocity.values.tolist() == [velocity]
            assert saved.attrs["velocity_basis"] == velocity
            xr.testing.assert_equal(saved.dsd, dsd.dsd.sel(velocity=[velocity]))
            xr.testing.assert_equal(saved.number_concentration, parameters.number_concentration.sel(velocity=[velocity]))
        for kind in ("integral_csv", "dsd_csv"):
            assert set(pd.read_csv(paths[f"{velocity}_{kind}"]).velocity) == {velocity}
        assert json.loads(paths[f"{velocity}_provenance"].read_text())["velocity_basis"] == velocity
        for plot, product in ((lite.plot_integral_parameters, parameters), (lite.plot_dsd, dsd)):
            fig = plot(product, velocity=velocity, output_dir=tmp_path / "output", dpi=30)
            plt.close(fig)
    pngs = list((tmp_path / "output/Plots").rglob("*.png"))
    assert len(pngs) == 4
    assert sum("measured_velocity" in p.name for p in pngs) == 2
    assert sum("terminal_velocity" in p.name for p in pngs) == 2
    assert plot_main([str(paths["terminal_netcdf"]), "--output-dir", str(tmp_path / "cli")]) == 0
    assert all("terminal_velocity" in p.name for p in (tmp_path / "cli").rglob("*.png"))
    with pytest.raises(SystemExit) as error:
        plot_main([str(paths["terminal_netcdf"]), "--velocity", "measured"])
    assert error.value.code == 2


def test_public_namespaces_and_lazy_plot_import():
    import subprocess
    import sys
    from gv_tools import io, core, graph, read_2dvd
    assert read_2dvd is io.ingest_raw
    assert core.calculate_products is lite.calculate_products
    assert callable(graph.plot_dsd)
    subprocess.run([sys.executable, "-c",
        "import sys; import gv_tools; from gv_tools import io, core, graph, py_2dvd; "
        "assert 'matplotlib.pyplot' not in sys.modules"], check=True,
        cwd=__import__("pathlib").Path(lite.__file__).parents[1])


def test_process_cli_and_provenance(tmp_path):
    from gv_tools import __version__
    from gv_tools.cli_2dvd import process_main
    output = tmp_path / "processed"
    assert process_main([str(write_input(tmp_path)), "--output-dir", str(output)]) == 0
    files = list(output.rglob("*.nc"))
    assert len(files) == 2
    with xr.open_dataset(files[0]) as saved:
        assert saved.attrs["package"] == "gv_tools"
        assert saved.attrs["package_version"] == __version__
        assert saved.attrs["source_implementation"] == "py_2dvd_lite 0.1.2"
