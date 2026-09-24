import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

from gv_tools import (
    InstrumentMetadata,
    create_adapter,
    plot_aio_quicklook,
    plot_aio_wind_rose,
    plot_parsivel_quicklook,
    plot_radar_ppi_quicklook,
    plot_radar_rhi_quicklook,
    plot_radar_bb_zdr_calibration,
    plot_mrr_time_height_quicklook,
    plot_ws800_full_quicklook,
)
from test_metone_aio_adapter import HEADER, ROW, metadata, sample


def test_metone_quicklooks(tmp_path):
    rows = "".join(ROW.replace("04+0000.", f"04+00{minute:02d}.").replace("06+0070.", f"06+{minute * 30:04d}.") for minute in range(6))
    dataset = create_adapter("metone_aio", metadata()).read(sample(tmp_path / "AIO_GAIL_202501.dat", HEADER + rows))["2025-01-01"]
    overview = tmp_path / "overview.png"
    windrose = tmp_path / "windrose.png"
    figure = plot_aio_quicklook(dataset, overview)
    wind_figure, wind_axis, frequencies = plot_aio_wind_rose(dataset, windrose)
    assert overview.is_file() and windrose.is_file()
    assert np.isclose(frequencies.sum(), 100.0)
    assert len(figure.axes) == 7
    assert figure.axes[0].get_ylabel() == "Temperature (°C)"
    assert figure.axes[1].get_ylabel() == "Pressure (hPa)"
    assert figure.axes[2].get_ylabel() == "Humidity (%)"
    assert figure.axes[3].get_ylabel() == "Wind speed (m/s)"
    assert figure.axes[4].get_ylabel() == "Wind direction (°)"
    assert figure.axes[5].name == "polar"
    assert "Observations:" in figure.axes[6].texts[0].get_text()
    for plot_title in (figure._suptitle.get_text(), wind_axis.get_title()):
        assert "GAIL" in plot_title
        assert "AIO_GAIL" in plot_title
        assert "2025-01-01" in plot_title


def test_aio_quicklook_show_or_save(tmp_path, capsys):
    dataset = parsivel_dataset().rename(
        {
            "radar_reflectivity": "air_temperature",
            "precipitation_rate": "air_pressure",
            "liquid_water_content": "relative_humidity",
            "number_concentration": "wind_speed",
            "drop_count": "wind_from_direction",
        }
    )
    destination = tmp_path / "aio.png"
    figure = plot_aio_quicklook(dataset, savefig=destination)
    assert destination.is_file()
    assert capsys.readouterr().out.strip() == f"Plot saved to {destination}"
    import matplotlib.pyplot as plt

    assert figure.number not in plt.get_fignums()


def test_ws800_full_quicklook_accepts_dataframe_and_dataset(tmp_path, capsys):
    times = pd.date_range("2026-08-31", periods=12, freq="10min")
    frame = pd.DataFrame(
        {
            "time": times,
            "air_temperature": np.linspace(20, 24, 12),
            "air_temp_min": np.linspace(19, 23, 12),
            "air_temp_max": np.linspace(21, 25, 12),
            "dew_point_temperature": np.linspace(16, 18, 12),
            "dew_point_min": np.linspace(15, 17, 12),
            "relative_humidity": np.linspace(80, 60, 12),
            "rel_hum_min": np.linspace(75, 55, 12),
            "abs_hum_act": np.linspace(10, 8, 12),
            "abs_hum_min": np.linspace(9, 7, 12),
            "mixing_ratio_act": np.linspace(8, 6, 12),
            "lightning_count": np.arange(12),
            "lightning_distance": np.linspace(30, 5, 12),
        }
    )
    frame.attrs.update(site_id="WFF", instrument_id="PIERS0042", model="Lufft WS800")
    destination = tmp_path / "ws800.png"
    figure = plot_ws800_full_quicklook(
        frame, savefig=destination, colors={"air_temperature": "crimson"}
    )
    assert destination.is_file()
    assert capsys.readouterr().out.strip() == f"Plot saved to {destination}"
    assert len(figure.axes) == 6
    assert [axis.get_title(loc="left") for axis in figure.axes] == [
        "Air temperature", "Dew point", "Relative humidity",
        "Absolute humidity", "Mixing ratio", "Lightning",
    ]
    assert [len(axis.lines) for axis in figure.axes] == [3, 2, 2, 2, 1, 2]
    assert all(axis.get_legend()._loc == 0 for axis in figure.axes)
    assert figure.axes[0].lines[0].get_color() == "crimson"
    assert all(value in figure._suptitle.get_text() for value in ("WFF", "PIERS0042", "2026-08-31"))

    dataset = frame.set_index("time").to_xarray()
    dataset.attrs.update(frame.attrs)
    displayed = plot_ws800_full_quicklook(dataset)
    assert displayed is not None
    import matplotlib.pyplot as plt

    plt.close(displayed)


def test_ws800_full_quicklook_marks_missing_optional_families():
    frame = pd.DataFrame({
        "time": pd.date_range("2026-08-31", periods=2, freq="h"),
        "air_temperature": [20.0, 21.0],
    })
    figure = plot_ws800_full_quicklook(frame)
    assert len(figure.axes) == 6
    assert all(axis.texts[0].get_text() == "No matching fields" for axis in figure.axes[1:])
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_radar_ppi_quicklook_uses_selected_sweep_and_overlays(tmp_path, capsys):
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_ppi_radar(40, 36, 2)
    values = np.ma.array(np.tile(np.linspace(-10, 50, radar.ngates), (radar.nrays, 1)))
    radar.add_field(
        "reflectivity",
        {"data": values, "units": "dBZ", "long_name": "equivalent reflectivity factor"},
    )
    radar.metadata.update(instrument_name="NPOL", site_name="WFF", gv_tools_scan_type="PPI")
    destination = tmp_path / "npol_ppi.png"
    figure = plot_radar_ppi_quicklook(
        radar,
        sweep=1,
        savefig=destination,
        max_range_km=20,
        range_ring_interval_km=5,
        range_ring_color="magenta",
        radial_spoke_interval_deg=90,
        radial_spoke_color="cyan",
        gridline_interval_deg=0.25,
        add_map_features=False,
        add_borders=False,
        map_resolution="110m",
    )
    assert destination.is_file()
    assert capsys.readouterr().out.strip().endswith(f"Plot saved to {destination}")
    assert "NPOL" in figure.axes[0].get_title()
    assert "Sweep 1 of 1" in figure.axes[0].get_title()
    assert "Elevation" in figure.axes[0].get_title()
    assert "WFF" in figure.axes[0].get_title()
    assert pyart.util.datetime_from_radar(radar).strftime("%Y-%m-%d") in figure.axes[0].get_title()


def test_radar_ppi_quicklook_validates_scan_and_sweep():
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_ppi_radar(10, 12, 1)
    radar.add_field("reflectivity", {"data": np.ma.zeros((radar.nrays, radar.ngates)), "units": "dBZ"})
    with pytest.raises(ValueError, match="sweep"):
        plot_radar_ppi_quicklook(radar, sweep=2)
    radar.scan_type = "rhi"
    with pytest.raises(ValueError, match="requires a PPI"):
        plot_radar_ppi_quicklook(radar)


def test_radar_ppi_quicklook_plots_xradar_ingest(tmp_path):
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("xradar")
    from gv_tools import ingest_radar

    source = pyart.testing.make_empty_ppi_radar(20, 36, 2)
    values = np.ma.array(np.tile(np.linspace(-10, 50, source.ngates), (source.nrays, 1)))
    source.add_field("reflectivity", {"data": values, "units": "dBZ"})
    radar_file = tmp_path / "xradar_ppi.nc"
    pyart.io.write_cfradial(radar_file, source)

    radar = ingest_radar(radar_file, XRADAR=True)
    destination = tmp_path / "xradar_ppi.png"
    figure = plot_radar_ppi_quicklook(
        radar,
        sweep=1,
        savefig=destination,
        add_map_features=False,
        add_borders=False,
    )

    assert destination.is_file()
    assert "Sweep 1 of 1" in figure.axes[0].get_title()


def test_radar_rhi_quicklook_plots_symbols_and_selected_sweep(tmp_path, capsys):
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_rhi_radar(40, 30, 2)
    values = np.ma.array(np.tile(np.linspace(-10, 50, radar.ngates), (radar.nrays, 1)))
    radar.add_field("reflectivity", {"data": values, "units": "dBZ", "long_name": "reflectivity"})
    radar.metadata.update(instrument_name="NPOL", site_name="WFF", gv_tools_scan_type="RHI")
    destination = tmp_path / "npol_rhi.png"
    figure = plot_radar_rhi_quicklook(
        radar,
        sweep=1,
        savefig=destination,
        max_range_km=20,
        max_height_km=10,
        symbols=[
            {"Location": (5, 2), "Color": "gold", "Symbol": "*", "Size": 180},
            {"Location": (12, 4), "Color": "cyan", "Symbol": "^", "Size": 90},
        ],
    )
    assert destination.is_file()
    assert capsys.readouterr().out.strip().endswith(f"Plot saved to {destination}")
    assert "Sweep 1 of 1" in figure.axes[0].get_title()
    assert "Azimuth" in figure.axes[0].get_title()
    assert all(value in figure.axes[0].get_title() for value in ("WFF", "NPOL"))
    assert pyart.util.datetime_from_radar(radar).strftime("%Y-%m-%d") in figure.axes[0].get_title()
    assert len(figure.axes[0].collections) >= 3


def test_radar_rhi_quicklook_plots_xradar_ingest(tmp_path):
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("xradar")
    from gv_tools import ingest_radar

    source = pyart.testing.make_empty_rhi_radar(30, 24, 2)
    values = np.ma.array(np.tile(np.linspace(-10, 40, source.ngates), (source.nrays, 1)))
    source.add_field("reflectivity", {"data": values, "units": "dBZ"})
    radar_file = tmp_path / "xradar_rhi.nc"
    pyart.io.write_cfradial(radar_file, source)
    radar = ingest_radar(radar_file, XRADAR=True)
    destination = tmp_path / "xradar_rhi.png"
    plot_radar_rhi_quicklook(radar, sweep=1, savefig=destination)
    assert destination.is_file()


def test_radar_rhi_quicklook_validates_scan_and_symbols():
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_rhi_radar(10, 12, 1)
    radar.add_field("reflectivity", {"data": np.ma.zeros((radar.nrays, radar.ngates)), "units": "dBZ"})
    with pytest.raises(ValueError, match="symbols"):
        plot_radar_rhi_quicklook(radar, symbols={"Location": (1, 2)})
    radar.scan_type = "ppi"
    with pytest.raises(ValueError, match="requires an RHI"):
        plot_radar_rhi_quicklook(radar)


def test_radar_bb_zdr_calibration_estimates_bias_and_correction(tmp_path, capsys):
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_ppi_radar(40, 36, 1)
    radar.elevation["data"][:] = 90.0
    radar.metadata.update(instrument_name="NPOL", site_name="WFF", gv_tools_scan_type="BB")
    rng = np.random.default_rng(42)
    zdr = np.ma.array(0.35 + rng.normal(0.0, 0.08, (radar.nrays, radar.ngates)))
    reflectivity = np.ma.array(np.full((radar.nrays, radar.ngates), 20.0))
    radar.add_field("differential_reflectivity", {"data": zdr, "units": "dB"})
    radar.add_field("reflectivity", {"data": reflectivity, "units": "dBZ"})
    destination = tmp_path / "bb_zdr_calibration.png"
    figure = plot_radar_bb_zdr_calibration(
        radar,
        savefig=destination,
        min_range_km=0.1,
        reflectivity_bounds_dbz=(10, 30),
    )
    result = figure.gv_tools_zdr_calibration
    assert destination.is_file()
    assert capsys.readouterr().out.strip().endswith(f"Plot saved to {destination}")
    assert result["sample_count"] > 100
    assert result["bias_db"] == pytest.approx(0.35, abs=0.02)
    assert result["additive_correction_db"] == pytest.approx(-0.35, abs=0.02)
    assert "Birdbath ZDR Calibration" in figure._suptitle.get_text()
    assert figure.axes[0].get_legend()._loc == 0
    assert figure.axes[1].get_legend()._loc == 0
    assert all(value in figure._suptitle.get_text() for value in ("WFF", "NPOL"))
    assert pyart.util.datetime_from_radar(radar).strftime("%Y-%m-%d") in figure._suptitle.get_text()


def test_radar_bb_zdr_calibration_validates_scan_and_sample_count():
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_ppi_radar(10, 12, 1)
    radar.add_field("ZDR", {"data": np.ma.zeros((radar.nrays, radar.ngates)), "units": "dB"})
    with pytest.raises(ValueError, match="requires a BB"):
        plot_radar_bb_zdr_calibration(radar)
    radar.metadata["gv_tools_scan_type"] = "BB"
    with pytest.raises(ValueError, match="valid ZDR samples"):
        plot_radar_bb_zdr_calibration(radar, min_range_km=10)


def test_radar_bb_zdr_calibration_plots_xradar_ingest(tmp_path):
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("xradar")
    from gv_tools import ingest_radar

    source = pyart.testing.make_empty_ppi_radar(20, 24, 1)
    source.elevation["data"][:] = 90.0
    source.add_field("ZDR", {"data": np.ma.array(np.full((source.nrays, source.ngates), 0.2)), "units": "dB"})
    radar_file = tmp_path / "birdbath.nc"
    pyart.io.write_cfradial(radar_file, source)
    radar = ingest_radar(radar_file, XRADAR=True)
    destination = tmp_path / "xradar_bb.png"
    figure = plot_radar_bb_zdr_calibration(radar, min_range_km=0.1, savefig=destination)
    assert destination.is_file()
    assert figure.gv_tools_zdr_calibration["additive_correction_db"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    ("model", "fields"),
    [("MRR2", ["MRR_RR", "MRR_LWC"]), ("MRRPro", ["RR", "LWC"])],
)
def test_mrr_time_height_quicklook_plots_requested_fields(tmp_path, capsys, model, fields):
    times = pd.date_range("2026-07-22", periods=12, freq="10min")
    heights = np.arange(8) * 100.0
    values = np.arange(times.size * heights.size, dtype=float).reshape(times.size, heights.size) / 20.0
    if model == "MRR2":
        dataset = xr.Dataset(
            {
                "MRR_H": (("time", "MRR rangegate"), np.broadcast_to(heights, values.shape)),
                "MRR_RR": (("time", "MRR rangegate"), values),
                "MRR_LWC": (("time", "MRR rangegate"), values / 10.0),
            },
            coords={"time": times},
            attrs={"mrr_model": model, "site_id": "WFF"},
        )
    else:
        dataset = xr.Dataset(
            {"RR": (("time", "range"), values), "LWC": (("time", "range"), values / 10.0)},
            coords={"time": times, "range": ("range", heights, {"units": "m"})},
            attrs={"mrr_model": model, "site_id": "WFF"},
        )
    for name in fields:
        dataset[name].attrs.update(long_name=name, units="test units")
    destination = tmp_path / f"{model}_profiles.png"
    figure = plot_mrr_time_height_quicklook(
        dataset,
        fields,
        savefig=destination,
        cmaps={fields[0]: "turbo", fields[1]: "viridis"},
        colorbar_bounds={fields[0]: (0, 5), fields[1]: (0, 0.5)},
        height_range_km=(0, 0.7),
        field_labels={fields[0]: "Rain rate", fields[1]: "Liquid water"},
    )
    assert destination.is_file()
    assert capsys.readouterr().out.strip().endswith(f"Plot saved to {destination}")
    assert len(figure.axes) == 4
    assert model in figure._suptitle.get_text()
    assert all(value in figure._suptitle.get_text() for value in ("WFF", "2026-07-22"))


def test_mrr_time_height_quicklook_validates_fields():
    dataset = xr.Dataset(
        {"RR": (("time", "range"), [[1.0, 2.0]])},
        coords={"time": [np.datetime64("2026-07-22")], "range": [0.0, 100.0]},
        attrs={"mrr_model": "MRRPro"},
    )
    with pytest.raises(ValueError, match="non-empty"):
        plot_mrr_time_height_quicklook(dataset, [])
    with pytest.raises(ValueError, match="unavailable"):
        plot_mrr_time_height_quicklook(dataset, ["LWC"])


def parsivel_dataset():
    time = pd.date_range("2026-07-21", periods=24, freq="h")
    diameter = [0.2, 0.5, 1.0]
    orders = [0, 1, 2, 3, 4, 5, 6]
    return xr.Dataset(
        {
            "drop_size_distribution": (("time", "drop_diameter"), np.arange(72).reshape(24, 3) + 1.0),
            "radar_reflectivity": ("time", np.linspace(5, 30, 24)),
            "precipitation_rate": ("time", np.linspace(0, 10, 24)),
            "liquid_water_content": ("time", np.linspace(0, 2, 24)),
            "number_concentration": ("time", np.linspace(10, 500, 24)),
            "drop_count": ("time", np.arange(24)),
            "drop_size_moment": (("time", "moment_order"), np.arange(168).reshape(24, 7)),
        },
        coords={"time": time, "drop_diameter": diameter, "moment_order": orders},
        attrs={"site_id": "WFF", "model": "Parsivel2"},
    )


def test_parsivel_quicklook_has_six_aligned_daily_panels(tmp_path):
    output = tmp_path / "parsivel.png"
    dataset = parsivel_dataset()
    dataset["drop_size_distribution"][0, 0] = 0
    figure = plot_parsivel_quicklook(
        dataset,
        output,
        cmap="plasma",
        colorbar_bounds=(2, 60),
        colorbar_location="left",
        dsd_yrange=(0.1, 1.1),
    )
    panels = figure.axes[:6]

    assert output.is_file()
    assert len(panels) == 6
    assert panels[0].get_title() == "WFF | Parsivel2 | 2026-07-21 — Parsivel quicklook"
    assert np.isclose(panels[-1].get_xlim()[1] - panels[-1].get_xlim()[0], 1.0)
    assert isinstance(panels[-1].xaxis.get_major_locator(), mdates.HourLocator)
    assert all(any(line.get_visible() for line in axis.get_xgridlines()) for axis in panels)
    assert np.allclose([axis.get_position().x0 for axis in panels], panels[0].get_position().x0)
    assert np.allclose([axis.get_position().width for axis in panels], panels[0].get_position().width)
    assert np.allclose(panels[0].get_ylim(), (0.1, 1.1))
    mesh = panels[0].collections[0]
    assert mesh.cmap.name == "plasma"
    assert (mesh.norm.vmin, mesh.norm.vmax) == (2, 60)
    assert np.ma.getmaskarray(mesh.get_array()).any()


@pytest.mark.parametrize("location", ["left", "right", "top", "bottom"])
def test_parsivel_quicklook_accepts_colorbar_locations(location):
    figure = plot_parsivel_quicklook(parsivel_dataset(), colorbar_location=location)
    assert len(figure.axes[0].child_axes) == 1


def test_parsivel_quicklook_validates_plot_ranges():
    with pytest.raises(ValueError, match="colorbar_bounds"):
        plot_parsivel_quicklook(parsivel_dataset(), colorbar_bounds=(5, 1))
    with pytest.raises(ValueError, match="dsd_yrange"):
        plot_parsivel_quicklook(parsivel_dataset(), dsd_yrange=(1, 1))


def test_parsivel_quicklook_savefig_requires_full_filename(tmp_path, capsys):
    destination = tmp_path / "figures" / "parsivel.png"
    figure = plot_parsivel_quicklook(parsivel_dataset(), savefig=destination)
    assert destination.is_file()
    assert figure is not None
    assert capsys.readouterr().out.strip() == f"Plot saved to {destination}"
    with pytest.raises(ValueError, match="full path.*filename"):
        plot_parsivel_quicklook(parsivel_dataset(), savefig="parsivel.png")
    with pytest.raises(ValueError, match="either output or savefig"):
        plot_parsivel_quicklook(parsivel_dataset(), destination, savefig=destination)


def test_parsivel_quicklook_registers_display_figure_only_when_not_saving(tmp_path):
    import matplotlib.pyplot as plt

    existing = plt.get_fignums()
    figure = plot_parsivel_quicklook(parsivel_dataset())
    assert figure.number in plt.get_fignums()
    plt.close(figure)
    assert plt.get_fignums() == existing

    plot_parsivel_quicklook(parsivel_dataset(), savefig=tmp_path / "saved.png")
    assert plt.get_fignums() == existing


def test_parsivel_notebook_relies_on_single_backend_display():
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "GV_Tools_Ingest_Demonstration.ipynb").read_text()
    )
    source = "".join(
        cell_source
        for cell in notebook["cells"]
        for cell_source in cell.get("source", [])
        if "parsivel_figure" in cell_source
    )
    assert "display(parsivel_figure)" not in source
    assert "plt.close(parsivel_figure)" not in source
