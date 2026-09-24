import gv_tools
import pandas as pd
import xarray as xr


HEADER = "Timestamp,AirTemp_Act,RelHum_Act,AbsPress_Act,WindSpd_Act,WindDir_Act,Precip_Abs,Precip_Diff,Precip_Intens,Precip_Type,GlobRad_Act\n"


def test_read_aio_routes_and_merges_lufft_ws800(tmp_path):
    first = tmp_path / "PIERS0042_WS800_20260831_a.csv"
    second = tmp_path / "PIERS0042_WS800_20260831_b.csv"
    first.write_text(HEADER + "20260831000004,25.1,87.4,1019.1,1.3,247.4,55.2,0,0,0,0\n")
    second.write_text(HEADER + "20260831000019,25.2,86.9,1019.2,0.7,223.3,55.2,0,0,0,0\n")

    frame = gv_tools.io.read_aio([first, second])

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["time", "air_temperature", "relative_humidity", "air_pressure", "wind_speed", "wind_from_direction"]
    assert frame.attrs["model"] == "Lufft WS800"
    assert frame.air_temperature.tolist() == [25.1, 25.2]
    dataset = gv_tools.io.read_aio([first, second], XARRAY=True)
    assert isinstance(dataset, xr.Dataset)
    assert dataset.sizes["time"] == 2


def test_read_aio_raw_lufft_ws800(tmp_path):
    source = tmp_path / "PIERS0042_WS800_20260831_daily.csv"
    source.write_text(HEADER + "20260831000004,25.1,87.4,1019.1,1.3,247.4,55.2,0,0,0,0\n")
    raw = gv_tools.io.read_aio_raw([source])
    assert raw.observations.shape[0] == 1
    assert raw.source_files == (source.name,)


def test_read_ws800_full_returns_pandas_or_xarray(tmp_path):
    source = tmp_path / "PIERS0042_WS800_20260831_daily.csv"
    source.write_text(HEADER + "20260831000004,25.1,87.4,1019.1,1.3,247.4,55.2,0.1,0.4,1,12\n")
    frame = gv_tools.io.read_ws800_full([source])
    dataset = gv_tools.io.read_ws800_full([source], XARRAY=True)
    assert isinstance(frame, pd.DataFrame)
    assert "precipitation_amount" in frame
    assert "global_radiation" in frame
    assert isinstance(dataset, xr.Dataset)
    assert "precipitation_amount" in dataset
    assert "global_radiation" in dataset


def test_read_ws800_full_merges_file_list(tmp_path):
    first = tmp_path / "PIERS0042_WS800_20260830_daily.csv"
    second = tmp_path / "PIERS0042_WS800_20260831_daily.csv"
    first.write_text(HEADER + "20260830235949,25.0,88,1019,1,200,55,0,0,0,0\n")
    second.write_text(HEADER + "20260831000004,25.1,87,1019,2,210,55,0,0,0,0\n")
    frame = gv_tools.io.read_ws800_full([first, second])
    dataset = gv_tools.io.read_ws800_full([first, second], XARRAY=True)
    assert frame.time.dt.strftime("%Y-%m-%d").tolist() == ["2026-08-30", "2026-08-31"]
    assert dataset.sizes["time"] == 2
